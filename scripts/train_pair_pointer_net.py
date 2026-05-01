"""
PPO training for the SplitPointerNet / SpatialPointerNet bin-packing agent.

Architecture
------------
No pretrained backbone. Trains entirely from random init with PPO + rollout-TD advantage.

At each step, for every remaining part i:
  save board → place part i → render resulting board SDF → restore
  pair_images = env.preview_pair_images()  →  (N, 2, 128, 128)

SplitPointerNet encodes board and parts separately:
  board  = pair_images[0, 0]                        →  BoardCNN (once per step)  →  (1, 128)
  deltas = pair_images[:, 1] - pair_images[:, 0]   →  PartCNN  →  (N, 128)
  fused  = MLP(cat[board_feat, part_feats])         →  (N, 128)
  ctx    = PartTransformer(fused, mask)             →  (N, 128)
  logits = part_head(ctx)                           →  (N,)

Episode loop
------------
  1. Sample N parts from the shape library.
  2. At each step:
       a. pair_images = env.preview_pair_images()  →  (N, 2, 128, 128)
       b. part_logits = model(pair_images, mask)
       c. Sample part_id ~ Categorical(part_logits).
       d. Ask U-Nesting to place the part.
  3. Per-step reward: R_t = packing_density(s_{t+1}) − packing_density(s_t)
     Value baseline: V(s) = neural value head on SplitPointerNet, trained on actual returns
     Advantage: GAE(λ=0.95) using actual rewards and bootstrapped values
  4. Loss = PPO_loss − entropy_coef·entropy + value_coef·MSE(V_pred, V_target)

Usage:
    maturin develop --release --features python-bindings
    python scripts/train_pair_pointer_net.py --json data/library.json --n-parts 20
"""

import argparse
import contextlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from u_nesting_gym import UNestingGymEnv
from pointer_net import SplitPointerNet, SpatialPointerNet

import wandb


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast(device: torch.device):
    """
    bfloat16 autocast on CUDA (Ampere+ gets Tensor Core speedup, no GradScaler needed).
    No-op on MPS (limited op coverage) and CPU.
    """
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _curriculum_n_parts(episode: int, args: argparse.Namespace) -> int:
    if args.n_parts_start >= args.n_parts:
        return args.n_parts
    t = min(1.0, episode / args.curriculum_episodes)
    return round(args.n_parts_start + t * (args.n_parts - args.n_parts_start))


def _build_remaining_mask(remaining_ids: list[int], n_parts: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(n_parts, device=device)
    for idx in remaining_ids:
        mask[idx] = 1.0
    return mask


def _sample_action(
    model: torch.nn.Module,
    obs_t: torch.Tensor,
    mask: torch.Tensor,
    greedy: bool,
) -> tuple:
    """
    Sample a (part, rotation) action from the model.

    Returns (part_id, part_dist, part_logits, rot_id, rot_dist, value).
    rot_id / rot_dist are None when the model does not have a rotation_head.
    value is None when the model does not have a value_head.
    """
    output = model(obs_t, mask)
    value = None
    if isinstance(output, tuple):
        if len(output) == 4:
            part_logits, rot_feats, ctx, value = output
        else:
            part_logits, rot_feats, ctx = output
    else:
        part_logits, rot_feats, ctx = output, None, None

    part_dist = Categorical(logits=part_logits)
    part_id   = part_logits.argmax() if greedy else part_dist.sample()

    rot_id = rot_dist = None
    if rot_feats is not None and hasattr(model, "rotation_head"):
        rot_logits = model.rotation_head(ctx[part_id], rot_feats[part_id])
        rot_dist   = Categorical(logits=rot_logits)
        rot_id     = rot_logits.argmax() if greedy else rot_dist.sample()

    return part_id, part_dist, part_logits, rot_id, rot_dist, value


def _action_log_prob_and_entropy(
    part_dist: Categorical,
    part_id: torch.Tensor,
    rot_dist: Categorical | None = None,
    rot_id: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Returns (joint_log_prob, part_entropy, rot_entropy_or_None)."""
    lp      = part_dist.log_prob(part_id)
    ent     = part_dist.entropy()
    rot_ent = None
    if rot_dist is not None and rot_id is not None:
        lp      = lp + rot_dist.log_prob(rot_id)
        rot_ent = rot_dist.entropy()
    return lp, ent, rot_ent


def _polygon_area(polygon: list) -> float:
    """Signed shoelace area (absolute value returned)."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _greedy_target(env: "UNestingGymEnv", remaining: list[int]) -> int:
    """Episode ID of the largest remaining part by polygon area."""
    episode_geoms = env._episode_geoms()
    return max(remaining, key=lambda ep_id: _polygon_area(episode_geoms[ep_id]["polygon"]))


def _run_episode(
    env: UNestingGymEnv,
    model: torch.nn.Module,
    lib_ids: list[int],
    device: torch.device,
    greedy: bool = False,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    capture_snapshots: bool = False,
    rotations_rad: list[float] | None = None,
) -> tuple[list, list, list, list[float], float, list, dict, list]:
    env.reset(lib_ids)
    n_parts = len(lib_ids)

    log_probs:      list[torch.Tensor] = []
    entropies:      list[torch.Tensor] = []
    rot_entropies:  list[torch.Tensor] = []
    observations:   list[tuple]        = []
    snapshots:      list               = []
    R_t_list:       list[float]        = []
    value_preds_ep: list[torch.Tensor] = []

    prev_density = env.packing_density()
    skip_ids: set[int] = set()

    for _ in range(n_parts):
        remaining = [r for r in env.remaining_item_ids() if r not in skip_ids]
        if not remaining:
            break

        mask = _build_remaining_mask(remaining, n_parts, device)

        if rotations_rad is not None:
            obs_np = env.preview_images_per_rotation(rotations_rad)
        else:
            obs_np = env.preview_pair_images()
        obs_t = torch.from_numpy(obs_np).to(device, non_blocking=True)

        part_id, part_dist, _, rot_id, rot_dist, value = _sample_action(model, obs_t, mask, greedy)

        if rotations_rad is not None and rot_id is not None:
            success = env.place_with_rotation(part_id.item(), rotations_rad[rot_id.item()])
        else:
            success = env.place_anywhere(part_id.item())

        if not success:
            skip_ids.add(part_id.item())
            continue

        value_preds_ep.append(
            value.detach() if value is not None else torch.tensor(0.0, device=device)
        )

        if not greedy:
            lp, ent, rot_ent = _action_log_prob_and_entropy(
                part_dist, part_id, rot_dist, rot_id)
            log_probs.append(lp)
            entropies.append(ent)
            if rot_ent is not None:
                rot_entropies.append(rot_ent)
            gt = _greedy_target(env, remaining)
            observations.append((
                obs_t.cpu(), mask.cpu(), part_id.cpu(),
                rot_id.cpu() if rot_id is not None else None,
                torch.tensor(gt),
            ))

        R_t = env.packing_density() - prev_density
        prev_density = env.packing_density()
        R_t_list.append(R_t)

        if capture_snapshots:
            snapshots.append((env.placed_polygons(), env.packing_density()))

    # Compute GAE advantages and value targets (bootstrap V=0 at terminal)
    T = len(R_t_list)
    A_t_list        = [0.0] * T
    value_targets_ep = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        v_t    = value_preds_ep[t].item() if t < len(value_preds_ep) else 0.0
        v_next = value_preds_ep[t + 1].item() if t + 1 < len(value_preds_ep) else 0.0
        delta  = R_t_list[t] + gamma * v_next - v_t
        gae    = delta + gamma * gae_lambda * gae
        A_t_list[t]         = gae
        value_targets_ep[t] = v_t + gae

    # Attach value targets to stored observations
    if not greedy and observations:
        observations = [
            (*obs, torch.tensor(vt, dtype=torch.float32))
            for obs, vt in zip(observations, value_targets_ep)
        ]

    episode_reward = env.packing_density()
    board_area     = env.plate_w * env.plate_h

    stats = {
        "R_t_mean":           float(np.mean(R_t_list))                  if R_t_list else 0.0,
        "advantage_mean":     float(np.mean(A_t_list))                  if A_t_list else 0.0,
        "advantage_pos_frac": float(np.mean([a > 0 for a in A_t_list])) if A_t_list else 0.0,
        "value_mean":         float(np.mean([v.item() for v in value_preds_ep])) if value_preds_ep else 0.0,
        "bbox_frac":          env.bbox_area() / board_area,
    }

    return log_probs, entropies, rot_entropies, A_t_list, episode_reward, snapshots, stats, observations


def _build_env(args: argparse.Namespace) -> UNestingGymEnv:
    if args.plate_width <= 150 or args.plate_height <= 150:
        print(
            f"[warn] plate {args.plate_width:.0f}×{args.plate_height:.0f} mm is very small. "
            "Aligner parts are ~50 mm wide; use e.g. --plate-width 280 --plate-height 200."
        )
    return UNestingGymEnv(
        args.json,
        plate_width=args.plate_width,
        plate_height=args.plate_height,
        sdf_clip_px=args.sdf_clip_px,
        rotations=args.rotations,
    )


def _build_model_and_optimizer(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    if args.model == "spatial":
        model = SpatialPointerNet().to(device)
    else:
        model = SplitPointerNet().to(device)
    if device.type == "cuda":
        # ~30–60% faster on NVIDIA; not supported on MPS
        model = torch.compile(model)
        print("torch.compile enabled (CUDA)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return model, optimizer


def _current_entropy_coef(episode: int, args: argparse.Namespace) -> float:
    if args.entropy_coef_final is not None:
        t = min(1.0, episode / args.episodes)
        return args.entropy_coef + t * (args.entropy_coef_final - args.entropy_coef)
    return args.entropy_coef


def _current_imitation_coef(episode: int, args: argparse.Namespace) -> float:
    if args.imitation_coef <= 0.0:
        return 0.0
    t = min(1.0, episode / args.imitation_episodes)
    return args.imitation_coef * (1.0 - t)


def _evaluate_actions(
    model: torch.nn.Module,
    observations: list[tuple],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    log_probs, part_entropies, rot_entropies, imitation_losses, value_preds = [], [], [], [], []
    for obs_entry in observations:
        obs_t, mask, part_id, rot_id, greedy_target, _value_target = obs_entry
        obs_t   = obs_t.to(device,   non_blocking=True)
        mask    = mask.to(device,    non_blocking=True)
        part_id = part_id.to(device, non_blocking=True)

        with _autocast(device):
            output = model(obs_t, mask)
            value = None
            if isinstance(output, tuple):
                if len(output) == 4:
                    part_logits, rot_feats, ctx, value = output
                else:
                    part_logits, rot_feats, ctx = output
            else:
                part_logits, rot_feats, ctx = output, None, None

            if rot_feats is not None and rot_id is not None and hasattr(model, "rotation_head"):
                rot_logits = model.rotation_head(ctx[part_id], rot_feats[part_id])
                rot_dist   = Categorical(logits=rot_logits)
                rot_entropies.append(rot_dist.entropy())
            else:
                rot_logits = rot_dist = None

        part_dist = Categorical(logits=part_logits)
        lp        = part_dist.log_prob(part_id)
        part_entropies.append(part_dist.entropy())

        if rot_dist is not None:
            lp = lp + rot_dist.log_prob(rot_id.to(device, non_blocking=True))

        log_probs.append(lp)
        imitation_losses.append(
            F.cross_entropy(part_logits.unsqueeze(0), greedy_target.unsqueeze(0).to(device, non_blocking=True))
        )
        value_preds.append(value if value is not None else torch.tensor(0.0, device=device))
    return log_probs, part_entropies, rot_entropies, imitation_losses, value_preds


def _ppo_loss(
    old_log_probs: list[torch.Tensor],
    new_log_probs: list[torch.Tensor],
    entropies: list[torch.Tensor],
    imitation_losses: list[torch.Tensor],
    advantages: list[float],
    entropy_coef: float,
    imitation_coef: float,
    n_parts: int,
    clip_eps: float,
    device: torch.device,
    rot_entropies: list[torch.Tensor] | None = None,
    rot_entropy_coef: float = 0.0,
    n_rotations: int = 8,
    value_preds: list[torch.Tensor] | None = None,
    value_targets: list[float] | None = None,
    value_coef: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    adv     = torch.tensor(advantages, dtype=torch.float32, device=device)
    old_lp  = torch.stack(old_log_probs).detach()
    new_lp  = torch.stack(new_log_probs)
    ratio   = (new_lp - old_lp).exp()
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss    = -torch.min(ratio * adv, clipped * adv).mean()
    part_entropy   = torch.stack(entropies).mean() / math.log(max(n_parts, 2))
    imitation_loss = torch.stack(imitation_losses).mean()
    total = policy_loss - entropy_coef * part_entropy + imitation_coef * imitation_loss
    rot_entropy = torch.tensor(0.0, device=device)
    if rot_entropies:
        rot_entropy = torch.stack(rot_entropies).mean() / math.log(max(n_rotations, 2))
        total = total - rot_entropy_coef * rot_entropy
    value_loss = torch.tensor(0.0, device=device)
    if value_preds and value_targets:
        vp = torch.stack(value_preds)
        vt = torch.tensor(value_targets, dtype=torch.float32, device=device)
        value_loss = F.mse_loss(vp, vt)
        total = total + value_coef * value_loss
    return total, part_entropy, imitation_loss, rot_entropy, value_loss


def _run_greedy_eval(
    env: UNestingGymEnv,
    model: torch.nn.Module,
    eval_lib_ids: list[int],
    device: torch.device,
    args: argparse.Namespace,
    rotations_rad: list[float] | None = None,
) -> tuple[float, float, list, int, int, list]:
    """
    Run the agent greedily (argmax) on eval_lib_ids.

    Uses a two-phase observation strategy when rotations are enabled:
      Phase 1 — preview_pair_images()      (N Rust calls) → part head → argmax part
      Phase 2 — preview_images_for_part()  (R Rust calls) → rotation head → argmax rot
    Total: N+R calls per step instead of N×R. No rollout_value() calls.

    After the agent episode ends, place_remaining() fills whatever is left
    (engine picks rotation). This gives two metrics at no extra episode cost:
      pure_density     — agent part+rotation selection only
      fallback_density — agent selection + engine fills remaining gaps
    """
    model.eval()
    env.reset(eval_lib_ids)
    n_parts    = len(eval_lib_ids)
    snapshots: list = []
    skip_ids:  set[int] = set()

    with torch.no_grad():
        for _ in range(n_parts):
            remaining = [r for r in env.remaining_item_ids() if r not in skip_ids]
            if not remaining:
                break

            mask = _build_remaining_mask(remaining, n_parts, device)

            # Phase 1: part selection via cheap pair images (N Rust calls)
            obs_pair = torch.from_numpy(env.preview_pair_images()).to(device, non_blocking=True)
            with _autocast(device):
                output = model(obs_pair, mask)
            if isinstance(output, tuple):
                part_logits, _, ctx = output[0], output[1], output[2]  # ignore value at [3]
            else:
                part_logits, ctx = output, None

            part_id = int(part_logits.argmax().item())

            # Phase 2: rotation selection for chosen part only (R Rust calls)
            if rotations_rad is not None and ctx is not None and hasattr(model, "rotation_head"):
                obs_part = torch.from_numpy(
                    env.preview_images_for_part(part_id, rotations_rad)
                ).to(device, non_blocking=True)
                with _autocast(device):
                    rot_feats  = model.rotation_feats_for_part(obs_part)   # (R, 128)
                    rot_logits = model.rotation_head(ctx[part_id], rot_feats)
                rot_id  = int(rot_logits.argmax().item())
                success = env.place_with_rotation(part_id, rotations_rad[rot_id])
            else:
                success = env.place_anywhere(part_id)

            if not success:
                skip_ids.add(part_id)
                continue

            snapshots.append((env.placed_polygons(), env.packing_density()))

    model.train()

    # Pure: agent part+rotation only
    pure_density  = env.packing_density()
    n_placed_pure = env.n_placed()

    # Fallback: engine fills whatever the agent couldn't place
    n_fallback    = env.place_remaining()
    fallback_density  = env.packing_density()
    n_placed_fallback = n_placed_pure + n_fallback

    remaining      = env.remaining_item_ids()
    unplaced_polys = []
    for ep_id in remaining:
        ep_geoms = env._episode_geoms()
        if ep_id < len(ep_geoms):
            geom_id  = ep_geoms[ep_id]["id"]
            previews = env._board.preview_all([geom_id])
            if previews and previews[0] is not None:
                unplaced_polys.append(previews[0])

    return pure_density, fallback_density, snapshots, n_placed_pure, n_placed_fallback, unplaced_polys


def _make_eval_figure(
    eval_snapshots: list,
    eval_n_placed: int,
    eval_lib_ids: list[int],
    episode: int,
    args: argparse.Namespace,
    greedy_polys: list,
    greedy_n_placed: int,
    greedy_density: float,
    unplaced_polys: list | None = None,
) -> plt.Figure:
    all_panels = [
        (greedy_polys, greedy_density, f"Greedy  {greedy_n_placed}/{len(eval_lib_ids)}\nd={greedy_density:.3f}")
    ] + [
        (polys, density,
         f"ep {episode+1}  agent {eval_n_placed}/{len(eval_lib_ids)}\n#{i+1}  d={density:.3f}"
         if i == 0 else f"#{i+1}  d={density:.3f}")
        for i, (polys, density) in enumerate(eval_snapshots)
    ]
    last_agent_idx = len(all_panels) - 1
    n_panels = len(all_panels)
    n_cols   = min(max(n_panels, 1), 10)
    n_rows   = math.ceil(n_panels / n_cols)
    aspect   = args.plate_height / args.plate_width
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.8, n_rows * (1.8 * aspect + 0.45) + 0.4),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (polys, density, title) in enumerate(all_panels):
        ax = axes[i // n_cols, i % n_cols]
        ax.set_visible(True)
        if polys:
            patches   = [MplPolygon(np.array(p), closed=True) for p in polys]
            facecolor = "tomato" if i == 0 else "steelblue"
            ax.add_collection(PatchCollection(
                patches, facecolor=facecolor, edgecolor="white", linewidth=0.5, alpha=0.85))
        if i == last_agent_idx and unplaced_polys:
            ghost = [MplPolygon(np.array(p), closed=True) for p in unplaced_polys]
            ax.add_collection(PatchCollection(
                ghost, facecolor="none", edgecolor="#aaaaaa", linewidth=0.7, linestyle="--"))
        ax.set_xlim(0, args.plate_width)
        ax.set_ylim(0, args.plate_height)
        ax.set_aspect("equal")
        ax.set_facecolor("#f0f0f0")
        ax.set_title(title, fontsize=6, pad=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    return fig


def _log_training_step(
    episode: int,
    args: argparse.Namespace,
    env: UNestingGymEnv,
    n_parts_ep: int,
    reward: float,
    loss: torch.Tensor,
    entropy_bonus: torch.Tensor,
    imitation_loss: torch.Tensor,
    imitation_coef: float,
    stats: dict,
    episode_time: float,
    rot_entropy: torch.Tensor | None = None,
    value_loss: torch.Tensor | None = None,
    val_expl_var: float = 0.0,
    val_mean_target: float = 0.0,
) -> None:
    n_placed = env.n_placed()

    # Density EMA (α=0.02 ≈ ~50-episode window); warm-start on first episode
    prev_ema = _log_training_step._density_ema
    density_ema = reward if prev_ema == 0.0 else prev_ema + 0.02 * (reward - prev_ema)
    _log_training_step._density_ema = density_ema

    end_char = "\n" if (episode + 1) % args.log_interval == 0 else "\r"
    print(
        f"[train] ep={episode+1:>5}/{args.episodes}  "
        f"agent={n_placed}/{n_parts_ep}  "
        f"d={reward:.4f}(ema={density_ema:.4f})  "
        f"ev={val_expl_var:+.3f}  "
        f"loss={loss.item():.4f}  ent={entropy_bonus.item():.4f}  "
        f"imit={imitation_loss.item():.4f}({imitation_coef:.2f})  "
        f"t={episode_time:.1f}s",
        end=end_char, flush=True,
    )

    hits, misses, cache_size = env._board._board.cache_stats()
    ep_misses = misses - _log_training_step._prev_misses
    ep_hits   = hits   - _log_training_step._prev_hits
    ep_total  = ep_hits + ep_misses
    ep_hit_rate = ep_hits / ep_total if ep_total > 0 else 0.0
    _log_training_step._prev_misses = misses
    _log_training_step._prev_hits   = hits

    log_dict = {
        # Agent nesting quality
        "agent/density":            reward,
        "agent/density_ema":        density_ema,
        "agent/parts_placed":       n_placed / n_parts_ep,
        # Value network health
        "value/explained_variance": val_expl_var,
        "value/mean_pred":          stats["value_mean"],
        "value/mean_target":        val_mean_target,
        "loss/value":               value_loss.item() if value_loss is not None else 0.0,
        # Policy losses
        "loss/total":               loss.item(),
        "loss/entropy":             entropy_bonus.item(),
        "loss/imitation":           imitation_loss.item(),
        "loss/imitation_coef":      imitation_coef,
        # Advantage signal
        "train/advantage_mean":     stats["advantage_mean"],
        "train/advantage_pos_frac": stats["advantage_pos_frac"],
        # Performance
        "perf/episode_time_s":      episode_time,
        "cache/hit_rate":           ep_hit_rate,
        "cache/misses_per_episode": ep_misses,
        "cache/size":               cache_size,
    }
    if rot_entropy is not None:
        log_dict["loss/rot_entropy"] = rot_entropy.item()
    wandb.log(log_dict, step=episode + 1)

_log_training_step._prev_misses  = 0
_log_training_step._prev_hits    = 0
_log_training_step._density_ema  = 0.0


def _log_eval_set(
    env: UNestingGymEnv,
    model: torch.nn.Module,
    eval_configs: list[list[int]],
    device: torch.device,
    args: argparse.Namespace,
    episode: int,
    rotations_rad: list[float] | None = None,
) -> float:
    n = len(eval_configs)
    print(f"[eval]  ep={episode+1:>5}/{args.episodes}  {n} configs …", end="", flush=True)

    ratios_pure, ratios_fb = [], []
    agent_densities, fb_densities, greedy_densities = [], [], []
    n_placed_list, n_placed_fb_list, n_greedy_list  = [], [], []
    plot_snapshots = plot_greedy_polys = plot_lib_ids = plot_unplaced_polys = None
    plot_n_placed = plot_n_greedy = 0
    plot_greedy_density = 0.0
    best_n_greedy = -1
    worst_snapshots = worst_greedy_polys = worst_lib_ids = worst_unplaced_polys = None
    worst_n_placed = worst_n_greedy = 0
    worst_greedy_density = 0.0
    worst_ratio = float("inf")

    for lib_ids in eval_configs:
        pure_density, fb_density, snapshots, n_placed, n_placed_fb, unplaced_polys = _run_greedy_eval(
            env, model, lib_ids, device, args, rotations_rad=rotations_rad,
        )
        env.reset(lib_ids)
        n_greedy       = env.place_remaining()
        greedy_polys   = env.placed_polygons()
        greedy_density = env.packing_density()

        ratio_pure = pure_density / greedy_density if greedy_density > 0 else float("nan")
        ratio_fb   = fb_density   / greedy_density if greedy_density > 0 else float("nan")

        ratios_pure.append(ratio_pure)
        ratios_fb.append(ratio_fb)
        agent_densities.append(pure_density)
        fb_densities.append(fb_density)
        greedy_densities.append(greedy_density)
        n_placed_list.append(n_placed)
        n_placed_fb_list.append(n_placed_fb)
        n_greedy_list.append(n_greedy)

        if n_greedy > best_n_greedy:
            best_n_greedy        = n_greedy
            plot_snapshots       = snapshots
            plot_greedy_polys    = greedy_polys
            plot_lib_ids         = lib_ids
            plot_n_placed        = n_placed
            plot_n_greedy        = n_greedy
            plot_greedy_density  = greedy_density
            plot_unplaced_polys  = unplaced_polys

        if not np.isnan(ratio_pure) and ratio_pure < worst_ratio:
            worst_ratio           = ratio_pure
            worst_snapshots       = snapshots
            worst_greedy_polys    = greedy_polys
            worst_lib_ids         = lib_ids
            worst_n_placed        = n_placed
            worst_n_greedy        = n_greedy
            worst_greedy_density  = greedy_density
            worst_unplaced_polys  = unplaced_polys

    mean_ratio_pure    = float(np.mean(ratios_pure))
    mean_ratio_fb      = float(np.mean(ratios_fb))
    pct_above_pure     = int(round(100 * np.mean([r > 1.0 for r in ratios_pure])))
    pct_above_fb       = int(round(100 * np.mean([r > 1.0 for r in ratios_fb])))
    mean_placed        = float(np.mean(n_placed_list))
    mean_placed_fb     = float(np.mean(n_placed_fb_list))
    mean_greedy_placed = float(np.mean(n_greedy_list))
    mean_agent         = float(np.mean(agent_densities))
    mean_fb            = float(np.mean(fb_densities))
    mean_greedy        = float(np.mean(greedy_densities))

    print(f"  pure={mean_agent:.4f}({mean_ratio_pure:.3f})  "
          f"fallback={mean_fb:.4f}({mean_ratio_fb:.3f})  "
          f"greedy={mean_greedy:.4f}  "
          f"placed={mean_placed:.1f}/{mean_placed_fb:.1f} vs {mean_greedy_placed:.1f}")

    fig_best = _make_eval_figure(
        plot_snapshots, plot_n_placed, plot_lib_ids, episode, args,
        plot_greedy_polys, plot_n_greedy, plot_greedy_density, plot_unplaced_polys,
    )
    fig_worst = _make_eval_figure(
        worst_snapshots, worst_n_placed, worst_lib_ids, episode, args,
        worst_greedy_polys, worst_n_greedy, worst_greedy_density, worst_unplaced_polys,
    )
    wandb.log({
        "eval/board_best":            wandb.Image(fig_best),
        "eval/board_worst":           wandb.Image(fig_worst),
        "eval/agent_density":         mean_agent,
        "eval/agent_density_fallback": mean_fb,
        "eval/greedy_density":        mean_greedy,
        "eval/vs_greedy":             mean_ratio_pure,
        "eval/vs_greedy_fallback":    mean_ratio_fb,
        "eval/beat_greedy_pct":       pct_above_pure,
        "eval/beat_greedy_pct_fallback": pct_above_fb,
    }, step=episode + 1)
    plt.close(fig_best)
    plt.close(fig_worst)

    return mean_ratio_pure


def train(args: argparse.Namespace) -> None:
    device = _device()
    print(f"Device: {device}")

    if args.n_parts_start is None:
        args.n_parts_start = args.n_parts
    if args.curriculum_episodes is None:
        args.curriculum_episodes = args.episodes
    if args.imitation_episodes is None:
        args.imitation_episodes = args.curriculum_episodes
    if args.out_best is None:
        args.out_best = f"data/{args.model}_pointer_net_best.pt"
    if args.out_last is None:
        args.out_last = f"data/{args.model}_pointer_net_last.pt"

    env              = _build_env(args)
    model, optimizer = _build_model_and_optimizer(args, device)
    rng              = random.Random(args.rng_seed)

    wandb.init(project="u-nesting-pair-pointer-net", config=vars(args))

    eval_rng     = random.Random(args.eval_seed)
    n_eval_parts = args.n_eval_parts if args.n_eval_parts is not None else args.n_parts
    eval_configs = [env.sample_episode_ids(n=n_eval_parts, rng=eval_rng)
                    for _ in range(args.n_eval_configs)]
    fixed_lib_ids = (env.sample_episode_ids(n=args.n_parts, rng=rng)
                     if args.fixed_parts else None)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(
        f"\n── Training config ──────────────────────────────────────────────────\n"
        f"  Plate      : {args.plate_width:.0f}×{args.plate_height:.0f}   "
        f"n_parts={args.n_parts}   episodes={args.episodes}\n"
        f"  Model      : {args.model}\n"
        f"  Params     : {n_trainable:,} trainable / {n_total:,} total\n"
        f"  Device     : {device}\n"
        f"  Eval       : every {args.eval_interval} episodes  "
        f"n_eval_configs={args.n_eval_configs}  n_eval_parts={n_eval_parts}\n"
        f"  Checkpoint : {args.out_best}\n"
        f"─────────────────────────────────────────────────────────────────────\n"
        f"\n── Metric guide ─────────────────────────────────────────────────────\n"
        f"  d=X(ema=Y)   Packing density this episode (0–1) and its smoothed trend.\n"
        f"               ema rises → agent is learning to nest more densely.\n"
        f"               Typical greedy baseline: ~0.35–0.45 for aligners.\n"
        f"\n"
        f"  ev=X         Value explained variance (−∞ to 1).\n"
        f"               How well the value network predicts future returns:\n"
        f"                 > 0.5  good — value net is genuinely helping PPO\n"
        f"                 ~ 0    bad  — predicting the mean, no useful signal\n"
        f"                 < 0    broken — worse than a constant, check lr/arch\n"
        f"               Expect low/negative early; should rise within ~500 eps.\n"
        f"\n"
        f"  loss=X       Total PPO loss (policy + value + entropy + imitation).\n"
        f"               Should decrease early then plateau — large spikes = instability.\n"
        f"\n"
        f"  ent=X        Policy entropy (normalised 0–1). Measures exploration.\n"
        f"               Too low (< 0.1) → agent collapsed to one action.\n"
        f"               Too high (> 0.9) → agent still acting randomly.\n"
        f"\n"
        f"  imit=X(Y)    Imitation loss vs greedy-largest-first (weight Y, decays to 0).\n"
        f"               Guides early exploration; ignore once weight reaches 0.\n"
        f"\n"
        f"  [eval]       Runs every {args.eval_interval} eps on fixed configs.\n"
        f"               pure=X(R)  agent density and ratio vs greedy baseline.\n"
        f"               R > 1.0 → agent beats greedy — the goal.\n"
        f"─────────────────────────────────────────────────────────────────────\n"
    )


    rotations_rad = (
        env.get_rotation_angles(args.n_rotations)
        if args.model == "split" and args.n_rotations > 0
        else None
    )
    rot_entropy_coef = args.rot_entropy_coef if args.rot_entropy_coef is not None else args.entropy_coef

    best_density      = 0.0
    last_eval_density = 0.0

    for episode in range(args.episodes):
        t0         = time.perf_counter()
        n_parts_ep = _curriculum_n_parts(episode, args)
        lib_ids    = fixed_lib_ids or env.sample_episode_ids(n=n_parts_ep, rng=rng)
        model.train()
        with torch.no_grad():
            # Rollout: no graph needed — log_probs are detached, values are detached
            log_probs, entropies, rot_entropies, advantages, reward, _, stats, observations = _run_episode(
                env, model, lib_ids, device, gamma=args.gamma,
                gae_lambda=args.gae_lambda, rotations_rad=rotations_rad,
            )

        entropy_coef   = _current_entropy_coef(episode, args)
        imitation_coef = _current_imitation_coef(episode, args)

        if log_probs:
            old_log_probs  = [lp.detach() for lp in log_probs]
            value_targets  = [obs[5].item() for obs in observations]
            for _ in range(args.ppo_epochs):
                new_log_probs, new_entropies, new_rot_entropies, new_imitation, new_value_preds = _evaluate_actions(
                    model, observations, device)
                loss, entropy_bonus, imitation_loss, rot_entropy, value_loss = _ppo_loss(
                    old_log_probs, new_log_probs, new_entropies, new_imitation,
                    advantages, entropy_coef, imitation_coef, n_parts_ep, args.ppo_clip, device,
                    rot_entropies=new_rot_entropies,
                    rot_entropy_coef=rot_entropy_coef,
                    n_rotations=args.n_rotations,
                    value_preds=new_value_preds,
                    value_targets=value_targets,
                    value_coef=args.value_coef,
                )
                if torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    loss = torch.tensor(0.0)
                    break
        else:
            loss = entropy_bonus = imitation_loss = rot_entropy = value_loss = torch.tensor(0.0)
            new_value_preds = []
            value_targets   = []

        # Value explained variance: 1 - Var(target - pred) / Var(target)
        # > 0.5 → value net is useful; ~0 → predicting mean; < 0 → broken
        val_expl_var    = 0.0
        val_mean_target = 0.0
        if value_targets:
            vt_arr = np.array(value_targets, dtype=np.float32)
            vp_arr = np.array([v.item() for v in new_value_preds], dtype=np.float32)
            var_t  = float(np.var(vt_arr))
            val_expl_var    = float(1.0 - np.var(vt_arr - vp_arr) / var_t) if var_t > 1e-8 else 0.0
            val_mean_target = float(np.mean(vt_arr))

        episode_time = time.perf_counter() - t0
        _log_training_step(episode, args, env, n_parts_ep, reward, loss, entropy_bonus,
                           imitation_loss, imitation_coef, stats, episode_time,
                           rot_entropy=rot_entropy, value_loss=value_loss,
                           val_expl_var=val_expl_var, val_mean_target=val_mean_target)

        if (episode + 1) % args.eval_interval == 0:
            last_eval_density = _log_eval_set(
                env, model, eval_configs, device, args, episode,
                rotations_rad=rotations_rad,
            )

        if last_eval_density > best_density:
            best_density = last_eval_density
            torch.save({"episode": episode + 1, "model": model.state_dict(),
                        "density": best_density}, args.out_best)

    torch.save({"episode": args.episodes, "model": model.state_dict()}, args.out_last)
    wandb.finish()
    print(f"Training done. Best density: {best_density:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--json",                default="data/library.json")
    p.add_argument("--plate-width",         type=float, default=100.0)
    p.add_argument("--plate-height",        type=float, default=100.0)
    p.add_argument("--n-parts",             type=int,   default=15)
    p.add_argument("--episodes",            type=int,   default=5000)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--entropy-coef",        type=float, default=0.05)
    p.add_argument("--entropy-coef-final",  type=float, default=None)
    p.add_argument("--log-interval",        type=int,   default=50)
    p.add_argument("--model",               default="split", choices=["split", "spatial"])
    p.add_argument("--out-best",            default=None)
    p.add_argument("--out-last",            default=None)
    p.add_argument("--rng-seed",            type=int,   default=42)
    p.add_argument("--eval-interval",       type=int,   default=100)
    p.add_argument("--n-eval-parts",        type=int,   default=None)
    p.add_argument("--fixed-parts",         action="store_true")
    p.add_argument("--eval-seed",           type=int,   default=7)
    p.add_argument("--sdf-clip-px",         type=int,   default=8)
    p.add_argument("--n-parts-start",       type=int,   default=None)
    p.add_argument("--curriculum-episodes", type=int,   default=None)
    p.add_argument("--n-eval-configs",      type=int,   default=20)
    p.add_argument("--ppo-clip",            type=float, default=0.2)
    p.add_argument("--ppo-epochs",          type=int,   default=4)
    p.add_argument("--imitation-coef",        type=float, default=0.3,
                   help="Initial imitation loss weight (decays to 0). Set 0 to disable.")
    p.add_argument("--imitation-episodes",   type=int,   default=None,
                   help="Episodes over which imitation coef decays to 0 (default: curriculum_episodes).")
    p.add_argument("--gae-lambda",           type=float, default=0.95,
                   help="GAE lambda for advantage estimation (default 0.95).")
    p.add_argument("--value-coef",           type=float, default=0.5,
                   help="Value loss coefficient in PPO loss (default 0.5).")
    p.add_argument("--rotations",           type=float, nargs="+", default=None,
                   metavar="DEG",
                   help="Allowed rotation angles in degrees for all parts "
                        "(e.g. --rotations 0 90 180 270). "
                        "If omitted, uses each part's own rotation list from the library.")
    p.add_argument("--n-rotations",         type=int,   default=8,
                   help="Number of evenly-spaced rotation angles the agent chooses from "
                        "(SplitPointerNet only, default 8). Set 0 to disable rotation head.")
    p.add_argument("--rot-entropy-coef",    type=float, default=None,
                   help="Entropy bonus coefficient for the rotation head "
                        "(default: same as --entropy-coef).")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
