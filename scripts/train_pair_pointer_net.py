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
  3. Best-of-K selection: run K rollouts per config, keep the best (highest density).
     Advantage: best_density − mean_density across K rollouts (same scalar for all steps).
  4. Loss = PPO_loss − entropy_coef·entropy + imitation_coef·imitation_loss

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


def _autocast(_device: torch.device):
    """No-op: autocast removed. Model compute is ~0.1% of episode time (env dominates)."""
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


def _sample_action_flat(
    joint_logits: torch.Tensor,
    greedy: bool,
) -> tuple[torch.Tensor, Categorical, torch.Tensor, torch.Tensor]:
    """
    Sample from flat (N*R,) distribution over joint (part, rotation) pairs.

    Returns (action_id, dist, part_id, rot_id).
    """
    N, R      = joint_logits.shape
    flat      = joint_logits.reshape(N * R)
    dist      = Categorical(logits=flat)
    action_id = flat.argmax() if greedy else dist.sample()
    return action_id, dist, action_id // R, action_id % R


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
) -> tuple[list, list, list[float], float, list, dict, list]:
    env.reset(lib_ids)
    n_parts = len(lib_ids)

    log_probs:    list[torch.Tensor] = []
    entropies:    list[torch.Tensor] = []
    observations: list[tuple]        = []
    snapshots:    list               = []
    n_steps: int = 0

    for _ in range(n_parts):
        remaining = env.remaining_item_ids()
        if not remaining:
            break

        mask = _build_remaining_mask(remaining, n_parts, device)

        if rotations_rad is not None:
            # New flat joint (N, R) interface — SplitPointerNet
            obs_np, scalars_np = env.preview_images_per_rotation(rotations_rad)
            obs_t     = torch.from_numpy(obs_np).to(device, non_blocking=True)
            scalars_t = torch.from_numpy(scalars_np).to(device, non_blocking=True)
            joint_logits            = model(obs_t, mask, scalars_t)  # (N, R)
            action_id, dist, part_id, rot_id = _sample_action_flat(joint_logits, greedy)
            success = env.place_with_rotation(part_id.item(), rotations_rad[rot_id.item()])
            if not success:
                break  # episode ends on first failure — no skip-and-retry during training
            if not greedy:
                log_probs.append(dist.log_prob(action_id))
                entropies.append(dist.entropy())
                gt = _greedy_target(env, remaining)
                observations.append((obs_t.cpu(), mask.cpu(), action_id.cpu(), torch.tensor(gt), scalars_t.cpu()))
        else:
            # Legacy path — SpatialPointerNet (no rotation head)
            obs_np = env.preview_pair_images()
            obs_t  = torch.from_numpy(obs_np).to(device, non_blocking=True)
            output     = model(obs_t, mask)
            part_logits = output[0] if isinstance(output, tuple) else output
            part_dist  = Categorical(logits=part_logits)
            part_id    = part_logits.argmax() if greedy else part_dist.sample()
            success    = env.place_anywhere(part_id.item())
            if not success:
                break  # episode ends on first failure
            if not greedy:
                log_probs.append(part_dist.log_prob(part_id))
                entropies.append(part_dist.entropy())
                gt = _greedy_target(env, remaining)
                observations.append((obs_t.cpu(), mask.cpu(), part_id.cpu(), None, torch.tensor(gt)))

        n_steps += 1

        if capture_snapshots:
            snapshots.append((env.placed_polygons(), env.packing_density()))

    # Every step gets the full episode return (final density) as its advantage.
    # The per-config baseline is subtracted later in the training loop, after
    # all K rollouts of the same config have been collected.
    board_area     = env.plate_w * env.plate_h
    episode_reward = env.packing_density() * env.bbox_area() / board_area  # placed_area / board_area
    A_t_list       = [episode_reward] * n_steps

    stats = {
        "bbox_frac": env.bbox_area() / board_area,
        "n_placed":  env.n_placed(),
    }

    return log_probs, entropies, A_t_list, episode_reward, snapshots, stats, observations


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
        model = SplitPointerNet(part_encoder=args.part_encoder).to(device)
    if device.type == "cuda":
        # ~30–60% faster on NVIDIA; not supported on MPS
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)
        print("torch.compile enabled (CUDA), TF32 matmul enabled")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return model, optimizer


def _current_entropy_coef(episode: int, args: argparse.Namespace) -> float:
    if args.entropy_coef_final is not None:
        anneal_episodes = args.entropy_anneal_episodes or args.episodes
        t = min(1.0, episode / anneal_episodes)
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
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    log_probs, entropies, rot_entropies, imitation_losses = [], [], [], []
    for obs_entry in observations:
        if len(obs_entry) == 5 and obs_entry[3] is not None and obs_entry[4].dim() == 3:
            # New flat joint format with scalars: (obs_t, mask, action_id, greedy_target, scalars_t)
            obs_t, mask, action_id, greedy_target, scalars_t = obs_entry
            obs_t     = obs_t.to(device,      non_blocking=True)
            mask      = mask.to(device,       non_blocking=True)
            action_id = action_id.to(device,  non_blocking=True)
            scalars_t = scalars_t.to(device,  non_blocking=True)

            with _autocast(device):
                joint_logits = model(obs_t, mask, scalars_t)              # (N, R)
            N, R = joint_logits.shape
            flat_dist = Categorical(logits=joint_logits.reshape(N * R))
            log_probs.append(flat_dist.log_prob(action_id))
            entropies.append(flat_dist.entropy())
            # Imitation: marginalise over rotations via logsumexp → part logits
            part_logits_imit = joint_logits.logsumexp(dim=1)              # (N,)
            imitation_losses.append(
                F.cross_entropy(
                    part_logits_imit.unsqueeze(0),
                    greedy_target.unsqueeze(0).to(device, non_blocking=True),
                )
            )
        elif len(obs_entry) == 4:
            # Old flat joint format without scalars: (obs_t, mask, action_id, greedy_target)
            obs_t, mask, action_id, greedy_target = obs_entry
            obs_t     = obs_t.to(device,     non_blocking=True)
            mask      = mask.to(device,      non_blocking=True)
            action_id = action_id.to(device, non_blocking=True)

            with _autocast(device):
                joint_logits = model(obs_t, mask)                          # (N, R)
            N, R = joint_logits.shape
            flat_dist = Categorical(logits=joint_logits.reshape(N * R))
            log_probs.append(flat_dist.log_prob(action_id))
            entropies.append(flat_dist.entropy())
            part_logits_imit = joint_logits.logsumexp(dim=1)              # (N,)
            imitation_losses.append(
                F.cross_entropy(
                    part_logits_imit.unsqueeze(0),
                    greedy_target.unsqueeze(0).to(device, non_blocking=True),
                )
            )
        else:
            # Legacy format: (obs_t, mask, part_id, rot_id, greedy_target)
            obs_t, mask, part_id, rot_id, greedy_target = obs_entry
            obs_t   = obs_t.to(device,   non_blocking=True)
            mask    = mask.to(device,    non_blocking=True)
            part_id = part_id.to(device, non_blocking=True)

            with _autocast(device):
                output = model(obs_t, mask)
                part_logits = output[0] if isinstance(output, tuple) else output

            part_dist = Categorical(logits=part_logits)
            log_probs.append(part_dist.log_prob(part_id))
            entropies.append(part_dist.entropy())
            imitation_losses.append(
                F.cross_entropy(
                    part_logits.unsqueeze(0),
                    greedy_target.unsqueeze(0).to(device, non_blocking=True),
                )
            )

    return log_probs, entropies, rot_entropies, imitation_losses


def _ppo_loss(
    old_log_probs: list[torch.Tensor],
    new_log_probs: list[torch.Tensor],
    entropies: list[torch.Tensor],
    imitation_losses: list[torch.Tensor],
    advantages: list[float],
    entropy_coef: float,
    imitation_coef: float,
    n_actions: int,
    clip_eps: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    adv            = torch.tensor(advantages, dtype=torch.float32, device=device)
    old_lp         = torch.stack(old_log_probs).detach()
    new_lp         = torch.stack(new_log_probs)
    ratio          = (new_lp - old_lp).exp()
    clipped        = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss    = -torch.min(ratio * adv, clipped * adv).mean()
    entropy        = torch.stack(entropies).mean() / math.log(max(n_actions, 2))
    imitation_loss = torch.stack(imitation_losses).mean()
    total          = policy_loss - entropy_coef * entropy + imitation_coef * imitation_loss
    rot_entropy    = torch.tensor(0.0, device=device)   # kept for logging compat
    return total, entropy, imitation_loss, rot_entropy, policy_loss


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

    Uses the same observation as training (preview_images_per_rotation when rotations
    are enabled) so the model sees identical inputs during eval and training.

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

            if rotations_rad is not None:
                # New flat joint (N, R) interface — SplitPointerNet
                obs_np, scalars_np = env.preview_images_per_rotation(rotations_rad)
                obs       = torch.from_numpy(obs_np).to(device, non_blocking=True)
                scalars_t = torch.from_numpy(scalars_np).to(device, non_blocking=True)
                with _autocast(device):
                    joint_logits = model(obs, mask, scalars_t)         # (N, R)
                N_out, R_out = joint_logits.shape
                action_id = int(joint_logits.reshape(N_out * R_out).argmax().item())
                part_id   = action_id // R_out
                rot_id    = action_id % R_out
                success   = env.place_with_rotation(part_id, rotations_rad[rot_id])
            else:
                obs = torch.from_numpy(env.preview_pair_images()).to(device, non_blocking=True)
                with _autocast(device):
                    output = model(obs, mask)
                part_logits = output[0] if isinstance(output, tuple) else output
                part_id = int(part_logits.argmax().item())
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
    if n_fallback > 0:
        snapshots.append((env.placed_polygons(), fallback_density))

    return pure_density, fallback_density, snapshots, n_placed_pure, n_placed_fallback, []


def _make_eval_figure(
    eval_snapshots: list,
    eval_n_placed: int,
    eval_n_placed_fb: int,
    eval_lib_ids: list[int],
    episode: int,
    args: argparse.Namespace,
    greedy_polys: list,
    greedy_n_placed: int,
    greedy_density: float,
    unplaced_polys: list | None = None,
) -> plt.Figure:
    final_density = eval_snapshots[-1][1] if eval_snapshots else 0.0
    all_panels = [
        (greedy_polys, greedy_density, f"Greedy  {greedy_n_placed}/{len(eval_lib_ids)}\nd={greedy_density:.3f}")
    ] + [
        (polys, density,
         f"ep {episode+1}  {eval_n_placed_fb}/{len(eval_lib_ids)}  d={final_density:.3f}\n#{i+1}  d={density:.3f}"
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
    policy_loss: torch.Tensor | None = None,
    episode_entropy: float | None = None,
    entropy_coef: float = 0.0,
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

    # entropy_bonus here is the RAW normalised entropy [0,1] (not coef×entropy).
    # Log both the raw value and the weighted loss contribution separately.
    entropy_norm     = entropy_bonus.item()          # normalised entropy in [0,1]
    entropy_weighted = entropy_coef * entropy_norm   # actual loss contribution

    log_dict = {
        # Agent nesting quality
        "agent/density":            reward,
        "agent/density_ema":        density_ema,
        "agent/parts_placed":       n_placed / n_parts_ep,
        # Policy losses
        "loss/total":               loss.item(),
        "loss/entropy_norm":        entropy_norm,      # raw normalised entropy [0,1]
        "loss/entropy_weighted":    entropy_weighted,  # coef × entropy (actual loss term)
        "loss/imitation":           imitation_loss.item(),
        "loss/imitation_coef":      imitation_coef,
        "train/entropy_coef":       entropy_coef,      # shows the anneal schedule
        # Performance
        "perf/episode_time_s":      episode_time,
        "cache/hit_rate":           ep_hit_rate,
        "cache/misses_per_episode": ep_misses,
        "cache/size":               cache_size,
    }
    if rot_entropy is not None:
        log_dict["loss/rot_entropy"] = rot_entropy.item()
    if policy_loss is not None:
        log_dict["loss/policy"] = policy_loss.item()
    if episode_entropy is not None:
        # Normalise per-episode entropy to [0,1] range using current config size
        n_actions = n_parts_ep * (args.n_rotations if hasattr(args, "n_rotations") else 8)
        log_dict["agent/entropy"] = episode_entropy / math.log(max(n_actions, 2))
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
    greedy_cache: dict | None = None,
) -> float:
    n = len(eval_configs)
    print(f"[eval]  ep={episode+1:>5}/{args.episodes}  {n} configs …", end="", flush=True)

    ratios_pure, ratios_fb = [], []
    agent_densities, fb_densities, greedy_densities = [], [], []
    n_placed_list, n_placed_fb_list, n_greedy_list  = [], [], []
    plot_snapshots = plot_greedy_polys = plot_lib_ids = plot_unplaced_polys = None
    plot_n_placed = plot_n_placed_fb = plot_n_greedy = 0
    plot_greedy_density = 0.0
    best_density_seen  = -float("inf")
    worst_density_seen =  float("inf")
    worst_snapshots = worst_greedy_polys = worst_lib_ids = worst_unplaced_polys = None
    worst_n_placed = worst_n_placed_fb = worst_n_greedy = 0
    worst_greedy_density = 0.0

    for i, lib_ids in enumerate(eval_configs):
        pure_density, fb_density, snapshots, n_placed, n_placed_fb, unplaced_polys = _run_greedy_eval(
            env, model, lib_ids, device, args, rotations_rad=rotations_rad,
        )
        if greedy_cache is not None and i in greedy_cache:
            greedy_density, greedy_polys, n_greedy = greedy_cache[i]
        else:
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

        if pure_density > best_density_seen:
            best_density_seen    = pure_density
            plot_snapshots       = snapshots
            plot_greedy_polys    = greedy_polys
            plot_lib_ids         = lib_ids
            plot_n_placed        = n_placed
            plot_n_placed_fb     = n_placed_fb
            plot_n_greedy        = n_greedy
            plot_greedy_density  = greedy_density
            plot_unplaced_polys  = unplaced_polys

        if pure_density < worst_density_seen:
            worst_density_seen    = pure_density
            worst_snapshots       = snapshots
            worst_greedy_polys    = greedy_polys
            worst_lib_ids         = lib_ids
            worst_n_placed        = n_placed
            worst_n_placed_fb     = n_placed_fb
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

    print(f"  agent={mean_agent:.4f}({mean_ratio_pure:.3f}x)  "
          f"+autofill={mean_fb:.4f}({mean_ratio_fb:.3f}x)  "
          f"greedy={mean_greedy:.4f}  "
          f"placed={mean_placed:.1f}/{mean_placed_fb:.1f} vs {mean_greedy_placed:.1f}")

    fig_best = _make_eval_figure(
        plot_snapshots, plot_n_placed, plot_n_placed_fb, plot_lib_ids, episode, args,
        plot_greedy_polys, plot_n_greedy, plot_greedy_density, plot_unplaced_polys,
    )
    fig_worst = _make_eval_figure(
        worst_snapshots, worst_n_placed, worst_n_placed_fb, worst_lib_ids, episode, args,
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

    # Precompute greedy baseline for each eval config once — it never changes.
    print("Precomputing greedy baselines for eval configs … ", end="", flush=True)
    greedy_cache: dict[int, tuple] = {}  # config index → (density, polys, n_placed)
    for i, lib_ids in enumerate(eval_configs):
        env.reset(lib_ids)
        n_placed       = env.place_remaining()
        greedy_cache[i] = (env.packing_density(), env.placed_polygons(), n_placed)
    print("done")
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
        f"  loss=X       Total PPO loss (policy + entropy + imitation).\n"
        f"               Should decrease early then plateau — large spikes = instability.\n"
        f"\n"
        f"  ent=X        Policy entropy (normalised 0–1). Measures exploration.\n"
        f"               Too low (< 0.1) → agent collapsed to one action.\n"
        f"               Too high (> 0.9) → agent still acting randomly.\n"
        f"\n"
        f"  imit=X(Y)    Imitation loss vs greedy-largest-first (weight Y, decays to 0).\n"
        f"               Guides early exploration; ignore once weight reaches 0.\n"
        f"\n"
        f"  [eval]       agent=X(Rx)   density from agent placements only, ratio vs greedy.\n"
        f"               +autofill=X   density after engine greedily fills what the agent left.\n"
        f"               greedy=X      pure greedy baseline (no agent) on same configs.\n"
        f"               placed=A/B vs G  avg parts placed across {args.n_eval_configs} configs:\n"
        f"                              A = agent only, B = agent+autofill, G = greedy.\n"
        f"                              Decimals are averages (e.g. 5.2 = avg of 5,5,6,5,... runs).\n"
        f"               Rx > 1.0 → agent beats greedy — the goal.\n"
        f"─────────────────────────────────────────────────────────────────────\n"
    )


    rotations_rad = (
        env.get_rotation_angles(args.n_rotations)
        if args.model == "split" and args.n_rotations > 0
        else None
    )
    best_density      = 0.0
    last_eval_density = 0.0

    # Accumulators for batched PPO updates
    batch_log_probs:    list = []
    batch_entropies:    list = []
    batch_advantages:   list = []
    batch_observations: list = []

    # Same-config multi-rollout tracking (best-of-K selection)
    config_rollout_idx: int  = 0
    configs_since_update: int = 0
    current_lib_ids:    list[int] | None = None
    n_parts_ep:         int  = 0
    # Buffer for K rollouts of the same config; each entry is
    # (episode_reward, log_probs, entropies, advantages, observations)
    cfg_rollouts: list[tuple] = []

    for episode in range(args.episodes):
        t0 = time.perf_counter()

        # Sample a new config only at the start of each config group
        if config_rollout_idx == 0:
            n_parts_ep      = _curriculum_n_parts(episode, args)
            current_lib_ids = fixed_lib_ids or env.sample_episode_ids(n=n_parts_ep, rng=rng)
        lib_ids = current_lib_ids

        model.train()
        with torch.no_grad():
            log_probs, entropies, advantages, reward, _, stats, observations = _run_episode(
                env, model, lib_ids, device, gamma=args.gamma,
                gae_lambda=args.gae_lambda, rotations_rad=rotations_rad,
            )

        episode_entropy = float(torch.stack(entropies).mean().item()) if entropies else None
        cfg_rollouts.append((reward, log_probs, entropies, advantages, observations, stats["n_placed"]))
        config_rollout_idx += 1

        if config_rollout_idx == args.rollouts_per_config:
            # Use top-K and bottom-K rollouts for contrast.
            # Top rollouts get positive advantage, bottom rollouts get negative.
            # Advantage = episode_density - mean_density (same scalar for all steps in that rollout).
            # Middle rollouts (density ≈ mean) are discarded — they add near-zero gradient signal.
            mean_density = sum(r[0] for r in cfg_rollouts) / len(cfg_rollouts)
            sorted_rollouts = sorted(cfg_rollouts, key=lambda x: x[0])
            k = max(1, min(args.contrast_k, len(cfg_rollouts) // 2))
            top_rollouts    = sorted_rollouts[-k:]   # highest density
            bottom_rollouts = sorted_rollouts[:k]    # lowest density
            best  = sorted_rollouts[-1]
            worst = sorted_rollouts[0]
            spread = best[0] - worst[0]
            for rollout in top_rollouts + bottom_rollouts:
                density, r_lp, r_ent, _, r_obs, _ = rollout
                adv = density - mean_density
                r_adv = [adv] * len(r_lp)
                batch_log_probs.extend(r_lp)
                batch_entropies.extend(r_ent)
                batch_advantages.extend(r_adv)
                batch_observations.extend(r_obs)
            wandb.log({
                "config/best_density":   best[0],
                "config/worst_density":  worst[0],
                "config/mean_density":   mean_density,
                "config/spread":         spread,
                "config/adv_best":       best[0]  - mean_density,
                "config/adv_worst":      worst[0] - mean_density,
            }, step=episode + 1)
            print(f"  → config: best={best[0]:.4f}({best[5]}p)  worst={worst[0]:.4f}({worst[5]}p)  spread={spread:.4f}")
            cfg_rollouts.clear()
            config_rollout_idx = 0
            configs_since_update += 1

            if configs_since_update < args.configs_per_update:
                # Accumulate more configs before updating
                entropy_coef   = _current_entropy_coef(episode, args)
                imitation_coef = _current_imitation_coef(episode, args)
                loss = entropy_bonus = imitation_loss = rot_entropy = policy_loss = torch.tensor(0.0)
                continue

            configs_since_update = 0

            # PPO update after accumulating configs_per_update configs
            entropy_coef   = _current_entropy_coef(episode, args)
            imitation_coef = _current_imitation_coef(episode, args)

            old_log_probs = [lp.detach() for lp in batch_log_probs]
            N_total = len(batch_observations)

            # Normalize advantages over the full batch before splitting into mini-batches
            adv_arr = np.array(batch_advantages, dtype=np.float32)
            adv_std = float(adv_arr.std())
            norm_adv: list[float] = (
                ((adv_arr - adv_arr.mean()) / (adv_std + 1e-8)).tolist()
                if adv_std > 1e-8 else batch_advantages
            )
            wandb.log({
                "train/batch_adv_std":      adv_std,
                "train/batch_adv_pos_frac": float(np.mean([a > 0 for a in batch_advantages])),
            }, step=episode + 1)

            mini_bs   = args.ppo_mini_batch if args.ppo_mini_batch > 0 else N_total
            nan_break = False

            t_ppo = time.perf_counter()
            print(f"  [ppo] updating on {N_total} steps × {args.ppo_epochs} epochs ...", end=" ", flush=True)

            for _ in range(args.ppo_epochs):
                optimizer.zero_grad(set_to_none=True)
                epoch_loss = epoch_ent = epoch_imit = epoch_rot_ent = epoch_policy = 0.0

                for start in range(0, N_total, mini_bs):
                    chunk_obs = batch_observations[start : start + mini_bs]
                    chunk_old = old_log_probs[start : start + mini_bs]
                    chunk_adv = norm_adv[start : start + mini_bs]

                    new_lp, new_ent, _new_rot_ent, new_imit = _evaluate_actions(
                        model, chunk_obs, device)

                    n_actions = (n_parts_ep * args.n_rotations
                                 if rotations_rad is not None else n_parts_ep)
                    chunk_loss, ent_b, imit_l, rot_e, pol_l = _ppo_loss(
                        chunk_old, new_lp, new_ent, new_imit, chunk_adv,
                        entropy_coef, imitation_coef, n_actions, args.ppo_clip, device,
                    )

                    if not torch.isfinite(chunk_loss):
                        nan_break = True
                        break

                    scale = len(chunk_obs) / N_total
                    (chunk_loss * scale).backward()

                    epoch_loss   += chunk_loss.item() * scale
                    epoch_ent    += ent_b.item() * scale
                    epoch_imit   += imit_l.item() * scale
                    epoch_rot_ent += rot_e.item() * scale
                    epoch_policy += pol_l.item() * scale

                if nan_break:
                    loss = torch.tensor(0.0)
                    break

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss           = torch.tensor(epoch_loss)
                entropy_bonus  = torch.tensor(epoch_ent)
                imitation_loss = torch.tensor(epoch_imit)
                rot_entropy    = torch.tensor(epoch_rot_ent)
                policy_loss    = torch.tensor(epoch_policy)

            print(f"done ({time.perf_counter() - t_ppo:.1f}s)  loss={epoch_loss:.4f}", flush=True)

            batch_log_probs.clear()
            batch_entropies.clear()
            batch_advantages.clear()
            batch_observations.clear()
            current_lib_ids = None
        else:
            entropy_coef   = _current_entropy_coef(episode, args)
            imitation_coef = _current_imitation_coef(episode, args)
            loss = entropy_bonus = imitation_loss = rot_entropy = policy_loss = torch.tensor(0.0)

        episode_time = time.perf_counter() - t0
        _log_training_step(episode, args, env, n_parts_ep, reward, loss, entropy_bonus,
                           imitation_loss, imitation_coef, stats, episode_time,
                           rot_entropy=rot_entropy, policy_loss=policy_loss,
                           episode_entropy=episode_entropy, entropy_coef=entropy_coef)

        if (episode + 1) % args.eval_interval == 0:
            last_eval_density = _log_eval_set(
                env, model, eval_configs, device, args, episode,
                rotations_rad=rotations_rad,
                greedy_cache=greedy_cache,
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
    p.add_argument("--entropy-coef-final",    type=float, default=None)
    p.add_argument("--entropy-anneal-episodes", type=int, default=None,
                   help="Episodes over which entropy coef decays to --entropy-coef-final "
                        "(default: same as --episodes).")
    p.add_argument("--log-interval",        type=int,   default=50)
    p.add_argument("--model",               default="split", choices=["split", "spatial"])
    p.add_argument("--part-encoder",        default="cnn",   choices=["cnn", "vit"],
                   help="Part image encoder: 'cnn' (default, _SpatialCNN) or 'vit' (_PatchViT, 16x16 patches).")
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
    p.add_argument("--ppo-clip",             type=float, default=0.2)
    p.add_argument("--ppo-epochs",          type=int,   default=4)
    p.add_argument("--rollouts-per-config", type=int,   default=8,
                   help="Run this many rollouts on the same parts config before sampling a new one (default 8). "
                        "Within-group advantage normalization removes config-difficulty variance, "
                        "leaving only policy-quality signal.")
    p.add_argument("--configs-per-update",   type=int,   default=2,
                   help="Accumulate this many configs before running a PPO update (default 2). "
                        "Larger values give more diverse batches (different part combos) "
                        "at the cost of less frequent updates.")
    p.add_argument("--contrast-k",          type=int,   default=1,
                   help="Use top-K and bottom-K rollouts for the PPO update (default 1 = best+worst only). "
                        "Increasing to 2 uses top-2 and bottom-2, giving more gradient data while "
                        "keeping clear positive/negative contrast. Must be <= rollouts_per_config // 2.")
    p.add_argument("--ppo-mini-batch",      type=int,   default=0,
                   help="Mini-batch size within each PPO epoch (0 = full batch, default 0). "
                        "Set e.g. 16 on GPU to avoid OOM on large configs.")
    p.add_argument("--imitation-coef",        type=float, default=0.3,
                   help="Initial imitation loss weight (decays to 0). Set 0 to disable.")
    p.add_argument("--imitation-episodes",   type=int,   default=500,
                   help="Episodes over which imitation coef decays to 0.")
    p.add_argument("--gae-lambda",           type=float, default=0.95,
                   help="GAE lambda for advantage estimation (default 0.95).")
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
