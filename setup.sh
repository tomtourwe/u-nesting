#!/usr/bin/env bash
set -euo pipefail

# ── 1. Python venv ────────────────────────────────────────────────────────────
echo "[setup] Step 1/4 — Python virtual environment"
if [ ! -d ".venv" ]; then
    echo "[setup] Creating .venv …"
    python3 -m venv .venv
else
    echo "[setup] .venv already exists, skipping creation"
fi
PIP=".venv/bin/pip"

echo "[setup] Upgrading pip …"
$PIP install --upgrade pip

echo "[setup] Installing PyTorch (cu121) …"
$PIP install torch --index-url https://download.pytorch.org/whl/cu121
echo "[setup] PyTorch installed"

echo "[setup] Installing requirements …"
$PIP install -r requirements.txt
echo "[setup] Python dependencies installed"

# ── 2. W&B login ─────────────────────────────────────────────────────────────
echo "[setup] Step 2/4 — Weights & Biases login"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[setup] WANDB_API_KEY not set — prompting for interactive login"
    .venv/bin/wandb login
else
    echo "[setup] WANDB_API_KEY found — skipping interactive login"
fi

# ── 3. Rust ───────────────────────────────────────────────────────────────────
echo "[setup] Step 3/4 — Rust toolchain"
if ! command -v rustup &>/dev/null; then
    echo "[setup] rustup not found — installing Rust …"
    mkdir -p ~/tmp
    export TMPDIR=~/tmp
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet
    echo "[setup] Rust installed"
else
    echo "[setup] Rust already installed: $(rustc --version)"
fi
# shellcheck source=/dev/null
source "$HOME/.cargo/env"

# ── 4. Build Maturin Python bindings ─────────────────────────────────────────
echo "[setup] Step 4/4 — Building Rust/Python bindings (maturin) …"
MATURIN_TMPDIR=$(mktemp -d)
TMPDIR="$MATURIN_TMPDIR" .venv/bin/maturin develop --release --features python-bindings
rm -rf "$MATURIN_TMPDIR"
echo "[setup] Maturin build complete"

# ── Done ──────────────────────────────────────────────────────────────────────
echo "[setup] Setup complete — activate the venv and train:"
echo "        source .venv/bin/activate"
echo "        python scripts/train_pair_pointer_net.py --json data/library.json --n-parts 15"
