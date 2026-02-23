#!/usr/bin/env bash
# ============================================================================
# Bayesian Calibration App — One-click local setup
# ============================================================================
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# This script will:
#   1. Create a conda environment with all dependencies
#   2. Install CmdStan (the Stan compiler)
#   3. Set the app to "local" mode (CSV uploads enabled)
#   4. Launch the app in your browser
# ============================================================================

set -e

ENV_NAME="bayes-cal"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Bayesian Calibration App — Local Setup                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# --- Check conda is available ---
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. Please install Anaconda or Miniconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# --- Create or update conda environment ---
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✅ Conda environment '${ENV_NAME}' already exists."
    echo "   Updating packages..."
    conda run -n "$ENV_NAME" pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
else
    echo "📦 Creating conda environment '${ENV_NAME}'..."
    conda create -n "$ENV_NAME" python=3.11 -y
    echo "📦 Installing Python packages..."
    conda run -n "$ENV_NAME" pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# --- Install CmdStan if needed ---
echo "🔧 Checking CmdStan..."
conda run -n "$ENV_NAME" python -c "
import cmdstanpy
try:
    path = cmdstanpy.cmdstan_path()
    print(f'✅ CmdStan already installed at: {path}')
except ValueError:
    print('📦 Installing CmdStan (this takes a few minutes)...')
    cmdstanpy.install_cmdstan()
    print('✅ CmdStan installed.')
"

# --- Set to local mode ---
echo "🖥️  Setting app to local mode (CSV uploads enabled)..."
sed -i.bak 's/^MODE = "cloud"/MODE = "local"/' "$SCRIPT_DIR/app_config.py"
rm -f "$SCRIPT_DIR/app_config.py.bak"

# --- Launch ---
echo ""
echo "🚀 Launching app..."
echo "   The app will open in your browser at http://localhost:8501"
echo "   Press Ctrl+C to stop."
echo ""
cd "$SCRIPT_DIR"
conda run -n "$ENV_NAME" streamlit run app.py
