#!/bin/bash
# ---------------------------------------------------------------------------
# launch_ogusa.command
#
# Double-click this file in macOS Finder to launch the OG-USA web app.
# The script activates the ogusa-dev conda environment, starts the Panel
# server, and opens the app in your default browser automatically.
#
# One-time setup (run once in a terminal):
#   chmod +x launch_ogusa.command
# ---------------------------------------------------------------------------

# Change to the repository root (same directory as this script)
cd "$(dirname "$0")"

# Activate conda – try both common conda init locations
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not locate conda.  Please ensure conda is installed."
    read -p "Press Enter to close..."
    exit 1
fi

conda activate ogusa-dev

echo "Starting OG-USA app..."
echo "Open http://localhost:5006/app in your browser if it does not open automatically."
echo "Press Ctrl+C in this window to stop the server."
echo ""

panel serve ogusa/app/app.py \
    --show \
    --address localhost \
    --port 5006 \
    --prefix /app \
    --allow-websocket-origin localhost:5006
