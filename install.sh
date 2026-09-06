#!/usr/bin/env bash
# Installs TextSpill: the Rust client, the ASR virtualenv and the user service.
# Safe to re-run; it upgrades an existing installation in place.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
share="${XDG_DATA_HOME:-$HOME/.local/share}/textspill"
bindir="$HOME/.local/bin"
units="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
python="${TEXTSPILL_PYTHON:-python3.12}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }

say "Building the textspill binary"
cargo build --release --manifest-path "$repo/Cargo.toml"
mkdir -p "$bindir"
install -m755 "$repo/target/release/textspill" "$bindir/textspill"

say "Linking the ASR daemon into $share"
mkdir -p "$share"
# Symlinks, not copies: editing the repo takes effect on the next service restart.
ln -sfn "$repo/asr/daemon.py" "$share/daemon.py"

if [ ! -x "$share/venv/bin/python" ]; then
    say "Creating the Python environment with $python"
    command -v "$python" >/dev/null || {
        echo "$python not found; set TEXTSPILL_PYTHON to an interpreter torch supports" >&2
        exit 1
    }
    "$python" -m venv "$share/venv"
fi

say "Installing Python dependencies (this downloads PyTorch, ~2-3 GB)"
"$share/venv/bin/pip" install --upgrade pip
"$share/venv/bin/pip" install --upgrade \
    "torch" \
    "transformers>=5.13.0" \
    "accelerate" \
    "librosa" \
    "soundfile"

say "Installing the user service"
mkdir -p "$units"
install -m644 "$repo/systemd/textspill-asr.service" "$units/textspill-asr.service"
systemctl --user daemon-reload

cat <<'NEXT'

Done. Remaining steps:

  1. Start the ASR daemon (first run downloads the model, ~1.5 GB):
       systemctl --user enable --now textspill-asr.service
       journalctl --user -u textspill-asr.service -f

  2. Enable the paste backend:
       sudo pacman -S ydotool
       sudo usermod -aG input "$USER"     # log out and back in
       systemctl --user enable --now ydotool.service

  3. Bind the hotkey, e.g. in ~/.config/hypr/bindings.conf:
       bindd = SUPER, D, Dictate, exec, textspill toggle
     (SUPER+SPACE is Omarchy's menu; `unbind = SUPER, SPACE` first if you want it.)

  4. Make sure ~/.local/bin is on your PATH.

NEXT
