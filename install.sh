#!/bin/sh
# AIForge installer for macOS and Linux.
#
#   curl -LsSf https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/install.sh | sh
#
# What it does, and nothing else (no admin password, no system files):
#   1. installs uv (https://docs.astral.sh/uv/) if it is missing: it brings its own Python
#   2. installs AIForge with uv, in its own environment (run it again to update)
#   3. adds an AIForge icon: ~/Applications/AIForge.app (macOS) or the app menu (Linux)
# Your models in ~/AIForge are never touched.
#
# Uninstall (keeps your models):
#   curl -LsSf https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/install.sh | sh -s -- --uninstall
set -eu

PACKAGE="aiforge-studio"
PYTHON="3.12"
COLUMNS_WANTED=130
ROWS_WANTED=40
# Where the icon goes (tests point these somewhere else)
MAC_APPS="${AIFORGE_APPS_DIR:-$HOME/Applications}"
LINUX_APPS="${AIFORGE_APPS_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/applications}"

say() { printf '%s\n' "$*"; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mError:\033[0m %s\n' "$*" >&2; exit 1; }

find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv; return; fi
  for candidate in "${UV_INSTALL_DIR:-}/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -x "$candidate" ]; then printf '%s\n' "$candidate"; return; fi
  done
}

install_uv() {
  step "1/3 · Installing uv (it brings its own Python)"
  command -v curl >/dev/null 2>&1 || fail "curl is needed. Install it, then run this again."
  curl -LsSf https://astral.sh/uv/install.sh | sh
}

mac_launcher() {
  app="$MAC_APPS/AIForge.app"
  mkdir -p "$app/Contents/MacOS"
  cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>AIForge</string>
  <key>CFBundleIdentifier</key><string>io.github.ducdinh-codehub.aiforge</string>
  <key>CFBundleExecutable</key><string>AIForge</string>
  <key>CFBundlePackageType</key><string>APPL</string>
</dict></plist>
PLIST
  # Opens Terminal with AIForge in a window big enough for it.
  cat > "$app/Contents/MacOS/AIForge" <<LAUNCH
#!/bin/sh
osascript -e 'tell application "Terminal"
  activate
  set w to do script "clear; exec \\"$1\\""
  set number of columns of front window to $COLUMNS_WANTED
  set number of rows of front window to $ROWS_WANTED
end tell'
LAUNCH
  chmod +x "$app/Contents/MacOS/AIForge"
  say "Added AIForge to $MAC_APPS (open it from Finder, Launchpad or Spotlight)."
}

linux_launcher() {
  mkdir -p "$LINUX_APPS"
  cat > "$LINUX_APPS/aiforge.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=AIForge
Comment=Build, train and use neural networks without code
Exec="$1"
Terminal=true
Categories=Education;Science;Development;
DESKTOP
  say "Added AIForge to your app menu."
}

uninstall() {
  step "Uninstalling AIForge"
  uv="$(find_uv)"
  if [ -n "$uv" ] && "$uv" tool list 2>/dev/null | grep -q "^$PACKAGE "; then
    "$uv" tool uninstall "$PACKAGE"
  else
    say "AIForge isn't installed with uv: nothing to remove there."
  fi
  rm -rf "$MAC_APPS/AIForge.app"
  rm -f "$LINUX_APPS/aiforge.desktop"
  say "Done. Your models in ~/AIForge were kept (delete that folder yourself if you want them gone)."
  say "uv itself stays installed; remove it with 'uv self uninstall' if you don't use it."
}

main() {
  case "${1:-}" in
    --uninstall) uninstall; exit 0 ;;
    "") ;;
    *) fail "Unknown option '$1' (use --uninstall to remove AIForge)" ;;
  esac
  case "$(uname -s)" in
    Darwin) os=mac ;;
    Linux) os=linux ;;
    *) fail "This installer is for macOS and Linux. On Windows use install.ps1 (see the README)." ;;
  esac

  say "This installs AIForge for you only (no admin password needed)."
  say "The first time it downloads Python and PyTorch: about 200 MB to 1 GB, a few minutes."

  uv="$(find_uv)"
  if [ -z "$uv" ]; then
    install_uv
    uv="$(find_uv)"
    [ -n "$uv" ] || fail "uv was installed but can't be found. Open a new terminal and run this again."
  else
    step "1/3 · uv is already installed ($uv)"
  fi

  step "2/3 · Installing AIForge"
  extra=""
  if [ "$os" = linux ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    # No NVIDIA GPU: the CPU build of PyTorch is ~200 MB instead of 2+ GB.
    say "No NVIDIA GPU found: using the smaller CPU build of PyTorch."
    extra="--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match"
  fi
  # shellcheck disable=SC2086  # $extra is several arguments on purpose
  "$uv" tool install --upgrade --python "$PYTHON" $extra "$PACKAGE"
  bin="$("$uv" tool dir --bin)"
  aiforge="$bin/aiforge"
  [ -x "$aiforge" ] || fail "The install finished but $aiforge is missing."
  case ":$PATH:" in
    *":$bin:"*) ;;
    *) "$uv" tool update-shell >/dev/null 2>&1 || true ;;  # so 'aiforge' also works in new terminals
  esac

  step "3/3 · Adding the AIForge icon"
  if [ "$os" = mac ]; then mac_launcher "$aiforge"; else linux_launcher "$aiforge"; fi

  step "✓ $("$aiforge" --version) is installed"
  say "Start it from the AIForge icon, or type 'aiforge' in a new terminal."
  say "New here? Press F2 in the app for the guided tour."
}

main "$@"
