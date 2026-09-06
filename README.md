# TextSpill

**System-wide voice-to-text for Linux. Speak, transcribe, spill text wherever your cursor is.**

Press a hotkey, talk in Swedish or English, press it again. The transcription lands at
your cursor — in a terminal, in Claude Code, in Codex, in Firefox, in Neovim.

TextSpill **never presses Enter**. It types the text and stops. What you do with it is
your call.

## How it works

```
        hotkey ──▶ textspill toggle
                        │
        ┌───────────────┼────────────────┐
        │               │                │
    pw-record       asr.sock          wl-copy
   16 kHz mono     (Unix socket)      + ydotool
    → WAV              │              Shift+Insert
                       ▼
            ┌──────────────────────┐
            │ textspill-asr.service│
            │  Qwen3-ASR stays warm│
            │  in RAM / VRAM       │
            └──────────────────────┘
```

Two processes, on purpose:

- **`textspill`** (Rust) is the thing your hotkey runs. It starts in a few milliseconds,
  records, asks, pastes, exits. It does no machine learning.
- **`textspill-asr.service`** (Python) loads Qwen3-ASR **once** and keeps it resident.
  A dictation never pays the model load time — that is the whole latency design.

They talk over `$XDG_RUNTIME_DIR/textspill/asr.sock` with one line of JSON each way.

## Install

### 1. System packages

```bash
sudo pacman -S --needed rust pipewire pipewire-audio wl-clipboard libnotify ydotool python312
```

| Package | Provides | Used for |
|---|---|---|
| `rust` | `cargo`, `rustc` | building `textspill` |
| `pipewire-audio` | `pw-record` | microphone capture |
| `wl-clipboard` | `wl-copy`, `wl-paste` | clipboard |
| `ydotool` | `ydotool`, `ydotoold` | the Shift+Insert keystroke |
| `libnotify` | `notify-send` | on-screen feedback |
| `python312` | Python 3.12 | the ASR virtualenv (PyTorch lags the newest Python) |

`wtype` is an optional alternative to `ydotool` — see [Paste backends](#paste-backends).

### 2. TextSpill

```bash
git clone https://github.com/cryptofarian/textspill
cd textspill
./install.sh
```

`install.sh` builds the binary into `~/.local/bin/textspill`, creates the ASR virtualenv
in `~/.local/share/textspill/venv`, symlinks the daemon there (so editing the repo takes
effect on the next restart) and installs the user service. It downloads PyTorch, so
expect a few GB and a few minutes.

Make sure `~/.local/bin` is on your `PATH`.

### 3. The ASR daemon

```bash
systemctl --user enable --now textspill-asr.service
journalctl --user -u textspill-asr.service -f
```

The first start downloads `Qwen/Qwen3-ASR-0.6B-hf` (~1.5 GB) into `~/.cache/huggingface`
and runs one warm-up inference. Wait for `listening on …` before dictating.

### 4. The paste backend

`ydotool` writes to `/dev/uinput`, which needs group membership:

```bash
sudo usermod -aG input "$USER"      # then log out and back in
systemctl --user enable --now ydotool.service
```

The `ydotool` package ships a udev rule that gives the `input` group access to
`/dev/uinput`. Check it took effect with `ls -l /dev/uinput` — it should be
`crw-rw---- root input`.

### 5. The hotkey

Bind `textspill toggle` in your compositor. TextSpill itself is compositor-agnostic; it
never grabs a key.

**Omarchy** — add to `~/.config/hypr/bindings.lua`, where personal overrides live:

```lua
o.bind("SUPER + PERIOD", "Dictate", "textspill toggle")
```

`SUPER + .` is free on a stock Omarchy install; the emoji picker is on
`SUPER + CTRL + E`, and only `SUPER + CTRL + .` (Transcode) uses the period key.
`SUPER + SPACE` is the Omarchy menu — if you want that key for dictation, unbind it
first:

```lua
hl.unbind("SUPER + SPACE")
o.bind("SUPER + SPACE", "Dictate", "textspill toggle")
```

Then `hyprctl reload` and confirm with `hyprctl configerrors` (silence means clean) and
`omarchy menu keybindings --print | grep Dictate`.

**Plain Hyprland** — in `hyprland.conf`:

```ini
bind = SUPER, PERIOD, exec, textspill toggle
```

**Sway / river / other wlroots:**

```
bindsym $mod+period exec textspill toggle
```

`textspill` must be on the compositor's `PATH`. Check with
`tr '\0' '\n' < /proc/$(pgrep -x Hyprland)/environ | grep ^PATH=` — if `~/.local/bin`
is missing, use the absolute path in the binding.

## Use

```bash
textspill toggle     # start recording, or stop → transcribe → paste
textspill start      # start recording
textspill stop       # stop, transcribe, copy, paste
textspill cancel     # stop and throw the audio away
textspill status     # idle | recording | transcribing
```

`status` never blocks, so it is safe to poll from a bar module.

### Push-to-talk

Hold the key instead of toggling. This needs no extra code — `start` and `stop` already
do the two halves — only Hyprland's `release` flag on a second binding:

```lua
-- Toggle: press to start, press again to transcribe. For longer dictations.
o.bind("SUPER + PERIOD", "Dictate", "textspill toggle")

-- Push-to-talk: hold while speaking, release to transcribe. For short ones.
o.bind("SUPER + SHIFT + PERIOD", "Dictate (hold)", "textspill start")
o.bind("SUPER + SHIFT + PERIOD", nil, "textspill stop", { release = true })
```

Both modes can coexist on different keys, as above. Do not put them on the *same* key:
the press would start a recording and the release would immediately stop it.

A tap shorter than the recorder's startup takes `stop` into the lock while `start` still
holds it. That is handled — `acquire_lock` waits rather than refusing — so the worst case
is a "Nothing recorded" notification, never a recording left running.

## Configuration

### Vocabulary

Qwen3-ASR is biased towards a list of terms you actually say, so it writes *Hyprland* and
*crate* instead of near-miss homophones. The shipped list is `asr/context.txt`. To
customise it:

```bash
mkdir -p ~/.config/textspill
cp asr/context.txt ~/.config/textspill/context.txt
$EDITOR ~/.config/textspill/context.txt
systemctl --user restart textspill-asr.service
```

The config copy takes precedence and survives updates. Keep the list to words you use —
a long list dilutes the bias.

### Language

None needed. Qwen3-ASR identifies the language itself, so Swedish and English can alternate
between dictations. The detected language is shown in the notification and logged.

### Environment variables

| Variable | Effect |
|---|---|
| `RUST_LOG=debug` | verbose client logging (`RUST_LOG=debug textspill toggle`) |
| `TEXTSPILL_PASTE_BACKEND` | `ydotool`, `wtype`, `none` or `auto` (default) |
| `TEXTSPILL_ASR_MODEL` | another checkpoint, e.g. `Qwen/Qwen3-ASR-1.7B-hf` |

## Paste backends

TextSpill sends **Shift+Insert**, the one paste binding terminals, browsers, editors and
GTK/Qt apps all agree on. Ctrl+V means "literal next" in some terminals.

Two backends are tried in order:

1. **`ydotool`** — synthesises at the kernel level via `/dev/uinput`. Works everywhere,
   XWayland included. Needs the `input` group and `ydotoold` running.
2. **`wtype`** — uses the wlroots virtual-keyboard protocol. No privileges needed, but
   XWayland clients cannot see it.

Set `TEXTSPILL_PASTE_BACKEND=none` to only fill the clipboard.

## Testing each piece

**Recording**

```bash
textspill start && sleep 3 && textspill cancel
# or capture and keep the file:
pw-record --rate 16000 --channels 1 --format s16 /tmp/t.wav   # Ctrl-C to stop
file /tmp/t.wav      # → RIFF … 16 bit, mono 16000 Hz
```

**The model, without the socket**

```bash
~/.local/share/textspill/venv/bin/python asr/daemon.py --transcribe /tmp/t.wav
# {"text": "…", "language": "Swedish"}
```

**The socket protocol, by hand**

```bash
printf '{"audio_path":"/tmp/t.wav"}\n' \
  | socat - UNIX-CONNECT:"$XDG_RUNTIME_DIR/textspill/asr.sock"
```

**Clipboard**

```bash
printf 'hej' | wl-copy -n && wl-paste -n | od -c    # no trailing \n
```

**Paste keystroke** — focus a terminal, then from another one:

```bash
printf 'hej' | wl-copy -n && sleep 2 && ydotool key 42:1 110:1 110:0 42:0
```

**State machine**

```bash
textspill status                                   # idle
textspill start && textspill status                # recording
echo 999999 > "$XDG_RUNTIME_DIR/textspill/recording.pid"
textspill status                                   # idle — stale PID cleaned up
```

## Tests

```bash
cargo test                  # 16 tests: state machine, PID handling, lock, IPC protocol
python3 asr/test_daemon.py  # 18 tests: request validation, socket protocol, vocabulary
```

The Python suite stubs out Qwen3-ASR, so it needs neither the virtualenv nor a GPU. The
IPC tests run against a real Unix socket with a scripted daemon, covering the replies that
matter: success, an error object, invalid JSON, a silent close, and a daemon that never
answers.

## End-to-end test

```bash
systemctl --user status textspill-asr.service      # must be active
systemctl --user status ydotool.service            # must be active
```

Open a terminal, type `echo ` and leave the cursor there. Then:

1. Press the hotkey. A **🎙 Recording** notification appears.
2. Say: *"Kan du refaktorera den här funktionen och lägga till bättre error handling?"*
3. Press the hotkey again. **Transcribing…**, then **Swedish: Kan du refaktorera…**
4. The sentence appears after `echo `. **The prompt is not submitted.**

With logging, to see where the time goes:

```bash
RUST_LOG=info textspill toggle
# … transcription received latency_ms=680 language="Swedish" chars=71
# … copied to clipboard chars=71
# … pasted Shift+Insert backend="ydotool"
```

## Troubleshooting

Wayland has a few sharp edges. Roughly in order of how often they bite:

**Nothing is pasted, but the text is on the clipboard.**
`ydotoold` is not running, or you are not in the `input` group yet (group changes need a
full logout). Check `systemctl --user status ydotool.service` and `groups`. As a
stopgap, `TEXTSPILL_PASTE_BACKEND=wtype` needs no privileges.

**Pasting works in native apps but not in Electron/Steam/older apps.** Those are XWayland
clients, which cannot see `wtype`'s virtual keyboard. Use `ydotool`.

**"ASR daemon is not reachable".** The service is not up. `journalctl --user -u
textspill-asr.service -e`. Your recording is *not* lost — it is kept at
`$XDG_RUNTIME_DIR/textspill/recording.wav` and the next `start` is what clears it.

**The clipboard is empty when run from the hotkey but works from a terminal.** The
compositor did not export `WAYLAND_DISPLAY`/`XDG_RUNTIME_DIR` into the exec environment.
On Hyprland with uwsm this is handled; otherwise check `systemctl --user show-environment`.

**Silence, or a "Nothing recorded" notification.** `pw-record` grabbed the wrong source.
List them with `wpctl status` and set a default with `wpctl set-default <id>`.
`$XDG_RUNTIME_DIR/textspill/pw-record.log` has its stderr.

**The first fraction of a second is missing.** `pw-record` needs roughly a quarter of a
second to create its PipeWire node and link it to the source, so a word begun the instant
the hotkey is pressed can be clipped. Pause briefly before speaking. Removing this delay
is what native PipeWire capture on the roadmap is for.

**The first dictation after a reboot is slow.** The daemon warms up at start, but the
model files still have to come off disk. Later ones are fast.

**A notification says "another textspill command is already running".** Two hotkey
presses raced, and the lock did its job — exactly one of them acted.

## Design notes

- **State is derived, not stored.** `textspill status` asks `/proc` whether the recorded
  PID is alive *and* still named `pw-record`, so a crashed recorder reads as `idle`, the
  stale file is cleaned up, and a recycled PID is never signalled.
- **One lock per command.** State-changing commands hold an `flock` on
  `textspill.lock`, so a double hotkey press cannot start two recorders. A second command
  waits up to three seconds rather than failing — push-to-talk releases the key
  milliseconds after pressing it — and then gives up with a clear message rather than
  blocking the hotkey. The kernel releases the lock on exit, so a crash cannot wedge it.
  `status` is read-only and never takes it.
- **Audio survives failure.** `recording.wav` is only deleted after a successful spill, or
  by the next `start`. If the daemon is down, the dictation is still on disk.
- **Newlines are flattened.** A newline on the clipboard is an Enter keypress when pasted
  into a shell. `sanitize()` in `src/main.rs` collapses them, and `wl-copy --trim-newline`
  is a second line of defence.
- **No shell.** Every subprocess is `std::process::Command` with separate arguments, and
  the transcription reaches `wl-copy` on stdin — never as a command-line argument.
- **Runtime files are private.** `$XDG_RUNTIME_DIR/textspill/` is created 0700 and refused
  if owned by someone else; the socket is bound under a 0177 umask and is 0600.

## Layout

```
textspill/
├── Cargo.toml
├── install.sh
├── src/
│   ├── main.rs        command dispatch, the record → transcribe → paste flow
│   ├── paths.rs       every path TextSpill touches
│   ├── state.rs       PID files, stale-state cleanup, the lock
│   ├── audio.rs       pw-record lifecycle
│   ├── ipc.rs         Unix socket client, timeouts, error mapping
│   ├── clipboard.rs   wl-copy
│   ├── input.rs       ydotool / wtype paste backends
│   └── notify.rs      notify-send
├── asr/
│   ├── daemon.py      warm Qwen3-ASR over a Unix socket
│   └── context.txt    vocabulary bias
└── systemd/
    └── textspill-asr.service
```

`audio.rs`, `clipboard.rs` and `input.rs` each wrap exactly one external tool, so
replacing `pw-record` with native PipeWire, or `wl-copy`/`ydotool` with native Wayland
protocols, is a change to one file.

## Roadmap

- Streaming ASR for lower perceived latency
- Native PipeWire capture instead of `pw-record`
- Context profiles (`textspill --profile coding toggle`) and project vocabularies
- A recording indicator for Waybar

## License

MIT
