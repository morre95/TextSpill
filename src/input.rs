//! Synthesising the paste keystroke into whatever window has focus.
//!
//! TextSpill sends **Shift+Insert**, not Ctrl+V: it is the one paste binding
//! that terminals, browsers, editors and GTK/Qt apps all agree on, and it does
//! not collide with a terminal's Ctrl+V literal-next.
//!
//! It never sends Enter. That is the whole point of the tool: the text lands in
//! the shell, in Claude Code, in Codex — and the human decides whether to run it.

use anyhow::{Result, bail};
use std::process::{Command, Stdio};

/// Which tool produced the keystroke, for logging and for the notification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// `ydotool`, via `/dev/uinput`. Works in XWayland windows too.
    Ydotool,
    /// `wtype`, via the wlroots virtual-keyboard protocol. No root/uinput
    /// setup, but invisible to XWayland clients.
    Wtype,
}

impl Backend {
    fn command(self) -> Command {
        let mut cmd = match self {
            // Linux input event codes: KEY_LEFTSHIFT=42, KEY_INSERT=110.
            // `<code>:1` presses, `<code>:0` releases.
            Backend::Ydotool => {
                let mut c = Command::new("ydotool");
                c.args(["key", "42:1", "110:1", "110:0", "42:0"]);
                c
            }
            Backend::Wtype => {
                let mut c = Command::new("wtype");
                c.args(["-M", "shift", "-k", "Insert", "-m", "shift"]);
                c
            }
        };
        cmd.stdin(Stdio::null()).stdout(Stdio::null());
        cmd
    }

    fn name(self) -> &'static str {
        match self {
            Backend::Ydotool => "ydotool",
            Backend::Wtype => "wtype",
        }
    }
}

/// Backends to try, in order. `ydotool` first because it reaches XWayland.
const BACKENDS: [Backend; 2] = [Backend::Ydotool, Backend::Wtype];

/// Presses Shift+Insert in the focused window.
///
/// Set `TEXTSPILL_PASTE_BACKEND` to `ydotool`, `wtype` or `none` to override
/// the automatic choice; `none` leaves the text on the clipboard only.
pub fn paste() -> Result<Option<Backend>> {
    let backends = selected_backends()?;
    if backends.is_empty() {
        tracing::info!("paste disabled by TEXTSPILL_PASTE_BACKEND=none");
        return Ok(None);
    }

    let mut failures = Vec::new();
    for backend in backends {
        match run(backend) {
            Ok(()) => {
                tracing::info!(backend = backend.name(), "pasted Shift+Insert");
                return Ok(Some(backend));
            }
            Err(reason) => {
                tracing::debug!(backend = backend.name(), %reason, "paste backend unavailable");
                failures.push(format!("{}: {reason}", backend.name()));
            }
        }
    }
    bail!("no paste backend worked ({})", failures.join("; "));
}

fn selected_backends() -> Result<Vec<Backend>> {
    match std::env::var("TEXTSPILL_PASTE_BACKEND").as_deref() {
        Err(_) | Ok("") | Ok("auto") => Ok(BACKENDS.to_vec()),
        Ok("ydotool") => Ok(vec![Backend::Ydotool]),
        Ok("wtype") => Ok(vec![Backend::Wtype]),
        Ok("none") => Ok(Vec::new()),
        Ok(other) => bail!("unknown TEXTSPILL_PASTE_BACKEND {other:?}"),
    }
}

fn run(backend: Backend) -> Result<(), String> {
    let output = backend
        .command()
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| e.to_string())?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(match stderr.trim() {
        "" => format!("exited with {}", output.status),
        msg => msg.to_string(),
    })
}
