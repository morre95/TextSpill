//! Synthesising the paste keystroke into whatever window has focus.
//!
//! Shift+Insert is the default. The Chatgpt desktop window uses Ctrl+V;
//! its Shift+Insert shortcut opens an unrelated dialog.
//!
//! It never sends Enter. That is the whole point of the tool: the text lands in
//! the shell, in Claude Code, in Codex — and the human decides whether to run it.

use anyhow::{Result, bail};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shortcut {
    ShiftInsert,
    CtrlV,
}

fn shortcut_for_window(window: &serde_json::Value) -> Shortcut {
    if window.get("class").and_then(|v| v.as_str()) == Some("Chatgpt") {
        Shortcut::CtrlV
    } else {
        Shortcut::ShiftInsert
    }
}

/// Optional Hyprland integration. Other compositors retain the default.
/// Bound the query so a stuck compositor cannot stall dictation indefinitely.
fn focused_shortcut() -> Shortcut {
    let query = || -> Option<Shortcut> {
        std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE")?;
        let mut child = Command::new("hyprctl")
            .args(["activewindow", "-j"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let deadline = Instant::now() + Duration::from_millis(500);
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
            }
        }
        let output = child.wait_with_output().ok()?;
        if !output.status.success() {
            return None;
        }
        let window = serde_json::from_slice(&output.stdout).ok()?;
        Some(shortcut_for_window(&window))
    };
    query().unwrap_or(Shortcut::ShiftInsert)
}

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
    fn command(self, shortcut: Shortcut) -> Command {
        let mut cmd = match self {
            // Linux input event codes: KEY_LEFTSHIFT=42, KEY_INSERT=110.
            // `<code>:1` presses, `<code>:0` releases.
            Backend::Ydotool => {
                let mut c = Command::new("ydotool");
                c.args(match shortcut {
                    Shortcut::ShiftInsert => ["key", "42:1", "110:1", "110:0", "42:0"],
                    // KEY_LEFTCTRL=29, KEY_V=47. Never send KEY_ENTER.
                    Shortcut::CtrlV => ["key", "29:1", "47:1", "47:0", "29:0"],
                });
                c
            }
            Backend::Wtype => {
                let mut c = Command::new("wtype");
                c.args(match shortcut {
                    Shortcut::ShiftInsert => ["-M", "shift", "-k", "Insert", "-m", "shift"],
                    Shortcut::CtrlV => ["-M", "ctrl", "-k", "v", "-m", "ctrl"],
                });
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

/// Sends the app's paste shortcut to the focused window.
///
/// Set `TEXTSPILL_PASTE_BACKEND` to `ydotool`, `wtype` or `none` to override
/// the automatic choice; `none` leaves the text on the clipboard only.
pub fn paste() -> Result<Option<Backend>> {
    let backends = selected_backends()?;
    if backends.is_empty() {
        tracing::info!("paste disabled by TEXTSPILL_PASTE_BACKEND=none");
        return Ok(None);
    }

    let shortcut = focused_shortcut();
    let mut failures = Vec::new();
    for backend in backends {
        match run(backend, shortcut) {
            Ok(()) => {
                tracing::info!(backend = backend.name(), ?shortcut, "paste shortcut sent");
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

fn run(backend: Backend, shortcut: Shortcut) -> Result<(), String> {
    let output = backend
        .command(shortcut)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_confirmed_desktop_class_uses_ctrl_v() {
        assert_eq!(
            shortcut_for_window(&serde_json::json!({"class":"Chatgpt"})),
            Shortcut::CtrlV
        );
        for class in ["Alacritty", "neovim", "Joplin", "firefox", ""] {
            assert_eq!(
                shortcut_for_window(&serde_json::json!({"class":class,"title":"Chatgpt"})),
                Shortcut::ShiftInsert
            );
        }
        assert_eq!(
            shortcut_for_window(&serde_json::json!({})),
            Shortcut::ShiftInsert
        );
    }

    #[test]
    fn ctrl_v_uses_balanced_keys_without_enter() {
        let cmd = Backend::Ydotool.command(Shortcut::CtrlV);
        assert_eq!(
            cmd.get_args().collect::<Vec<_>>(),
            ["key", "29:1", "47:1", "47:0", "29:0"]
        );
        let cmd = Backend::Wtype.command(Shortcut::CtrlV);
        assert_eq!(
            cmd.get_args().collect::<Vec<_>>(),
            ["-M", "ctrl", "-k", "v", "-m", "ctrl"]
        );
    }
}
