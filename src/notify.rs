//! Desktop feedback, via `notify-send`.
//!
//! Notifications are advisory: if the notification daemon is missing or broken,
//! TextSpill still records, transcribes and pastes. Every call here therefore
//! swallows failure into a log line instead of an error.

use std::process::{Command, Stdio};

const APP_NAME: &str = "TextSpill";

/// Replaces the previous TextSpill notification rather than stacking a new one.
/// Understood by mako and dunst; harmlessly ignored by daemons that do not.
const SYNC_HINT: &str = "string:x-canonical-private-synchronous:textspill";

/// How long a transcript preview stays on screen.
const TIMEOUT_MS: &str = "4000";

#[derive(Clone, Copy)]
pub enum Urgency {
    Low,
    Normal,
    Critical,
}

impl Urgency {
    fn as_str(self) -> &'static str {
        match self {
            Urgency::Low => "low",
            Urgency::Normal => "normal",
            Urgency::Critical => "critical",
        }
    }
}

pub fn recording() {
    send(Urgency::Low, "🎙 Recording", "");
}

pub fn transcribing() {
    send(Urgency::Low, "Transcribing…", "");
}

pub fn cancelled() {
    send(Urgency::Low, "Recording cancelled", "");
}

/// Shown after the text has been pasted.
pub fn spilled(language: Option<&str>, text: &str) {
    let summary = match language {
        Some(language) => format!("{language}: {}", preview(text)),
        None => preview(text),
    };
    send(Urgency::Normal, &summary, "");
}

/// Shown when the clipboard holds the text but the keystroke did not land.
pub fn clipboard_only(reason: &str) {
    send(
        Urgency::Critical,
        "Text is on the clipboard",
        &format!("Paste it manually with Shift+Insert — {reason}"),
    );
}

pub fn error(message: &str) {
    send(Urgency::Critical, "TextSpill failed", message);
}

pub fn info(summary: &str) {
    send(Urgency::Normal, summary, "");
}

fn send(urgency: Urgency, summary: &str, body: &str) {
    let result = Command::new("notify-send")
        .args(["--app-name", APP_NAME])
        .args(["--urgency", urgency.as_str()])
        .args(["--expire-time", TIMEOUT_MS])
        .args(["--hint", SYNC_HINT])
        // `--` keeps a summary that begins with `-` from being read as a flag.
        .arg("--")
        .arg(summary)
        .arg(body)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if let Err(e) = result {
        tracing::debug!(error = %e, "notify-send unavailable");
    }
}

/// First line of a transcript, short enough for a notification bubble.
fn preview(text: &str) -> String {
    const MAX_CHARS: usize = 60;
    let mut out: String = text.chars().take(MAX_CHARS).collect();
    if text.chars().count() > MAX_CHARS {
        out.push('…');
    }
    out
}
