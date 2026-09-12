//! Deepgram transport. Credentials and server response bodies are never logged.
use crate::{
    config::{self, Config},
    transcription::Transcription,
};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{fs::File, io::Read, path::Path, time::Duration};

pub const HTTP_URL: &str = "https://api.deepgram.com/v1/listen";
pub const WS_URL: &str = "wss://api.deepgram.com/v1/listen";

/// Debug builds can exercise the complete CLI against a loopback test server.
/// Release builds always use Deepgram's TLS endpoints.
pub fn endpoint(streaming: bool) -> String {
    #[cfg(debug_assertions)]
    if let Ok(base) = std::env::var("_TEXTSPILL_TEST_DEEPGRAM_ENDPOINT")
        && let Ok(url) = reqwest::Url::parse(&base)
        && url.scheme() == "http"
        && url.host_str() == Some("127.0.0.1")
    {
        return if streaming {
            base.replacen("http://", "ws://", 1)
        } else {
            base
        };
    }
    if streaming {
        WS_URL.into()
    } else {
        HTTP_URL.into()
    }
}

pub fn transcribe(config: &Config, wav: &Path) -> Result<Transcription> {
    transcribe_at(
        config,
        wav,
        &endpoint(false),
        &config::api_key()?,
        Duration::from_secs(120),
    )
}

fn transcribe_at(
    config: &Config,
    wav: &Path,
    url: &str,
    key: &str,
    timeout: Duration,
) -> Result<Transcription> {
    config.validate_language()?;
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let mut auth = reqwest::header::HeaderValue::from_str(&format!("Token {key}"))
        .map_err(|_| anyhow::anyhow!("invalid Deepgram API key"))?;
    auth.set_sensitive(true);
    let response = client
        .post(url)
        .query(&[
            ("model", "nova-3"),
            ("language", config.deepgram_language.as_str()),
            ("punctuate", "true"),
        ])
        .header(reqwest::header::AUTHORIZATION, auth)
        .header(reqwest::header::CONTENT_TYPE, "audio/wav")
        .body(File::open(wav)?)
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                anyhow::anyhow!("Deepgram request timed out; audio kept")
            } else {
                anyhow::anyhow!("Deepgram connection failed; check network access")
            }
        })?;
    if !response.status().is_success() {
        bail!("{}", status_error(response.status().as_u16()));
    }
    let mut bytes = Vec::new();
    response
        .take(1_048_577)
        .read_to_end(&mut bytes)
        .context("could not read Deepgram response")?;
    if bytes.len() > 1_048_576 {
        bail!("Deepgram response is too large");
    }
    parse_batch(&bytes, &config.deepgram_language)
}

pub fn status_error(status: u16) -> String {
    let hint = match status {
        401 | 403 => "check your API key and its transcription permissions",
        402 => "check your Deepgram account balance",
        429 => "rate limit reached; try again later",
        _ => "request failed; audio kept for recovery",
    };
    format!("Deepgram HTTP {status}: {hint}")
}

#[derive(Deserialize)]
struct Batch {
    results: Results,
}
#[derive(Deserialize)]
struct Results {
    channels: Vec<Channel>,
}
#[derive(Deserialize)]
pub struct Channel {
    pub alternatives: Vec<Alternative>,
}
#[derive(Deserialize)]
pub struct Alternative {
    pub transcript: String,
}

fn parse_batch(bytes: &[u8], language: &str) -> Result<Transcription> {
    let parsed: Batch =
        serde_json::from_slice(bytes).context("invalid Deepgram transcription response")?;
    let alternative = parsed
        .results
        .channels
        .into_iter()
        .next()
        .and_then(|c| c.alternatives.into_iter().next())
        .context("Deepgram response has no transcription alternative")?;
    Ok(Transcription {
        text: alternative.transcript,
        language: Some(language.into()),
    })
}

#[derive(Deserialize)]
pub struct StreamResult {
    pub is_final: bool,
    pub start: f64,
    pub duration: f64,
    pub channel: Channel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
    };
    #[test]
    fn batch_response_requires_transcript_but_accepts_silence() {
        assert!(parse_batch(br#"{"results":{"channels":[]}}"#, "sv").is_err());
        assert!(parse_batch(b"not json", "sv").is_err());
        let result = parse_batch(
            br#"{"results":{"channels":[{"alternatives":[{"transcript":""}]}]}}"#,
            "sv",
        )
        .unwrap();
        assert_eq!(result.text, "");
    }

    #[test]
    fn http_sends_wav_and_handles_success_auth_rate_limit_and_timeout() {
        let wav = std::env::temp_dir().join(format!("ts-http-{}.wav", std::process::id()));
        std::fs::write(&wav, b"RIFF test WAV").unwrap();
        for status in [200, 401, 429, 500, 0] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}/v1/listen", listener.local_addr().unwrap());
            let handle = thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(line.contains("model=nova-3") && line.contains("language=sv"));
                let mut size = 0;
                let mut auth = false;
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    let lower = line.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("content-length: ") {
                        size = value.trim().parse().unwrap();
                    }
                    if lower == "authorization: token test-secret\r\n" {
                        auth = true;
                    }
                }
                assert!(auth);
                let mut body = vec![0; size];
                reader.read_exact(&mut body).unwrap();
                assert_eq!(body, b"RIFF test WAV");
                if status == 0 {
                    thread::sleep(Duration::from_millis(400));
                    return;
                }
                let body = r#"{"results":{"channels":[{"alternatives":[{"transcript":"hej"}]}]}}"#;
                write!(socket, "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            });
            let result = transcribe_at(
                &Config::default(),
                &wav,
                &url,
                "test-secret",
                Duration::from_millis(200),
            );
            if status == 200 {
                assert_eq!(result.unwrap().text, "hej");
            } else {
                let error = format!("{:#}", result.unwrap_err());
                assert!(!error.contains("test-secret"));
            }
            handle.join().unwrap();
        }
        std::fs::remove_file(wav).unwrap();
    }
}
