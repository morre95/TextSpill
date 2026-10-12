//! The PCM format shared by capture, WAV snapshots and streaming transport.

pub const SAMPLE_RATE: u32 = 16_000;
pub const CHANNELS: u16 = 1;
pub const BITS_PER_SAMPLE: u16 = 16;
pub const BYTES_PER_SAMPLE: u16 = BITS_PER_SAMPLE / 8;
pub const BLOCK_ALIGN: u16 = CHANNELS * BYTES_PER_SAMPLE;
pub const BYTES_PER_SECOND: u64 = SAMPLE_RATE as u64 * BLOCK_ALIGN as u64;
pub const PCM_FORMAT: u16 = 1;
pub const CAPTURE_FORMAT: &str = "s16";
pub const STREAM_ENCODING: &str = "linear16";
pub const WAV_HEADER_BYTES: u64 = 44;

pub const fn bytes_for_millis(millis: u64) -> u64 {
    (SAMPLE_RATE as u64 * millis / 1000) * BLOCK_ALIGN as u64
}

/// Round to a complete sample frame, matching transcript timing to captured PCM.
pub fn bytes_for_seconds(seconds: f64) -> u64 {
    ((seconds * SAMPLE_RATE as f64).round() as u64).saturating_mul(BLOCK_ALIGN as u64)
}

pub fn seconds_for_bytes(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_SECOND as f64
}
