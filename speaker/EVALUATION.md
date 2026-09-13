# CAM++ / WeSep CPU evaluation — 2026-09-13

**Decision: CPU feasibility passed this smoke test; the quality/calibration gate
has not passed. Product integration is held at the plan's evaluation stage.**

This is not evidence that CAM++ is unsuitable. It is evidence that a speaker
match alone cannot certify the quality of the chosen separator's output, and
that the current short example corpus is insufficient for a production decision.

## Measured setup

- Host CPU: Intel Core i7-1260P, 12 cores / 16 logical processors; Linux x86-64.
- CPU only: four PyTorch threads; one ONNX Runtime thread.
- CAM++: WeSpeaker VoxCeleb ONNX, 27.9 MiB.
- Separator: WeSep BSRNN with its original ECAPA encoder; the model archive is
  249.5 MiB. CAM++ is used after extraction, not inside the separator.
- Python 3.12; `torch`/`torchaudio` 2.5.1+cpu; ONNX Runtime 1.20.1;
  NumPy 1.26.4. Source revisions, model checksums and installation steps are
  pinned in this directory.
- One warm-up pass, then three measurements per input. Each input is 2 seconds,
  except the official mixture, which is 4 seconds. Enrollment is 3 seconds.

The complete measured report is [evaluation-2026-09-13.json](evaluation-2026-09-13.json).
No recordings or embeddings are included in the repository.

## Timing

The combined extraction and verification RTF ranged from **0.494 to 0.526**.
On the 4-second mixture, extraction took a median **2.020 seconds** and CAM++
verification took **0.085 seconds**, including feature computation.
Model initialization took 1.53 seconds. Peak process RSS was approximately
**795 MiB**, including the Python runtime, loading and warm-up.

An initial separator-only sweep over the 4.01-second official example measured
median extraction times of 7.115, 3.818, 2.095 and 3.238 seconds at 1, 2, 4 and 8
threads respectively. Four threads were the best of these tested choices.
The reproducible runner explicitly restores the thread limit after importing
Silero VAD, whose import otherwise silently resets PyTorch to one thread.

These are short-run CPU results, not a sustained live-session latency or memory
guarantee. Overlapping windows, audio transport and either ASR backend remain
unmeasured.

## Voice matching and separation

Cosine scores range from −1 to 1; they are not calibrated probabilities. SI-SDR
compares the extracted waveform to a sample-aligned clean target, allowing a
global scale change. It measures reconstruction error, not transcription accuracy.

| Scenario | CAM++ before extraction | CAM++ after extraction | Output SI-SDR |
|---|---:|---:|---:|
| Target alone | 0.756 | 0.759 | 31.42 dB |
| Other speaker alone | 0.037 | −0.052 | — |
| Two overlapping speakers | 0.110 | 0.739 | 16.13 dB |
| Louder competing speaker | 0.045 | 0.696 | 11.62 dB |
| Official mixed example | 0.149 | 0.614 | No clean reference |
| Three simultaneous speakers | 0.137 | 0.622 | **−0.69 dB** |
| Two other speakers, target absent | 0.061 | −0.049 | — |
| Target followed by another speaker | 0.213 | 0.684 | 31.46 dB |
| Silence | No score | No score | — |

The three-speaker case improved only from −3.05 dB to −0.69 dB SI-SDR.
Its reconstruction error remains larger than the projected target energy, even
though its CAM++ score is comparable to that of the official mixed example.
At an illustrative threshold of 0.60 it would be accepted. This threshold has
**not** been selected for TextSpill. Raising it can reject such a case, but can
also discard target speech; this corpus cannot quantify that tradeoff.

The waveform error may include both residual interfering speech and target
distortion. No ASR was run, so the evaluation does not establish which unwanted
words, if any, would be transcribed. It also does not establish a false-accept
rate or prove that a higher threshold would solve the multi-speaker problem.

## Limits and required next evidence

The fixtures are real public voice recordings mixed synthetically. They are
English and Chinese, not Swedish. Enrollment uses only 3 seconds, while the
planned product uses 15–30 seconds. Positive clean evaluation audio comes from
a non-overlapping portion of the same recording, making verification easier
than an independent recording or microphone change.

Before the product integration gate can pass:

1. Evaluate 15–30 second enrollment against separately recorded Swedish and
   English target speech and other speakers, including similar voices.
2. Include several real room recordings and known clean-source mixtures with
   multiple simultaneous speakers, noise, different levels and microphones.
3. Use separate calibration and held-out evaluation sets to measure false
   acceptance and target-speech loss; do not tune on this tiny smoke set.
4. Measure contamination and target-word accuracy through both ASR backends.
   A clean-sounding voice or a high embedding score is insufficient evidence.
5. Verify sustained throughput with overlapping live windows. If extraction
   quality still fails, evaluate an improved separator or an additional quality
   rejection step before integrating automatic delivery. Keep CPU-only and
   rejection-on-uncertainty requirements.

## Implementation and validation status

Implemented: pinned downloads, CPU model adapters, matching against an enrollment
embedding, reproducible fixture generation, timing/quality reports, input/model
validation, and eight automated tests. The setup and runner were exercised in
a newly created isolated environment.

Not implemented: profile enrollment CLI, warm speaker service, automatic
filtering, Deepgram/local ASR wiring, preview fields, or installer integration.
This follows the plan's instruction to report unsuccessful or unresolved model
evaluation before proceeding with product integration. Existing dictation
behavior remains as it was.
