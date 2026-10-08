# Overnight Assistant — Project Status and Modernization Roadmap

Updated: 2026-10-08

## Mission

Overnight Assistant is a privacy-first, local/edge safety assistant intended to detect a possible fall or distress event without storing or streaming ordinary room video.

This repository began as an in-room camera-vision prototype for elderly fall detection. The current code is a proof of concept, not a validated medical or emergency-response system.

## Verified current implementation

- Python + OpenCV camera loop.
- YOLOv3/Darknet person detection from local `yolov3-coco` model files.
- A detected person's bounding-box aspect ratio is used as a simple horizontal/vertical heuristic.
- Horizontal detections are logged; there is not yet a robust temporal fall classifier.
- The README states that video should be processed and discarded rather than retained.
- A static HTML site is included, but the safety engine is not yet integrated into a production dashboard/alert workflow.

## What AI assistance changes

Do not make a cloud LLM responsible for deciding whether a fall occurred. The safety-critical detection path should remain deterministic/local and continue working without Internet access.

Use modern AI where it is useful:

1. **Pose estimation** — replace bounding-box orientation alone with body landmarks/keypoints.
2. **Temporal reasoning** — distinguish a real fall from sleeping, sitting, bending, tying shoes, reaching toward the floor, or lying in bed by considering motion over time.
3. **Multi-signal confidence** — combine person detection, pose geometry, vertical displacement, velocity, time near the floor, and recovery/non-recovery.
4. **AI-assisted verification** — optionally use a local secondary model to classify an already-triggered event or produce a human-readable explanation. It must not silently override the core detector.
5. **AI development assistance** — use coding/research assistants to analyze failures, propose tests, summarize event metadata, and improve models from labeled test data while preserving a reproducible source of truth in Git.

## Proposed event pipeline

`camera -> person/pose inference -> temporal state machine -> candidate event -> confirmation window -> alert adapter`

Suggested states:

`NORMAL -> DESCENDING -> DOWN -> VERIFYING -> ALERT / RECOVERED`

A candidate fall should require multiple independent signals rather than `width / height > threshold` alone. Example signals include:

- torso/keypoint orientation;
- hip/head height change relative to the frame or calibrated floor;
- downward velocity;
- transition time from upright to down;
- persistence near floor level;
- absence of rapid recovery;
- confidence/visibility of landmarks.

All thresholds must be configuration values and must be tested rather than treated as universal medical constants.

## Privacy architecture

Default target behavior:

- inference occurs locally;
- raw frames remain in memory only long enough for inference;
- raw video is not saved by default;
- no remote camera feed is required;
- event logs contain timestamps, detector state, confidence/features, system health, and alert outcome — not room imagery;
- optional event-image/video capture, if ever added, must be explicitly opt-in and separated from the default privacy-preserving mode.

## Modernization phases

### Phase 0 — make the prototype reproducible

- Add a dependency manifest and supported Python/platform matrix.
- Document where model files come from and their hashes/licenses.
- Add configuration instead of hard-coded paths and thresholds.
- Add CLI arguments for camera/device and test-video input.
- Separate capture, inference, decision logic, logging, and alerts into modules.
- Add automated tests for decision logic.

### Phase 1 — pose + temporal detector

- Evaluate a current pose-estimation backend suitable for local inference (for example MediaPipe Pose Landmarker or a modern YOLO pose model).
- Preserve the detector backend behind an interface so it can be replaced later.
- Implement the temporal state machine.
- Build replay support so recorded public/test datasets can be evaluated offline without changing detector code.

### Phase 2 — false-positive control

Create labeled scenarios including:

- normal walking;
- sitting and standing;
- getting into/out of bed;
- lying/sleeping;
- bending/reaching;
- kneeling;
- intentional floor activity;
- simulated falls in multiple directions;
- partial occlusion, darkness, unusual camera angles, and person leaving frame.

Track precision, recall/sensitivity, specificity, false alarms per hour/night, missed events, detection latency, and recovery handling. Accuracy alone is not enough for a rare-event detector.

### Phase 3 — alerting and health monitoring

- Pluggable alert interface rather than detector-specific notification code.
- Confirmation/escalation timers.
- Camera disconnected/covered/stalled detection.
- Model load failure and inference watchdog.
- Local heartbeat/health status.
- Clear fail-safe state when monitoring is unavailable.

### Phase 4 — optional sensor fusion

Keep the architecture open to privacy-preserving corroborating sensors such as radar, thermal, wearable/IMU, floor/pressure, or other local sensors. They should be adapters, not requirements for the initial camera prototype.

## AI/LLM boundary

LLMs can help with explanations, development, log summaries, test generation, and caregiver-facing text. They must not be the only fall detector, must not require sending continuous private imagery to a remote service, and must not turn an uncertain event into a medical diagnosis.

## Definition of a useful V1

V1 is not 'the camera recognizes a horizontal person.' A useful V1 must:

- start reliably;
- detect and track one person locally;
- classify transitions over time;
- suppress common non-fall horizontal/low-body activities;
- trigger a testable alert event;
- expose monitoring/system-health state;
- retain no raw video by default;
- run repeatable evaluation against labeled clips;
- produce metrics and an error report;
- document known failure modes.

## Immediate next implementation target

Build a replayable detector core before adding UI polish:

1. `DetectorBackend` interface returning person/pose observations.
2. `FallStateMachine` consuming timestamped observations.
3. JSON/YAML configuration for thresholds/timers.
4. CLI that accepts either camera input or a video file.
5. structured JSONL event log.
6. unit tests using synthetic observation sequences.
7. benchmark script producing a confusion matrix, event-level precision/recall, latency, and false-alert rate.

This creates a stable base on which MediaPipe, YOLO pose, sensor fusion, or later models can be compared without rewriting the whole application.

## Safety note

Until independently tested and validated, Overnight Assistant should be described as an experimental assistive safety project, not a medical device or a replacement for supervision/emergency services.
