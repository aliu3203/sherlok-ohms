# sherlok-ohms: project context

Hackathon project ("Best Use of QNX"). A Raspberry Pi running **QNX OS** watches a breadboard with a camera and checks that the circuit matches a schema. GPIO LEDs: green = OK, yellow = can't verify, red = wrong. **RUNBOOK.md is the operational guide; keep it in sync with the code.**

Contest rules: (1) the product must run QNX OS; (2) it must include an open-source AI module listed at oss.qnx.com. We use **tflite-runtime**, plus optionally **mediapipe** for hands.

## Platform facts (verified)
- Pi 5 + Camera Module 3 on DISP0, QNX 8.0.5 Quick Start image. Camera = unit 3 (`/dev/sensor/camera3`).
- Camera from Python: `qnx_camera_helper.py` (QNX codelab `mediapipe-camera-python`, downloaded into the repo root, not committed). `cv2.VideoCapture` does NOT work on QNX.
- apk packages: tflite-runtime 2.21 + python3-tflite-runtime, opencv 4.12 + python3-opencv, python3-numpy, mediapipe 0.10.26, onnxruntime. Python 3.14.
- GPIO: `rpi_gpio` module (BCM 17 / 27 / 22 = green / yellow / red).

## Design (as built)
- **Geometry decides where:** 4 clicked corner holes (+ rail ends) → homography → canonical top-down board, 24 px pitch, ~400 hole centers (`sherlok/board.py`). Nodes: a–e of column N = `A{N}`, f–j = `B{N}`, rails = `R:{name}`.
- **Hole model (AI):** a TFLite CNN on 48×48 crops per hole → P(empty / covered / plugged) (`sherlok/vision.py: HoleClassifier`).
  - Trained on our own camera's frames. **Labels come from the schema of the built circuit**: schema pins = plugged; changed-but-not-pin = covered; unchanged = empty (`train/make_dataset.py`).
  - Schema labels are the true holes on the board plane, so the model learns perspective/parallax offsets of raised wires. A fixed-camera dataset is required.
- **Grouping:** diff vs `ref/empty.png` (Lab, brightness-normalized) → blobs. Covered/plugged holes are assigned to the nearest blob (≤ 1.5 pitch). Ends = plugged holes (model) or, without a model, farthest changed holes along the blob axis (fallback; weak with raised wires).
- **Check:** compare by node, not hole (`sherlok/check.py`). Exact matches first, then "misplaced" (one end right), then missing / extra / short. Severity = error if part conf ≥ `high_conf` (0.8), else warn (yellow). Debounce: red after 2 s persistent, green after 1 s.
- Schemas: `.txt` (`ID TYPE PIN PIN` per line) or `.json`.

## Known limitations / next ideas
- The model scores holes independently. A fully occluded lead → low confidence → yellow (intended).
- Next ideas: aggregate P(plugged) per strip before deciding; use P(covered) to suppress false ends; wire color.
- The camera must be high and straight down; parallax offset ≈ wire height × off-center distance / camera height.

## Workflow
`collect.py` (Pi, guided capture into `captures/<schema>/`) → laptop `train/train_all.py` → `models/holes.tflite` → `main.py --model ...`. Test without hardware: `tools/fake_board.py` (flat synthetic wires only).
