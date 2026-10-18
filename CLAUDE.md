# sherlok-ohms — project context

Hackathon project ("Best Use of QNX"). A Raspberry Pi running **QNX OS** watches a breadboard with a camera and checks that the circuit being built matches a provided schema. If it doesn't, it signals an error (GPIO LED/buzzer).

Hard contest rules: (1) the product must run QNX OS; (2) it must include an open-source AI module listed at oss.qnx.com.

## Platform facts (verified 2026-10-18)
- The Pi does NOT run QNX by default. Flash the QNX 8.0 Quick Start Target Image: free QNX Everywhere license → QNX Software Center → `com.qnx.qnx800.quickstart.rpi5` (or `.rpi4`) → write the `.img` to a microSD of 32 GB or more.
- Target: **Raspberry Pi 5 + Camera Module 3 on DISP0, QNX 8.0.5 image**. Camera = unit 3 (`/dev/sensor/camera3`, config `/etc/config/sensor/sensor_rpi5.conf`).
- Camera from Python: `qnx_camera_helper.py` (`QnxCamera(unit=3).open(); rgb, info = cam.read()`) from the QNX codelab "mediapipe-camera-python" (github.com/qnx/codelabs). It wraps `libcamapi.so.1` via ctypes and outputs RGB numpy frames (2304×1296 NV12 stream). `cv2.VideoCapture(0)` does NOT work on QNX.
- AI/vision packages in QNX ports (github.com/qnx-ports/aports, `extra/`), installed with `apk add`: tflite-runtime 2.21 (+python3-tflite-runtime), onnxruntime 1.24.4, ncnn, mediapipe 0.10.26, opencv 4.12 (+contrib, so cv2.aruco is available), numpy, pillow, scipy, llama.cpp. Python 3.14.
- TFLite on a Pi 5 under QNX: SSD MobileNet ≈ 38 ms/frame with `num_threads=4`. The chosen inference runtime is **tflite-runtime**, with onnxruntime as backup.
- GPIO: `import rpi_gpio as GPIO; GPIO.setmode(GPIO.BCM); GPIO.setup(pin, GPIO.OUT); GPIO.output(pin, GPIO.HIGH)`.
- Train models on a laptop or Colab, then copy the `.tflite` files to the Pi.

## Architecture (decided)
Core principle: **geometry decides *where*; ML answers yes/no questions at known locations; code combines the evidence.** No model ever has to regress an exact hole from a downscaled image. YOLO at 640 px input gives ≈ 8 px per 2.54 mm hole, which is too coarse; at full resolution it's ≈ 28 px per hole.

1. **Fiducials → hole grid (deterministic).** 4 ArUco markers at the breadboard corners → `cv2.findHomography` → `warpPerspective` to a top-down board. Every hole (rows a–j × columns 1–63, plus rails) has a known pixel center. Target ≥ 20 px per hole pitch. A 400-point half-size board doubles that.
2. **Per-hole occupancy (ML, primary).** Crop ~32×32 px around each hole and classify empty / occupied (+ wire color). Start with a classical baseline: diff against an empty-board reference. Then train a tiny CNN → TFLite int8, running all ~830 crops in one batch. Labels are nearly free (an empty-board photo gives all `empty`).
3. **Part detector (ML, coarse).** YOLO11n (detect, or pose with 2 keypoints = lead insertion points). Classes: wire, resistor, led, button, capacitor, ic, hand. It says *what* is there and roughly where, never which hole. Train on the warped board image using 150–300 photos from our own mounted Pi camera. When labeling, keypoint = where the bare lead enters the board. Export with `format="tflite", int8=True`.
4. **Evidence fusion + assignment (code).** For each part end k and hole h:
   `score(k,h) ∝ P_position(h) × P_occupied(h) × P_color_match(h)`
   - P_position = Gaussian around the predicted keypoint; measure σ on validation data.
   - Constraint: each occupied hole holds one lead → global matching with `scipy.optimize.linear_sum_assignment`.
   - An occupied hole that no part claims means a stray or extra wire.
   - Average scores over frames, since the board is static.
5. **Netlist + schema check.** Breadboard rules: per column, a–e = one node and f–j = another; each rail = one node (check for split rails). Wires merge nodes (union-find).
   - Schema JSON: exact holes (`{"parts":[{"id":"R1","type":"resistor","pins":["a5","a10"]}]}`) or a netlist compared via `networkx.is_isomorphic(..., edge_match=type)`.
   - First demo milestone: compare occupancy alone against the exact-hole schema.
6. **Signal.** GPIO LEDs plus an optional on-screen overlay.

## Confidence handling (decided)
- **Aggregate probabilities per electrical node (strip), not per hole.** a5 vs b5 is the same node, so that split doesn't matter.
- For each end, compute p1 (top node) and p2 (runner-up). Starting thresholds, to tune on real photos:
  - **High:** p1 > 0.85 and p1 − p2 > 0.5 → accept; compare with the schema.
  - **Medium/ambiguous:** probability split across 2–3 nodes → if the expected node is among them, "probably OK, keep watching"; if none of them is expected → likely error, alarm after it persists.
  - **Low/no evidence:** p1 < 0.2 or the end isn't detected (hand, occlusion, unseated lead) → "can't verify". Never a red alarm.
- Guard against **confidently wrong** readings: require independent clues to agree (position, occupancy, color) plus consistency over time, and check calibration on a validation set.
- Lights: **green** = high confidence and matches; **red** = high-confidence mismatch, or ambiguous with every candidate wrong, persisting about 2 s; **yellow** = can't verify (e.g. a `hand` is detected).

## Pipeline
Camera Module 3 → QnxCamera (full-res RGB) → ArUco warp → occupancy classifier + YOLO part detector (tflite-runtime) → evidence fusion / assignment → netlist → schema compare → GPIO

## Order of work
1. Flash QNX 8.0.5. Camera smoke test (codelab `demo_capture_faces.py`). `apk add` tflite-runtime + opencv. Blink an LED.
2. Build the mount, print ArUco markers, write the warp + hole grid. Measure px per hole.
3. Occupancy (classical baseline → tiny CNN). Exact-hole schema check = first demo.
4. YOLO parts + fusion/assignment + netlist check.
5. Tune thresholds, rehearse the demo.

## References
- github.com/qnx/codelabs: `mobilenet-tflite-qnx`, `mediapipe-camera-python` (camera helper), `mediapipe-camera-sample`
- github.com/qnx-ports/aports (extra/)
- QNX Raspberry Pi image docs: https://www.qnx.com/developers/docs/qnxeverywhere/
- GPIO + Python on QNX: https://www.qnx.com/developers/docs/qnxeverywhere/com.qnx.doc.interfacing/topic/rpi/rpi_GPIO_python_qnx.html
