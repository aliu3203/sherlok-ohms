# Sherlok Ohms: runbook

A Pi on QNX watches a breadboard and checks it against a schema. LEDs show the result:
**green** = matches, **yellow** = can't verify (hand in view / unsure), **red** = wrong connection, persisting 2 s.

## How it works
1. **Grid (geometry).** You click 4 corner holes once. That gives every hole's exact position on the board surface.
2. **Hole model (AI, TensorFlow Lite).** For each hole, a small CNN looks at a crop and outputs P(empty), P(covered: a wire lies over it), P(plugged: a lead goes in).
   - It's trained on photos from **our** camera, so it learns how a plugged hole looks from this angle, including the shift caused by wire height and perspective.
   - Labels come from the schema of what you built. You never label holes by hand.
3. **Grouping + check (code).** Changed regions vs. the empty-board photo are grouped into parts. Each part's plugged holes are its ends. Ends map to strips (a–e and f–j of a column, rails), and strips are compared with the schema.
4. **Optional: MediaPipe hand detector (pretrained).** A hand over the board gives yellow.

Without a trained model the code falls back to "farthest changed holes = ends". That only works for flat wires; the model is the real path.

---

## A. One-time setup (≈20 min)

**Pi (QNX 8.0.5, Pi 5, Camera Module 3 on DISP0):**
```
uname -a ; ls /dev/sensor/camera3          # QNX + camera present
sudo apk update
sudo apk add opencv python3-opencv python3-numpy tflite-runtime python3-tflite-runtime
```
Copy the repo over (`scp -r sherlok-ohms qnxuser@<pi-ip>:~/`). Then put QNX's camera helper in the repo root:
```
cd ~/sherlok-ohms
curl -L -o cam.tgz https://raw.githubusercontent.com/qnx/codelabs/main/markdown/mediapipe-camera-python/mediapipe-camera-python-companion.tar.gz
tar -xzf cam.tgz qnx_camera_helper.py && rm cam.tgz
```
LEDs, each through a ~330 Ω resistor to GND: BCM 17 = green, 27 = yellow, 22 = red.

**Physical setup. This matters as much as the model:**
- Camera **high (≥ 35 cm) and pointing straight down**, rigidly mounted. Board centered in the frame and taped down.
- Even, diffuse light.
- Flat pre-cut jumpers, parts pushed flush to the board, wires not crossing.
- If the camera or board moves, redo the calibration below and recapture the empty board.

## B. Empty board + calibration (≈5 min)
1. On the Pi, with the board **empty**, run `python3 collect.py` (no arguments). It saves `ref/empty.png`.
2. Copy `ref/empty.png` to a laptop, then:
   ```
   pip install opencv-python numpy
   python3 tools/calibrate.py ref/empty.png --columns 30 --rails tp,tn --rail-count 25
   python3 tools/show_grid.py ref/empty.png      # open debug/grid.png: every dot must sit on a hole
   ```
   - `--columns 63` for a full-size board.
   - `tp`/`tn` = the + and − rails next to row a, `bp`/`bn` = next to row j. Hole 1 of a rail is the one nearest column 1.
3. Copy `board.json` back to the Pi.

## C. Collect training data (≈20 min, the important part)
1. **Write schemas**, one small text file per circuit. Pins are hole names, one part per line:
   ```
   # schemas/led_demo.txt
   W1   wire      tp3   a5
   R1   resistor  c5    c9
   D1   led       e9    e11
   W2   wire      a11   tn8
   ```
   Make **4–6 schemas**, and put parts in *different* holes across them (shifted columns, different rows). Three examples are already in `schemas/`, including a deliberately wrong build.
2. **On the Pi:**
   ```
   python3 collect.py schemas/*.txt
   ```
   For each schema it prints the parts. Build exactly that, keep hands away, press Enter, and it takes 15 frames 1 s apart.
   - Between frames, nudge a wire or shift the light a little, but keep everything plugged in.
   - `s` skips a schema, `q` quits. Re-running adds more frames.
   - Frames go to `captures/<schema name>/`. **The folder name is the label: don't rename folders, and don't change a build without changing its schema.**

## D. Train (≈5–10 min, laptop)
```
scp -r qnxuser@<pi-ip>:~/sherlok-ohms/captures .     # also needs ref/empty.png + board.json locally
pip install tensorflow numpy opencv-python
python3 train/train_all.py
```
- It holds out one whole build for testing and prints a confusion matrix. **Check the `plugged` row**: most of it should land in the plugged column.
- Copy `models/holes.tflite` to the Pi in `sherlok-ohms/models/`.
- If the plugged row is poor, collect more builds (more holes covered) and retrain.

## E. Run
```
python3 main.py --source qnx --schema schemas/led_demo.txt --model models/holes.tflite
```
- The terminal prints state changes and issues, e.g. `R1 misplaced: expected A5-A9, found A10-A5`.
- Every 5 frames it writes `debug/latest.png` (scp it to look).
- Hands (optional): `sudo apk add mediapipe`, download the model (URL in `sherlok/hands.py`), add `--hands models/hand_landmarker.task`.

## Testing without hardware
```
python3 tools/fake_board.py --schema schemas/led_demo.txt --out fake
python3 main.py --source fake/built.png --empty fake/empty.png --board fake/board.json --schema schemas/led_demo.txt --once
```
`--once` processes a single frame, so the debounced state shows yellow; read the `frame=` field. The fake board draws flat wires, so it doesn't test height or perspective; only real captures do.

**Camera fighting you?** `collect.py` and `main.py` both take `--source webcam` (laptop) or `--source some/folder`.

## Tuning (`thresholds.json`)
- **Too many reds:** raise `high_conf` (default 0.8). Below it, mismatches give yellow instead.
- **Holes falsely "changed":** raise `vision.hole_thresh` and `vision.px_thresh`.
- **Part ends attached to the wrong part:** lower `vision.assign_pitch` (default 1.5).
- **Hand not giving yellow:** lower `vision.obstructed_frac`.

## Files
| Path | What it is |
|---|---|
| `collect.py` | Guided data collection on the Pi |
| `main.py` | Live check |
| `capture.py` | Raw frame grabber |
| `sherlok/board.py` | Grid / homography |
| `sherlok/vision.py` | Change detection, hole model, grouping |
| `sherlok/check.py` | Schema comparison, debounce, `.txt`/`.json` schemas |
| `train/train_all.py` | Runs `make_dataset.py` (auto-labels from schemas) then `train_holes.py` (CNN → TFLite) |
| `tools/` | `calibrate`, `show_grid`, `fake_board` |
