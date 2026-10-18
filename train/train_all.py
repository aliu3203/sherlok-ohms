"""One command on the laptop: captures -> labeled hole crops -> trained TFLite model.

  pip install tensorflow numpy opencv-python
  python3 train/train_all.py            # holds out the last build for an honest test score
Output: models/holes.tflite  -> copy to the Pi's sherlok-ohms/models/
"""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
builds = sorted(p.name for p in (root / "captures").iterdir() if p.is_dir() and p.name != "empty")
if not builds:
    sys.exit("no captures/<schema>/ folders found - run collect.py on the Pi and copy captures/ here")
args = sys.argv[1:]
hold = ["--holdout", builds[-1]] if len(builds) >= 3 and "--holdout" not in args else []
for cmd in ([sys.executable, "train/make_dataset.py"],
            [sys.executable, "train/train_holes.py", *hold, *args]):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, check=True)
print("\nDone: models/holes.tflite. Look at the confusion matrix above: the 'plugged' row is what matters.")
