"""Train the tiny hole classifier (empty / covered / plugged) and export TFLite.
Runs on a laptop CPU in a few minutes, or in Colab.

  pip install tensorflow numpy
  python3 train/train_holes.py --data train/holes.npz --out models/holes.tflite [--holdout led_demo]

--holdout <build>: keep one whole build out for testing (frames of one build are near-duplicates).
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASSES = ("empty", "covered", "plugged")


AUG = keras.Sequential([layers.RandomBrightness(0.25, value_range=(0, 255)), layers.RandomContrast(0.25),
                        layers.RandomTranslation(0.06, 0.06), layers.RandomFlip()])


def build(size: int) -> keras.Model:
    inp = layers.Input((size, size, 3))
    x = layers.Rescaling(1 / 255.0)(inp)
    for f in (16, 32, 64):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/holes.npz")
    ap.add_argument("--out", default="models/holes.tflite")
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()

    d = np.load(a.data)
    x, y, g = d["x"].astype(np.float32), d["y"], d["group"]
    if a.holdout:
        te = g == a.holdout
    else:
        te = np.random.default_rng(0).random(len(y)) < 0.15
    xtr, ytr, xte, yte = x[~te], y[~te], x[te], y[te]
    counts = np.bincount(ytr, minlength=len(CLASSES))
    cw = {i: float(len(ytr) / (len(CLASSES) * max(c, 1))) for i, c in enumerate(counts)}
    print("train per class", counts.tolist(), "test", len(yte))

    m = build(x.shape[1])
    m.compile(optimizer=keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    ds = (tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(len(ytr)).batch(64)
          .map(lambda xb, yb: (tf.clip_by_value(AUG(xb, training=True), 0, 255), yb)))
    sw = np.array([cw[int(t)] for t in ytr], np.float32)
    ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(sw).batch(64))).map(
        lambda xy, w: (xy[0], xy[1], w))
    m.fit(ds, validation_data=(xte, yte), epochs=a.epochs, verbose=2)

    pred = m.predict(xte, verbose=0).argmax(1)
    conf = np.zeros((3, 3), int)
    for t, p in zip(yte, pred):
        conf[t, p] += 1
    print("confusion (rows=true, cols=pred)", CLASSES)
    print(conf)

    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    import pathlib
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_bytes(conv.convert())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
