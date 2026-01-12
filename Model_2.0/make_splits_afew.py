import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
CSV_PATH = "afew_va_frames.csv"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
# ----------------------------------------

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

df = pd.read_csv(CSV_PATH)

# ---------- Extract video id from path ----------
def get_video_id(path):
    # .../AFEW-VA/307/00000.png → 307
    return os.path.basename(os.path.dirname(path))

df["video_id"] = df["img_path"].apply(get_video_id)

videos = df["video_id"].unique()

# ---------- Video-level split ----------
train_vids, temp_vids = train_test_split(
    videos,
    test_size=(1 - TRAIN_RATIO),
    random_state=SEED,
    shuffle=True
)

val_vids, test_vids = train_test_split(
    temp_vids,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=SEED,
    shuffle=True
)

# ---------- Assign splits ----------
df["split"] = "train"
df.loc[df.video_id.isin(val_vids), "split"] = "val"
df.loc[df.video_id.isin(test_vids), "split"] = "test"

# ---------- Sanity checks ----------
print("Video counts:")
print("Train:", len(train_vids))
print("Val  :", len(val_vids))
print("Test :", len(test_vids))

print("\nFrame counts:")
print(df["split"].value_counts())

# ---------- Save ----------
df.to_csv(CSV_PATH, index=False)

print("\n folder-level splits saved")
