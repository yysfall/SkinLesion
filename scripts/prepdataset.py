import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

metadata_path = "HAM10000_metadata.csv"
img_dir1 = "HAM10000_images_part_1"
img_dir2 = "HAM10000_images_part_2"
output_dir = "data/processed"

classes = ["nv", "mel"]

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

df = pd.read_csv(metadata_path)
df.columns = df.columns.str.strip()

print("Available classes:", df["dx"].unique())

df = df[df["dx"].isin(classes)]
df["filename"] = df["image_id"] + ".jpg"

train_val, test = train_test_split(
    df, test_size=0.15, stratify=df["dx"], random_state=42
)

train, val = train_test_split(
    train_val, test_size=0.15, stratify=train_val["dx"], random_state=42
)

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

def copy_images(dataframe, split):
    missing = 0
    for _, row in dataframe.iterrows():
        fname = row["filename"]
        label = row["dx"]

        path1 = os.path.join(img_dir1, fname)
        path2 = os.path.join(img_dir2, fname)

        dest = os.path.join(output_dir, split, label, fname)

        if os.path.exists(path1):
            shutil.copy(path1, dest)
        elif os.path.exists(path2):
            shutil.copy(path2, dest)
        else:
            missing += 1

    if missing > 0:
        print(f"{missing} images missing in {split}")

copy_images(train, "train")
copy_images(val, "val")
copy_images(test, "test")

print("Dataset preparation complete!")