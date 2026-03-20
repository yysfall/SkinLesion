import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


metadata_path = "HAM10000_metadata.csv"
images_folder1 = "HAM10000_images_part_1"
images_folder2 = "HAM10000_images_part_2"
output_base = "data/processed"

target_classes = ["nv", "mel"]  # benign vs malignant
subset_per_class = 50
test_ratio = 0.15
val_ratio = 0.15


for split in ["train", "val", "test"]:
    for cls in target_classes:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)


df = pd.read_csv(metadata_path)


df.columns = df.columns.str.strip()


possible_cols = [c for c in df.columns if "dx" in c.lower() or "diagnosis" in c.lower()]
if not possible_cols:
    raise ValueError(f"No column found for class labels. Columns found: {df.columns.tolist()}")
class_col = possible_cols[0]
print(f"Using '{class_col}' as class column")


df = df[df[class_col].isin(target_classes)]
df["filename"] = df["image_id"].astype(str) + ".jpg"


samples = []

for cls in target_classes:
    cls_df = df[df[class_col] == cls]
    n_sample = min(subset_per_class, len(cls_df))
    sampled = cls_df.sample(n=n_sample, random_state=42)
    samples.append(sampled)

df_subset = pd.concat(samples, ignore_index=True)

print("Selected subset per class:")
print(df_subset[class_col].value_counts())

train_val, test = train_test_split(
    df_subset, test_size=test_ratio, random_state=42, stratify=df_subset[class_col]
)
train, val = train_test_split(
    train_val, test_size=val_ratio, random_state=42, stratify=train_val[class_col]
)


def copy_images(df_subset, split_name):
    missing = []
    for _, row in df_subset.iterrows():
        fname = row["filename"]
        cls = row[class_col]
        paths_to_check = [
            os.path.join(images_folder1, fname),
            os.path.join(images_folder2, fname)
        ]
        copied = False
        for p in paths_to_check:
            if os.path.exists(p):
                dest = os.path.join(output_base, split_name, cls, fname)
                shutil.copy(p, dest)
                copied = True
                break
        if not copied:
            missing.append(fname)
    if missing:
        print(f"Warning: {len(missing)} images not found in folders:")
        print(missing)

copy_images(train, "train")
copy_images(val, "val")
copy_images(test, "test")

print("Subset preparation complete!")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")