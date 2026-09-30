import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

XML_DIR = "data/ZJU-Leaper/Annotations/xmls"
IMAGE_DIR = "data/ZJU-Leaper/Images"

OUTPUT_DIR = "split_dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "annotations"), exist_ok=True)


# ============================================================
# READ XML DATA
# ============================================================

samples = []

for xml_filename in os.listdir(XML_DIR):

    if not xml_filename.lower().endswith(".xml"):
        continue

    xml_path = os.path.join(XML_DIR, xml_filename)

    try:
        root = ET.parse(xml_path).getroot()

        filename = root.findtext("filename")
        pattern_name = root.findtext("pattern_name")

        if not filename or not pattern_name:
            print(f"Skipping invalid XML: {xml_filename}")
            continue

        # ----------------------------------------------------
        # Determine defective / non-defective
        # ----------------------------------------------------
        #
        # Your dataset:
        # defective    -> has <bbox>
        # non-defective -> no <bbox>
        #
        bbox = root.find("bbox")

        if bbox is not None:
            defect_status = "defective"
        else:
            defect_status = "non-defective"

        # Combined stratification label
        stratify_label = f"{pattern_name}__{defect_status}"

        samples.append({
            "xml": xml_filename,
            "image": filename,
            "pattern": pattern_name,
            "defect": defect_status,
            "stratify": stratify_label
        })

    except Exception as e:
        print(f"Error reading {xml_filename}: {e}")


print(f"\nTotal samples found: {len(samples)}")


# ============================================================
# CHECK STRATIFICATION GROUPS
# ============================================================

print("\nStratification groups:")

counter = Counter(sample["stratify"] for sample in samples)

for label, count in sorted(counter.items()):
    print(f"{label:<40} {count}")


# ============================================================
# FIRST SPLIT
#
# 70% TRAIN
# 30% TEMP
# ============================================================

train_samples, temp_samples = train_test_split(
    samples,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=[s["stratify"] for s in samples],
    random_state=42
)


# ============================================================
# SECOND SPLIT
#
# TEMP = 30%
#
# Split TEMP equally:
# 15% VAL
# 15% TEST
#
# Therefore:
# test_size = 0.5
# ============================================================

val_samples, test_samples = train_test_split(
    temp_samples,
    test_size=0.5,
    stratify=[s["stratify"] for s in temp_samples],
    random_state=42
)


splits = {
    "train": train_samples,
    "val": val_samples,
    "test": test_samples
}


# ============================================================
# COPY FILES
# ============================================================

def copy_dataset(samples, split):

    image_output = os.path.join(
        OUTPUT_DIR,
        split,
        "images"
    )

    annotation_output = os.path.join(
        OUTPUT_DIR,
        split,
        "annotations"
    )

    for sample in samples:

        # -------------------------
        # Copy XML
        # -------------------------

        xml_src = os.path.join(
            XML_DIR,
            sample["xml"]
        )

        xml_dst = os.path.join(
            annotation_output,
            sample["xml"]
        )

        shutil.copy2(xml_src, xml_dst)

        # -------------------------
        # Copy image
        # -------------------------

        image_src = os.path.join(
            IMAGE_DIR,
            sample["image"]
        )

        image_dst = os.path.join(
            image_output,
            sample["image"]
        )

        if os.path.exists(image_src):

            shutil.copy2(
                image_src,
                image_dst
            )

        else:
            print(
                f"WARNING: Image not found: "
                f"{sample['image']}"
            )


# ============================================================
# PERFORM SPLIT
# ============================================================

for split_name, split_samples in splits.items():

    print(
        f"\nCopying {split_name}: "
        f"{len(split_samples)} samples"
    )

    copy_dataset(
        split_samples,
        split_name
    )


# ============================================================
# PRINT FINAL DISTRIBUTION
# ============================================================

print("\n")
print("=" * 90)
print("FINAL DATASET DISTRIBUTION")
print("=" * 90)

print(
    f"{'Pattern Class':<25}"
    f"{'Split':<10}"
    f"{'Defective':>12}"
    f"{'Non-defective':>16}"
    f"{'Total':>12}"
)

print("-" * 90)


for pattern in sorted(
    set(s["pattern"] for s in samples)
):

    for split_name in ["train", "val", "test"]:

        split_samples = [
            s for s in splits[split_name]
            if s["pattern"] == pattern
        ]

        defective = sum(
            s["defect"] == "defective"
            for s in split_samples
        )

        non_defective = sum(
            s["defect"] == "non-defective"
            for s in split_samples
        )

        total = defective + non_defective

        print(
            f"{pattern:<25}"
            f"{split_name:<10}"
            f"{defective:>12}"
            f"{non_defective:>16}"
            f"{total:>12}"
        )

    print("-" * 90)


# ============================================================
# OVERALL TOTALS
# ============================================================

print("\nOVERALL SPLIT TOTALS")
print("-" * 60)

for split_name in ["train", "val", "test"]:

    split_samples = splits[split_name]

    defective = sum(
        s["defect"] == "defective"
        for s in split_samples
    )

    non_defective = sum(
        s["defect"] == "non-defective"
        for s in split_samples
    )

    total = len(split_samples)

    print(
        f"{split_name:<10}"
        f"Defective: {defective:<10}"
        f"Non-defective: {non_defective:<10}"
        f"Total: {total}"
    )

print("-" * 60)

print(
    f"{'ALL':<10}"
    f"Defective: {sum(s['defect'] == 'defective' for s in samples):<10}"
    f"Non-defective: {sum(s['defect'] == 'non-defective' for s in samples):<10}"
    f"Total: {len(samples)}"
)