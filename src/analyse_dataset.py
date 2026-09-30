import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "split_dataset"

SPLITS = ["train", "val", "test"]

# Expected structure:
#
# dataset/
# ├── train/
# │   ├── images/
# │   └── annotations/
# ├── val/
# │   ├── images/
# │   └── annotations/
# └── test/
#     ├── images/
#     └── annotations/


# ============================================================
# HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_image_files(folder):
    return [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]


def analyze_xml(xml_path):
    """
    Read pattern and defect information from XML.
    """

    root = ET.parse(xml_path).getroot()

    pattern = root.findtext(
        "pattern_name",
        default="Unknown"
    ).strip()

    filename = root.findtext(
        "filename",
        default=""
    ).strip()

    # Defective if bbox exists
    bbox = root.find("bbox")

    if bbox is not None:
        defect = "defective"
    else:
        defect = "non-defective"

    return pattern, defect, filename


# ============================================================
# GLOBAL STATISTICS
# ============================================================

global_pattern_counts = Counter()
global_defect_counts = Counter()

global_dimensions = Counter()

total_images = 0
total_xml = 0

total_missing_images = 0
total_missing_xml = 0
total_corrupt_images = 0

duplicate_images = []

# Store filenames seen across splits
all_image_names = set()


# ============================================================
# ANALYZE EACH SPLIT
# ============================================================

for split in SPLITS:

    print("\n" + "=" * 80)
    print(f"ANALYZING: {split.upper()}")
    print("=" * 80)

    image_dir = os.path.join(
        DATASET_DIR,
        split,
        "images"
    )

    xml_dir = os.path.join(
        DATASET_DIR,
        split,
        "annotations"
    )

    if not os.path.exists(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        continue

    if not os.path.exists(xml_dir):
        print(f"ERROR: Annotation directory not found: {xml_dir}")
        continue

    image_files = get_image_files(image_dir)

    xml_files = [
        f for f in os.listdir(xml_dir)
        if f.lower().endswith(".xml")
    ]

    print(f"\nImages found      : {len(image_files)}")
    print(f"XML files found   : {len(xml_files)}")

    # --------------------------------------------------------
    # Counters for this split
    # --------------------------------------------------------

    pattern_counts = Counter()
    defect_counts = Counter()
    dimensions = Counter()

    missing_images = []
    missing_xml = []
    corrupt_images = []

    # --------------------------------------------------------
    # Image lookup
    # --------------------------------------------------------

    image_lookup = {
        os.path.splitext(f)[0]: f
        for f in image_files
    }

    xml_lookup = {
        os.path.splitext(f)[0]: f
        for f in xml_files
    }

    # --------------------------------------------------------
    # Check duplicate filenames across splits
    # --------------------------------------------------------

    for filename in image_files:

        if filename in all_image_names:
            duplicate_images.append(filename)

        all_image_names.add(filename)

    # --------------------------------------------------------
    # Analyze XML files
    # --------------------------------------------------------

    for xml_filename in xml_files:

        xml_path = os.path.join(
            xml_dir,
            xml_filename
        )

        total_xml += 1

        try:

            pattern, defect, image_filename = analyze_xml(
                xml_path
            )

            pattern_counts[pattern] += 1
            defect_counts[defect] += 1

            global_pattern_counts[pattern] += 1
            global_defect_counts[defect] += 1

            # ------------------------------------------------
            # Check referenced image
            # ------------------------------------------------

            image_path = os.path.join(
                image_dir,
                image_filename
            )

            if not os.path.exists(image_path):

                missing_images.append(
                    image_filename
                )

                continue

            # ------------------------------------------------
            # Read image
            # ------------------------------------------------

            try:

                with Image.open(image_path) as img:

                    # Verify actual image
                    img.verify()

                # Open again because verify() invalidates file
                with Image.open(image_path) as img:

                    width, height = img.size

                    dimensions[(width, height)] += 1
                    global_dimensions[(width, height)] += 1

            except Exception:

                corrupt_images.append(
                    image_filename
                )

        except Exception as e:

            print(
                f"ERROR parsing {xml_filename}: {e}"
            )

    # --------------------------------------------------------
    # Check images without XML
    # --------------------------------------------------------

    for image_filename in image_files:

        basename = os.path.splitext(
            image_filename
        )[0]

        if basename not in xml_lookup:

            missing_xml.append(
                image_filename
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_images += len(image_files)

    total_missing_images += len(missing_images)
    total_missing_xml += len(missing_xml)
    total_corrupt_images += len(corrupt_images)

    # --------------------------------------------------------
    # Print pattern distribution
    # --------------------------------------------------------

    print("\nPATTERN DISTRIBUTION")
    print("-" * 70)

    print(
        f"{'Pattern':<25}"
        f"{'Defective':>12}"
        f"{'Non-defective':>16}"
        f"{'Total':>12}"
    )

    print("-" * 70)

    for pattern in sorted(pattern_counts):

        # Need to calculate defect counts per pattern
        pattern_defective = 0
        pattern_non_defective = 0

        for xml_filename in xml_files:

            xml_path = os.path.join(
                xml_dir,
                xml_filename
            )

            try:

                p, d, _ = analyze_xml(xml_path)

                if p == pattern:

                    if d == "defective":
                        pattern_defective += 1
                    else:
                        pattern_non_defective += 1

            except Exception:
                pass

        total = (
            pattern_defective +
            pattern_non_defective
        )

        print(
            f"{pattern:<25}"
            f"{pattern_defective:>12}"
            f"{pattern_non_defective:>16}"
            f"{total:>12}"
        )

    # --------------------------------------------------------
    # Defect distribution
    # --------------------------------------------------------

    print("\nDEFECT DISTRIBUTION")
    print("-" * 50)

    print(
        f"Defective      : "
        f"{defect_counts['defective']}"
    )

    print(
        f"Non-defective  : "
        f"{defect_counts['non-defective']}"
    )

    print(
        f"Total          : "
        f"{sum(defect_counts.values())}"
    )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    print("\nIMAGE DIMENSIONS")
    print("-" * 50)

    for (width, height), count in dimensions.most_common():

        print(
            f"{width} x {height:<10} : {count}"
        )

    # --------------------------------------------------------
    # Problems
    # --------------------------------------------------------

    print("\nDATASET INTEGRITY")
    print("-" * 50)

    print(
        f"Missing images : "
        f"{len(missing_images)}"
    )

    print(
        f"Missing XML    : "
        f"{len(missing_xml)}"
    )

    print(
        f"Corrupt images : "
        f"{len(corrupt_images)}"
    )

    if missing_images:

        print("\nFirst missing images:")

        for filename in missing_images[:10]:
            print(f"  {filename}")

    if missing_xml:

        print("\nFirst images without XML:")

        for filename in missing_xml[:10]:
            print(f"  {filename}")

    if corrupt_images:

        print("\nFirst corrupt images:")

        for filename in corrupt_images[:10]:
            print(f"  {filename}")


# ============================================================
# GLOBAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 80)
print("GLOBAL DATASET SUMMARY")
print("=" * 80)

print(f"\nTotal images      : {total_images}")
print(f"Total XML files   : {total_xml}")

print(
    f"Defective         : "
    f"{global_defect_counts['defective']}"
)

print(
    f"Non-defective     : "
    f"{global_defect_counts['non-defective']}"
)

print(
    f"Missing images    : "
    f"{total_missing_images}"
)

print(
    f"Missing XML       : "
    f"{total_missing_xml}"
)

print(
    f"Corrupt images    : "
    f"{total_corrupt_images}"
)


# ============================================================
# GLOBAL PATTERN DISTRIBUTION
# ============================================================

print("\n")
print("GLOBAL PATTERN DISTRIBUTION")
print("-" * 70)

print(
    f"{'Pattern':<25}"
    f"{'Total':>12}"
)

print("-" * 70)

for pattern, count in sorted(
    global_pattern_counts.items()
):

    print(
        f"{pattern:<25}"
        f"{count:>12}"
    )


# ============================================================
# GLOBAL IMAGE DIMENSIONS
# ============================================================

print("\n")
print("MOST COMMON IMAGE DIMENSIONS")
print("-" * 60)

for (width, height), count in global_dimensions.most_common(15):

    print(
        f"{width} x {height:<10} : {count}"
    )


# ============================================================
# DUPLICATES
# ============================================================

print("\n")
print("DUPLICATE FILENAMES ACROSS SPLITS")
print("-" * 60)

if duplicate_images:

    print(
        f"Found {len(duplicate_images)} duplicates."
    )

    for filename in duplicate_images[:20]:
        print(f"  {filename}")

else:

    print("No duplicate filenames found.")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n")
print("=" * 80)

if (
    total_missing_images == 0
    and total_missing_xml == 0
    and total_corrupt_images == 0
    and len(duplicate_images) == 0
):

    print("DATASET CHECK: PASSED")

else:

    print("DATASET CHECK: ISSUES FOUND")

print("=" * 80)