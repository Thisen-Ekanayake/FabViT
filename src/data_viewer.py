import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# Folder containing XML files
XML_FOLDER = "./data/ZJU-Leaper/Annotations/xmls/"

# pattern_name -> [defective_count, non_defective_count]
stats = defaultdict(lambda: [0, 0])

for filename in os.listdir(XML_FOLDER):
    if not filename.lower().endswith(".xml"):
        continue

    xml_path = os.path.join(XML_FOLDER, filename)

    try:
        root = ET.parse(xml_path).getroot()

        pattern_name = root.findtext("pattern_name", default="Unknown").strip()

        # Defective if <bbox> exists
        bbox = root.find("bbox")

        if bbox is not None:
            stats[pattern_name][0] += 1
        else:
            stats[pattern_name][1] += 1

    except ET.ParseError:
        print(f"Warning: Could not parse {filename}")

# Print table
print()
print(f"{'Pattern Class':<25} {'Defective':>12} {'Non-defective':>15} {'Total Images':>15}")
print("-" * 70)

total_defective = 0
total_non_defective = 0

for pattern_name in sorted(stats):
    defective, non_defective = stats[pattern_name]
    total = defective + non_defective

    total_defective += defective
    total_non_defective += non_defective

    print(
        f"{pattern_name:<25} "
        f"{defective:>12} "
        f"{non_defective:>15} "
        f"{total:>15}"
    )

# Total row
total = total_defective + total_non_defective

print("-" * 70)
print(
    f"{'TOTAL':<25} "
    f"{total_defective:>12} "
    f"{total_non_defective:>15} "
    f"{total:>15}"
)