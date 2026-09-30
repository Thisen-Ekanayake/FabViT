import os
import xml.etree.ElementTree as ET

from PIL import Image

import torch
from torch.utils.data import Dataset

from config import (
    PATTERN_TO_IDX,
    DEFECT_TO_IDX,
)


class FabricDataset(Dataset):

    def __init__(
        self,
        image_dir,
        xml_dir,
        transform=None
    ):

        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.transform = transform

        self.samples = []

        self._load_samples()

    # ========================================================
    # LOAD XML FILES
    # ========================================================

    def _load_samples(self):

        xml_files = sorted([
            f
            for f in os.listdir(self.xml_dir)
            if f.lower().endswith(".xml")
        ])

        for xml_file in xml_files:

            xml_path = os.path.join(
                self.xml_dir,
                xml_file
            )

            try:

                root = ET.parse(
                    xml_path
                ).getroot()

                image_name = root.findtext(
                    "filename"
                )

                pattern_name = root.findtext(
                    "pattern_name"
                )

                if not image_name or not pattern_name:
                    continue

                pattern_name = pattern_name.strip()

                if pattern_name not in PATTERN_TO_IDX:
                    print(
                        f"Unknown pattern: {pattern_name}"
                    )
                    continue

                # ------------------------------------------------
                # Defective if bbox exists
                # ------------------------------------------------

                bbox = root.find("bbox")

                if bbox is not None:
                    defect_name = "defective"
                else:
                    defect_name = "non-defective"

                image_path = os.path.join(
                    self.image_dir,
                    image_name
                )

                if not os.path.exists(image_path):

                    print(
                        f"Missing image: {image_path}"
                    )

                    continue

                self.samples.append({
                    "image_path": image_path,
                    "pattern": PATTERN_TO_IDX[
                        pattern_name
                    ],
                    "defect": DEFECT_TO_IDX[
                        defect_name
                    ],
                    "pattern_name": pattern_name,
                    "defect_name": defect_name,
                })

            except Exception as e:

                print(
                    f"Error reading {xml_file}: {e}"
                )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):
        return len(self.samples)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "pattern": torch.tensor(
                sample["pattern"],
                dtype=torch.long
            ),
            "defect": torch.tensor(
                sample["defect"],
                dtype=torch.long
            ),
        }