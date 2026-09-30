# ============================================================
# EXPORT DATA FOR THE 3D VISUALIZER (viz/index.html)
#
# Runs the trained SmallViT on the test split, picks a few
# samples and writes their real attention maps, token features
# and predictions to viz/data.js. The original sample images
# are copied to viz/samples/.
#
#   python src/export_viz.py   (from the repo root)
# ============================================================

import base64
import io
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

import config
from dataset import FabricDataset
from model import SmallViT
from transforms import val_transform


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "viz" / "data.js"
SAMPLE_IMAGE_DIR = OUTPUT_PATH.parent / "samples"
CHECKPOINT_PATH = config.CHECKPOINT_DIR / "best_model.pth"

NUM_DEFECTIVE_SAMPLES = 5
NUM_CLEAN_SAMPLES = 2
NUM_MISTAKE_SAMPLES = 1


# ============================================================
# MODEL
# ============================================================

device = torch.device(
    config.DEVICE
    if torch.cuda.is_available()
    else "cpu"
)

model = SmallViT(
    image_size=config.IMAGE_SIZE,
    embed_dim=config.EMBED_DIM,
    depth=config.DEPTH,
    num_heads=config.NUM_HEADS,
    mlp_ratio=config.MLP_RATIO,
    dropout=config.DROPOUT,
    num_pattern_classes=config.NUM_PATTERN_CLASSES,
    num_defect_classes=config.NUM_DEFECT_CLASSES
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device).eval()

print(
    f"Loaded {CHECKPOINT_PATH} "
    f"(epoch {checkpoint['epoch']}, "
    f"val loss {checkpoint['val_loss']:.4f})"
)


# ============================================================
# FORWARD PASS THAT KEEPS INTERMEDIATES
# ============================================================

@torch.no_grad()
def forward_with_internals(images):
    """
    Same computation as SmallViT.forward, but also returns the
    token sequence after every stage and the per-head attention
    weights of every encoder layer.
    """

    stages = []
    attentions = []

    x = model.patch_embed(images)

    # Patch embedding has no CLS yet: pad with zeros so every
    # stage has 197 tokens (CLS last, like the visualizer)
    stages.append(
        torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    )

    cls_tokens = model.cls_token.expand(x.size(0), -1, -1)

    x = torch.cat([cls_tokens, x], dim=1)
    x = x + model.pos_embed

    stages.append(x)

    for layer in model.transformer.layers:

        # Pre-norm: attention sees norm1(x)
        h = layer.norm1(x)

        _, weights = layer.self_attn(
            h,
            h,
            h,
            need_weights=True,
            average_attn_weights=False
        )

        # B, heads, 197, 197
        attentions.append(weights)

        x = layer(x)

        stages.append(x)

    x = model.norm(x)

    stages.append(x)

    features = x[:, 0]

    outputs = {
        "pattern": model.pattern_head(features),
        "defect": model.defect_head(features),
    }

    return outputs, stages, attentions


# ============================================================
# EVALUATE TEST SET
# ============================================================

test_dataset = FabricDataset(
    config.TEST_IMAGE_DIR,
    config.TEST_XML_DIR,
    transform=val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS
)

pattern_probs = []
defect_probs = []

with torch.no_grad():

    for batch in test_loader:

        outputs = model(batch["image"].to(device))

        pattern_probs.append(outputs["pattern"].softmax(1).cpu())
        defect_probs.append(outputs["defect"].softmax(1).cpu())

pattern_probs = torch.cat(pattern_probs)
defect_probs = torch.cat(defect_probs)

pattern_true = torch.tensor([s["pattern"] for s in test_dataset.samples])
defect_true = torch.tensor([s["defect"] for s in test_dataset.samples])

pattern_ok = pattern_probs.argmax(1) == pattern_true
defect_ok = defect_probs.argmax(1) == defect_true

metrics = {
    "test_samples": len(test_dataset),
    "pattern_acc": pattern_ok.float().mean().item(),
    "defect_acc": defect_ok.float().mean().item(),
    "epoch": checkpoint["epoch"],
    "val_loss": checkpoint["val_loss"],
}

print(
    f"Test: pattern acc {metrics['pattern_acc']:.4f} | "
    f"defect acc {metrics['defect_acc']:.4f} "
    f"({len(test_dataset)} images)"
)


# ============================================================
# PICK SAMPLES
# ============================================================

def pick_samples():

    both_ok = pattern_ok & defect_ok
    confidence = (
        pattern_probs.max(1).values
        * defect_probs.max(1).values
    )

    order = confidence.argsort(descending=True).tolist()

    picked = []
    seen_patterns = set()

    # Confident, correct defective samples with distinct patterns
    for i in order:

        if len(picked) >= NUM_DEFECTIVE_SAMPLES:
            break

        pattern = test_dataset.samples[i]["pattern"]

        if both_ok[i] and defect_true[i] == 1 and pattern not in seen_patterns:
            picked.append((i, "defective"))
            seen_patterns.add(pattern)

    # Correct non-defective samples
    for i in order:

        if sum(tag == "clean" for _, tag in picked) >= NUM_CLEAN_SAMPLES:
            break

        pattern = test_dataset.samples[i]["pattern"]

        if both_ok[i] and defect_true[i] == 0 and pattern not in seen_patterns:
            picked.append((i, "clean"))
            seen_patterns.add(pattern)

    # Confident mistakes
    mistakes = [i for i in order if not both_ok[i]]

    for i in mistakes[:NUM_MISTAKE_SAMPLES]:
        picked.append((i, "mistake"))

    return picked


def read_bboxes(image_path, image_size):

    xml_path = (
        Path(config.TEST_XML_DIR)
        / (Path(image_path).stem + ".xml")
    )

    if not xml_path.exists():
        return []

    width, height = image_size
    sx = config.IMAGE_SIZE / width
    sy = config.IMAGE_SIZE / height

    boxes = []

    for bbox in ET.parse(xml_path).getroot().findall("bbox"):

        boxes.append([
            round(float(bbox.findtext("xmin")) * sx, 1),
            round(float(bbox.findtext("ymin")) * sy, 1),
            round(float(bbox.findtext("xmax")) * sx, 1),
            round(float(bbox.findtext("ymax")) * sy, 1),
        ])

    return boxes


def image_to_data_uri(image):

    buffer = io.BytesIO()

    image.resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE)
    ).save(buffer, format="JPEG", quality=88)

    encoded = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# TOKEN FEATURES -> RGB (PCA per stage)
# ============================================================

def stages_to_rgb(stages):
    """
    Projects each stage's 197 x 192 tokens onto their top 3
    principal components and maps them to 0-255 RGB. Component
    signs are aligned with the previous stage so colors stay
    comparable from layer to layer.
    """

    colors = []
    previous = None

    for index, tokens in enumerate(stages):

        # Visualizer order: 196 patches, then CLS
        if index == 0:
            tokens = tokens.clone()
        else:
            tokens = torch.cat([tokens[1:], tokens[:1]])

        fit = tokens[:196] - tokens[:196].mean(0)

        _, _, v = torch.linalg.svd(fit, full_matrices=False)

        projected = (tokens - tokens[:196].mean(0)) @ v[:3].T

        if previous is not None:

            signs = torch.sign(
                (projected[:196] * previous[:196]).sum(0)
            )

            projected = projected * torch.where(signs == 0, 1.0, signs)

        previous = projected

        low = torch.quantile(projected[:196], 0.02, dim=0)
        high = torch.quantile(projected[:196], 0.98, dim=0)

        scaled = ((projected - low) / (high - low + 1e-8)).clamp(0, 1)

        rgb = (scaled * 255).round().int()

        # Patch embedding stage has no CLS token
        if index == 0:
            rgb[196] = 0

        colors.append(rgb.flatten().tolist())

    return colors


# ============================================================
# EXPORT
# ============================================================

samples = []

SAMPLE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

for index, tag in pick_samples():

    sample = test_dataset.samples[index]

    # Keep the original test image next to the visualizer
    shutil.copy(sample["image_path"], SAMPLE_IMAGE_DIR)

    raw = Image.open(sample["image_path"]).convert("RGB")

    image = val_transform(raw).unsqueeze(0).to(device)

    outputs, stages, attentions = forward_with_internals(image)

    # CLS row of each layer: heads x 197, reordered to patches then CLS
    cls_attention = []

    for weights in attentions:

        row = weights[0, :, 0, :].cpu()
        row = torch.cat([row[:, 1:], row[:, :1]], dim=1)

        cls_attention.append([
            [round(v, 5) for v in head.tolist()]
            for head in row
        ])

    samples.append({
        "id": Path(sample["image_path"]).stem,
        "tag": tag,
        "image": image_to_data_uri(raw),
        "bboxes": read_bboxes(sample["image_path"], raw.size),
        "true_pattern": sample["pattern"],
        "true_defect": sample["defect"],
        "pattern_probs": [
            round(v, 5)
            for v in outputs["pattern"].softmax(1)[0].tolist()
        ],
        "defect_probs": [
            round(v, 5)
            for v in outputs["defect"].softmax(1)[0].tolist()
        ],
        "attention": cls_attention,
        "token_rgb": stages_to_rgb([s[0].cpu() for s in stages]),
    })

    print(
        f"  {samples[-1]['id']}  {tag:<9}  "
        f"{sample['pattern_name']} / {sample['defect_name']}"
    )

payload = {
    "metrics": metrics,
    "pattern_classes": config.PATTERN_CLASSES,
    "defect_classes": config.DEFECT_CLASSES,
    "samples": samples,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH.write_text(
    "// Generated by src/export_viz.py. Do not edit by hand.\n"
    "window.VIZ_DATA = "
    + json.dumps(payload, separators=(",", ":"))
    + ";\n"
)

print(
    f"Wrote {OUTPUT_PATH} "
    f"({OUTPUT_PATH.stat().st_size / 1024:.0f} KB, {len(samples)} samples)"
)
