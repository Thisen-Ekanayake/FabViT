import torchvision.transforms as T

from config import IMAGE_SIZE


# ============================================================
# TRAIN TRANSFORMS
# ============================================================

train_transform = T.Compose([
    T.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    T.RandomHorizontalFlip(
        p=0.5
    ),

    T.RandomVerticalFlip(
        p=0.2
    ),

    T.RandomRotation(
        degrees=10
    ),

    T.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.03
    ),

    T.ToTensor(),

    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# VALIDATION / TEST
# ============================================================

val_transform = T.Compose([
    T.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    T.ToTensor(),

    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])