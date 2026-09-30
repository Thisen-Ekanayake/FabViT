import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192
    ):

        super().__init__()

        assert image_size % patch_size == 0

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.projection(x)

        # B, C, H, W
        x = x.flatten(2)

        # B, N, C
        x = x.transpose(1, 2)

        return x


class SmallViT(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        num_pattern_classes=19,
        num_defect_classes=2
    ):

        super().__init__()

        # ----------------------------------------------------
        # Patch embedding
        # ----------------------------------------------------

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        num_patches = (
            self.patch_embed.num_patches
        )

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )

        # ----------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim
            )
        )

        self.pos_dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(
                    embed_dim * mlp_ratio
                ),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=depth
            )
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        # ----------------------------------------------------
        # Classification heads
        # ----------------------------------------------------

        self.pattern_head = nn.Linear(
            embed_dim,
            num_pattern_classes
        )

        self.defect_head = nn.Linear(
            embed_dim,
            num_defect_classes
        )

        self._init_weights()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _init_weights(self):

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        x = self.patch_embed(x)

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        x = x + self.pos_embed

        x = self.pos_dropout(x)

        x = self.transformer(x)

        x = self.norm(x)

        # CLS representation
        features = x[:, 0]

        pattern_logits = self.pattern_head(
            features
        )

        defect_logits = self.defect_head(
            features
        )

        return {
            "pattern": pattern_logits,
            "defect": defect_logits
        }