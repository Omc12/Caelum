import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2  —  reflect padding kills border artifacts."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      padding_mode="reflect", bias=False),          # ← reflect, not zeros
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """
    Attention gate — tells the decoder which spatial regions to focus on.
    Suppresses irrelevant activations before skip connections.
    """
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1, inplace=True))
        return x * psi


class LightUNet(nn.Module):
    """
    Residual U-Net: instead of predicting the full enhanced image, the network
    predicts a *delta* (enhancement residual) that is added back to the input.

    Why residual learning?
    ─────────────────────
    • Without it the network can map any input to vivid colours it "saw" during
      training, even when the input is already well-exposed.
    • With it the default behaviour is the identity (delta ≈ 0), so subtle
      images stay subtle and only genuinely dull regions get boosted.
    • This is the same trick used in ResNets and SR networks (EDSR, RRDB).
    """
    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────
        self.down1 = DoubleConv(3,   64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64,  128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # ── Bottleneck ────────────────────────────────────
        self.bottleneck = DoubleConv(512, 1024)

        # ── Attention gates ───────────────────────────────
        self.att1 = AttentionGate(f_g=512, f_l=512, f_int=256)
        self.att2 = AttentionGate(f_g=256, f_l=256, f_int=128)
        self.att3 = AttentionGate(f_g=128, f_l=128, f_int=64)
        self.att4 = AttentionGate(f_g=64,  f_l=64,  f_int=32)

        # ── Decoder ───────────────────────────────────────
        self.up1      = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2      = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3      = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4      = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        # ── Residual head: predict delta, not full output ─
        # tanh outputs in [-1, 1]; we scale to a small range so the
        # residual nudges the image rather than replacing it.
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.delta_scale = 0.15   # Capped to 0.15 (was 0.3) to heavily prevent oversaturation and artificial colors

    def forward(self, x):
        # ── Encoder ──
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        # ── Bottleneck ──
        b = self.bottleneck(self.pool4(d4))

        # ── Decoder with attention-gated skip connections ──
        u1 = self.up1(b)
        u1 = torch.cat([u1, self.att1(u1, d4)], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, self.att2(u2, d3)], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, self.att3(u3, d2)], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, self.att4(u4, d1)], dim=1)
        u4 = self.conv_up4(u4)

        # Standard UNet output to allow the model to completely redraw/denoise
        # instead of passing input noise through a residual connection
        return torch.sigmoid(self.final_conv(u4))