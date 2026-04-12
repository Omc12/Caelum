import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class LightUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Downsampling)
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck (The deepest mathematical representation)
        self.bottleneck = DoubleConv(256, 512)

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256) # 512 because of skip connection (256+256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)

        # Final output layer (Maps back to 3 RGB channels)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Down
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Up (With Skip Connections concatenated)
        u1 = self.up1(b)
        u1 = torch.cat([u1, d3], dim=1)
        c1 = self.conv_up1(u1)

        u2 = self.up2(c1)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv_up2(u2)

        u3 = self.up3(c2)
        u3 = torch.cat([u3, d1], dim=1)
        c3 = self.conv_up3(u3)

        out = self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh()
        )
        return out