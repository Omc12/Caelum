import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data_pipeline import download_pexels_sky, SkyEnhancementDataset
from unet import LightUNet
from tqdm import tqdm
import os

# ── Hyperparameters ──────────────────────────────────────────
BATCH_SIZE    = 16
EPOCHS        = 25
LEARNING_RATE = 1e-4
NUM_IMAGES    = 8000   # Total images to download from Pexels
IMAGE_DIR     = "sky_images"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────

def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

def train_model():
    print(f"Using device: {DEVICE}")

    # 1. Download / load sky images from Pexels
    filepaths = download_pexels_sky(num_images=NUM_IMAGES, save_dir=IMAGE_DIR)

    if len(filepaths) == 0:
        raise RuntimeError("No images found! Check your Pexels API key.")

    # 2. Build dataset with train/val split (90/10)
    full_dataset = SkyEnhancementDataset(filepaths)
    val_size     = max(1, int(0.1 * len(full_dataset)))
    train_size   = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
        persistent_workers=True if DEVICE == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
        persistent_workers=True if DEVICE == "cuda" else False
    )

    # 3. Model, loss, optimizer
    model     = LightUNet().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Reduce LR if val loss plateaus for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    # 4. Training loop
    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train", leave=False)

        for x_dull, y_perfect in loop:
            x_dull    = x_dull.to(DEVICE, non_blocking=True)
            y_perfect = y_perfect.to(DEVICE, non_blocking=True)

            predictions = model(x_dull)
            l1_loss     = criterion(predictions, y_perfect)
            tv_loss     = total_variation_loss(predictions)
            loss        = l1_loss + (0.05 * tv_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train = train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_dull, y_perfect in val_loader:
                x_dull    = x_dull.to(DEVICE, non_blocking=True)
                y_perfect = y_perfect.to(DEVICE, non_blocking=True)
                predictions = model(x_dull)
                val_loss   += criterion(predictions, y_perfect).item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Save every epoch
        torch.save(model.state_dict(), f"checkpoints/sky_unet_epoch_{epoch+1}.pth")

        # Save best model separately
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "checkpoints/sky_unet_best.pth")
            print(f"  ✓ New best model saved (val loss: {best_val_loss:.4f})")

    print("\nTraining complete!")
    print(f"Best model saved to checkpoints/sky_unet_best.pth (val loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train_model()