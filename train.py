import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data_pipeline import download_pexels_sky, SkyEnhancementDataset
from unet import LightUNet
from tqdm import tqdm
import os

# ── Hyperparameters ──────────────────────────────────────────
BATCH_SIZE    = 12       # safe for 4070 Super 12GB at 512px with AMP
EPOCHS        = 30
LEARNING_RATE = 1e-4
NUM_IMAGES    = 8000
IMAGE_SIZE    = 512      # was 256 — higher res = less upscale blur and grain
IMAGE_DIR     = "sky_images"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────


def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def ssim_loss(pred, target, window_size=11):
    """
    Structural Similarity loss — penalises blurry/flat outputs and
    preserves fine cloud texture.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p  = torch.nn.functional.avg_pool2d(pred,   window_size, 1, window_size // 2)
    mu_t  = torch.nn.functional.avg_pool2d(target, window_size, 1, window_size // 2)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p2 = torch.nn.functional.avg_pool2d(pred   * pred,   window_size, 1, window_size // 2) - mu_p2
    sigma_t2 = torch.nn.functional.avg_pool2d(target * target, window_size, 1, window_size // 2) - mu_t2
    sigma_pt = torch.nn.functional.avg_pool2d(pred   * target, window_size, 1, window_size // 2) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))
    return 1 - ssim_map.mean()


def saturation_reg_loss(pred, target):
    """
    NEW — Saturation regularisation.

    In OpenCV LAB (normalised to [0,1]):
      • Channel 0 = L (lightness)
      • Channels 1 & 2 = A and B (colour opponents); neutral = 128/255 ≈ 0.502

    Chroma ≈ sqrt((A - 0.5)² + (B - 0.5)²)

    This loss penalises the model if its predicted chroma is significantly
    higher than the ground-truth chroma, directly preventing over-saturation.
    It is asymmetric: only the excess is penalised, not under-saturation.
    """
    neutral = 0.502  # 128 / 255
    pred_chroma   = torch.sqrt((pred[:, 1] - neutral) ** 2 + (pred[:, 2] - neutral) ** 2 + 1e-6)
    target_chroma = torch.sqrt((target[:, 1] - neutral) ** 2 + (target[:, 2] - neutral) ** 2 + 1e-6)
    # Penalise only over-saturation (pred chroma > target chroma)
    excess = torch.clamp(pred_chroma - target_chroma, min=0)
    return excess.mean()


def combined_loss(pred, target):
    l1   = nn.L1Loss()(pred, target)
    ssim = ssim_loss(pred, target)
    tv   = total_variation_loss(pred)
    sat  = saturation_reg_loss(pred, target)

    # L1        → colour accuracy
    # SSIM      → cloud structure / smooth gradients
    # TV        → prevent noisy/pixelated output
    # sat_reg   → prevent over-saturation blowout   ← NEW
    return l1 + (0.5 * ssim) + (0.05 * tv) + (0.2 * sat)


def train_model():
    print(f"Using device: {DEVICE}")

    filepaths = download_pexels_sky(num_images=NUM_IMAGES, save_dir=IMAGE_DIR)
    if len(filepaths) == 0:
        raise RuntimeError("No images found! Check your Pexels API key.")

    full_dataset = SkyEnhancementDataset(filepaths, image_size=IMAGE_SIZE)
    val_size     = max(1, int(0.1 * len(full_dataset)))
    train_size   = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(DEVICE == "cuda"),
        persistent_workers=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(DEVICE == "cuda"),
        persistent_workers=(DEVICE == "cuda")
    )

    model     = LightUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5, verbose=True
    )

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    # Automatic Mixed Precision — uses float16 for most ops on the 4070 Super,
    # giving ~40-50% faster training with no quality loss.
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train", leave=False)

        for x_dull, y_perfect in loop:
            x_dull    = x_dull.to(DEVICE, non_blocking=True)
            y_perfect = y_perfect.to(DEVICE, non_blocking=True)

            # AMP autocast: runs forward pass in float16 where safe
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                predictions = model(x_dull)
                loss        = combined_loss(predictions, y_perfect)

            optimizer.zero_grad()
            scaler.scale(loss).backward()   # scaled gradients avoid float16 underflow
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_dull, y_perfect in val_loader:
                x_dull    = x_dull.to(DEVICE, non_blocking=True)
                y_perfect = y_perfect.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    val_loss += combined_loss(model(x_dull), y_perfect).item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        torch.save(model.state_dict(), f"checkpoints/sky_unet_epoch_{epoch+1}.pth")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "checkpoints/sky_unet_best.pth")
            print(f"  ✓ Best model saved (val: {best_val_loss:.4f})")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print("Best model → checkpoints/sky_unet_best.pth")


if __name__ == "__main__":
    train_model()