import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_pipeline import download_open_images_sky, SkyEnhancementDataset
from unet import LightUNet
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    filepaths = download_open_images_sky(num_samples=5000)
    dataset = SkyEnhancementDataset(filepaths)
    # Added pin_memory=True for faster GPU transfer. Lower num_workers to 0 if lagging persists.
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True if DEVICE == "cuda" else False,
        persistent_workers=True if DEVICE == "cuda" else False
    )

    def total_variation_loss(img):
        tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return tv_h + tv_w

    # 2. Initialize Model, Loss, and Optimizer
    model = LightUNet().to(DEVICE)
    # Enable CUDA Benchmarking for faster convolution ops
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        
    criterion = nn.L1Loss() # L1 Loss prevents blurry outputs
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, leave=True)
        epoch_loss = 0

        for x_dull, y_perfect in loop:
            # non_blocking=True helps speed up host-to-device transfers
            x_dull, y_perfect = x_dull.to(DEVICE, non_blocking=True), y_perfect.to(DEVICE, non_blocking=True)

            # Forward pass
            predictions = model(x_dull)
            l1_loss = criterion(predictions, y_perfect)

            tv_loss = total_variation_loss(predictions)

            loss = l1_loss + (0.05 * tv_loss)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.4f}")
        
        # Save checkpoint to the checkpoints folder
        checkpoint_path = os.path.join("checkpoints", f"sky_unet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    print("Training Complete. Final model saved to checkpoints/ folder.")

if __name__ == "__main__":
    train_model()