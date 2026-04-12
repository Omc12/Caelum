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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Initialize Model, Loss, and Optimizer
    model = LightUNet().to(DEVICE)
    criterion = nn.L1Loss() # L1 Loss prevents blurry outputs
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, leave=True)
        epoch_loss = 0

        for x_dull, y_perfect in loop:
            x_dull, y_perfect = x_dull.to(DEVICE), y_perfect.to(DEVICE)

            # Forward pass
            predictions = model(x_dull)
            loss = criterion(predictions, y_perfect)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"sky_unet_epoch_{epoch+1}.pth")

    print("Training Complete. Final model saved.")

if __name__ == "__main__":
    train_model()