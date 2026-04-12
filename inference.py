import torch
import cv2
import numpy as np
from torchvision import transforms
from unet import LightUNet

def enhance_image(image_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    model = LightUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and prep image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])
    
    img_resized = cv2.resize(img, (256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))

    # Reverse the normalization to turn it back into a viewable image
    output_hsv = output.squeeze().cpu().numpy()
    output_hsv = np.transpose(output_hsv, (1, 2, 0))

    output_hsv[:, :, 0] *= 360.0

    output_rgb = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2RGB)

    final_output = np.clip(output_rgb * 255, 0, 255).astype(np.uint8)

    # Resize back to original dimensions
    final_output = cv2.resize(final_output, original_size)
    final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

    cv2.imwrite("enhanced_sky_hsv.jpg", final_output)
    print("Saved enhanced image to enhanced_sky.jpg")

if __name__ == "__main__":
    enhance_image("sunset-background.webp", "checkpoints/sky_unet_epoch_10.pth")
