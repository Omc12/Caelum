import torch
import cv2
import numpy as np
from unet import LightUNet

def enhance_image(image_path, model_path="checkpoints/sky_unet_best.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = LightUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])

    # Preprocess — must match data_pipeline.py exactly
    img_resized = cv2.resize(img, (256, 256))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab /= 255.0  # normalize to [0, 1]

    input_tensor = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    # Postprocess — reverse the normalization
    output_lab = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_lab = np.clip(output_lab * 255.0, 0, 255).astype(np.uint8)

    # Convert LAB → RGB → BGR for saving
    output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)
    final_output = cv2.resize(output_rgb, original_size)
    final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

    output_path = "enhanced_sky.jpg"
    cv2.imwrite(output_path, final_output)
    print(f"Saved enhanced image to {output_path}")
    return output_path

if __name__ == "__main__":
    enhance_image("sunset-background.webp", "checkpoints/sky_unet_best.pth")