import torch
import cv2
import numpy as np
from unet import LightUNet


def enhance_image(image_path, model_path="checkpoints/sky_unet_best.pth", output_path="enhanced_sky.jpg"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = LightUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # ── Load image ────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])

    # ── Preprocess ────────────────────────────────────────────────────────────
    # The model is fully convolutional, so we can process at varied resolutions.
    # However, very large images will cause Out Of Memory (OOM) errors.
    # We cap the maximum processing dimension to prevent this.
    h, w, _ = img.shape
    max_dim = 1536  # Safe maximum for most mid-range/high-end GPUs
    
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Ensure dimensions are multiples of 16 (due to 4 MaxPool layers).
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab /= 255.0

    input_tensor = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        with torch.autocast(device_type=device if device == "cuda" else "cpu", enabled=(device == "cuda")):
            output = model(input_tensor)

    # Free up memory immediately after inference
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Postprocess ───────────────────────────────────────────────────────────
    output_lab = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_lab = np.clip(output_lab * 255.0, 0, 255).astype(np.uint8)

    output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)

    # Resize back to exact original resolution
    output_rgb = cv2.resize(output_rgb, original_size, interpolation=cv2.INTER_LANCZOS4)

    # ── Gentle post-processing ────────────────────────────────────────────────

    # 1. Denoise with stronger settings to remove more noise/grain
    output_rgb = cv2.fastNlMeansDenoisingColored(output_rgb, None,
                                                  h=10, hColor=10,
                                                  templateWindowSize=7,
                                                  searchWindowSize=21)

    # 2. Safety clamp
    output_rgb = np.clip(output_rgb, 0, 255).astype(np.uint8)

    # 3. Blend 80% enhanced + 20% original to ground the colours and prevent oversaturation.
    img_original_resized = cv2.resize(
        cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),
        original_size, interpolation=cv2.INTER_LANCZOS4
    )
    output_rgb = cv2.addWeighted(output_rgb, 0.80, img_original_resized, 0.20, 0)

    # ── Save ──────────────────────────────────────────────────────────────────
    final_output = np.clip(output_rgb, 0, 255).astype(np.uint8)
    final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, final_output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved enhanced image to {output_path}")
    return output_path


if __name__ == "__main__":
    enhance_image("samples\pexels-ian-panelo-7538388.jpg", "checkpoints/sky_unet_best.pth")