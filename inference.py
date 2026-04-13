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

    # ── Preprocess (must exactly match data_pipeline.py) ─────────────────────
    img_resized = cv2.resize(img, (256, 256))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab /= 255.0

    input_tensor = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        output = model(input_tensor)

    # ── Postprocess ───────────────────────────────────────────────────────────
    output_lab = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_lab = np.clip(output_lab * 255.0, 0, 255).astype(np.uint8)

    output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)

    # Resize back to original resolution using high-quality Lanczos
    output_rgb = cv2.resize(output_rgb, original_size, interpolation=cv2.INTER_LANCZOS4)

    # ── Gentle post-processing ────────────────────────────────────────────────

    # 1. Mild unsharp mask to recover sharpness lost at 256px processing.
    #    OLD: addWeighted(output, 1.4, blurred, -0.4, 0)  ← far too aggressive
    #    NEW: 1.15 / -0.15 — subtle crispness only.
    blurred    = cv2.GaussianBlur(output_rgb, (0, 0), 2)
    output_rgb = cv2.addWeighted(output_rgb, 1.15, blurred, -0.15, 0)

    # 2. Safety clamp after unsharp mask
    output_rgb = np.clip(output_rgb, 0, 255).astype(np.uint8)

    # 3. Gentle colour grading: very small S-curve via LUT on the L channel.
    #    Lifts shadows slightly and adds micro-contrast without blowing highlights.
    output_lab2 = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB)
    l, a, b     = cv2.split(output_lab2)
    lut         = np.array([
        int(np.clip(i + 6 * np.sin(np.pi * i / 255.0), 0, 255))
        for i in range(256)
    ], dtype=np.uint8)
    l          = cv2.LUT(l, lut)
    output_rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # 4. Blend 85% enhanced + 15% original to guard against any remaining
    #    colour cast — ensures the output is always at least as good as input.
    img_original_resized = cv2.resize(
        cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),
        original_size, interpolation=cv2.INTER_LANCZOS4
    )
    output_rgb = cv2.addWeighted(output_rgb, 0.85, img_original_resized, 0.15, 0)

    # ── Save ──────────────────────────────────────────────────────────────────
    final_output = np.clip(output_rgb, 0, 255).astype(np.uint8)
    final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, final_output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved enhanced image to {output_path}")
    return output_path


if __name__ == "__main__":
    enhance_image("pexels-ian-panelo-7538388.jpg", "checkpoints/sky_unet_best.pth")