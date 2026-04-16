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
    # The neural network uses a residual connection (x + delta) and has limited capacity 
    # to subtract noise. If we feed it a noisy input, the output WILL be noisy because 
    # the noise is mathematically passed through the residual layer (`x`).
    # We must wipe the base noise from the input image before the model ever sees it.
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

# ── Detail Preservation ───────────────────────────────────────────────────
    # Creating artificial masks always risks smudging or deleting fine cloud wisps.
    # Instead of complex masking, we use a single, highly-targeted Bilateral Filter.
    # Bilateral filters mathematically preserve sharp edges (clouds) natively, 
    # but strictly smooth out flat low-contrast noise (canvas grain in the sky).
    
    img_clean = cv2.fastNlMeansDenoisingColored(img, None, h=5, hColor=5, templateWindowSize=5, searchWindowSize=15)
    
    img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
    original_size = (img_clean.shape[1], img_clean.shape[0])

    # ── Preprocess ────────────────────────────────────────────────────────────
    # Since the model was trained on downscaled 256x256 images, we process the 
    # image globally at a safe intermediate resolution (768).
    h, w, _ = img_clean.shape
    max_dim = 768
    
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    
    img_resized = cv2.resize(img_clean, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab /= 255.0
    input_tensor = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        output = model(input_tensor)

    # Free up memory immediately after inference
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Apply Enhancement Delta ────────────────────────────────────────────────
    # Convert stitched model output from LAB back to RGB
    output_lab = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_lab = np.clip(output_lab * 255.0, 0, 255).astype(np.uint8)
    output_rgb_lowres = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    
    input_rgb_lowres = img_resized.astype(np.float32)
    
    # 1. Isolate the pure color/lightness enhancement delta
    delta = output_rgb_lowres - input_rgb_lowres
    
    # 2. Heavily blur the delta locally to destroy any microscopic inverse-noise 
    # that the model tried to map, leaving ONLY smooth gradient color shifts.
    delta_blurred = cv2.GaussianBlur(delta, (5, 5), 0)

    # 3. Smoothly upscale this pure, gradient enhancement delta to original resolution
    delta_highres = cv2.resize(delta_blurred * 0.75, original_size, interpolation=cv2.INTER_CUBIC)
    
    # 4. Add the perfectly smooth AI color shifts onto the aggressively denoised original photo
    output_rgb = img_clean.astype(np.float32) + delta_highres

    # ── Save ──────────────────────────────────────────────────────────────────
    final_output = np.clip(output_rgb, 0, 255).astype(np.uint8)
    final_output = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, final_output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved enhanced image to {output_path}")

if __name__ == "__main__":
    enhance_image("samples/pexels-hilal-2150529123-34216162.jpg", "checkpoints/sky_unet_best.pth")