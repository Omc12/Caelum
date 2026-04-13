# Caelum

Caelum is a **sky photo enhancement** project built in **PyTorch**. It trains a lightweight **Attention + Residual U-Net** to transform **dull/hazy/under-saturated sky photos** into cleaner, more vibrant results while avoiding the “over-saturation blowout” that many enhancement models produce.

The project includes:
- A dataset pipeline that **downloads sky photos from Pexels** and creates paired training data by applying **realistic degradations**.
- A training script with a **multi-term loss** (L1 + SSIM + total variation + saturation regularization) and **AMP** for faster GPU training.
- An inference script that enhances a single image and applies gentle post-processing (denoise, unsharp mask, light color grading, and blending).

## How it works (high level)

- **Training pairs** are created by taking a clean sky photo and producing a “dull” version via:
  - mild saturation/brightness reduction
  - mild haze
  - optional blur
  - optional noise (gaussian / poisson / jpeg artifacts)
- The model operates in **LAB color space** (normalized), and uses **residual learning**: it predicts a small delta and adds it back to the input (helps preserve already-good skies).

## Project layout

- `data_pipeline.py` — downloads Pexels images and defines `SkyEnhancementDataset`
- `unet.py` — `LightUNet` (U-Net + attention gates + residual head)
- `train.py` — trains the model and saves checkpoints
- `inference.py` — runs enhancement on a single image using a saved checkpoint
- `reconstruct_model.py` — helper to reconstruct a checkpoint from `.part*` files (if you split model weights)

## Requirements

Python 3.9+ recommended.

Key dependencies:
- `torch`
- `opencv-python` (`cv2`)
- `numpy`
- `requests`
- `tqdm`

If you want a quick start, install:
```bash
pip install torch opencv-python numpy requests tqdm
```

## Configure Pexels (important)

Training uses the Pexels API to download sky images.

1. Create/get a Pexels API key.
2. Set it in `data_pipeline.py` (currently referenced as `PEXELS_API_KEY`).

**Tip:** Don’t commit real API keys to GitHub. Prefer an environment variable or a local config file.

## Train

By default, training:
- downloads ~8000 images into `sky_images/`
- trains at 512×512
- saves checkpoints in `checkpoints/`

Run:
```bash
python train.py
```

Checkpoints saved:
- `checkpoints/sky_unet_epoch_XX.pth` (every epoch)
- `checkpoints/sky_unet_best.pth` (best validation loss)

## Inference (enhance an image)

Run:
```bash
python inference.py
```

Or call the function from code:
```python
from inference import enhance_image

enhance_image(
    image_path="your_input.jpg",
    model_path="checkpoints/sky_unet_best.pth",
    output_path="enhanced_sky.jpg"
)
```

Notes:
- Inference preprocess must match training (resizes to 512×512, converts to LAB, normalizes).
- Output is resized back to original resolution and lightly post-processed to reduce grain and preserve natural color.

## Reconstructing a split checkpoint (optional)

If your model file is stored as parts (e.g. `sky_unet_best.pth.part1`, `...part2`, etc.):

```bash
python reconstruct_model.py
```

This concatenates parts back into `checkpoints/sky_unet_best.pth` (based on the default path used by the script).

## Safety / caveats

- This repo currently looks like an **experiment / prototype**: there’s no packaged CLI, no pinned `requirements.txt`, and the Pexels key handling should be hardened before sharing publicly.
- Training downloads a lot of images; be mindful of API rate limits (the downloader handles HTTP 429 by sleeping and retrying).

## License

No license file is currently included. If you want others to use or contribute, add a license (MIT/Apache-2.0 are common for code).

---

If you want, tell me your preferred license + whether this is meant to be a “research repo” or a “tool people install”, and I’ll tailor the README wording and add:
- a proper `requirements.txt`
- a `.env`-style key workflow
- a small CLI usage section
