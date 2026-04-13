import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import requests
import shutil
from tqdm import tqdm

cv2.setNumThreads(0)

# ─────────────────────────────────────────────
PEXELS_API_KEY = "GbGsIvuhDBFKJzo7dEELjUIj1u1eP0dyw0153bE8QOwhRWSsRb5cCNyV"
# ─────────────────────────────────────────────


def download_pexels_sky(num_images=8000, save_dir="sky_images"):
    if PEXELS_API_KEY == "YOUR_PEXELS_API_KEY":
        raise ValueError("Please set your Pexels API key in data_pipeline.py!")

    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    if len(existing) >= num_images:
        print(f"Already have {len(existing)} images, skipping download.")
        return [os.path.join(save_dir, f) for f in existing]

    print(f"Downloading {num_images} sky images from Pexels...")
    queries = [
        "sunset sky",
        "dramatic sky clouds",
        "blue sky clouds",
        "sunrise sky",
        "stormy sky",
        "golden hour sky",
        "cloudy sky",
        "clear sky",
    ]
    per_query = num_images // len(queries)
    headers   = {"Authorization": PEXELS_API_KEY}
    filepaths = []

    for query in queries:
        print(f"\nDownloading '{query}' images...")
        page       = 1
        downloaded = 0
        pbar       = tqdm(total=per_query, desc=query)

        while downloaded < per_query:
            try:
                response = requests.get(
                    "https://api.pexels.com/v1/search",
                    headers=headers,
                    params={"query": query, "per_page": 80, "page": page, "orientation": "landscape"},
                    timeout=10
                )
                if response.status_code == 429:
                    import time; time.sleep(60)
                    continue

                photos = response.json().get("photos", [])
                if not photos:
                    break

                for photo in photos:
                    if downloaded >= per_query:
                        break
                    img_id    = photo["id"]
                    img_url   = photo["src"]["large2x"]
                    save_path = os.path.join(save_dir, f"{img_id}.jpg")
                    if os.path.exists(save_path):
                        filepaths.append(save_path)
                        downloaded += 1
                        pbar.update(1)
                        continue
                    try:
                        r = requests.get(img_url, stream=True, timeout=15)
                        with open(save_path, "wb") as f:
                            shutil.copyfileobj(r.raw, f)
                        filepaths.append(save_path)
                        downloaded += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Failed: {e}")
                page += 1
            except Exception as e:
                print(f"Request error: {e}")
                break
        pbar.close()

    print(f"\nTotal downloaded: {len(filepaths)}")
    return filepaths


def load_local_sky_images(image_dir="sky_images"):
    import glob
    filepaths = []
    for ext in ["jpg", "jpeg", "png", "webp"]:
        filepaths += glob.glob(f"{image_dir}/**/*.{ext}", recursive=True)
    print(f"Found {len(filepaths)} local sky images")
    return filepaths


class SkyEnhancementDataset(Dataset):
    def __init__(self, filepaths, image_size=512):
        self.filepaths  = self._validate_files(filepaths)
        self.image_size = image_size
        print(f"Dataset ready: {len(self.filepaths)} valid images")

    def _validate_files(self, filepaths):
        return [fp for fp in filepaths if os.path.exists(fp) and os.path.getsize(fp) > 1024]

    def degrade_image(self, image):
        """
        Degrades a vibrant sky to simulate real-world dull/hazy phone photos.

        KEY FIX vs original:
        ────────────────────
        The original used sat_factor ∈ [0.2, 0.4] — an 60-80% saturation
        cut.  That is far more severe than any real camera produces, so the
        model learned to *always* boost saturation hard, causing the vivid
        blue/red blowout seen in the enhanced output.

        Real phone underexposure / haze causes roughly 20-45% saturation
        loss at most.  Keeping the training distribution close to real
        degradations prevents the model from over-compensating.
        """
        # Mild-to-moderate saturation reduction (realistic range)
        sat_factor = np.random.uniform(0.55, 0.80)   # was [0.20, 0.40] ← too extreme
        bri_factor = np.random.uniform(0.80, 0.95)   # slight underexposure

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= sat_factor
        hsv[:, :, 2] *= bri_factor
        dull_hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        dull_rgb = cv2.cvtColor(dull_hsv, cv2.COLOR_HSV2RGB)

        # Mild atmospheric haze (realistic: 5-15%)
        haze_strength = np.random.uniform(0.05, 0.15)   # was [0.10, 0.20]
        haze = np.full_like(dull_rgb, 200)               # warm white, not grey
        degraded = cv2.addWeighted(dull_rgb, 1.0 - haze_strength, haze, haze_strength, 0)

        # Optional: slight Gaussian blur to simulate lens softness
        if np.random.rand() > 0.6:
            k = np.random.choice([3, 5])
            degraded = cv2.GaussianBlur(degraded, (k, k), 0)

        # ── Noise augmentation (NEW) ──────────────────────────────────────────
        # Simulates real sensor noise from phone cameras in low-light / hazy
        # conditions. Training with noisy inputs teaches the model to suppress
        # grain rather than amplify it (which is what caused the visible texture
        # artifact in the enhanced output).
        #
        # Three noise types, each applied randomly and independently:
        #
        #   1. Gaussian noise  — continuous sensor readout noise (most common)
        #   2. Poisson noise   — shot noise from low photon count (dim skies)
        #   3. JPEG artefacts  — compression ringing along cloud edges
        #
        # Applied only to the DULL image; the clean target is untouched.
        noise_type = np.random.choice(["gaussian", "poisson", "jpeg", "none"],
                                      p=[0.40, 0.20, 0.20, 0.20])

        if noise_type == "gaussian":
            sigma = np.random.uniform(2, 10)          # std in [0,255] space
            noise = np.random.normal(0, sigma, degraded.shape).astype(np.float32)
            degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        elif noise_type == "poisson":
            # Scale to [0,1], add Poisson, scale back
            scale   = np.random.uniform(0.03, 0.10)
            img_f   = degraded.astype(np.float32) / 255.0
            noisy   = np.random.poisson(img_f / scale) * scale
            degraded = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)

        elif noise_type == "jpeg":
            quality  = np.random.randint(30, 70)      # low quality → blocky artefacts
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, enc   = cv2.imencode(".jpg",
                                    cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR),
                                    encode_param)
            degraded = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR),
                                    cv2.COLOR_BGR2RGB)
        # "none" → no noise added (keeps ~20% of samples clean for stability)

        return degraded

    def augment(self, image, seed):
        """Apply identical random augmentation to both dull and perfect images."""
        rng  = np.random.RandomState(seed)
        if rng.rand() > 0.5:
            image = cv2.flip(image, 1)
        h, w  = image.shape[:2]
        crop  = rng.randint(0, int(h * 0.1) + 1)
        if crop > 0:
            image = image[crop:h-crop, crop:w-crop]
            image = cv2.resize(image, (self.image_size, self.image_size))
        return image

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.filepaths[idx])
            if image is None:
                raise ValueError()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self.filepaths)))

        dull_image = self.degrade_image(image)

        # Shared seed ensures identical spatial augmentation on both images
        seed       = np.random.randint(0, 100000)
        image      = self.augment(image,      seed)
        dull_image = self.augment(dull_image, seed)

        # LAB: separates lightness from colour — no hue-circularity problem
        dull_lab    = cv2.cvtColor(dull_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        perfect_lab = cv2.cvtColor(image,      cv2.COLOR_RGB2LAB).astype(np.float32)

        dull_lab    /= 255.0
        perfect_lab /= 255.0

        x_dull    = torch.from_numpy(dull_lab).permute(2, 0, 1)
        y_perfect = torch.from_numpy(perfect_lab).permute(2, 0, 1)
        return x_dull, y_perfect