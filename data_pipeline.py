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
        "sunset sky", "dramatic sky clouds", "blue sky clouds", "sunrise sky",
        "stormy sky", "golden hour sky", "cloudy sky", "clear sky",
    ]
    per_query = num_images // len(queries)
    headers   = {"Authorization": PEXELS_API_KEY}
    filepaths = []

    for query in queries:
        print(f"\nDownloading '{query}' images...")
        page, downloaded = 1, 0
        pbar = tqdm(total=per_query, desc=query)
        while downloaded < per_query:
            try:
                response = requests.get(
                    "https://api.pexels.com/v1/search", headers=headers,
                    params={"query": query, "per_page": 80, "page": page,
                            "orientation": "landscape"}, timeout=10
                )
                if response.status_code == 429:
                    import time; time.sleep(60); continue
                photos = response.json().get("photos", [])
                if not photos: break
                for photo in photos:
                    if downloaded >= per_query: break
                    img_id    = photo["id"]
                    save_path = os.path.join(save_dir, f"{img_id}.jpg")
                    if os.path.exists(save_path):
                        filepaths.append(save_path); downloaded += 1; pbar.update(1); continue
                    try:
                        r = requests.get(photo["src"]["large2x"], stream=True, timeout=15)
                        with open(save_path, "wb") as f:
                            shutil.copyfileobj(r.raw, f)
                        filepaths.append(save_path); downloaded += 1; pbar.update(1)
                    except Exception as e:
                        print(f"Failed: {e}")
                page += 1
            except Exception as e:
                print(f"Request error: {e}"); break
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
    """
    Lean dataset — no cache, no memmap, no workers.
    On Windows num_workers=0 is faster than spawning workers because
    spawning copies the entire process and re-imports everything.
    At 256px each image is tiny (~192KB in RAM) so disk IO is fast enough.
    """
    def __init__(self, filepaths, image_size=256):
        self.image_size = image_size
        self.filepaths  = [fp for fp in filepaths
                           if os.path.exists(fp) and os.path.getsize(fp) > 1024]
        print(f"Dataset ready: {len(self.filepaths)} images")

    def degrade_image(self, image):
        sat_factor = np.random.uniform(0.55, 0.80)
        bri_factor = np.random.uniform(0.80, 0.95)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= sat_factor
        hsv[:, :, 2] *= bri_factor
        dull_rgb = cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB
        )

        haze_strength = np.random.uniform(0.05, 0.15)
        degraded = cv2.addWeighted(
            dull_rgb, 1.0 - haze_strength,
            np.full_like(dull_rgb, 200), haze_strength, 0
        )

        if np.random.rand() > 0.6:
            degraded = cv2.GaussianBlur(degraded, (3, 3), 0)

        if np.random.rand() > 0.4:
            sigma    = np.random.uniform(1, 6)
            noise    = np.random.randn(*degraded.shape).astype(np.float32) * sigma
            degraded = np.clip(
                degraded.astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)

        return degraded

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.filepaths[idx])
            if image is None:
                raise ValueError()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size),
                               interpolation=cv2.INTER_LINEAR)
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self.filepaths)))

        dull_image = self.degrade_image(image)

        # Shared flip augmentation
        if np.random.rand() > 0.5:
            image      = cv2.flip(image,      1)
            dull_image = cv2.flip(dull_image, 1)

        dull_lab    = cv2.cvtColor(dull_image, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        perfect_lab = cv2.cvtColor(image,      cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0

        return (torch.from_numpy(dull_lab).permute(2, 0, 1),
                torch.from_numpy(perfect_lab).permute(2, 0, 1))