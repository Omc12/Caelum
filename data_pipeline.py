import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import requests
import shutil
from tqdm import tqdm

# Prevent OpenCV from hoarding CPU threads when used with PyTorch DataLoader
cv2.setNumThreads(0)

# ─────────────────────────────────────────────
# CONFIG — paste your free key from pexels.com/api
PEXELS_API_KEY = "YOUR_PEXELS_API_KEY"
# ─────────────────────────────────────────────

def download_pexels_sky(num_images=8000, save_dir="sky_images"):
    """
    Downloads sky/sunset/cloud images from Pexels.
    Get a free API key at https://www.pexels.com/api/
    """
    if PEXELS_API_KEY == "YOUR_PEXELS_API_KEY":
        raise ValueError("Please set your Pexels API key in data_pipeline.py!")

    os.makedirs(save_dir, exist_ok=True)

    # Check how many we already have (resume support)
    existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    if len(existing) >= num_images:
        print(f"Already have {len(existing)} images, skipping download.")
        return [os.path.join(save_dir, f) for f in existing]

    print(f"Downloading {num_images} sky images from Pexels...")

    # Diverse sky queries for better model generalization
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
    headers = {"Authorization": PEXELS_API_KEY}
    filepaths = []

    for query in queries:
        print(f"\nDownloading '{query}' images...")
        page = 1
        downloaded = 0
        pbar = tqdm(total=per_query, desc=query)

        while downloaded < per_query:
            try:
                response = requests.get(
                    "https://api.pexels.com/v1/search",
                    headers=headers,
                    params={"query": query, "per_page": 80, "page": page, "orientation": "landscape"},
                    timeout=10
                )

                if response.status_code == 429:
                    print("Rate limited, waiting 60s...")
                    import time; time.sleep(60)
                    continue

                data = response.json()
                photos = data.get("photos", [])

                if not photos:
                    break  # No more results for this query

                for photo in photos:
                    if downloaded >= per_query:
                        break

                    img_url = photo["src"]["large"]  # ~1280px wide
                    img_id = photo["id"]
                    save_path = os.path.join(save_dir, f"{img_id}.jpg")

                    if os.path.exists(save_path):
                        filepaths.append(save_path)
                        downloaded += 1
                        pbar.update(1)
                        continue

                    try:
                        img_response = requests.get(img_url, stream=True, timeout=15)
                        with open(save_path, "wb") as f:
                            shutil.copyfileobj(img_response.raw, f)
                        filepaths.append(save_path)
                        downloaded += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Failed to download {img_url}: {e}")

                page += 1

            except Exception as e:
                print(f"Request error: {e}")
                break

        pbar.close()

    print(f"\nTotal images downloaded: {len(filepaths)}")
    return filepaths


def load_local_sky_images(image_dir="sky_images"):
    """
    Alternative: load images from a local folder.
    Use this if you already have a dataset downloaded manually.
    """
    import glob
    filepaths = []
    for ext in ["jpg", "jpeg", "png", "webp"]:
        filepaths += glob.glob(f"{image_dir}/**/*.{ext}", recursive=True)
        filepaths += glob.glob(f"{image_dir}/**/*.{ext.upper()}", recursive=True)
    print(f"Found {len(filepaths)} local sky images")
    return filepaths


class SkyEnhancementDataset(Dataset):
    def __init__(self, filepaths, image_size=256):
        # Filter out any corrupt or unreadable files upfront
        self.filepaths = self._validate_files(filepaths)
        self.image_size = image_size
        print(f"Dataset ready: {len(self.filepaths)} valid images")

    def _validate_files(self, filepaths):
        valid = []
        for fp in filepaths:
            if os.path.exists(fp) and os.path.getsize(fp) > 1024:  # skip tiny/corrupt files
                valid.append(fp)
        return valid

    def degrade_image(self, image):
        """Degrades the image to create the dull input for training."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.3   # Drop saturation by 70%
        hsv[:, :, 2] *= 0.8   # Drop brightness slightly

        dull_hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        dull_rgb = cv2.cvtColor(dull_hsv, cv2.COLOR_HSV2RGB)

        # Add a slight gray haze
        haze = np.full_like(dull_rgb, 128)
        final_dull = cv2.addWeighted(dull_rgb, 0.85, haze, 0.15, 0)
        return final_dull

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
        except Exception:
            # Return a random valid sample if this one is corrupt
            return self.__getitem__(np.random.randint(0, len(self.filepaths)))

        dull_image = self.degrade_image(image)

        # Convert to LAB color space — avoids hue circularity issues of HSV
        # and separates lightness from color (better than RGB for enhancement)
        dull_lab  = cv2.cvtColor(dull_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        perfect_lab = cv2.cvtColor(image,     cv2.COLOR_RGB2LAB).astype(np.float32)

        # OpenCV uint8 LAB stores all channels in [0, 255] — just normalize to [0, 1]
        dull_lab    /= 255.0
        perfect_lab /= 255.0

        x_dull    = torch.from_numpy(dull_lab).permute(2, 0, 1)
        y_perfect = torch.from_numpy(perfect_lab).permute(2, 0, 1)

        return x_dull, y_perfect