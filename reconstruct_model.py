import os
import glob

def reconstruct_file(base_name="checkpoints/sky_unet_best.pth"):
    # Find all parts for the given base name
    parts = sorted(glob.glob(f"{base_name}.part*"))
    
    if not parts:
        print(f"No parts found for {base_name}.")
        return

    print(f"Found {len(parts)} parts. Reconstructing {base_name}...")
    
    with open(base_name, 'wb') as outfile:
        for part in parts:
            print(f"Appending {part}...")
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
            
    print(f"Successfully reconstructed {base_name}!")

if __name__ == "__main__":
    reconstruct_file()
