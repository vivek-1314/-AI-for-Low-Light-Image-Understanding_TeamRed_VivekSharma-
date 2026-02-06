import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

# -------------------------------
# Dataset
# -------------------------------
class ImageEnhanceDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp = Image.open(os.path.join(self.input_dir, self.input_files[idx])).convert("RGB")
        tgt = Image.open(os.path.join(self.target_dir, self.target_files[idx])).convert("RGB")
        if self.transform:
            inp = self.transform(inp)
            tgt = self.transform(tgt)
        return inp, tgt

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

val_input = os.path.join(BASE_DIR, "data/val/inputs")
val_target = os.path.join(BASE_DIR, "data/val/targets")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints/best_model.pth")


print("\n--- Validation Setup ---")
# -------------------------------
# Loader
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

val_dataset = ImageEnhanceDataset(val_input, val_target, transform)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


print("\n--- Loading Model ---")
# -------------------------------
# Model
# -------------------------------
from uformer import Uformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = Uformer(in_ch=3, out_ch=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

lpips_fn = lpips.LPIPS(net='alex').to(device)

print("\n--- Starting Metrics ---")

# -------------------------------
# Metrics
# -------------------------------
psnr_list, ssim_list, lpips_list = [], [], []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        o = outputs.permute(0,2,3,1).cpu().numpy()
        t = targets.permute(0,2,3,1).cpu().numpy()
        print("outer loop")
        for oi, ti in zip(o, t):
            oi_u8 = (oi * 255).astype(np.uint8)
            ti_u8 = (ti * 255).astype(np.uint8)

            psnr_list.append(psnr(ti_u8, oi_u8))
            ssim_list.append(ssim(ti_u8, oi_u8, channel_axis=-1))

            lpips_list.append(
                lpips_fn(
                    torch.from_numpy(oi).permute(2,0,1).unsqueeze(0).to(device),
                    torch.from_numpy(ti).permute(2,0,1).unsqueeze(0).to(device)
                ).item()
            )
            print("innter loop")

print("\n--- Validation Metrics ---")
print(f"PSNR  >= 26.2 : {np.mean(psnr_list):.3f}")
print(f"SSIM  >= 0.9  : {np.mean(ssim_list):.3f}")
print(f"LPIPS <= 0.095: {np.mean(lpips_list):.3f}")
