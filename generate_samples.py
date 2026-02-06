import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

# -------------------------------
# Paths
# -------------------------------
IMAGE_PATH = "data/train/inputs/AI_SUMMIT_ENHANCEMENT_0000.jpg"
MODEL_PATH = "checkpoints/best_model.pth"
OUTPUT_PATH = "enhanced_output.png"

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# Model
# -------------------------------
from uformer import Uformer

model = Uformer(in_ch=3, out_ch=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------------
# Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# -------------------------------
# Load image
# -------------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
inp = transform(img).unsqueeze(0).to(device)

# -------------------------------
# Inference
# -------------------------------
with torch.no_grad():
    out = model(inp)
    out = torch.clamp(out, 0, 1)

# -------------------------------
# Save BEFORE | AFTER
# -------------------------------
comparison = make_grid(
    [inp.cpu()[0], out.cpu()[0]],
    nrow=2
)

save_image(comparison, OUTPUT_PATH)

print(f"✅ Comparison image saved at: {OUTPUT_PATH}")