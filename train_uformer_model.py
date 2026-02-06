import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True

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

train_input = os.path.join(BASE_DIR, "data/train/inputs")
train_target = os.path.join(BASE_DIR, "data/train/targets")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------
# Transforms & Loader
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = ImageEnhanceDataset(train_input, train_target, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# -------------------------------
# Model
# -------------------------------
from uformer import Uformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Uformer(in_ch=3, out_ch=3).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

# -------------------------------
# Training Loop
# -------------------------------
num_epochs = 1
best_loss = float("inf")

print("🚀 Starting training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for step, (inputs, targets) in enumerate(train_loader):
        if step % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step}")

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f}")

    # -------- Save checkpoint --------
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict()
    }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"))

    # -------- Save best model --------
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print("✅ Best model saved")

print("✅ Training completed")
