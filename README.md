AI for Low-Light / Underwater Image Enhancement
Team Red – Vivek Sharma

Overview
This project focuses on low-light and underwater image enhancement using a Transformer-based architecture (Uformer).
The goal is to improve image quality while meeting baseline evaluation metrics such as PSNR, SSIM, LPIPS, and UCIQE.

🧠 Model Architecture
Model Used: Uformer (U-shaped Transformer for Image Restoration)
Input: RGB low-light / underwater images
Output: Enhanced RGB images
Loss Function: L1 Loss
Optimizer: Adam
Precision: Mixed Precision Training (AMP)

Uformer was chosen for its strong ability to capture global context and local features, making it well-suited for image enhancement tasks.

📂 Dataset Usage & Split
To enable faster experimentation:
Total dataset used: 100%dataset (~5500 images)
Training split: ~80% of total dataset
Validation split: ~20% of total dataset
Image size: 256 × 256
Paired data: (Input → Target)

⚠️ Full dataset training was not performed due to GPU time and disk quota limitations.


⚙️ Training Configuration

1st Cycle of training
Parameter	Value
Batch Size	4
Epochs	4
Learning Rate	1e-4
Optimizer	Adam
Loss	L1 Loss
Device	GPU (CUDA)


📊 Validation Metrics Obtained of 1st Cycle
Metric	Obtained Value	Baseline
PSNR	~19.48	≥ 26.2
SSIM	~0.82	≥ 0.90
LPIPS	~0.16	≤ 0.095
UCIQE	Not Obtained	≥ 0.42

Note:
Metrics are below baseline, primarily due to:
> Limited training data
> Low number of epochs
> No UCIQE-aware loss during training

COMMANDS For testing the trained model

1️⃣ Create Conda Environment (CLEAN)
conda create -n uw-enhance-gpu python=3.10 -y

Activate it:
conda activate uw-enhance-gpu

2️⃣ Upgrade pip 
python -m pip install --upgrade pip

3️⃣ Install PyTorch (CUDA 12.1 – matches CDAC)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121