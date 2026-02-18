import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
from tqdm import tqdm

# ==========================================================
# 1. HARDCODED CONFIGURATION (3:40 AM WIN)
# ==========================================================
N_CLASSES = 2
BACKBONE_SIZE = "small"
MODEL_PATH = r"C:\Users\hp\Downloads\Offroad_Segmentation_Scripts (1)\segmentation_head.pth"
DATA_DIR = r"C:\Users\hp\Downloads\Offroad_Segmentation_Scripts (1)\Offroad_Segmentation_testImages"
OUTPUT_DIR = "./test_results"

# Mapping: Only Dry Grass (300) is '1', everything else is '0'
VALUE_MAP = {0: 0, 100: 0, 200: 0, 300: 1, 500: 0, 550: 0, 700: 0, 800: 0, 7100: 0, 10000: 0}

# ==========================================================
# 2. DATASET & MODEL ARCHITECTURE
# ==========================================================
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Forced absolute paths to kill the FileNotFoundError
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = np.array(Image.open(os.path.join(self.masks_dir, data_id)))
        
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        for raw, val in VALUE_MAP.items(): binary_mask[mask == raw] = val
        
        if self.transform: img = self.transform(img)
        return img, torch.from_numpy(binary_mask).long(), data_id

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 128, 7, padding=3), nn.GELU())
        self.block = nn.Sequential(nn.Conv2d(128, 128, 7, padding=3, groups=128), nn.GELU(), nn.Conv2d(128, 128, 1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.classifier(self.block(self.stem(x)))

# ==========================================================
# 3. EVALUATION EXECUTION
# ==========================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Standard resize for DINOv2
    w, h = 476, 266 
    transform = transforms.Compose([
        transforms.Resize((h, w)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valset = MaskDataset(DATA_DIR, transform=transform)
    loader = DataLoader(valset, batch_size=2, shuffle=False)

    print("Loading Backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    
    print("Loading Weights...")
    classifier = SegmentationHeadConvNeXt(384, N_CLASSES, w // 14, h // 14).to(device)
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    classifier.eval()
    
    print("🚀 All systems ready. Starting evaluation...")
    iou_scores = []
    with torch.no_grad():
        for imgs, labels, ids in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)
            feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(feat)
            outputs = F.interpolate(logits, size=labels.shape[1:], mode="bilinear")
            preds = torch.argmax(outputs, dim=1)

            for i in range(imgs.shape[0]):
                intersection = (preds[i] & labels[i].to(device)).sum().float()
                union = (preds[i] | labels[i].to(device)).sum().float()
                iou_scores.append((intersection / (union + 1e-6)).cpu().item())

    print("\n" + "="*35)
    print(f"   FINAL TEST IoU: {np.mean(iou_scores):.4f}")
    print("="*35)
    print("FINISHED. YOU CAN NOW CLOSE YOUR LAPTOP.")

    # Quick fix to save at least 5 comparison images
    print("Saving sample visualizations...")
    os.makedirs("./final_visuals", exist_ok=True)
    with torch.no_grad():
        for imgs, labels, ids in loader:
            imgs = imgs.to(device)
            feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(feat)
            preds = torch.argmax(F.interpolate(logits, size=labels.shape[1:], mode="bilinear"), dim=1)
            
            # Save just the first batch (2-5 images)
            for i in range(imgs.shape[0]):
                pred_np = preds[i].cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(f"./final_visuals/{ids[i]}", pred_np)
            break # Stop after one batch to save time
    print("Check the 'final_visuals' folder!")

if __name__ == "__main__": main()