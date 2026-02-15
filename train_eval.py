import os
import sys
import subprocess
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import io
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---

def get_xmp_rating(filepath):
    """Extract XMP:Rating from a file using exiftool."""
    try:
        cmd = ["exiftool", "-Rating", "-json", filepath]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        if data:
            rating = data[0].get("Rating", 0)
            return float(rating)
    except Exception as e:
        print(f"Warning: Could not extract rating from {filepath}: {e}")
    return 0.0

def load_image(filepath):
    """Load image, handling CR2 by extracting preview."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.cr2':
        # Try different preview tags
        for tag in ["-PreviewImage", "-JpgFromRaw", "-ThumbnailImage"]:
            cmd = ["exiftool", tag, "-b", filepath]
            result = subprocess.run(cmd, capture_output=True)
            if result.stdout:
                try:
                    return Image.open(io.BytesIO(result.stdout)).convert('RGB')
                except:
                    continue
        raise ValueError(f"Could not extract preview from {filepath}")
    else:
        return Image.open(filepath).convert('RGB')

# --- Dataset ---

class ImageRatingDataset(Dataset):
    def __init__(self, file_list, ratings, transform=None):
        self.file_list = file_list
        self.ratings = ratings
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        rating = self.ratings[idx]

        try:
            image = load_image(filepath)
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return a dummy black image if loading fails
            image = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        return image, torch.tensor([rating], dtype=torch.float32)

# --- Model ---

class MobileNetV3Regression(nn.Module):
    def __init__(self):
        super(MobileNetV3Regression, self).__init__()
        # Use weights=MobileNet_V3_Large_Weights.DEFAULT for latest V3-Large
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        # Replace the classifier head with a regression head
        # MobileNetV3 Large classifier has 1280 input features
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)

# --- Training and Evaluation Functions ---

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, ratings in tqdm(loader, desc="Training", leave=False):
        images, ratings = images.to(device), ratings.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_ratings = []

    with torch.no_grad():
        for images, ratings in tqdm(loader, desc="Evaluating", leave=False):
            images, ratings = images.to(device), ratings.to(device)
            outputs = model(images)
            loss = criterion(outputs, ratings)
            running_loss += loss.item() * images.size(0)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_ratings.extend(ratings.cpu().numpy().flatten())

    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_ratings)))
    return running_loss / len(loader.dataset), mae, all_preds

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MobileNetV3 for image rating regression.")
    parser.add_argument("train_dir", help="Directory with images for training and validation.")
    parser.add_argument("unseen_dir", help="Directory with unseen images for prediction.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # 1. Collect all images and ratings from train_dir
    print(f"Collecting images from {args.train_dir}...")
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.cr2', '.tiff')
    all_files = []
    for root, _, files in os.walk(args.train_dir):
        for f in files:
            if f.lower().endswith(image_extensions):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print("No images found in training directory.")
        return

    print(f"Found {len(all_files)} images. Extracting ratings...")
    all_ratings = [get_xmp_rating(f) for f in tqdm(all_files, desc="Ratings")]

    # 2. Split into train and validation (80/20)
    train_files, val_files, train_ratings, val_ratings = train_test_split(
        all_files, all_ratings, test_size=0.2, random_state=42
    )

    # 3. Data Loaders
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageRatingDataset(train_files, train_ratings, transform=transform)
    val_ds = ImageRatingDataset(val_files, val_ratings, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 4. Initialize Model, Optimizer, Loss
    model = MobileNetV3Regression().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_mae = float('inf')
    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_mae, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  --> Saved new best model with MAE: {best_mae:.4f}")

    # 6. Predict on unseen_dir
    print(f"\nPredicting on unseen images in {args.unseen_dir}...")
    unseen_files = []
    for root, _, files in os.walk(args.unseen_dir):
        for f in files:
            if f.lower().endswith(image_extensions):
                unseen_files.append(os.path.join(root, f))

    if not unseen_files:
        print("No images found in unseen directory.")
    else:
        unseen_ratings = [get_xmp_rating(f) for f in tqdm(unseen_files, desc="Unseen Ratings")]
        unseen_ds = ImageRatingDataset(unseen_files, unseen_ratings, transform=transform)
        unseen_loader = DataLoader(unseen_ds, batch_size=args.batch_size, shuffle=False)

        model.load_state_dict(torch.load("best_model.pth"))
        unseen_loss, unseen_mae, unseen_preds = evaluate(model, unseen_loader, criterion, DEVICE)

        print("\nResults for unseen images:")
        for f, pred, actual in zip(unseen_files, unseen_preds, unseen_ratings):
            print(f"{f}: Predicted={pred:.2f}, Actual={actual:.1f}")

        print(f"\nOverall Unseen MAE: {unseen_mae:.4f}")

if __name__ == "__main__":
    main()
