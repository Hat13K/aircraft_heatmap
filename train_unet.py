import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
import os
import glob
from torchinfo import summary
from model_unetS import UNet
import numpy as np
import cv2

# Truncated image hatasını önlemek için
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    
    else "cpu"
)
print(f"Using {device} device")

# Custom Dataset Class for loading images and masks
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(512,384)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size

        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(image_path))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # mask_np = np.array(mask)
        # mask_dilated = cv2.dilate(mask_np, np.ones((5, 5), np.uint8), iterations=6)  # Dilation
        # mask_eroded = cv2.erode(mask_dilated, np.ones((5, 5), np.uint8), iterations=5)  # Erosion
        # mask = Image.fromarray(mask_eroded)

        # Resize images and masks
        image_resized = image.resize(self.target_size)
        mask_resized = mask.resize(self.target_size)

        if self.transform:
            image = self.transform(image_resized)
            mask = self.transform(mask_resized)

        return image, mask

learning_rate = 1e-3
model = UNet(1).to(device)
# checkpoint_path='/Users/hakrts/Desktop/proje/yeni/unet5_h.pth'
# checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
# model.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
# epoch= checkpoint['epoch']

# Print model summary
print("Model Summary:")
summary(model, input_size=(1, 3, 512,384), device=device)

# Define loss function
# pos_weight = torch.tensor([3], device=device)
criterion = nn.BCELoss()

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader=None, num_epochs=100, early_stopping_patience=4):
    patience_counter = 0
    checkpoint_path = '/Users/hakrts/Desktop/proje/yeni/unet5.pth'
    best_loss = float('inf')
    
    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)  # Tüm tensörleri cihazda
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation (only if val_loader is provided)
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)
                val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping and checkpointing based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = model.state_dict()
                patience_counter = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'learning_rate': learning_rate,
                }
                torch.save(checkpoint, checkpoint_path)
                print('Saved best model checkpoint')
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print('Early stopping triggered')
                break
        else:
            # Print training results without validation
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')
            if train_loss < best_loss:
                best_loss = train_loss
                best_model_wts = model.state_dict()
                patience_counter = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'learning_rate': learning_rate,
                }
                torch.save(checkpoint, checkpoint_path)
                print('Saved best model checkpoint')
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print('Early stopping triggered')
                break
            
            

    # Load the best model weights if validation was used
    if val_loader:
        model.load_state_dict(best_model_wts)



# Directories
train_image_dir = 'path_to_images'
train_mask_dir = 'path_to_masks'


# Data transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Datasets and DataLoaders with target size
train_dataset = CustomDataset(train_image_dir, train_mask_dir, transform=transform, target_size=(512,384))

train_loader = DataLoader(train_dataset, batch_size=4)

# Train the model
model = train_model(model, criterion, optimizer, train_loader, num_epochs=40)


