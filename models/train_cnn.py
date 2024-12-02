import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
import os
import glob
from torchinfo import summary
from model_cnn import SimpleCNN
import json


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
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(768,512)):
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

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except OSError as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        image_resized = image.resize(self.target_size)
        mask_resized = mask.resize(self.target_size)

        if self.transform:
            image = self.transform(image_resized)
            mask = self.transform(mask_resized)

        return image, mask


learning_rate = 1e-3
model = SimpleCNN().to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Print model summary
print("Model Summary:")
summary(model, input_size=(1, 3, 768,512), device=device)

# Define loss function and optimizer
criterion = nn.BCELoss()

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, early_stopping_patience=5):
    patience_counter = 0
    checkpoint_path = '/Users/hakrts/Desktop/proje/yeni/cnn_checkpoint222.pth'
    best_loss=float('inf')    
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
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)  # Append train loss to list

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
            val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)  # Append validation loss to list

        # Print training and validation results
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f},   Val Loss: {val_loss:.4f}')

        # Early stopping and checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
            # Save the best model weights and optimizer state
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

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save training and validation losses to JSON
    with open('losses_cnn.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

    return model


# Directories
train_image_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/train/images'
train_mask_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/train/masks'
test_image_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/test/images'
test_mask_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/test/masks'

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Datasets and DataLoaders with target size (örneğin 256x256 olarak ayarlandı)
train_dataset = CustomDataset(train_image_dir, train_mask_dir, transform=transform, target_size=(768,512))
test_dataset = CustomDataset(test_image_dir, test_mask_dir, transform=transform, target_size=(768,512))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Train the model
model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, early_stopping_patience=10)
torch.save(model.state_dict(), 'cnn_model33.pt')
