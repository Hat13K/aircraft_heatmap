import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_16
from PIL import Image
import glob
import os
import json

# Cihaz ayarı
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Özel Veri Seti Sınıfı
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(image_path))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask.float()
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTSegmentationSmaller(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ViTSegmentationSmaller, self).__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool=None)
        self.feature_dim = self.vit.embed_dim       
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False), 
            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        
        batch_size, num_patches, _ = x.shape
        patch_size = int(num_patches**0.5)
        
        x = x.transpose(1, 2).reshape(batch_size, self.feature_dim, patch_size, patch_size)
        
        outputs = self.segmentation_head(x)
        return outputs    
model = ViTSegmentationSmaller(in_channels=768, out_channels=1).to(device)
model.to(device)

# Eğitim ayarları
learning_rate = 1e-4
criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Eğitim döngüsü
# Eğitim döngüsü
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, early_stopping_patience=5):
    patience_counter = 0
    checkpoint_path = '/Users/hakrts/Desktop/proje/yeni/vit2_checkpoint.pth'
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
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

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
        val_losses.append(val_loss)

        # Print training and validation results
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f},   Val Loss: {val_loss:.4f}')

        # Early stopping and checkpointing
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

    model.load_state_dict(best_model_wts)

    with open('losses_vit2.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

    return model


# Dizinler
train_image_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/train/images'
train_mask_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/train/masks'
test_image_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/test/images'
test_mask_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/test/masks'

# Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT için gerekli boyut
    transforms.ToTensor()
])

# Veri setleri ve DataLoader'lar
train_dataset = CustomDataset(train_image_dir, train_mask_dir, transform=transform)
test_dataset = CustomDataset(test_image_dir, test_mask_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Modeli eğit
model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100)
torch.save(model.state_dict(), 'vit2_model.pt')
