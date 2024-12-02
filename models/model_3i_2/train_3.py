# Gerekli kütüphaneler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import json

# Cihaz ayarı
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)




# UNet Model
class UNetWeightedOutput(nn.Module):
    def __init__(self):
        super(UNetWeightedOutput, self).__init__()

        # Encoder katmanları
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck katmanı
        self.bottleneck = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Decoder katmanları
        self.dec1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Ağırlıklar
        self.weight1 = nn.Parameter(torch.tensor(0.65))
        self.weight2 = nn.Parameter(torch.tensor(0.2))
        self.weight3 = nn.Parameter(torch.tensor(0.15))

    def forward(self, x1, x2, x3):
        # Encoder işlemleri
        enc1_x1 = F.relu(self.enc1(x1))
        enc1_x2 = F.relu(self.enc1(x2))
        enc1_x3 = F.relu(self.enc1(x3))

        # Ağırlıklı bir özellik havuzu oluştur
        weighted_features = (self.weight1 * enc1_x1) + (self.weight2 * enc1_x2) + (self.weight3 * enc1_x3)

        # Havuzlanan özellikler üzerinden encoder işlemi
        enc2 = F.relu(self.enc2(self.pool(weighted_features)))

        # Bottleneck işlemi
        bottleneck_output = F.relu(self.bottleneck(enc2))

        # Skip connection: Spatial boyutları eşitliyoruz
        bottleneck_upsampled = F.interpolate(bottleneck_output, size=enc1_x1.shape[2:], mode='bilinear', align_corners=True)

        # Kanal boyutunda birleştirme
        dec1_input = torch.cat((bottleneck_upsampled, enc1_x1), dim=1)
        dec1 = F.relu(self.dec1(dec1_input))

        # Sonuç üretimi
        output = self.dec2(F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=True))
        output = F.interpolate(output, size=(384, 512), mode='bilinear', align_corners=True)

        return output


# Model örneği
model = UNetWeightedOutput().to(device)
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

        self.transform = transforms.Compose([
            transforms.Resize((384, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return max(0, len(self.image_paths) - 2)  # 3 görüntü sekansı için 2 çıkart

    def __getitem__(self, idx):
        image_paths = self.image_paths[idx:idx + 3]  # 3 görüntü al
        mask_path = os.path.join(self.mask_dir, os.path.basename(self.image_paths[idx+2]))  # 3. görüntüye karşılık gelen maskeyi al

        images = [self.transform(Image.open(path).convert("RGB")) for path in image_paths]
        mask = self.transform(Image.open(mask_path).convert("L"))

        return images[2], images[1], images[0], mask 

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, early_stopping_patience=5):
    patience_counter = 0
    checkpoint_path = 'triple_checkpoint.pth'
    best_loss = float('inf')

    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images_t, images_t1, images_t2, masks in train_loader:
            images_t = images_t.to(device)
            images_t1 = images_t1.to(device)
            images_t2 = images_t2.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images_t, images_t1, images_t2)
            # outputs = outputs.squeeze(1)  # Çıktıyı yeniden boyutlandır
            loss = criterion(outputs, masks)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_t, images_t1, images_t2, masks in val_loader:
                images_t = images_t.to(device)
                images_t1 = images_t1.to(device)
                images_t2 = images_t2.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images_t, images_t1, images_t2)
                # outputs = outputs.squeeze(1)  # Çıktıyı yeniden boyutlandır
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print('Early stopping triggered')
            break

    # Save loss history to JSON
    with open('losses_triple.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

    return model

# Directories
train_image_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/train/images'
train_mask_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/train/masks'
test_image_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/test/images'
test_mask_dir = '/Users/hakrts/Desktop/proje/yeni/final_dataset/test/masks'

# Create datasets and dataloaders
train_dataset = CustomDataset(train_image_dir, train_mask_dir)
val_dataset = CustomDataset(test_image_dir, test_mask_dir)  # Assuming test set is used for validation
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100)
