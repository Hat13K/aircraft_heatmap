import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data_preprocess_aug import CustomDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchinfo import summary
import json  # For saving losses
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

class UNetWithResNet34(nn.Module):
    def __init__(self, n_class, pretrained=True):
        super().__init__()

        # Load pre-trained ResNet34 encoder (requires torchvision)
        from torchvision.models import resnet34

        self.encoder = resnet34(pretrained=pretrained)

        # Modify encoder output channels (assuming final layer is conv2d)
        self.encoder.fc = nn.Conv2d(self.encoder.fc.in_channels, 64, kernel_size=1)

        # Decoder (adjust channels based on encoder output)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(96, 32, kernel_size=3, padding=1)  # Cat with encoder skip connection
        self.bn11d = nn.BatchNorm2d(32)
        self.d12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(32)

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(48, 16, kernel_size=3, padding=1)  # Cat with encoder skip connection
        self.bn21d = nn.BatchNorm2d(16)
        self.d22 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(32, 8, kernel_size=3, padding=1)  # Cat with encoder skip connection
        self.bn31d = nn.BatchNorm2d(8)
        self.d32 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(8)

        self.upconv4 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(16, 8, kernel_size=3, padding=1)  # Cat with encoder skip connection
        self.bn41d = nn.BatchNorm2d(8)
        self.d42 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(8)
        self.dropout5 = nn.Dropout(p=0.2)

        self.outconv = nn.Conv2d(8, n_class, kernel_size=1)

    def forward(self, x):
        # Extract features from ResNet34 stages (assuming 4 stages)
        encoder_outs = [
            self.encoder.conv1(x),
            self.encoder.layer1(encoder_outs[0]),
            self.encoder.layer2(encoder_outs[1]),
            self.encoder.layer3(encoder_outs[2]),
        ]

        # Pass through encoder
        xe52 = self.encoder.fc(encoder_outs[3])

        # Decoder with skip connections
        xu1 = self.upconv1(xe52)
        xe42 = F.interpolate(encoder_outs[2], size=xu1.shape[2:], mode='bilinear', align_corners=False)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.bn11d(self.d11(xu11)))
        xd12 = F.relu(self.bn12d(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        xe32 = F.interpolate(encoder_outs[1], size=xu2.shape[2:], mode='bilinear', align_corners=False)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.bn21d(self.d21(xu22)))
        xd22 = F.relu(self.bn22d(self.d22(xd21)))
        xd22 = self.dropout4(xd22)

        xu3 = self.upconv3(xd22)
        xe22 = F.interpolate(encoder_outs[0], size=xu3.shape[2:], mode='bilinear', align_corners=False)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.bn31d(self.d31(xu33)))
        xd32 = F.relu(self.bn32d(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, x], dim=1)  
        xd41 = F.relu(self.bn41d(self.d41(xu44)))
        xd42 = F.relu(self.bn42d(self.d42(xd41)))
        xd42 = self.dropout5(xd42)

        out = self.outconv(xd42)

        return out


def train_model(model, criterion, optimizer, dataloaders, num_epochs=100, early_stopping_patience=3):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    patience_counter = 0

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels, _, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)  # Ensure labels are tensors on device

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if outputs.size() != labels.size():
                        # Adjust labels shape if necessary
                        labels = labels.unsqueeze(1) if labels.dim() == 3 else labels
                        outputs = outputs.unsqueeze(1) if outputs.dim() == 3 else outputs
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # Save losses
            if phase == 'train':
                train_losses.append(epoch_loss)
            if phase == 'val':
                val_losses.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    patience_counter = 0

                    # Save the best model weights
                    torch.save(model.state_dict(), 'unet_best_model.pth')
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print('Early stopping triggered')
                    model.load_state_dict(best_model_wts)
                    # Save losses to file
                    with open('losses.json', 'w') as f:
                        json.dump({'train': train_losses, 'val': val_losses}, f)
                    return model

        print()

    print(f'Best val loss: {best_loss:4f}')

    model.load_state_dict(best_model_wts)
    # Save losses to file
    with open('losses2.json', 'w') as f:
        json.dump({'train': train_losses, 'val': val_losses}, f)
    return model


if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Parameters
    num_classes = 1  # Assuming binary segmentation
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    beta1 = 0.9
    beta2 = 0.999
    # Paths
    image_dir = '/Users/hakrts/Desktop/proje/images'
    mask_dir = '/Users/hakrts/Desktop/proje/masks'

    data_transforms = transforms.Compose([
        transforms.Resize((352, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Assuming pre-trained ResNet expects normalized input
    ])

    # Create the dataset object (assuming CustomDataset is defined in data_preprocess.py)
    dataset = CustomDataset(image_dir, mask_dir, transform=data_transforms)

    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.15, random_state=210911174)
    datasets = {
        'train': torch.utils.data.Subset(dataset, train_idx),
        'val': torch.utils.data.Subset(dataset, val_idx)
    }

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    model = UNetWithResNet34(num_classes).to(device)
    print("Model Summary:")
    summary(model, input_size=(1, 3, 352, 480), device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Train model
    model = train_model(model, criterion, optimizer, dataloaders, scheduler=scheduler)