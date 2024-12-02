import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data_preprocess import CustomDataset 

import torch.nn.functional as F
import torchvision.transforms as transforms

# Define UNetSimple model
class UNetSimple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetSimple, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = F.relu(self.encoder_conv1(x))
        enc2 = F.relu(self.encoder_conv2(self.pool(enc1)))

        # Decoder path with skip connections
        dec1 = self.upconv1(enc2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        dec2 = F.relu(self.decoder_conv1(dec1))

        # Final convolution followed by sigmoid activation
        output = self.final_conv(dec2)
        output = torch.sigmoid(output)

        return output

# Define UNetRNN model
class UNetRNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, rnn_layers, h, w):
        super(UNetRNN, self).__init__()

        self.hidden_size = hidden_size
        self.h = h
        self.w = w

        # RNN Layer
        self.rnn = nn.LSTM(input_size=in_channels * h * w,  # Input size for LSTM
                           hidden_size=hidden_size,
                           num_layers=rnn_layers,
                           batch_first=True)

        # Fully connected layer to reshape RNN output
        self.fc = nn.Linear(hidden_size, hidden_size * h * w)

        # UNet model
        self.unet = UNetSimple(in_channels=hidden_size, out_channels=out_channels)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        # Reshape input for RNN
        x_rnn = x.view(batch_size, seq_len, -1)  # Flatten H and W

        # Print shapes for debugging
        print(f"x_rnn shape: {x_rnn.shape}")

        # RNN forward pass
        rnn_out, _ = self.rnn(x_rnn)

        # Print shapes for debugging
        print(f"rnn_out shape: {rnn_out.shape}")

        # Use last timestep's output
        rnn_out = rnn_out[:, -1, :]  # Get last output of RNN

        # Print shapes for debugging
        print(f"rnn_out (last timestep) shape: {rnn_out.shape}")

        # Fully connected layer to reshape for UNet input
        rnn_out = self.fc(rnn_out)
        rnn_out = rnn_out.view(batch_size, self.hidden_size, self.h, self.w)  # Reshape for UNet

        # Print shapes for debugging
        print(f"rnn_out reshaped for UNet shape: {rnn_out.shape}")

        output = self.unet(rnn_out)
        return output
    

# Eğitim ve doğrulama işlevleri
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, early_stopping_patience=3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Checkpoint kaydet
            print('Model saved as best_model.pth')
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print('Early stopping')
                break

# Ana fonksiyon
def main():
    # Hiperparametreler ve ayarlar
    in_channels = 3
    out_channels = 1
    lstm_hidden_dim = 128
    lstm_num_layers = 2
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.001

    # Veri dönüşümleri ve veri yükleyiciler
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_dir = '/Users/hakrts/Desktop/proje/data/images'
    mask_dir = '/Users/hakrts/Desktop/proje/data/masks'
    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, kayıp fonksiyonu ve optimizer
    model = UNet_LSTM(in_channels=in_channels, out_channels=out_channels, lstm_hidden_dim=lstm_hidden_dim, lstm_num_layers=lstm_num_layers)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Modeli eğit
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs)

if __name__ == '__main__':
    main()
