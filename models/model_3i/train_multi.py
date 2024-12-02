from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
# from model_multi import UNetThreeInputs
from model_multi import UNetThreeInputs
from data_preprocess3 import CustomDataset
import torch.nn as nn
import torch
import torch.optim as optim
import os 
import json
from torchinfo import summary


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
model = UNetThreeInputs(3, 1).to(device)


image_dir = '/Users/hakrts/Desktop/proje/images'
mask_dir = '/Users/hakrts/Desktop/proje/masks'

# Define loss function and optimizer
#criterion = nn.BCELoss(reduction="sum")
criterion = nn.BCELoss()
learning_rate = 1e-3
dataset = CustomDataset(image_dir, mask_dir)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Update the training loop to handle three inputs
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, early_stopping_patience=5):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    patience_counter = 0
    checkpoint_path = 'model_checkpoint_t3.pth'
    
    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training
        model.train()
        train_loss = 0.0
        for images_t, images_t1, images_t2, masks in train_loader:
            images_t, images_t1, images_t2, masks = images_t.to(device), images_t1.to(device), images_t2.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images_t, images_t1, images_t2)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images_t.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)  # Append train loss to list

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_t, images_t1, images_t2, masks in val_loader:
                images_t, images_t1, images_t2, masks = images_t.to(device), images_t1.to(device), images_t2.to(device), masks.to(device)
                outputs = model(images_t, images_t1, images_t2)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images_t.size(0)
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
    with open('losses_t.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

    return model


model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=1000, early_stopping_patience=3)
