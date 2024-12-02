import torch
import torch.nn as nn
import torch.nn.functional as F

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

class UNetThreeInputs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetThreeInputs, self).__init__()

        # Three U-Net instances
        self.unet_t = UNetSimple(in_channels, out_channels)
        self.unet_t1 = UNetSimple(in_channels, out_channels)
        self.unet_t2 = UNetSimple(in_channels, out_channels)

        # Adjust the final output size to match the number of input channels
        self.final_conv = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1)  # 3 * out_channels

    def forward(self, x_t, x_t1, x_t2):
        # Get U-Net outputs for each input image
        out_t = self.unet_t(x_t)
        out_t1 = self.unet_t1(x_t1)
        out_t2 = self.unet_t2(x_t2)

        # Concatenate outputs along the channel dimension
        combined = torch.cat((out_t, out_t1, out_t2), dim=1)  # Combine the outputs of all three U-Net instances

        # Apply final convolution to reduce the channel dimension
        final_out = self.final_conv(combined)

        return final_out
