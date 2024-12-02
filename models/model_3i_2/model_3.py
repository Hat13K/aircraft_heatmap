# Gerekli kütüphaneler
import torch
import torch.nn as nn
import torch.nn.functional as F


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