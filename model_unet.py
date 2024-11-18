import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(8)
        self.e12 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.e21 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(16)
        self.e22 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(32)
        self.e32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.e41 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(64)
        self.e42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(128)
        self.e52 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(p=0.2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn11d = nn.BatchNorm2d(64)
        self.d12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(32)
        self.d22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(16)
        self.d32 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(16)

        self.upconv4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(8)
        self.d42 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(8)
        self.dropout5 = nn.Dropout(p=0.2)

        self.outconv = nn.Conv2d(8, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.bn11(self.e11(x)))
        xe12 = F.leaky_relu(self.bn12(self.e12(xe11)), negative_slope=0.01)
        xp1 = self.pool1(xe12)
        xp1 = self.dropout1(xp1)

        xe21 = torch.tanh(self.bn21(self.e21(xp1)))
        xe22 = F.relu(self.bn22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = F.leaky_relu(self.bn31(self.e31(xp2)), negative_slope=0.01)
        xe32 = torch.tanh(self.bn32(self.e32(xe31)))
        xp3 = self.pool3(xe32)
        xp3 = self.dropout2(xp3)

        xe41 = F.relu(self.bn41(self.e41(xp3)))
        xe42 = F.leaky_relu(self.bn42(self.e42(xe41)), negative_slope=0.01)
        xp4 = self.pool4(xe42)

        xe51 = torch.tanh(self.bn51(self.e51(xp4)))
        xe52 = F.relu(self.bn52(self.e52(xe51)))
        xe52 = self.dropout3(xe52)

        # Decoder
        xu1 = self.upconv1(xe52)
        xe42 = F.interpolate(xe42, size=xu1.shape[2:], mode='bilinear', align_corners=False)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.leaky_relu(self.bn11d(self.d11(xu11)), negative_slope=0.01)
        xd12 = torch.tanh(self.bn12d(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        xe32 = F.interpolate(xe32, size=xu2.shape[2:], mode='bilinear', align_corners=False)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.bn21d(self.d21(xu22)))
        xd22 = F.leaky_relu(self.bn22d(self.d22(xd21)), negative_slope=0.01)
        xd22 = self.dropout4(xd22)

        xu3 = self.upconv3(xd22)
        xe22 = F.interpolate(xe22, size=xu3.shape[2:], mode='bilinear', align_corners=False)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = torch.tanh(self.bn31d(self.d31(xu33)))
        xd32 = F.relu(self.bn32d(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        xe12 = F.interpolate(xe12, size=xu4.shape[2:], mode='bilinear', align_corners=False)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.leaky_relu(self.bn41d(self.d41(xu44)), negative_slope=0.01)
        xd42 = torch.tanh(self.bn42d(self.d42(xd41)))
        xd42 = self.dropout5(xd42)

        output = self.outconv(xd42)

        output = torch.sigmoid(output)

        return output
