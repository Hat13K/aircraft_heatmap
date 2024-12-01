import cv2
import numpy as np
import random
import torch
from torchvision import transforms
from model_unet import UNet
from collections import deque

# Cihaz belirleme
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Model ve dönüşüm ayarları
transform = transforms.Compose([
    transforms.ToTensor()
])

def vibrate_image(frame, shake_amount):
    """Titreşim efekti uygular."""
    rows, cols = frame.shape[:2]
    dx = random.randint(-shake_amount, shake_amount)
    dy = random.randint(-shake_amount, shake_amount)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (cols, rows))

def get_mse(frame1, frame2):
    """MSE (Mean Squared Error) hesaplaması."""
    return np.mean((frame1.astype(np.float32) - frame2.astype(np.float32))**2)
def stabilize_frames(queue, reference_frame, threshold=4000):
    """Stabilize edilmiş kareyi, kuyruğun ikinci yarısından itibaren MSE'ye göre seçer."""
    min_mse = float('inf')
    best_frame = queue[0]
    start_index = len(queue) // 2  # Kuyruğun ikinci yarısından başla
    for i in range(start_index, len(queue)):
        frame = queue[i]
        mse = get_mse(frame, reference_frame)
        if mse < min_mse:
            
            min_mse = mse
            best_frame = frame
    return best_frame

def visualize_vibration_and_stabilized_prediction(video_path, model_path, shake_amount=25):
    # Modeli yükle
    model = UNet(1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Video yükle
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open video file.")
        return

    img_width, img_height = 512, 384  # Model input boyutları
    queue = deque(maxlen=10)
    ref_frame = None

    # Video işleme döngüsü
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Modelin beklediği boyutta görüntüleri yeniden boyutlandır
        frame_resized = cv2.resize(frame, (img_width, img_height))
        vibrated_frame = vibrate_image(frame_resized, shake_amount)
        
        # Stabilize edilen görüntüyü işleyin
        if ref_frame is None:
            ref_frame = vibrated_frame
        queue.append(vibrated_frame)
        
        # Stabilizasyonu MSE'ye göre yap (kuyruğun ikinci yarısından itibaren)
        stabilized_frame = stabilize_frames(queue, ref_frame)
        ref_frame = stabilized_frame

        # Model girdi verisi: 512x384 boyutunda
        input_tensor_vibrated = transform(vibrated_frame).unsqueeze(0).to(device)
        input_tensor_stabilized = transform(stabilized_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            output_vibrated = model(input_tensor_vibrated).squeeze().cpu().numpy()
            output_stabilized = model(input_tensor_stabilized).squeeze().cpu().numpy()

        # Model çıktısını 512x384 boyutuna yeniden boyutlandır
        vibrated_frame = cv2.resize(vibrated_frame, (256, 192))
        output_vibrated = cv2.resize(output_vibrated, (256, 192))
        stabilized_frame = cv2.resize(stabilized_frame, (256, 192))
        output_stabilized = cv2.resize(output_stabilized, (256, 192))

        # Çıktıyı renkli formata dönüştür
        # output_vibrated = cv2.cvtColor((output_vibrated * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # output_stabilized = cv2.cvtColor((output_stabilized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        output_vibrated = cv2.applyColorMap((output_vibrated * 255).astype(np.uint8), cv2.COLORMAP_JET)
        output_stabilized = cv2.applyColorMap((output_stabilized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Görüntüleri yan yana birleştir
        vibrated_combined = cv2.hconcat([vibrated_frame, output_vibrated])
        stabilized_combined = cv2.hconcat([stabilized_frame, output_stabilized])
        final_output = cv2.vconcat([vibrated_combined, stabilized_combined])
        # final_output = cv2.hconcat([output_vibrated, output_stabilized])

        cv2.imshow('Vibration and Stabilized Outputs', final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Parametreler
video_path = '/Users/hakrts/Desktop/proje/v_Mayis.mp4'
model_path = '/Users/hakrts/Desktop/proje/yeni/unet2_h.pth'

visualize_vibration_and_stabilized_prediction(video_path, model_path)
