import numpy as np
import cv2
import torch
from torchvision import transforms
from model_unet import UNet
import time

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

def visualize_prediction_video(video_path, model_path):

    model = UNet(1).to(device)

    checkpoint_path=model_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open video file.")
        return
    
    start_time = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.resize(frame, (512,384))
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            output = output.squeeze().cpu().numpy()
        
        # Normalize the output for better visualization
        # output = (output - output.min()) / (output.max() - output.min()) * 255
        # output = output.astype(np.uint8)
        
        # # Convert output to color map
        # output_colored = cv2.applyColorMap(output, cv2.COLORMAP_JET)
        output_colored = cv2.applyColorMap(np.uint8(output * 255), cv2.COLORMAP_JET)

        # Stack images horizontally
        combined = cv2.vconcat([cv2.cvtColor(image, cv2.COLOR_BGR2RGB), output_colored])
        
        # Display the combined image
        cv2.imshow('Prediction', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Video file closed.")
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    print(f"Final FPS: {fps:.2f}")

# video_path = 'v_solo.mp4'
video_path = 'v_Mayis.mp4'
model_path = 'unet2_h.pth'
visualize_prediction_video(video_path, model_path)
