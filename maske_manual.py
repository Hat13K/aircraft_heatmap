import cv2
import os
import numpy as np

def yolo_to_image_coords(image_size, bbox):
    """YOLO formatındaki koordinatları görüntü boyutuna göre dönüştür."""
    img_w, img_h = image_size
    x_center, y_center, w, h = bbox
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h
    return int(x_center), int(y_center), int(w), int(h)

def create_ellipse_mask(image, bbox_coords):
    """Belirtilen bbox'a göre elips maskesi oluştur ve pikselleri işleme al."""
    img_h, img_w = image.shape[:2]
    x_center, y_center, w, h = bbox_coords

    # Boş bir maske oluştur (siyah arka plan)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask2 = np.zeros((img_h, img_w), dtype=np.uint8)

    # Elipsi çiz
    cv2.ellipse(mask, (x_center, y_center), (w // 2, h // 2), 0, 0, 360, 255, -1)
    cv2.ellipse(mask2, (x_center, y_center), (w // 4, h // 4), 0, 0, 360, 255, -1)

    # En dıştaki sınırı bulur
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_pixels = gray_image[mask2 == 255]
    mean_intensity = np.mean(masked_pixels)
    print(mean_intensity)   

    # Sonuç maskesini oluştur
    result1 = np.zeros_like(gray_image)
    result2 = np.zeros_like(gray_image)

    result1[mask == 255] = np.where(gray_image[mask == 255] <= mean_intensity, 255, 0)
    result2[mask == 255] = np.where(gray_image[mask == 255] > mean_intensity, 255, 0)

    # Convert single channel results to three channels for concatenation
    result1_colored = cv2.cvtColor(result1, cv2.COLOR_GRAY2BGR)
    result2_colored = cv2.cvtColor(result2, cv2.COLOR_GRAY2BGR)

    # Combine original image with the two results
    combined = np.hstack((image, result1_colored, result2_colored))

    # Show combined image for 4 seconds
    cv2.imshow('Original Image, Result1, Result2', combined)
    cv2.waitKey(4000)  # Display for 4000 milliseconds (4 seconds)
    cv2.destroyAllWindows()

    print("Maskeleri değerlendirin:")
    print("1: Result1'i maske olarak kabul et")
    print("2: Result2'yi maske olarak kabul et")
    print("3: Threshold değerini güncelle")

    while True:
        choice = input("Seçiminizi yapın (1, 2 veya 3): ")

        if choice == '1':
            selected_mask = result1
            print("Result1 seçildi.")
            break
        elif choice == '2':
            selected_mask = result2
            print("Result2 seçildi.")
            break
        elif choice == '3':
            new_mean_intensity = input("Yeni mean_intensity değerini girin: ")
            try:
                mean_intensity = int(new_mean_intensity)
                print(f"Yeni mean_intensity değeri: {mean_intensity}")
                
                # Recalculate the masks using the new mean intensity
                result1[mask == 255] = np.where(gray_image[mask == 255] <= mean_intensity, 255, 0)
                result2[mask == 255] = np.where(gray_image[mask == 255] > mean_intensity, 255, 0)

                # Convert updated results to three channels for concatenation
                result1_colored = cv2.cvtColor(result1, cv2.COLOR_GRAY2BGR)
                result2_colored = cv2.cvtColor(result2, cv2.COLOR_GRAY2BGR)

                # Update combined view with new results
                combined = np.hstack((image, result1_colored, result2_colored))
                cv2.imshow('Original Image, Result1, Result2', combined)
                cv2.destroyAllWindows()

            except ValueError:
                print("Geçersiz değer! Lütfen bir tamsayı girin.")
        else:
            print("Geçersiz seçim! Lütfen 1, 2 veya 3 girin.")

    return selected_mask


def read_yolo_txt(txt_file_path):
    """YOLO formatındaki koordinatları txt dosyasından oku."""
    coords = []
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            x_center, y_center, width, height = map(float, parts[1:])  # İlk kısım class_id, kullanmıyoruz
            coords.append((x_center, y_center, width, height))
    return coords

def apply_masks_to_image(image_path, txt_file_path):
    """YOLO formatındaki koordinatlara göre maskeleri uygula."""
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]

    # Koordinatları txt dosyasından oku
    yolo_coords_list = read_yolo_txt(txt_file_path)

    # Tüm bounding box'lar için maskeler oluşturulacak
    final_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for yolo_coords in yolo_coords_list:
        # YOLO koordinatlarını görüntü koordinatlarına çevir
        bbox_coords = yolo_to_image_coords((img_w, img_h), yolo_coords)

        # Maske oluştur ve birleştir
        mask = create_ellipse_mask(image, bbox_coords)

        # Maskeleri birleştir
        final_mask = cv2.bitwise_or(final_mask, mask)

    return final_mask

def process_images(image_dir, label_dir, output_mask_dir):
    """Tüm görüntüleri ve etiketleri otomatik olarak işle ve maskeleri kaydet."""
    # Eğer maske dizini yoksa, oluştur
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
   
    # Görüntü dizinindeki tüm dosyaları işle
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + '.txt'  # Aynı isimdeki label dosyası
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                print(f"Processing: {image_filename} with {label_filename}")
                mask = apply_masks_to_image(image_path, label_path)

                # Maskeleri kaydet
                mask_output_path = os.path.join(output_mask_dir, image_filename)
                cv2.imwrite(mask_output_path, mask)
                
            else:
                print(f"Label not found for {image_filename}, skipping.")

# Kullanım örneği:
image_dir = '/path_to_image'
label_dir = 'path_to_label'
output_mask_dir = 'path_to_masks'

# Tüm görüntüleri ve etiketleri işleyip maskeleri kaydet
process_images(image_dir, label_dir, output_mask_dir)
