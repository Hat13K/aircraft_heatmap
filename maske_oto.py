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

    # Elipsin kenarındaki pikselleri bul (en dıştaki sınır)
    kernel = np.ones((3, 3), np.uint8)
    outer_border = cv2.dilate(mask, kernel, iterations=1) - mask  # En dıştaki sınırı bulur
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # En dıştaki elips sınırının ortalama piksel değerini al
    border_pixels = gray_image[outer_border == 255]
    mean_circumference = np.mean(border_pixels)
    print('arka plan:',mean_circumference)

    masked_pixels = gray_image[mask2 == 255]
    mean_intensity = np.mean(masked_pixels) 
    print(mean_intensity)   

    # Sonuç maskesini oluştur
    result1 = np.zeros_like(gray_image)
    result2 = np.zeros_like(gray_image)
    
    # Arka plan rengini belirle
    if mean_circumference > 80:  # Ortalama değere göre arka planın beyaz mı siyah mı olduğunu belirle
        # Arka plan beyaz
        print('hedef koyu')
        result1[mask == 255] = np.where(gray_image[mask == 255] <= mean_intensity, 255, 0)
        result2[mask == 255] = np.where(gray_image[mask == 255] >= mean_intensity, 255, 0)
    else:
        # Arka plan koyu
        print('hedef açık')
        result1[mask == 255] = np.where(gray_image[mask == 255] >= mean_intensity, 255, 0)
        result2[mask == 255] = np.where(gray_image[mask == 255] <= mean_intensity, 255, 0)

    return result1, result2



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
    final_mask1 = np.zeros((img_h, img_w), dtype=np.uint8)
    final_mask2 = np.zeros((img_h, img_w), dtype=np.uint8)

    for yolo_coords in yolo_coords_list:
        # YOLO koordinatlarını görüntü koordinatlarına çevir
        bbox_coords = yolo_to_image_coords((img_w, img_h), yolo_coords)

        # Maske oluştur ve birleştir
        mask1, mask2 = create_ellipse_mask(image, bbox_coords)

        # Maskeleri birleştir
        final_mask1 = cv2.bitwise_or(final_mask1, mask1)
        final_mask2 = cv2.bitwise_or(final_mask2, mask2)

    return final_mask1, final_mask2

def process_images(image_dir, label_dir, output_mask_dir1, output_mask_dir2):
    """Tüm görüntüleri ve etiketleri otomatik olarak işle ve maskeleri kaydet."""
    # Eğer maske dizini yoksa, oluştur
    if not os.path.exists(output_mask_dir1):
        os.makedirs(output_mask_dir1)
    if not os.path.exists(output_mask_dir2):
        os.makedirs(output_mask_dir2)

    # Görüntü dizinindeki tüm dosyaları işle
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + '.txt'  # Aynı isimdeki label dosyası
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                print(f"Processing: {image_filename} with {label_filename}")
                mask1, mask2 = apply_masks_to_image(image_path, label_path)

                # Maskeleri kaydet
                mask_output_path1 = os.path.join(output_mask_dir1, image_filename)
                cv2.imwrite(mask_output_path1, mask1)
                mask_output_path2 = os.path.join(output_mask_dir2, image_filename)
                cv2.imwrite(mask_output_path2, mask2)
            else:
                print(f"Label not found for {image_filename}, skipping.")

# Kullanım örneği:
image_dir = 'path_to_images'
label_dir = 'path_to_labels'
output_mask_dir1 = 'path_to_masks1'
output_mask_dir2 = 'path_to_masks2'

# Tüm görüntüleri ve etiketleri işleyip maskeleri kaydet
process_images(image_dir, label_dir, output_mask_dir1, output_mask_dir2)
