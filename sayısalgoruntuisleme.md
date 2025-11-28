  # 1.1. Kütüphanelerin İçe Aktarılması [cite: 10]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # OpenCV
import os
from PIL import Image

# Görüntülerin notebook içinde görünmesi için
%matplotlib inline

# 1.2. Veri Setinin Yüklenmesi

# --- HOCAM İÇİN NOT ---
# Aşağıdaki dosya yolu (path) benim çalışma ortamıma göredir.
# Kodu kendi bilgisayarınızda çalıştırırken lütfen burayı kendi klasör yolunuzla değiştiriniz.
# ----------------------

dataset_path = '/content/skin_data' # (Veya senin kullandığın '/content/drive/...' yolu)

# 1.2. Veri Setinin Yüklenmesi [cite: 15]
# LÜTFEN DİKKAT: Veri setini Colab'e yükledikten sonra ana klasör yolunu buraya yazmalısın.
# Örneğin: '/content/drive/MyDrive/ISIC_Dataset/Train' veya sadece '/content/skin-cancer9-classesisic'
dataset_path = '/content/drive/MyDrive/Colab Notebooks/Skin cancer ISIC The International Skin Imaging Collaboration' # <-- BURAYI KENDİ YOLUNLA GÜNCELLE

data = []

# Klasördeki tüm görüntüleri tara
# (Dataset yapısına göre klasör isimleri etiket olabilir, şimdilik sadece dosya yollarını alıyoruz)
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            data.append(file_path)

# DataFrame oluşturma [cite: 16]
train_df = pd.DataFrame(data, columns=['filepath'])

# 1.3. Veri Özelliklerinin İncelenmesi [cite: 19]

# İlk 5 satırı görüntüle [cite: 17]
print("--- İlk 5 Veri ---")
print(train_df.head())

# Toplam görüntü sayısı [cite: 18]
print(f"\nToplam Görüntü Sayısı: {len(train_df)}")

# Örnek bir görüntünün özelliklerini inceleyelim (Çözünürlük ve Kanal Sayısı) [cite: 20, 21]
if len(train_df) > 0:
    sample_img_path = train_df['filepath'][0]
    img = cv2.imread(sample_img_path)

    # OpenCV BGR okur, RGB'ye çevirip bakmak mantıklı ama shape değişmez
    if img is not None:
        height, width, channels = img.shape
        print(f"\nÖrnek Görüntü Analizi:")
        print(f"Dosya Yolu: {sample_img_path}")
        print(f"Çözünürlük (Yükseklik x Genişlik): {height}x{width}")
        print(f"Kanal Sayısı: {channels} (3 ise RGB, 1 ise Grayscale)") #

        # Dosya boyutu [cite: 22]
        file_size = os.path.getsize(sample_img_path) / 1024 # KB cinsinden
        print(f"Dosya Boyutu: {file_size:.2f} KB")
    else:
        print("Görüntü okunamadı. Dosya yolu hatalı olabilir.")
else:
    print("Veri seti bulunamadı. Lütfen 'dataset_path' değişkenini kontrol edin.")

    # 2. Görüntü Yükleme ve Görselleştirme
# 2.1. Rastgele Görüntüler Seçme ve Gösterme

import random

# DataFrame'den rastgele 9 görüntü seçelim
random_images_df = train_df.sample(n=9, random_state=42)
img_paths = random_images_df['filepath'].values

plt.figure(figsize=(10, 20)) # Görsel boyutunu ayarlıyoruz

for i, img_path in enumerate(img_paths):
    # Görüntüyü oku (OpenCV BGR okur)
    img_bgr = cv2.imread(img_path)

    # RGB'ye çevir (Görselleştirme için)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Grayscale'e çevir
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Görselleştirme (Yan yana)
    # RGB
    plt.subplot(9, 2, 2*i + 1)
    plt.imshow(img_rgb)
    plt.title(f"Görüntü {i+1} - RGB")
    plt.axis('off')

    # Grayscale
    plt.subplot(9, 2, 2*i + 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title(f"Görüntü {i+1} - Grayscale")
    plt.axis('off')

plt.tight_layout()
plt.show()

# 2.2. ve 2.3. İstatistiksel Özellikler ve Histogram Çizimi

# Sonuçları tutmak için boş bir liste
stats_data = []

# Grafiklerin düzgün görünmesi için döngü (Burada örnek olarak ilk 3 görüntüye detaylı bakacağız,
# çünkü 9 görüntünün hepsinin histogramını rapora koymak çok yer kaplar.
# Ancak hoca "hepsi" dediyse sayı 9 kalabilir. Şimdilik 3 tanesini çizdiriyoruz.)
for i, img_path in enumerate(img_paths[:3]): # İstersen [:3] kısmını silip hepsini yapabilirsin.

    # Görüntü Okuma
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- 2.2 İstatistik Hesaplama ---
    # RGB İstatistikleri
    rgb_min = img_rgb.min()
    rgb_max = img_rgb.max()
    rgb_mean = img_rgb.mean()
    rgb_std = img_rgb.std()

    # Grayscale İstatistikleri
    gray_min = img_gray.min()
    gray_max = img_gray.max()
    gray_mean = img_gray.mean()
    gray_std = img_gray.std()

    # Tabloya ekle
    stats_data.append({
        "Görüntü": f"Img {i+1}",
        "RGB Mean": round(rgb_mean, 2), "RGB Std": round(rgb_std, 2),
        "Gray Mean": round(gray_mean, 2), "Gray Std": round(gray_std, 2)
    })

    # --- 2.3 Histogram Çizimi ---
    plt.figure(figsize=(12, 4))

    # a) RGB Histogramı
    plt.subplot(1, 2, 1)
    colors = ('r', 'g', 'b')
    for j, color in enumerate(colors):
        # calcHist(image, channel, mask, histSize, ranges)
        hist = cv2.calcHist([img_rgb], [j], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.title(f"Görüntü {i+1} - RGB Histogramı")
    plt.xlabel("Piksel Değeri")
    plt.ylabel("Frekans")

    # b) Grayscale Histogramı
    plt.subplot(1, 2, 2)
    plt.hist(img_gray.ravel(), 256, [0, 256], color='gray')
    plt.title(f"Görüntü {i+1} - Grayscale Histogramı")
    plt.xlabel("Piksel Değeri")

    plt.show()

# İstatistik Tablosunu Göster
stats_df = pd.DataFrame(stats_data)
print("\n--- 2.2. İstatistiksel Karşılaştırma Tablosu ---")
print(stats_df)

# 3. Görüntü İşleme ve İyileştirme
# Örnek olarak ilk görüntüyü seçiyoruz
sample_img_path = img_paths[0]
original_bgr = cv2.imread(sample_img_path)
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

# --- FONKSİYONLAR ---

# 3.1. Kontrast Germe Fonksiyonu (Min-Max Stretching)
def contrast_stretching(img):
    # Görüntüdeki min ve max piksel değerlerini bul
    min_val = np.min(img)
    max_val = np.max(img)
    # Formül: (pixel - min) * (255 / (max - min))
    stretched = (img - min_val) * (255.0 / (max_val - min_val))
    return np.uint8(stretched)

# 3.3. Gamma Düzeltme Fonksiyonu
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    # Hız için Lookup Table (LUT) kullanıyoruz
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- UYGULAMA ---

# A) KONTRAST GERME (Stretching)
# RGB için her kanala ayrı uygulanır
r, g, b = cv2.split(original_rgb)
r_str = contrast_stretching(r)
g_str = contrast_stretching(g)
b_str = contrast_stretching(b)
rgb_stretched = cv2.merge((r_str, g_str, b_str))

# Grayscale için
gray_stretched = contrast_stretching(original_gray)


# B) HISTOGRAM EŞİTLEME (Equalization)
# Grayscale için
gray_eq = cv2.equalizeHist(original_gray)

# RGB için (YCrCb yöntemi - İstenen Yöntem)
img_yuv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2YCrCb)
# Sadece Y (Luminance/Parlaklık) kanalını eşitle
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# Tekrar RGB'ye dön
rgb_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB)


# C) GAMMA DÜZELTME
gamma_05_rgb = adjust_gamma(original_rgb, gamma=0.5)
gamma_20_rgb = adjust_gamma(original_rgb, gamma=2.0)


# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(15, 12))

# 1. Satır: Orijinal vs Kontrast Germe
plt.subplot(3, 4, 1); plt.imshow(original_rgb); plt.title("Orijinal RGB"); plt.axis('off')
plt.subplot(3, 4, 2); plt.imshow(rgb_stretched); plt.title("3.1 RGB Stretching"); plt.axis('off')
plt.subplot(3, 4, 3); plt.imshow(original_gray, cmap='gray'); plt.title("Orijinal Gray"); plt.axis('off')
plt.subplot(3, 4, 4); plt.imshow(gray_stretched, cmap='gray'); plt.title("3.1 Gray Stretching"); plt.axis('off')

# 2. Satır: Histogram Eşitleme
plt.subplot(3, 4, 5); plt.imshow(original_rgb); plt.title("Orijinal RGB"); plt.axis('off')
plt.subplot(3, 4, 6); plt.imshow(rgb_eq); plt.title("3.2 RGB Hist. Eq (Y kanalı)"); plt.axis('off')
plt.subplot(3, 4, 7); plt.imshow(original_gray, cmap='gray'); plt.title("Orijinal Gray"); plt.axis('off')
plt.subplot(3, 4, 8); plt.imshow(gray_eq, cmap='gray'); plt.title("3.2 Gray Hist. Eq"); plt.axis('off')

# 3. Satır: Gamma Düzeltme (Sadece RGB Örneği Yeterli Olabilir, Yer Kazanmak İçin)
plt.subplot(3, 4, 9); plt.imshow(original_rgb); plt.title("Gamma 1.0 (Orijinal)"); plt.axis('off')
plt.subplot(3, 4, 10); plt.imshow(gamma_05_rgb); plt.title("3.3 Gamma 0.5 (Karanlık)"); plt.axis('off')
plt.subplot(3, 4, 11); plt.imshow(gamma_20_rgb); plt.title("3.3 Gamma 2.0 (Parlak)"); plt.axis('off')

plt.tight_layout()
plt.show()

# 4. Gürültü Azaltma ve 5. Döndürme İşlemleri
import random

# Örnek görüntü (Daha önce seçtiğimiz 'sample_img_path' kullanılıyor)
img_bgr = cv2.imread(sample_img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# --- 4. GÜRÜLTÜ AZALTMA (BLUR) ---

# 4.1. Median Blur (Tuz-Biber gürültüsüne karşı etkilidir)
# Kernel boyutu tek sayı olmalı (örn: 5)
median_rgb = cv2.medianBlur(img_rgb, 5)
median_gray = cv2.medianBlur(img_gray, 5)

# 4.2. Gaussian Blur (Genel yumuşatma)
# (5, 5) kernel boyutu
gaussian_rgb = cv2.GaussianBlur(img_rgb, (5, 5), 0)
gaussian_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Görselleştirme - Gürültü Azaltma
plt.figure(figsize=(12, 8))
plt.suptitle("4. Gürültü Azaltma Karşılaştırması", fontsize=16)

# Satır 1: RGB Karşılaştırma
plt.subplot(2, 3, 1); plt.imshow(img_rgb); plt.title("Orijinal RGB"); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(median_rgb); plt.title("Median Blur (RGB)"); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(gaussian_rgb); plt.title("Gaussian Blur (RGB)"); plt.axis('off')

# Satır 2: Grayscale Karşılaştırma
plt.subplot(2, 3, 4); plt.imshow(img_gray, cmap='gray'); plt.title("Orijinal Gray"); plt.axis('off')
plt.subplot(2, 3, 5); plt.imshow(median_gray, cmap='gray'); plt.title("Median Blur (Gray)"); plt.axis('off')
plt.subplot(2, 3, 6); plt.imshow(gaussian_gray, cmap='gray'); plt.title("Gaussian Blur (Gray)"); plt.axis('off')

plt.tight_layout()
plt.show()


# --- 5. DÖNDÜRME VE AYNA ÇEVİRME ---

# 5.1. Rastgele Döndürme (0-10 derece arası)
angle = random.uniform(0, 10)
rows, cols = img_rgb.shape[:2]
# Döndürme matrisi oluşturma (Merkez noktası etrafında)
M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

rotated_rgb = cv2.warpAffine(img_rgb, M, (cols, rows))
rotated_gray = cv2.warpAffine(img_gray, M, (cols, rows))

# 5.2. Yatay Ayna Çevirme (Horizontal Flip)
# 1 kodu yatay çevirme anlamına gelir
flipped_rgb = cv2.flip(img_rgb, 1)
flipped_gray = cv2.flip(img_gray, 1)

# Görselleştirme - Geometrik İşlemler
plt.figure(figsize=(12, 8))
plt.suptitle(f"5. Geometrik İşlemler (Açı: {angle:.2f}°)", fontsize=16)

# RGB
plt.subplot(2, 3, 1); plt.imshow(img_rgb); plt.title("Orijinal RGB"); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(rotated_rgb); plt.title(f"Döndürülmüş ({angle:.1f}°)"); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(flipped_rgb); plt.title("Yatay Flip (Ayna)"); plt.axis('off')

# Gray
plt.subplot(2, 3, 4); plt.imshow(img_gray, cmap='gray'); plt.title("Orijinal Gray"); plt.axis('off')
plt.subplot(2, 3, 5); plt.imshow(rotated_gray, cmap='gray'); plt.title("Döndürülmüş (Gray)"); plt.axis('off')
plt.subplot(2, 3, 6); plt.imshow(flipped_gray, cmap='gray'); plt.title("Yatay Flip (Gray)"); plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Frekans Alanında Filtreleme (FFT)
import numpy as np

# Grayscale görüntü üzerinde çalışılır
# (Daha önce seçtiğimiz 'sample_img_path' kullanılıyor)
img = cv2.imread(sample_img_path, 0) # 0 parametresi direkt grayscale okur

# 6.1. Fourier Dönüşümü (FFT)
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft) # Düşük frekansları merkeze taşı

# Görselleştirme için spektrum (Magnitude Spectrum)
# Logaritmik ölçekte bakılır çünkü değerler çok büyüktür
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# 6.2. Alçak Geçiren Filtre (Low Pass Filter - LPF) Maskesi Oluşturma
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2 # Merkez noktası

# Maske: Merkezde 1 (beyaz), kenarlarda 0 (siyah)
# 60 piksellik bir yarıçap belirleyelim (Değeri değiştirirsen bulanıklık değişir)
mask = np.zeros((rows, cols), np.uint8)
r = 60
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

# Maskeyi frekans alanına uygula
fshift = dft_shift * mask

# Filtrelenmiş spektrumu görselleştirme (Opsiyonel ama raporda şık durur)
fshift_spectrum = 20 * np.log(np.abs(fshift) + 1) # +1 log(0) hatasını önlemek için

# 6.3. Ters Fourier Dönüşümü (Inverse FFT)
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back) # Karmaşık sayıdan gerçel sayıya dön

# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(12, 10))

# Orijinal
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Orijinal Grayscale')
plt.axis('off')

# Frekans Spektrumu
plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Frekans Spektrumu (FFT)')
plt.axis('off')

# Uygulanan Maske (Frekans alanında)
plt.subplot(2, 2, 3)
plt.imshow(fshift_spectrum, cmap='gray')
plt.title('Filtrelenmiş Spektrum (Maske Uygulandı)')
plt.axis('off')

# Sonuç (Ters FFT)
plt.subplot(2, 2, 4)
plt.imshow(img_back, cmap='gray')
plt.title('6.3 Ters FFT Sonucu (Low Pass Filter)')
plt.axis('off')

plt.tight_layout()
plt.show()

# 7. Keskinleştirme ve Enterpolasyon

# Örnek görüntü (Daha önceki sample_img_path)
img_bgr = cv2.imread(sample_img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# --- 7.1. Unsharp Masking ile Keskinleştirme ---
# OpenCV'de hazır unsharp mask fonksiyonu yoktur, manuel hesaplanır:
# Formül: Sharp = Original + (Original - Blurred) * Amount

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

# RGB Keskinleştirme
sharp_rgb = unsharp_mask(img_rgb)

# Grayscale Keskinleştirme
sharp_gray = unsharp_mask(img_gray)


# --- 7.2. Bicubic Enterpolasyon (2 Kat Büyütme) ---
# fx=2, fy=2 demek genişlik ve yüksekliği 2 ile çarp demektir.
# INTER_CUBIC parametresi Bicubic algoritmasını seçer.

# RGB Büyütme
resized_rgb = cv2.resize(sharp_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Grayscale Büyütme
resized_gray = cv2.resize(sharp_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(12, 12))

# 1. Satır: Orijinal vs Keskinleştirilmiş (RGB)
plt.subplot(3, 2, 1); plt.imshow(img_rgb); plt.title("Orijinal RGB")
plt.axis('off')
plt.subplot(3, 2, 2); plt.imshow(sharp_rgb); plt.title("7.1 Unsharp Mask (Keskin RGB)")
plt.axis('off')

# 2. Satır: Orijinal vs Keskinleştirilmiş (Gray)
plt.subplot(3, 2, 3); plt.imshow(img_gray, cmap='gray'); plt.title("Orijinal Grayscale")
plt.axis('off')
plt.subplot(3, 2, 4); plt.imshow(sharp_gray, cmap='gray'); plt.title("7.1 Unsharp Mask (Keskin Gray)")
plt.axis('off')

# 3. Satır: Büyütülmüş Görüntüler (Crop alıp gösteriyoruz ki detay belli olsun)
# Görüntü büyüdüğü için sadece bir kısmını (crop) gösterelim
h, w, _ = resized_rgb.shape
crop_rgb = resized_rgb[h//4:h//2, w//4:w//2] # Merkezden bir parça
crop_gray = resized_gray[h//4:h//2, w//4:w//2]

plt.subplot(3, 2, 5); plt.imshow(crop_rgb); plt.title("7.2 Bicubic Zoom (RGB Crop)")
plt.axis('off')
plt.subplot(3, 2, 6); plt.imshow(crop_gray, cmap='gray'); plt.title("7.2 Bicubic Zoom (Gray Crop)")
plt.axis('off')

plt.tight_layout()
plt.show()

# Çözünürlük Kontrolü
print(f"Orijinal Boyut: {img_rgb.shape}")
print(f"Büyütülmüş Boyut: {resized_rgb.shape}")
