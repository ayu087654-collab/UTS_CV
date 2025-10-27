import cv2
import numpy as np
import os

# === 1. Buat folder output ===
os.makedirs("output", exist_ok=True)

# === 2. Baca gambar karakter dan background ===
img = cv2.imread("sapi.png")           # karakter sapi lucu
bg = cv2.imread("background.jpg")      # background padang rumput

# === 3. Cek apakah gambar berhasil dibuka ===
if img is None or bg is None:
    print("❌ Gagal membuka gambar! Pastikan 'sapi.png' dan 'background.jpg' ada di folder yang sama.")
    exit()

# Samakan ukuran background dengan gambar karakter
bg = cv2.resize(bg, (img.shape[1], img.shape[0]))

# === 4. TRANSFORMASI ===

# a. Translasi (geser)
rows, cols = img.shape[:2]
M_trans = np.float32([[1, 0, 80], [0, 1, 50]])  # geser kanan 80px, bawah 50px
img_trans = cv2.warpAffine(img, M_trans, (cols, rows))
cv2.imwrite("output/1_translasi.jpg", img_trans)

# b. Rotasi (putar 45 derajat)
M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
img_rot = cv2.warpAffine(img, M_rot, (cols, rows))
cv2.imwrite("output/2_rotasi.jpg", img_rot)

# c. Resize (ubah ukuran jadi 0.5x)
img_resize = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imwrite("output/3_resize.jpg", img_resize)

# d. Crop (potong sebagian tengah)
h, w = img.shape[:2]
img_crop = img[h//4:h//4*3, w//4:w//4*3]
cv2.imwrite("output/4_crop.jpg", img_crop)

# === 5. OPERASI ARITMATIKA / BITWISE ===

# a. cv2.add() → menambahkan piksel antara dua gambar
add_result = cv2.add(img, bg)
cv2.imwrite("output/5_add.jpg", add_result)

# b. cv2.bitwise_and() → masking antara gambar sapi dan background
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

bg_masked = cv2.bitwise_and(bg, bg, mask=mask_inv)
char_masked = cv2.bitwise_and(img, img, mask=mask)
bitwise_result = cv2.add(bg_masked, char_masked)
cv2.imwrite("output/6_bitwise_and.jpg", bitwise_result)

# === 6. HASIL FINAL ===
# Gabungkan karakter hasil rotasi & resize ke background menggunakan operasi addWeighted
# biar efeknya lembut dan realistis
final = cv2.addWeighted(bg, 0.6, img_rot, 0.4, 0)
cv2.imwrite("output/7_final.jpg", final)

# === 7. Tampilkan hasil (opsional) ===
cv2.imshow("Translasi", img_trans)
cv2.imshow("Rotasi", img_rot)
cv2.imshow("Resize", img_resize)
cv2.imshow("Crop", img_crop)
cv2.imshow("Add", add_result)
cv2.imshow("Bitwise AND", bitwise_result)
cv2.imshow("Final Result", final)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Semua hasil disimpan di folder 'output/'")
