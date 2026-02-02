import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- CONFIGURACIÓN --------
csv_path = r"G:\U. Nacional y opcion de grado\ADELANTO RIGUROSO\GANs_Echocardiogram\data\raw\FileList.csv"
video_dir = r"G:\U. Nacional y opcion de grado\BasedeDatos\EchoNet-Dynamic\Videos"

IMG_SIZE = (112, 112)

# --------------------------
# Creación de máscara
# --------------------------
def mask(shape, top, left, right, h, w, center, radius):
    maskT = np.zeros(shape, dtype=np.uint8)
    points = np.array([top, left, right], dtype=np.int32)
    mask_triangular = cv2.fillConvexPoly(maskT, points, 255)

    maskC = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(maskC, center, radius, 255, -1)
    maskC[:center[1], :] = 0

    return cv2.bitwise_or(mask_triangular, maskC)

def apply_mask(img, mask):
    return cv2.bitwise_and(img, mask)

# Parámetros de la máscara
original_shape = (112, 112)
top = (60, 5)
left = (-15, 80)
right = (127, 80)
center = (60, 80)
radius = 148
combined_mask = mask(original_shape, top, left, right, 112, 112, center, radius)

# --------------------------
# Función para detectar sístole y diástole
# --------------------------
def detectar_sistole_diastole(frames_gray, combined_mask, thresh=30):
    counts = []
    binarized_frames = []
    for frame in frames_gray:
        masked = apply_mask(frame, combined_mask)
        _, binary = cv2.threshold(masked, thresh, 255, cv2.THRESH_BINARY)
        counts.append(int(np.sum(binary > 0)))
        binarized_frames.append(binary)
    sistole_idx = int(np.argmax(counts))
    diastole_idx = int(np.argmin(counts))
    return sistole_idx, diastole_idx, binarized_frames

# -------- PROCESAR SOLO EL PRIMER VIDEO --------
df = pd.read_csv(csv_path)
video_name = "0X1A2A76BDB5B98BED" 
video_path = os.path.join(video_dir, f"{video_name}.avi")

cap = cv2.VideoCapture(video_path)
frames_gray = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    frames_gray.append(resized)
cap.release()

# Detectar sístole y diástole
sistole_idx, diastole_idx, binarized_frames = detectar_sistole_diastole(frames_gray, combined_mask)

# Obtener imágenes
img_sistole = binarized_frames[sistole_idx]
img_diastole = binarized_frames[diastole_idx]

# -------- VISUALIZACIÓN --------
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img_diastole, cmap="gray")
plt.title("Diástole")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_sistole, cmap="gray")
plt.title("Sístole")
plt.axis("off")

plt.suptitle(f"Video: {video_name}")
plt.show()
