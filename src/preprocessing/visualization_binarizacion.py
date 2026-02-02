import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Funciones de máscara (las mismas que usabas antes)
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

# --------------------------
# Detección de sístole y diástole con máscara y binarización
# --------------------------
def detectar_sistole_diastole(frames_gray, combined_mask):
    counts = []
    binarizadas = []
    for frame in frames_gray:
        frame_masked = apply_mask(frame, combined_mask)
        _, binary = cv2.threshold(frame_masked, 30, 255, cv2.THRESH_BINARY)
        binarizadas.append(binary)
        counts.append(np.sum(binary > 0))  # píxeles blancos

    sistole_idx = np.argmax(counts)   # mayor número de píxeles
    diastole_idx = np.argmin(counts)  # menor número de píxeles
    return sistole_idx, diastole_idx, binarizadas, counts

# --------------------------
# Parámetros de máscara
# --------------------------
original_shape = (112, 112)  # Para la máscara
top = (60, 5)
left = (-15, 80)
right = (127, 80)
center = (60, 80)
radius = 148
combined_mask = mask(original_shape, top, left, right, 112, 112, center, radius)

# --------------------------
# Paths
# --------------------------
csv_path = r"G:\U. Nacional y opcion de grado\ADELANTO RIGUROSO\GANs_Echocardiogram\data\raw\FileList.csv"
video_dir = r"G:\U. Nacional y opcion de grado\BasedeDatos\EchoNet-Dynamic\Videos"

df = pd.read_csv(csv_path)
video_name = df.iloc[18, 0]  # Cambia el índice para otro video
video_file = f"{video_name}.avi"
video_path = os.path.join(video_dir, video_file)

# --------------------------
# Cargar frames en gris (112x112)
# --------------------------
cap = cv2.VideoCapture(video_path)
frames_gray = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
    frames_gray.append(frame_resized)
cap.release()

# --------------------------
# Detectar sístole y diástole
# --------------------------
sistole_idx, diastole_idx, binarizadas, counts = detectar_sistole_diastole(frames_gray, combined_mask)

# Frames de interés
indices = [0, sistole_idx, diastole_idx, len(frames_gray) - 1]
titulos = ["Primer frame", "Sístole", "Diástole", "Último frame"]

# Mostrar conteo de píxeles blancos
for i, idx in enumerate(indices):
    print(f"{titulos[i]} → {counts[idx]} píxeles blancos")

# --------------------------
# Mostrar en Matplotlib
# --------------------------
fig, axes = plt.subplots(len(indices), 2, figsize=(6, 8))
for i, idx in enumerate(indices):
    # Columna izquierda: original
    axes[i, 0].imshow(frames_gray[idx], cmap='gray')
    axes[i, 0].set_title(titulos[i])
    axes[i, 0].axis('off')

    # Columna derecha: enmascarada y binarizada
    axes[i, 1].imshow(binarizadas[idx], cmap='gray')
    axes[i, 1].set_title(f"{titulos[i]} (mask+bin)")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
