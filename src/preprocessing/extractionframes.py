import os
import cv2
import numpy as np
import pandas as pd

# -------- CONFIGURACIÓN --------
csv_path = r"G:\U. Nacional y opcion de grado\ADELANTO RIGUROSO\GANs_Echocardiogram\data\raw\FileList.csv"
video_dir = r"G:\U. Nacional y opcion de grado\BasedeDatos\EchoNet-Dynamic\Videos"
output_dir = r"G:\U. Nacional y opcion de grado\ADELANTO RIGUROSO\GANs_Echocardiogram\data\frames_extraidos"

# tamaño intermedio y final
IMG_SIZE = (112, 112)    # tamaño al que se leen los frames
FINAL_SIZE = (128, 128)  # tamaño final guardado 

os.makedirs(output_dir, exist_ok=True)


# --------------------------
# Creación de máscara (tu versión)
# --------------------------
def mask(shape, top, left, right, center, axes, angle, startAngle, endAngle):
    maskT = np.zeros(shape, dtype=np.uint8)
    points = np.array([top, left, right], dtype=np.int32)
    mask_triangular = cv2.fillConvexPoly(maskT.copy(), points, 255)

    maskE = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(maskE, center, axes, angle, startAngle, endAngle, 255, -1)

    return cv2.bitwise_or(mask_triangular, maskE)

def apply_mask(img, mask):
    return cv2.bitwise_and(img, mask)

# --------------------------
# Parámetros de tu máscara
# --------------------------
original_shape = (112, 112)
top = (60, 5)
left = (-15, 80)
right = (127, 80)
center = (55, 78)
axes = (70, 40)
angle = 0
startAngle = 0
endAngle = 180

combined_mask = mask(original_shape, top, left, right, center, axes, angle, startAngle, endAngle)


# --------------------------
# Función para detectar sístole/diástole
# --------------------------
def detectar_sistole_diastole(frames_gray, combined_mask, thresh=30):
    counts = []
    for frame in frames_gray:
        masked = apply_mask(frame, combined_mask)          # aplica tu máscara
        _, binary = cv2.threshold(masked, thresh, 255, cv2.THRESH_BINARY)
        counts.append(int(np.sum(binary > 0)))
    # sístole = más píxeles blancos, diástole = menos píxeles blancos
    sistole_idx = int(np.argmax(counts))
    diastole_idx = int(np.argmin(counts))
    return sistole_idx, diastole_idx, counts


# -------- PROCESAR TODOS LOS VIDEOS --------
df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    video_name = str(row[0])  # nombre sin extensión
    video_path = os.path.join(video_dir, f"{video_name}.avi")

    if not os.path.exists(video_path):
        print(f"⚠ No se encontró: {video_path}")
        continue

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

    if len(frames_gray) == 0:
        print(f"⚠ Video vacío o no legible: {video_name}")
        continue

    # Detectar sístole/diástole usando la máscara combinada
    try:
        sistole_idx, diastole_idx, counts = detectar_sistole_diastole(frames_gray, combined_mask)
    except Exception as e:
        print(f"Error al detectar sístole/diástole en {video_name}: {e}")
        continue

    # Indices de frames a guardar: primer, sistole, diastole, último
    indices = [0, sistole_idx, diastole_idx, len(frames_gray) - 1]

    # Guardar frames procesados
    for i, idx in enumerate(indices):
        masked = apply_mask(frames_gray[idx], combined_mask)                 # enmascarado (112x112)
        # Escalar a FINAL_SIZE con Lanczos4
        upscaled = cv2.resize(masked, FINAL_SIZE, interpolation=cv2.INTER_LANCZOS4)
        # Suavizado bilateral para mantener bordes y reducir artefactos
        smooth = cv2.bilateralFilter(upscaled, d=5, sigmaColor=15, sigmaSpace=15)
        filename = f"{video_name}_f{i}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, smooth)

    # Mostrar resumen por video (opcional)
    print(f"Procesado {video_name} → frames guardados: indices {indices} | counts (ejemplos): {counts[:5]}...")

print("Extracción completada.")
