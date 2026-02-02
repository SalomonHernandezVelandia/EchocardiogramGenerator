import cv2
import numpy as np
import matplotlib.pyplot as plt

# Tamaño base
shape = (112, 112)
top = (60, 5)
left = (-15, 80)
right = (127, 80)
center = (55, 78)
axes = (70, 40)  # (ancho, alto) del óvalo
angle = 0
startAngle = 0
endAngle = 180  # semicircular inferior

# Máscara triangular
maskT = np.zeros(shape, dtype=np.uint8)
points = np.array([top, left, right], dtype=np.int32)
mask_triangular = cv2.fillConvexPoly(maskT.copy(), points, 255)

# Máscara elíptica (semióvalo inferior)
maskC = np.zeros(shape, dtype=np.uint8)
cv2.ellipse(maskC, center, axes, angle, startAngle, endAngle, 255, -1)

# Máscara combinada
mask_combined = cv2.bitwise_or(mask_triangular, maskC)

# Mostrar las tres máscaras
fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(mask_triangular, cmap='gray')
axs[0].set_title('Máscara Triangular')
axs[1].imshow(maskC, cmap='gray')
axs[1].set_title('Máscara Elíptica')
axs[2].imshow(mask_combined, cmap='gray')
axs[2].set_title('Máscara Combinada')

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
