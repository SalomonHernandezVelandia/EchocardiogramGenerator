import os
import random
from pathlib import Path
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[2]

REAL_DIR = BASE_DIR / "data" / "frames_extraidos"
FAKE_DIR = BASE_DIR / "experiments" / "stylegan2_ada" / "E4" / "best_generated"
OUTPUT_DIR = BASE_DIR / "assets" / "evaluations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# FUNCIONES
# =========================

def get_random_image(folder):
    images = [f for f in folder.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    if not images:
        raise ValueError(f"No se encontraron imágenes en {folder}")
    return random.choice(images)

def concat_images(img1, img2):
    # Asegurar mismo alto
    max_height = max(img1.height, img2.height)

    def resize_to_height(img, height):
        ratio = height / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, height))

    img1 = resize_to_height(img1, max_height)
    img2 = resize_to_height(img2, max_height)

    total_width = img1.width + img2.width
    new_img = Image.new("RGB", (total_width, max_height))

    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    return new_img

# Para 4 imagens, 2 reales y 2 falsas
def concat_multiple_images(images):
    # Ajustar todas al mismo alto
    max_height = max(img.height for img in images)

    resized = []
    for img in images:
        ratio = max_height / img.height
        new_width = int(img.width * ratio)
        resized.append(img.resize((new_width, max_height)))

    total_width = sum(img.width for img in resized)
    new_img = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in resized:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img

# =========================
# MAIN
# =========================

def main():
    # real_paths = [get_random_image(REAL_DIR) for _ in range(2)]
    # fake_paths = [get_random_image(FAKE_DIR) for _ in range(2)]
    real_paths = [get_random_image(REAL_DIR) for _ in range(1)]
    fake_paths = [get_random_image(FAKE_DIR) for _ in range(3)]

    image_items = []
    for p in real_paths:
        image_items.append(("REAL", Image.open(p).convert("RGB")))

    for p in fake_paths:
        image_items.append(("FAKE", Image.open(p).convert("RGB")))

    random.shuffle(image_items)
    labels = [item[0] for item in image_items]
    images = [item[1] for item in image_items]
    combined = concat_multiple_images(images)

    # Guardar
    output_file = OUTPUT_DIR / f"evaluation_{random.randint(10000,99999)}.png"
    combined.save(output_file)

    # Consola
    print("===== RESULTADO =====")
    print("Orden de izquierda a derecha:")

    for i, label in enumerate(labels, start=1):
        print(f"Posición {i}: {label}")

    print(f"\nGuardado en: {output_file}")

# Para solo 1 imagen real y una falsa
# def main():
#     real_path = get_random_image(REAL_DIR)
#     fake_path = get_random_image(FAKE_DIR)
#     real_img = Image.open(real_path).convert("RGB")
#     fake_img = Image.open(fake_path).convert("RGB")
#     if random.random() > 0.5:
#         left_img, right_img = real_img, fake_img
#         left_label, right_label = "REAL", "FAKE"
#     else:
#         left_img, right_img = fake_img, real_img
#         left_label, right_label = "FAKE", "REAL"

#     combined = concat_images(left_img, right_img)

#     output_file = OUTPUT_DIR / f"evaluation_{random.randint(10000,99999)}.png"
#     combined.save(output_file)
#     print("===== RESULTADO =====")
#     print(f"Izquierda: {left_label}")
#     print(f"Derecha: {right_label}")
#     print(f"Guardado en: {output_file}")

if __name__ == "__main__":
    main()