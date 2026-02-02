import os
import zipfile

# Ruta de la carpeta a comprimir
folder_path = r"G:\U. Nacional y opcion de grado\ADELANTO RIGUROSO\GANs_Echocardiogram\data\frames_extraidos"

# Ruta del archivo ZIP resultante
zip_path = folder_path + ".zip"

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Guardar ruta relativa para no incluir la carpeta completa en el zip
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)
    print(f"✅ Carpeta comprimida en: {zip_path}")

if __name__ == "__main__":
    zip_folder(folder_path, zip_path)