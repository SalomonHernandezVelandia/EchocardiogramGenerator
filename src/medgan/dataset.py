import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch
import os

def dstget(opt):
    """
    Devuelve un DataLoader para el dataset especificado en opt.dataroot.
    Adaptado para ecocardiogramas en escala de grises (1 canal, 128x128).
    """

    # Transformaciones para imágenes médicas
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # Asegura 1 canal
        transforms.Resize((opt.img_size, opt.img_size)),  # Redimensiona a 128x128 (o lo que indiques en --img_size)
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normaliza a rango [-1,1]
    ])

    # Dataset desde carpeta
    if not os.path.exists(opt.dataroot):
        raise ValueError(f"La ruta {opt.dataroot} no existe. Verifica tu dataset.")

    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        pin_memory=True
    )

    return dataloader