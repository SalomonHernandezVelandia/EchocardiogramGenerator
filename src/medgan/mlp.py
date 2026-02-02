# MODELOS GANs
import torch
import torch.nn as nn


###############################################################################
# Generador MLP
###############################################################################
class MLP_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        """
        isize: tamaño de la imagen (ej. 128)
        nz: tamaño del vector latente
        nc: número de canales (1 para escala de grises)
        ngf: tamaño de capa oculta del generador
        ngpu: número de GPUs
        """
        super(MLP_G, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.isize = isize
        self.nz = nz

        main = nn.Sequential(
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, nc * isize * isize),
            nn.Tanh()  # salida normalizada [-1,1]
        )
        self.main = main

    def forward(self, input):
        # input: (batch, nz, 1, 1) o (batch, nz)
        input = input.view(input.size(0), -1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)


###############################################################################
# Discriminador MLP
###############################################################################
class MLP_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        """
        isize: tamaño de la imagen (ej. 128)
        nz: tamaño del vector latente (no se usa aquí, pero se mantiene por compatibilidad)
        nc: número de canales (1 para escala de grises)
        ndf: tamaño de capa oculta del discriminador
        ngpu: número de GPUs
        """
        super(MLP_D, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.isize = isize
        self.nz = nz

        main = nn.Sequential(
            nn.Linear(nc * isize * isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1)  # salida: score real/fake
        )
        self.main = main

    def forward(self, input):
        # input: (batch, nc, isize, isize)
        input = input.view(input.size(0), -1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)  # un score por imagen
