# Echocardiogram Image Generation using Generative Learning Strategies

<p align="center">
  <img src="assets/generated_stylegan.gif" width="100%"/>
</p>

[![Python](https://img.shields.io/badge/Python-3.10+-2E7D32?logo=python&logoColor=white)]
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F57C00?logo=jupyter&logoColor=white)]
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Neural%20Networks-6A1B9A)]
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch&logoColor=white)]


## Related Publications

### Conference Paper (IEEE SIPAIM)

**Comparative Study of Methods for Generating Echocardiographic Images**  
S. Hernández, et al.  
SIPAIM, IEEE, 2025  

<p align="center">  

  <a href="https://ieeexplore.ieee.org/document/11283212">
    <img src="https://img.shields.io/badge/IEEE-Xplore-00629B" alt="IEEE Xplore">
  </a>

  <a href="https://doi.org/10.1109/SIPAIM67325.2025.11283212">
    <img src="https://img.shields.io/badge/DOI-10.1109%2FSIPAIM67325.2025.11283212-555555" alt="DOI">
  </a>

  <a href="https://www.researchgate.net/publication/398847593">
    <img src="https://img.shields.io/badge/Preprint-ResearchGate-00CCBB?logo=researchgate" alt="ResearchGate Preprint">
  </a>
</p>

> This article represents the first and most concise version of the research, presenting the results of StyleGAN and MedGAN as generation architectures alongside VQGAN and Pix2Pix as reconstruction architectures. 

### Thesis (Extended and Robust Study)

**Generación de imágenes de ecocardiogramas mediante estrategias de aprendizaje generativo**  
Salomón Hernández Velandia  
(Thesis manuscript – not yet formally published)

> The thesis significantly extends the SIPAIM publication by incorporating additional architectures, deeper experimental analysis, and a more comprehensive evaluation framework.

A preprint version of the thesis is available in the `publications/thesis/` directory.

---

## Table of Contents

- [Echocardiogram Image Generation using Generative Learning Strategies](#echocardiogram-image-generation-using-generative-learning-strategies)
  - [Related Publications](#related-publications)
  - [Table of Contents](#table-of-contents)
  - [Tech Stack](#tech-stack)
  - [Research Context](#research-context)
  - [Thesis (Extended and Robust Study)](#thesis-extended-and-robust-study)


## Tech Stack

This repository contains the code, experimental setup, and results associated with the research project:

**"Generación de imágenes de ecocardiogramas mediante estrategias de aprendizaje generativo"**

This work was developed as a thesis project and extends previous research published at SIPAIM (IEEE), exploring multiple generative architectures for synthetic echocardiographic image generation.

---


## Research Context

The generation of synthetic echocardiographic images is an effective alternative for data augmentation, avoiding the limitations of traditional data augmentation techniques such as affine transformations, which can alter, distort, or falsify medical images and cause critical spatial errors. By using generative models such as GANs, it is possible to create entirely new and diverse images that preserve anatomical and morphological properties, facilitating the comparative evaluation of algorithms and the study of reproducibility in this field of medical imaging.

This project presents a comprehensive comparative study of different generative architectures applied to apical four-chamber echocardiography, the objective is to expand and diversify the available datasets, identify the most efficient model and configuration for generating new images, and thus enrich the resources dedicated to the analysis of cardiac function.

### The architectures explored include:
- StyleGAN2-ADA
- MedGAN
- WGAN
- VQGAN

Each architecture was evaluated under eight different hyperparameter configurations and training strategies, the process included preprocessing the Echonet-Dynamic dataset, which consisted of converting it to grayscale, resizing it to 128×128 pixels, leveraging a power of 2 to optimize calculations in the GANs, binarization to extract the frames corresponding to diastole and systole for each patient, and selecting the first and last frames of each sequence to diversify the training data. All of this processing can be found in `src/preprocessing/extractionframes.py`

---

## Implemented Configurations

| Architecture | FID Curve |
|-------------|----------|
| **StyleGAN2-ADA** | [View Image](results/stylegan2-ada/line_graph/FID_stylegan2ada.png) |
| **MedGAN** | [View Image](results/medgan/line_graph/FID_medgan.png) |
| **WGAN** | [View Image](results/wgan/line_graph/FID_wgan.png) |
| **VQGAN** | [View Image](results/vqgan/line_graph/FID_vqgan.png) |



<h3 align="center">FID Comparison</h3>

<p align="center">
  <img src="results/stylegan2_ada/line_graph/FID_style.png" width="350">
  <img src="results/medgan/line_graph/FID_medgan.png" width="350">
  <img src="results/wgan/line_graph/FID_wgan.png" width="350">
  <img src="results/vqgan/line_graph/FID_vqgan.png" width="350">
</p>

---


<p align="center">  <!-- GitHub Repo -->
  <!-- <a href="https://github.com/SalomonHernandezVelandia/EchocardiogramGenerator">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub Repository">
  </a> -->

  <!-- LinkedIn -->
  <a href="https://www.linkedin.com/in/salomon-hernandez-velandia-827417196/">
    <img src="https://img.shields.io/badge/LinkedIn-Salomon_Hernandez-0A66C2?logo=linkedin" alt="LinkedIn Profile">
  </a>
</p>

## 🗂️ Repository Structure

```text
ecocardiogram-gan-thesis/
│
├── README.md
├── LICENSE
├── LICENSE_DATA
├── CITATION.cff
├── requirements.txt
|
├── experiments/
│   ├── stylegan2_ada/
│   │   ├── E1/
│   │   │   ├── checkpoints/
│   │   │   │   └── .....pth
│   │   │   ├── generated_samples/
│   │   │   │   ├── generated_0.png
│   │   │   │   └── generated_25.png
│   │   │   ├── metrics_csv/
│   │   │   │   ├── metrics.csv
│   │   │   │   └── losses.csv
│   │   │   └── samples/
│   │   │
│   │   ├── E2/
│   │   ├── E3/
│   │   └── E4/
│   │
│   ├── medgan/
│   │   ├── M1/
│   │   └── ...
│   │
│   ├── wgan/
│   │   └── ...
│   │
│   └── vqgan/
│       └── ...
│
├── external/
│   ├── README.md
│   └── stylegan2-ada/   # submodule o instrucción de clonación
│
├── publications/
│   ├── sipaim/
│   └── thesis/
│
├── training/
│   ├── StyleGan2_Ada.ipynb
│   ├── MedGAN.ipynb
│   └── WGAN.ipynb
│
├── src/
│   ├── medgan/
│   │   ├── dcgan.py
│   │   ├── mlp.py
│   │   └── dztaset.py
│   |
│   ├── preprocessing/
│   │   ├── comprobacion_sistole.py
│   │   ├── convertirZIP.py
│   │   ├── extractionframes.py
│   │   ├── visualizacion_mask.py
│   │   └── visualizacion_binarizacion.py
│   │
├── results/
│   ├── stylegan2_ada/
│   |   ├── line_graph/
│   |   ├── losses/
│   |   ├── radar_graph/
│   |   └── violin_boxplot_graph/
│   |
│   ├── medgan/
│   |   ├── line_graph/
│   |   ├── losses/
│   |   ├── radar_graph/
│   |   └── violin_boxplot_graph/
│   |
│   ├── wgan/
│   |   ├── line_graph/
│   |   ├── losses/
│   |   ├── radar_graph/
│   |   └── violin_boxplot_graph/
│   |
│   └── vqgan/
│       ├── line_graph/
│       ├── losses/
│       ├── radar_graph/
│       └── violin_boxplot_graph/
│   


├── configs/                # Configuration files for different architectures and experiments
├── generated_samples/      # Synthetic echocardiographic images
├── checkpoints/            # Trained model checkpoints (if applicable)
├── publications/           # Thesis and paper preprints
├── external/               # External repositories (e.g., StyleGAN2-ADA, VQGAN)
├── scripts/                # Utility scripts for setup, training, and evaluation
