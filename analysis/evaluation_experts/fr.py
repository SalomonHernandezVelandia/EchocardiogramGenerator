import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv(
    r"G:\PYTHON\Proyectos Personales\EchocardiogramGenerator\analysis\evaluation_experts\Ecocardiogramas_Evaluation_Limpio.csv"
)

df = df.dropna(how="all").reset_index(drop=True)

ground_truth = df.iloc[0]
experts = df.iloc[1:]

QUESTION_START = 0
QUESTION_END = 17

fooling_per_image = []

for col_idx in range(QUESTION_START, QUESTION_END + 1):

    correct_answer = ground_truth.iloc[col_idx]

    total_selections = 0
    synthetic_selected = 0

    for exp_idx in range(len(experts)):

        expert_answer = experts.iloc[exp_idx, col_idx]

        # Caso columnas 10–14 (dos selecciones)
        if 10 <= col_idx <= 14:

            correct_set = set(str(correct_answer).replace(" ", "").split("_"))
            expert_set = set(str(expert_answer).replace(" ", "").split("_"))

            for selection in expert_set:
                total_selections += 1
                if selection not in correct_set:
                    synthetic_selected += 1

        else:
            # Una sola selección
            total_selections += 1
            if expert_answer != correct_answer:
                synthetic_selected += 1

    fooling_rate = synthetic_selected / total_selections
    fooling_per_image.append(fooling_rate)

    print(f"Columna {col_idx} → Fooling Rate: {fooling_rate:.3f}")

# 📊 Gráfica estilo tesis
plt.figure(figsize=(11,5))

x_positions = np.arange(len(fooling_per_image))

bars = plt.bar(
    x_positions,
    fooling_per_image,
    edgecolor='black',
    linewidth=1.1
)

plt.ylabel("Fooling Rate", fontsize=12)
plt.title("Fooling Rate por Imagen", fontsize=14, pad=15)

# ---- Etiquetas personalizadas ----
xtick_labels = []
for col_idx in range(QUESTION_START, QUESTION_END + 1):

    if 10 <= col_idx <= 14:
        xtick_labels.append(f"Img {col_idx}\n(2 imgs)")
    else:
        xtick_labels.append(f"Img {col_idx}")

plt.xticks(
    x_positions,
    xtick_labels,
    rotation=0,
    fontsize=9
)

plt.ylim(0, 1)

# Grid sutil en eje Y
plt.grid(axis='y', linestyle='--', alpha=0.35)

# ---- Porcentaje encima de cada barra ----
for i, value in enumerate(fooling_per_image):
    plt.text(
        i,
        value + 0.02,
        f"{value*100:.1f}%",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()

# ---- Guardar imagen en alta resolución ----
save_path = r"G:\PYTHON\Proyectos Personales\EchocardiogramGenerator\analysis\evaluation_experts\graficos"
os.makedirs(save_path, exist_ok=True)

file_name = "Fooling_Rate.png"
full_path = os.path.join(save_path, file_name)

plt.savefig(full_path, dpi=300, bbox_inches='tight')

plt.show()