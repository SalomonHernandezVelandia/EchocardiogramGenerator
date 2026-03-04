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

PAIR_START = 10
PAIR_END = 14

valid_columns = [
    i for i in range(QUESTION_START, QUESTION_END + 1)
    if not (PAIR_START <= i <= PAIR_END)
]

for exp_idx in range(len(experts)):

    row = experts.iloc[exp_idx]

    y_true = []
    y_pred = []

    for col_idx in valid_columns:
        y_true.append(ground_truth.iloc[col_idx])
        y_pred.append(row.iloc[col_idx])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = np.unique(y_true)

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for i, real_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum((y_true == real_label) & (y_pred == pred_label))

    # ---- Métricas en consola ----
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    accuracy = (TP + TN) / np.sum(cm)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    fooling_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

    print("\n" + "="*60)
    print(f"Especialista {exp_idx+1}")
    print("="*60)
    print(f"Accuracy      : {accuracy:.3f}")
    print(f"Sensibilidad  : {sensitivity:.3f}")
    print(f"Especificidad : {specificity:.3f}")
    print(f"Fooling Rate  : {fooling_rate:.3f}")
    print("="*60)

    # ---- Normalización por fila ----
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # ---- Graficar ----
    plt.figure()
    im = plt.imshow(cm_normalized)
    plt.colorbar(im)

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Clasificación del especialista")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - Especialista {exp_idx+1}")

    for i in range(len(labels)):
        for j in range(len(labels)):

            count = cm[i, j]
            percentage = cm_normalized[i, j] * 100

            # Obtener color real del colormap
            rgba = im.cmap(im.norm(cm_normalized[i, j]))
            r, g, b, _ = rgba

            # Calcular luminancia
            luminance = 0.2126*r + 0.7152*g + 0.0722*b

            # Texto blanco si fondo oscuro
            color = "white" if luminance < 0.5 else "black"

            # Número absoluto (sin negrilla)
            plt.text(j, i-0.08, f"{count}",
                     ha="center", va="center",
                     color=color, fontsize=11)

            # Porcentaje más pequeño
            plt.text(j, i+0.22, f"{percentage:.1f}%",
                     ha="center", va="center",
                     color=color, fontsize=7)

    plt.tight_layout()

    # ---- Guardar imagen ----
    save_path = r"G:\PYTHON\Proyectos Personales\EchocardiogramGenerator\analysis\evaluation_experts\graficos"
    os.makedirs(save_path, exist_ok=True)

    file_name = f"Matriz_Confusion_Especialista_{exp_idx+1}.png"
    full_path = os.path.join(save_path, file_name)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.show()