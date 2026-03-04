import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv(
    r"G:\PYTHON\Proyectos Personales\EchocardiogramGenerator\analysis\evaluation_experts\Ecocardiogramas_Evaluation_Limpio.csv"
)

df = df.dropna(how="all").reset_index(drop=True)

ground_truth = df.iloc[0]
experts = df.iloc[1:]

# ---- DEFINIR RANGO REAL DE PREGUNTAS ----
QUESTION_START = 0
QUESTION_END = 17   # ← última columna que SÍ debe evaluarse (inclusive)

def score_pair_response(correct, expert):
    if pd.isna(expert):
        return 0.0

    correct_set = set(str(correct).replace(" ", "").split("_"))
    expert_set = set(str(expert).replace(" ", "").split("_"))

    matches = correct_set.intersection(expert_set)

    if len(matches) == 2:
        return 1.0
    elif len(matches) == 1:
        return 0.5
    else:
        return 0.0


accuracies = []

for exp_idx in range(len(experts)):

    row = experts.iloc[exp_idx]
    total_score = 0.0
    total_questions = QUESTION_END - QUESTION_START + 1

    print("\n" + "="*60)
    print(f"Evaluando Experto {exp_idx+1}")
    print("="*60)

    for col_idx in range(QUESTION_START, QUESTION_END + 1):

        correct_answer = ground_truth.iloc[col_idx]
        expert_answer = row.iloc[col_idx]

        # Columnas 11 a 15 → índices 10 a 14
        if 10 <= col_idx <= 14:
            score = score_pair_response(correct_answer, expert_answer)
        else:
            score = 1.0 if correct_answer == expert_answer else 0.0

        total_score += score

        print(f"Columna {col_idx}")
        print(f"  Correcta : {correct_answer}")
        print(f"  Experto  : {expert_answer}")
        print(f"  Puntaje  : {score}")
        print("-"*40)

    accuracy = total_score / total_questions
    accuracies.append(accuracy)

    print(f"Accuracy Experto {exp_idx+1}: {accuracy}")


# 📊 Gráfica mejorada
plt.figure(figsize=(8,5))

bars = plt.bar(
    range(len(accuracies)),
    accuracies,
    edgecolor='black',
    linewidth=1
)

plt.xlabel("Especialista", fontsize=11)
plt.ylabel("Exactitud (Accuracy)", fontsize=11)
plt.title("Exactitud por Especialista", fontsize=13, pad=12)

plt.xticks(
    range(len(accuracies)),
    [f"Esp {i+1}" for i in range(len(accuracies))]
)

plt.ylim(0, 1)

# Grid sutil
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Mostrar porcentaje encima de cada barra
for i, acc in enumerate(accuracies):
    plt.text(
        i,
        acc + 0.02,
        f"{acc*100:.1f}%",
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.tight_layout()

# ---- Guardar imagen en alta resolución ----
save_path = r"G:\PYTHON\Proyectos Personales\EchocardiogramGenerator\analysis\evaluation_experts\graficos"
os.makedirs(save_path, exist_ok=True)

file_name = "Accuracy.png"
full_path = os.path.join(save_path, file_name)

plt.savefig(full_path, dpi=300, bbox_inches='tight')

plt.show()