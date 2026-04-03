import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# LABEL CLEANING FUNCTION
def normalize_labels(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


# LOAD MAIN DATA

gold = pd.read_csv("1_comments_class.csv")
true_labels = normalize_labels(gold["label"])


# LOAD MODEL PREDICTIONS
model_files = {
    "RoBERTa": "1_roberta.csv",
    "Gemini": "1_Gemini.csv",
    "Vader": "1_vader.csv",
    "Hybrid": "1_hybrid.csv"
}

model_labels = {}
accuracies = {}

LABELS = sorted(true_labels.unique())
print("Canonical labels:", LABELS)


# LOAD + CLEAN PREDICTIONS
for model, path in model_files.items():
    df = pd.read_csv(path)
    preds = normalize_labels(df["label"])

    model_labels[model] = preds
    accuracies[model] = accuracy_score(true_labels, preds)

    print(f"{model} Accuracy: {accuracies[model]:.4f}")


# CONFUSION MATRICES (DIAGRAM)
for model, preds in model_labels.items():
    cm = confusion_matrix(true_labels, preds, labels=LABELS)

    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=True
    )

    plt.title(f"{model} Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ACCURACY COMPARISON BAR CHART
plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison Across Models")
plt.show()


# PIE CHARTS
for model, acc in accuracies.items():
    plt.figure()
    plt.pie(
        [acc, 1 - acc],
        labels=["Correct", "Incorrect"],
        autopct="%1.1f%%"
    )
    plt.title(f"{model} Accuracy Breakdown")
    plt.show()
