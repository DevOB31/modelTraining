"""
train_robertuito_multilabel.py
-------------------------------
Tarea 2: Clasificación multi-etiqueta de categorías pedagógicas.

Etiquetas (5):
  - DESARROLLO DEL CONOCIMIENTO
  - DESEMPEÑO DOCENTE
  - PROCESOS DE EVALUACIÓN
  - INTEGRACIÓN INTERPERSONAL
  - SIN CATEGORIA

Diferencias clave vs Tarea 1 (multiclase):
  - Labels → vector binario de 5 posiciones  [1, 0, 1, 0, 0]
  - Loss   → BCEWithLogitsLoss (no CrossEntropy)
  - Pred   → sigmoid() + umbral 0.5 (no argmax)
  - Métricas → F1-micro, F1-macro, Hamming Loss
"""

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    classification_report,
)
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================
MODEL_NAME  = "pysentimiento/robertuito-base-uncased"
DATA_PATH   = "../data/dataset_tesis_limpio.csv"
OUTPUT_DIR  = "./resultados_robertuito_categorias"
SAVE_PATH   = "./modelo_robertuito_categorias_final"
MAX_LENGTH  = 128
SEED        = 42  # Condición controlada del experimento

# Orden fijo de etiquetas — CRÍTICO: debe ser el mismo en todos los modelos
ETIQUETAS = [
    "DESARROLLO DEL CONOCIMIENTO",
    "DESEMPEÑO DOCENTE",
    "PROCESOS DE EVALUACIÓN",
    "INTEGRACIÓN INTERPERSONAL",
    "SIN CATEGORIA",
]
NUM_LABELS = len(ETIQUETAS)  # 5


# ==========================================
# 1. LOAD DATA (Carga de datos)
# ==========================================
def cargar_datos(path: str) -> pd.DataFrame:
    print("Cargando datos...")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = df.dropna(subset=["text", "categories"])
    print(f"  Filas cargadas: {len(df)}")
    return df


# ==========================================
# 2. PREPROCESSING (Preprocesamiento)
# Convertir la columna 'categories' a vector binario
# Ejemplo: "DESEMPEÑO DOCENTE | PROCESOS DE EVALUACIÓN"
#       →  [0, 1, 1, 0, 0]
# ==========================================
def binarizar_etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    print("Binarizando etiquetas multi-etiqueta...")

    def fila_a_vector(cadena: str) -> list:
        etiquetas_fila = [e.strip() for e in cadena.split(" | ")]
        return [1 if etiqueta in etiquetas_fila else 0 for etiqueta in ETIQUETAS]

    vectores = df["categories"].apply(fila_a_vector).tolist()
    df["labels"] = vectores

    # Verificación rápida
    total = len(df)
    print(f"  Distribución de etiquetas:")
    for i, etiqueta in enumerate(ETIQUETAS):
        count = sum(v[i] for v in vectores)
        print(f"    {etiqueta}: {count} ({count/total*100:.1f}%)")

    return df


# ==========================================
# 3. TOKENIZER
# ==========================================
def tokenizar(dataset: Dataset, tokenizer) -> Dataset:
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
    return dataset.map(tokenize_function, batched=True)


# ==========================================
# 4. MODELO PERSONALIZADO
# Necesitamos sobreescribir el cálculo de la pérdida
# para usar BCEWithLogitsLoss en lugar de CrossEntropyLoss
# ==========================================
class ModeloMultiEtiqueta(AutoModelForSequenceClassification.__class__):
    pass


class TrainerMultiEtiqueta(Trainer):
    """
    Trainer personalizado que reemplaza CrossEntropyLoss por
    BCEWithLogitsLoss, necesaria para clasificación multi-etiqueta.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extraer y convertir labels a float (requerido por BCEWithLogitsLoss)
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ==========================================
# 5. MÉTRICAS para multi-etiqueta
# ==========================================
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids

    # Aplicar sigmoid y umbral 0.5 para obtener predicciones binarias
    probs = 1 / (1 + np.exp(-logits))   # sigmoid
    preds = (probs >= 0.5).astype(int)

    f1_micro  = f1_score(labels, preds, average="micro",  zero_division=0)
    f1_macro  = f1_score(labels, preds, average="macro",  zero_division=0)
    h_loss    = hamming_loss(labels, preds)

    return {
        "f1_micro":    f1_micro,
        "f1_macro":    f1_macro,
        "hamming_loss": h_loss,
    }


# ==========================================
# MAIN
# ==========================================
def main():
    # --- Datos ---
    df = cargar_datos(DATA_PATH)
    df = binarizar_etiquetas(df)

    # Split estratificado no es directo en multi-etiqueta;
    # usamos random con seed fija para reproducibilidad
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED
    )

    train_dataset = Dataset.from_pandas(train_df[["text", "labels"]])
    test_dataset  = Dataset.from_pandas(test_df[["text", "labels"]])

    # --- Tokenizer ---
    print(f"\nCargando tokenizer para {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizando datasets...")
    train_dataset = tokenizar(train_dataset, tokenizer)
    test_dataset  = tokenizar(test_dataset,  tokenizer)

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # --- Modelo ---
    print("\nCargando modelo...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",  # Clave para multi-etiqueta
    )

    # --- Training Arguments ---
    # Mismos hiperparámetros que Tarea 1 para comparación justa
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",   # Métrica principal para multi-etiqueta
        seed=SEED,
    )

    # --- Trainer personalizado ---
    trainer = TrainerMultiEtiqueta(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Entrenamiento ---
    print("\nIniciando entrenamiento (Fine-Tuning RoBERTuito Multi-Etiqueta)...")
    trainer.train()

    # --- Evaluación final ---
    print("\nEvaluando modelo final...")
    eval_result = trainer.evaluate()
    print(f"\nResultados de evaluación:")
    for key, val in eval_result.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # --- Reporte por etiqueta ---
    print("\nGenerando reporte por etiqueta...")
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)

    print("\n" + classification_report(
        labels, preds,
        target_names=ETIQUETAS,
        zero_division=0
    ))

    # --- Guardar modelo ---
    print(f"\nGuardando modelo en {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("✅ Entrenamiento completado con éxito.")


if __name__ == "__main__":
    main()