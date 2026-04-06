"""
entrenamiento-categorias-electricidad.py
------------------------------------------
Tarea 2: Clasificación multi-etiqueta de categorías pedagógicas.
Modelo: Electricidad (mrm8488/electricidad-base-discriminator)

Etiquetas (5):
  - DESARROLLO DEL CONOCIMIENTO
  - DESEMPEÑO DOCENTE
  - PROCESOS DE EVALUACIÓN
  - INTEGRACIÓN INTERPERSONAL
  - SIN CATEGORIA

Misma arquitectura multi-etiqueta que los modelos anteriores pero con
Electricidad (ELECTRA en español) como modelo base.
Hiperparámetros idénticos para comparación justa.
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
    ElectraTokenizerFast,
    Trainer,
    TrainingArguments,
)

# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================
MODEL_NAME = "mrm8488/electricidad-base-discriminator"
DATA_PATH  = "../data/dataset_tesis_limpio.csv"
OUTPUT_DIR = "./resultados_electricidad_categorias"
SAVE_PATH  = "./modelo_electricidad_categorias_final"
MAX_LENGTH = 128
SEED       = 42

ETIQUETAS = [
    "DESARROLLO DEL CONOCIMIENTO",
    "DESEMPEÑO DOCENTE",
    "PROCESOS DE EVALUACIÓN",
    "INTEGRACIÓN INTERPERSONAL",
    "SIN CATEGORIA",
]
NUM_LABELS = len(ETIQUETAS)


# ==========================================
# 1. LOAD DATA
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
# 2. BINARIZACIÓN DE ETIQUETAS
# ==========================================
def binarizar_etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    print("Binarizando etiquetas multi-etiqueta...")

    def fila_a_vector(cadena: str) -> list:
        etiquetas_fila = [e.strip() for e in cadena.split(" | ")]
        return [1 if etiqueta in etiquetas_fila else 0 for etiqueta in ETIQUETAS]

    vectores = df["categories"].apply(fila_a_vector).tolist()
    df["labels"] = vectores

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
# 4. TRAINER PERSONALIZADO (BCEWithLogitsLoss)
# ==========================================
class TrainerMultiEtiqueta(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ==========================================
# 5. MÉTRICAS
# ==========================================
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)

    f1_micro = f1_score(labels, preds, average="micro",  zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro",  zero_division=0)
    h_loss   = hamming_loss(labels, preds)

    return {
        "f1_micro":     f1_micro,
        "f1_macro":     f1_macro,
        "hamming_loss": h_loss,
    }


# ==========================================
# MAIN
# ==========================================
def main():
    # --- Datos ---
    df = cargar_datos(DATA_PATH)
    df = binarizar_etiquetas(df)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED
    )
    print(f"\n  Train: {len(train_df)} muestras | Test: {len(test_df)} muestras")

    train_dataset = Dataset.from_pandas(train_df[["text", "labels"]])
    test_dataset  = Dataset.from_pandas(test_df[["text", "labels"]])

    # --- Tokenizer ---
    print(f"\nCargando tokenizer para {MODEL_NAME}...")
    tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)

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
        problem_type="multi_label_classification",
    )

    # --- Training Arguments ---
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
        metric_for_best_model="f1_micro",
        seed=SEED,
    )

    # --- Trainer ---
    trainer = TrainerMultiEtiqueta(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Entrenamiento ---
    print(f"\nIniciando entrenamiento (Fine-Tuning Electricidad — Categorías)...")
    trainer.train()

    # --- Evaluación ---
    print("\nEvaluando modelo final...")
    eval_result = trainer.evaluate()
    print("\nResultados de evaluación:")
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

    # --- Guardar ---
    print(f"\nGuardando modelo en {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("✅ Entrenamiento completado con éxito.")


if __name__ == "__main__":
    main()