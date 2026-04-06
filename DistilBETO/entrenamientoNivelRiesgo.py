"""
entrenamiento-nivelDeRiesgo-distilbeto.py
------------------------------------------
Tarea 1: Clasificación de nivel de riesgo (multiclase).
Modelo: DistilBETO (dccuchile/distilbert-base-spanish-uncased)

Clases:
  - BAJO  (0)
  - MEDIO (1)
  - ALTO  (2)

Diferencias clave vs BETO y RoBERTuito:
  - DistilBETO es la versión destilada de BETO: ~40% menos parámetros,
    ~60% más rápido en inferencia, mantiene ~97% del rendimiento de BETO.
  - Es "uncased": no distingue mayúsculas/minúsculas (todo se convierte
    a minúsculas antes de tokenizar). Útil para texto informal con
    capitalización inconsistente como los comentarios estudiantiles.
  - Arquitectura: 6 capas Transformer en lugar de 12 (BERT/BETO).
  - Pre-entrenado con texto en español usando destilación de conocimiento
    desde BETO como modelo maestro.
  - Mismos hiperparámetros que RoBERTuito y BETO para comparación justa.
"""

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================
MODEL_NAME = "dccuchile/distilbert-base-spanish-uncased"
DATA_PATH  = "../data/dataset_tesis_limpio.csv"
OUTPUT_DIR = "./resultados_distilbeto_riesgo"
SAVE_PATH  = "./modelo_distilbeto_riesgo_final"
MAX_LENGTH = 128
SEED       = 42  # Condición controlada del experimento


# ==========================================
# 1. LOAD DATA
# ==========================================
def cargar_datos(path: str) -> pd.DataFrame:
    print("Cargando datos...")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    print(f"  Filas cargadas: {len(df)}")
    return df


# ==========================================
# 2. PREPROCESSING
# ==========================================
def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocesando datos...")
    label_map = {"BAJO": 0, "MEDIO": 1, "ALTO": 2}

    etiquetas_encontradas = set(df["label"].unique())
    if not etiquetas_encontradas.issubset(set(label_map.keys())):
        print(f"  Advertencia: etiquetas inesperadas → {etiquetas_encontradas - set(label_map.keys())}")
        df = df[df["label"].isin(label_map.keys())].copy()

    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    print(f"  Distribución de clases:")
    for nombre, codigo in label_map.items():
        count = (df["label"] == codigo).sum()
        print(f"    {nombre}: {count} ({count/len(df)*100:.1f}%)")

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
# 4. MÉTRICAS
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy":  acc,
        "f1_macro":  f1,
        "precision": precision,
        "recall":    recall,
    }


# ==========================================
# MAIN
# ==========================================
def main():
    # --- Datos ---
    df = cargar_datos(DATA_PATH)
    df = preprocesar(df)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    print(f"\n  Train: {len(train_df)} muestras | Test: {len(test_df)} muestras")

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    test_dataset  = Dataset.from_pandas(test_df[["text", "label"]])

    # --- Tokenizer ---
    print(f"\nCargando tokenizer para {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizando datasets...")
    train_dataset = tokenizar(train_dataset, tokenizer)
    test_dataset  = tokenizar(test_dataset,  tokenizer)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch",  columns=["input_ids", "attention_mask", "label"])

    # --- Modelo ---
    print("\nCargando modelo...")
    id2label = {0: "BAJO", 1: "MEDIO", 2: "ALTO"}
    label2id = {"BAJO": 0, "MEDIO": 1, "ALTO": 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    # --- Training Arguments ---
    # Mismos hiperparámetros que RoBERTuito y BETO para comparación justa
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
        metric_for_best_model="f1_macro",
        seed=SEED,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Entrenamiento ---
    print(f"\nIniciando entrenamiento (Fine-Tuning DistilBETO — Nivel de Riesgo)...")
    trainer.train()

    # --- Evaluación ---
    print("\nEvaluando modelo final...")
    eval_result = trainer.evaluate()
    print("\nResultados de evaluación:")
    for key, val in eval_result.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # --- Guardar ---
    print(f"\nGuardando modelo en {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("✅ Entrenamiento completado con éxito.")


if __name__ == "__main__":
    main()