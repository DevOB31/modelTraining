"""
entrenamiento-nivelDeRiesgo-electricidad.py
--------------------------------------------
Tarea 1: Clasificación de nivel de riesgo (multiclase).
Modelo: Electricidad (mrm8488/electricidad-base-discriminator)

Clases:
  - BAJO  (0)
  - MEDIO (1)
  - ALTO  (2)

Diferencias clave vs RoBERTuito, BETO y DistilBETO:
  - Electricidad es la versión española de ELECTRA, basada en arquitectura
    discriminadora en lugar de enmascaradora (MLM).
  - Durante el pre-entrenamiento, ELECTRA aprende a detectar tokens
    "falsos" insertados por un generador, lo que produce representaciones
    semánticas más eficientes con menor coste computacional que BERT.
  - Usa ElectraTokenizerFast en lugar del tokenizer genérico.
  - Es "uncased": convierte todo a minúsculas antes de tokenizar.
  - Mismos hiperparámetros que los modelos anteriores para comparación justa.

Nota sobre el código de HuggingFace:
  La documentación oficial muestra ElectraForPreTraining, que es para la
  tarea original de detección de tokens. Para clasificación de texto usamos
  AutoModelForSequenceClassification, que añade automáticamente la cabeza
  de clasificación sobre el encoder de ELECTRA.
"""

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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
DATA_PATH  = "../data/dataset_tesis_version_3_categories_new.csv"
OUTPUT_DIR = "./resultados_electricidad_riesgo"
SAVE_PATH  = "./modelo_electricidad_riesgo_final"
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
# Usamos ElectraTokenizerFast explícitamente
# ya que es el tokenizer nativo de ELECTRA.
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
    # ElectraTokenizerFast es el tokenizer nativo para modelos ELECTRA.
    print(f"\nCargando tokenizer para {MODEL_NAME}...")
    tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)

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
    # Mismos hiperparámetros que los modelos anteriores para comparación justa
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
    print(f"\nIniciando entrenamiento (Fine-Tuning Electricidad — Nivel de Riesgo)...")
    trainer.train()

    # --- Evaluación ---
    print("\nEvaluando modelo final...")
    eval_result = trainer.evaluate()
    print("\nResultados de evaluación:")
    for key, val in eval_result.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # --- Guardar ---
    # Guardamos también el tokenizer para usarlo en predicción
    print(f"\nGuardando modelo en {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("✅ Entrenamiento completado con éxito.")


if __name__ == "__main__":
    main()