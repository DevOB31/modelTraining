import os
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    # ==========================================
    # 1. LOAD DATA (Carga de datos)
    # ==========================================
    print("Loading data...")
    nombre_archivo = "../data/dataset_tesis_version_3_categories_new.csv"
    try:
        # Añadimos encoding='utf-8' para evitar problemas con las tildes y la 'ñ'
        df = pd.read_csv(nombre_archivo, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: '{nombre_archivo}' not found.")
        return
    
    # ==========================================
    # 2. PREPROCESSING (Preprocesamiento)
    # ==========================================
    print("Preprocessing data...")
    # Adaptado a tus etiquetas de alerta
    label_map = {"BAJO": 0, "MEDIO": 1, "ALTO": 2}

    # Verificar que no haya etiquetas extrañas en el CSV
    if not set(df["label"].unique()).issubset(set(label_map.keys())):
        print(f"Warning: Found unknown labels. Expected {list(label_map.keys())}, found {df['label'].unique()}")
        df = df[df["label"].isin(label_map.keys())].copy()

    # Mapear a números enteros
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=['text', 'label'])
    df["label"] = df["label"].astype(int)

    # Split dataset (stratify es clave para tu dataset desbalanceado)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Convertir a formato Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # ==========================================
    # 3. TOKENIZER (Tokenización por lotes)
    # ==========================================
    # Usamos el modelo especializado en sentimientos que acordamos
    model_name = "pysentimiento/robertuito-base-uncased"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # ==========================================
    # 4. MODEL SETUP (Configuración del Modelo)
    # ==========================================

    print("Loading model...")
    id2label = {0: "BAJO", 1: "MEDIO", 2: "ALTO"}
    label2id = {"BAJO": 0, "MEDIO": 1, "ALTO": 2}

    # Cargamos el modelo base indicándole que cree una cabeza de clasificación para 3 clases
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    # ==========================================
    # 5. METRICS (Métricas Macro para el desbalance)
    # ==========================================

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1_macro": f1, "precision": precision, "recall": recall}
    
    # ==========================================
    # 6. TRAINING ARGUMENTS
    # ==========================================
    training_args = TrainingArguments(
        output_dir="./resultados_robertuito_base",
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
    )

    # ==========================================
    # 7. TRAINER
    # ==========================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ==========================================
    # 8. TRAIN & EVALUATE
    # ==========================================
    print("\nStarting training (Fine-Tuning RoBERTuito Base)...")
    trainer.train()

    print("\nEvaluating final model...")
    eval_result = trainer.evaluate()
    print(f"\nEvaluation results: {eval_result}\n")

    # ==========================================
    # 9. SAVE MODEL
    # ==========================================
    save_path = "./modelo_robertuito_base_final"
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done! Entrenamiento completado con éxito.")

if __name__ == "__main__":
    main()