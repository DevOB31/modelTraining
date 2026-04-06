"""
predecir-robertuito-categorias.py
-----------------------------------
Prueba el modelo RoBERTuito entrenado para clasificación
multi-etiqueta de categorías pedagógicas.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "./modelo_robertuito_categorias_final"
MAX_LENGTH = 128
UMBRAL     = 0.5   # Probabilidad mínima para activar una etiqueta

ETIQUETAS = [
    "DESARROLLO DEL CONOCIMIENTO",
    "DESEMPEÑO DOCENTE",
    "PROCESOS DE EVALUACIÓN",
    "INTEGRACIÓN INTERPERSONAL",
    "SIN CATEGORIA",
]

# ==========================================
# CARGAR MODELO Y TOKENIZER
# ==========================================
print("Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Modelo cargado en: {device}\n")


# ==========================================
# FUNCIÓN DE PREDICCIÓN
# ==========================================
def predecir(texto: str, umbral: float = UMBRAL) -> dict:
    """
    Recibe un texto y devuelve las categorías predichas
    junto con sus probabilidades.
    """
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    # Sigmoid para obtener probabilidades independientes por etiqueta
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    resultado = {
        "predicciones": [],
        "todas": {},
    }

    for etiqueta, prob in zip(ETIQUETAS, probs):
        resultado["todas"][etiqueta] = float(prob)
        if prob >= umbral:
            resultado["predicciones"].append((etiqueta, float(prob)))

    # Ordenar por probabilidad descendente
    resultado["predicciones"].sort(key=lambda x: x[1], reverse=True)

    # Si ninguna etiqueta supera el umbral, tomar la de mayor probabilidad
    if not resultado["predicciones"]:
        mejor = max(resultado["todas"].items(), key=lambda x: x[1])
        resultado["predicciones"].append(mejor)
        resultado["nota"] = "Ninguna etiqueta superó el umbral. Se seleccionó la de mayor probabilidad."

    return resultado


# ==========================================
# INTERFAZ DE PRUEBA INTERACTIVA
# ==========================================
def mostrar_resultado(texto: str, resultado: dict):
    print("\n" + "=" * 60)
    print(f"Comentario: \"{texto[:80]}{'...' if len(texto) > 80 else ''}\"")
    print("=" * 60)

    print("\nCategorías detectadas:")
    for etiqueta, prob in resultado["predicciones"]:
        barra = "█" * int(prob * 20)
        print(f"  ✅ {etiqueta:<35} {prob:.2%}  {barra}")

    if "nota" in resultado:
        print(f"\n  ⚠️  {resultado['nota']}")

    print("\nProbabilidades completas:")
    for etiqueta, prob in sorted(resultado["todas"].items(), key=lambda x: x[1], reverse=True):
        activado = "✅" if prob >= UMBRAL else "  "
        print(f"  {activado} {etiqueta:<35} {prob:.4f}")
    print()


# ==========================================
# EJEMPLOS PREDEFINIDOS PARA PRUEBA RÁPIDA
# ==========================================
EJEMPLOS = [
    "El profesor explica muy bien los temas y tiene dominio del contenido, pero los exámenes son muy difíciles y no avisa con tiempo.",
    "Siempre llega tarde y no respeta a los estudiantes. Tuve que hablar con el coordinador.",
    "Buen profesor, explica con ejemplos de la vida real y fomenta la participación en clase.",
    "Los criterios de evaluación no son claros y las notas tardan mucho en publicarse.",
]


def main():
    print("=" * 60)
    print("   Predictor de Categorías — RoBERTuito Multi-Etiqueta")
    print("=" * 60)
    print(f"Umbral de activación: {UMBRAL} ({UMBRAL*100:.0f}%)")
    print(f"Etiquetas: {', '.join(ETIQUETAS)}\n")

    while True:
        print("Opciones:")
        print("  [1] Escribir un comentario")
        print("  [2] Probar con ejemplos predefinidos")
        print("  [3] Salir")
        opcion = input("\nElige una opción: ").strip()

        if opcion == "1":
            texto = input("\nEscribe el comentario del estudiante:\n> ").strip()
            if texto:
                resultado = predecir(texto)
                mostrar_resultado(texto, resultado)

        elif opcion == "2":
            print("\nEjemplos predefinidos:\n")
            for i, ejemplo in enumerate(EJEMPLOS, 1):
                print(f"  [{i}] {ejemplo[:70]}...")
            idx = input("\nElige un ejemplo (1-4): ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(EJEMPLOS):
                texto = EJEMPLOS[int(idx) - 1]
                resultado = predecir(texto)
                mostrar_resultado(texto, resultado)

        elif opcion == "3":
            print("\n¡Hasta luego!")
            break

        else:
            print("  Opción no válida, intenta de nuevo.\n")


if __name__ == "__main__":
    main()