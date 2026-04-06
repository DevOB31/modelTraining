"""
predecir-distilbeto.py
-----------------------
Usa los dos modelos DistilBETO entrenados simultáneamente:
  - Modelo 1: Nivel de riesgo     (BAJO / MEDIO / ALTO)
  - Modelo 2: Categorías          (multi-etiqueta, 5 dimensiones)

Uso:
  python predecir-distilbeto.py
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==========================================
# CONFIGURACIÓN
# ==========================================
RIESGO_MODEL_PATH     = "./modelo_distilbeto_riesgo_final"
CATEGORIAS_MODEL_PATH = "./modelo_distilbeto_categorias_final"
MAX_LENGTH            = 128
UMBRAL_CATEGORIAS     = 0.5

ETIQUETAS_RIESGO = {0: "BAJO", 1: "MEDIO", 2: "ALTO"}

COLORES_RIESGO = {
    "BAJO":  "✅",
    "MEDIO": "⚠️ ",
    "ALTO":  "🚨",
}

ETIQUETAS_CATEGORIAS = [
    "DESARROLLO DEL CONOCIMIENTO",
    "DESEMPEÑO DOCENTE",
    "PROCESOS DE EVALUACIÓN",
    "INTEGRACIÓN INTERPERSONAL",
    "SIN CATEGORIA",
]

EJEMPLOS = [
    "El profesor explica muy bien los temas y tiene dominio del contenido, pero los exámenes son muy difíciles y no avisa con tiempo.",
    "Siempre llega tarde y no respeta a los estudiantes. Tuve que hablar con el coordinador.",
    "Buen profesor, explica con ejemplos de la vida real y fomenta la participación en clase.",
    "Los criterios de evaluación no son claros y las notas tardan mucho en publicarse.",
    "El man es un completo inepto, no sabe nada y además le cae mal todo el mundo.",
]


# ==========================================
# CARGAR MODELOS
# ==========================================
def cargar_modelos():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    print("Cargando modelo de riesgo (DistilBETO)...")
    tok_riesgo = AutoTokenizer.from_pretrained(RIESGO_MODEL_PATH)
    mod_riesgo = AutoModelForSequenceClassification.from_pretrained(RIESGO_MODEL_PATH)
    mod_riesgo.to(device).eval()

    print("Cargando modelo de categorías (DistilBETO)...")
    tok_cats = AutoTokenizer.from_pretrained(CATEGORIAS_MODEL_PATH)
    mod_cats = AutoModelForSequenceClassification.from_pretrained(CATEGORIAS_MODEL_PATH)
    mod_cats.to(device).eval()

    print("✅ Modelos cargados correctamente.\n")
    return device, tok_riesgo, mod_riesgo, tok_cats, mod_cats


# ==========================================
# PREDICCIÓN DE RIESGO
# ==========================================
def predecir_riesgo(texto, tokenizer, model, device):
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

    probs     = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    clase_idx = int(np.argmax(probs))
    clase     = ETIQUETAS_RIESGO[clase_idx]

    return {
        "clase": clase,
        "probabilidades": {
            ETIQUETAS_RIESGO[i]: float(probs[i]) for i in range(3)
        },
    }


# ==========================================
# PREDICCIÓN DE CATEGORÍAS
# ==========================================
def predecir_categorias(texto, tokenizer, model, device, umbral=UMBRAL_CATEGORIAS):
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

    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    detectadas = [
        (ETIQUETAS_CATEGORIAS[i], float(probs[i]))
        for i in range(len(ETIQUETAS_CATEGORIAS))
        if probs[i] >= umbral
    ]
    detectadas.sort(key=lambda x: x[1], reverse=True)

    if not detectadas:
        idx_max = int(np.argmax(probs))
        detectadas = [(ETIQUETAS_CATEGORIAS[idx_max], float(probs[idx_max]))]

    return {
        "detectadas": detectadas,
        "todas": {
            ETIQUETAS_CATEGORIAS[i]: float(probs[i])
            for i in range(len(ETIQUETAS_CATEGORIAS))
        },
    }


# ==========================================
# MOSTRAR RESULTADO
# ==========================================
def mostrar_resultado(texto, res_riesgo, res_cats):
    print("\n" + "=" * 65)
    print(f'Comentario: "{texto[:75]}{"..." if len(texto) > 75 else ""}"')
    print("=" * 65)

    nivel = res_riesgo["clase"]
    icono = COLORES_RIESGO[nivel]
    prob  = res_riesgo["probabilidades"][nivel]
    print(f"\n  Nivel de riesgo: {icono} {nivel}  ({prob:.2%})")
    print("  Probabilidades por clase:")
    for clase, p in sorted(
        res_riesgo["probabilidades"].items(), key=lambda x: x[1], reverse=True
    ):
        barra  = "█" * int(p * 20)
        activo = "→" if clase == nivel else " "
        print(f"    {activo} {clase:<8}  {p:.2%}  {barra}")

    print(f"\n  Categorías detectadas:")
    for cat, p in res_cats["detectadas"]:
        barra = "█" * int(p * 20)
        print(f"    ✅ {cat:<35}  {p:.2%}  {barra}")

    print(f"\n  Probabilidades completas (categorías):")
    for cat, p in sorted(
        res_cats["todas"].items(), key=lambda x: x[1], reverse=True
    ):
        activo = "✅" if p >= UMBRAL_CATEGORIAS else "  "
        print(f"    {activo} {cat:<35}  {p:.4f}")
    print()


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 65)
    print("   Predictor DistilBETO — Riesgo + Categorías Pedagógicas")
    print("=" * 65)

    device, tok_r, mod_r, tok_c, mod_c = cargar_modelos()

    while True:
        print("Opciones:")
        print("  [1] Escribir un comentario")
        print("  [2] Probar con ejemplos predefinidos")
        print("  [3] Salir")
        opcion = input("\nElige una opción: ").strip()

        if opcion == "1":
            texto = input("\nEscribe el comentario del estudiante:\n> ").strip()
            if texto:
                res_r = predecir_riesgo(texto, tok_r, mod_r, device)
                res_c = predecir_categorias(texto, tok_c, mod_c, device)
                mostrar_resultado(texto, res_r, res_c)

        elif opcion == "2":
            print("\nEjemplos predefinidos:")
            for i, ej in enumerate(EJEMPLOS, 1):
                print(f"  [{i}] {ej[:65]}...")
            idx = input(f"\nElige un ejemplo (1-{len(EJEMPLOS)}): ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(EJEMPLOS):
                texto = EJEMPLOS[int(idx) - 1]
                res_r = predecir_riesgo(texto, tok_r, mod_r, device)
                res_c = predecir_categorias(texto, tok_c, mod_c, device)
                mostrar_resultado(texto, res_r, res_c)
            else:
                print("  Opción no válida.\n")

        elif opcion == "3":
            print("\n¡Hasta luego!")
            break

        else:
            print("  Opción no válida, intenta de nuevo.\n")


if __name__ == "__main__":
    main()