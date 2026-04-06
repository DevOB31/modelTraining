"""
limpiar_dataset.py
------------------
Limpieza y estandarización del dataset para la Tarea 2 (multi-etiqueta).

Problemas detectados en el análisis exploratorio:
  - 'SIN CATEGORIA' y 'SIN CATEGORÍA'  → estandarizar a 'SIN CATEGORIA'
  - 'MOTIVACIÓN'                        → estandarizar a 'SIN CATEGORIA'
  - 'Desarrollo del Conocimiento'       → 'DESARROLLO DEL CONOCIMIENTO'
  - 'DESAROLLO DEL CONOCIMIENTO'        → 'DESARROLLO DEL CONOCIMIENTO'
  - 'MEDIO,DESARROLLO DEL CONOCIMIENTO' → separar correctamente

Salida: dataset_tesis_limpio.csv  (listo para entrenar)
"""

import pandas as pd
from collections import Counter

# ==========================================
# 1. CARGAR DATOS
# ==========================================
INPUT_PATH  = "data/dataset_tesis_version_3_categories_new.csv"
OUTPUT_PATH = "data/dataset_tesis_limpio.csv"

df = pd.read_csv(INPUT_PATH, encoding="utf-8")
print(f"Filas originales: {len(df)}")

# ==========================================
# 2. MAPA DE CORRECCIONES
# Cualquier valor que no esté aquí se deja igual.
# ==========================================
CORRECCIONES = {
    # Acento inconsistente
    "SIN CATEGORÍA":                  "SIN CATEGORIA",
    # Categoría fuera del esquema → sin categoría
    "MOTIVACIÓN":                     "SIN CATEGORIA",
    # Capitalización incorrecta
    "Desarrollo del Conocimiento":    "DESARROLLO DEL CONOCIMIENTO",
    # Typo
    "DESAROLLO DEL CONOCIMIENTO":     "DESARROLLO DEL CONOCIMIENTO",
}

# Etiquetas válidas finales (5 en total)
ETIQUETAS_VALIDAS = {
    "DESARROLLO DEL CONOCIMIENTO",
    "DESEMPEÑO DOCENTE",
    "PROCESOS DE EVALUACIÓN",
    "INTEGRACIÓN INTERPERSONAL",
    "SIN CATEGORIA",
}

# ==========================================
# 3. FUNCIÓN DE LIMPIEZA POR FILA
# ==========================================
def limpiar_categorias(valor: str) -> str:
    """
    Recibe la cadena raw de categorías de una fila y devuelve
    la cadena estandarizada con el separador ' | '.
    """
    # Caso especial: 'MEDIO,DESARROLLO DEL CONOCIMIENTO' (error de formato)
    # Se detecta porque contiene una coma y no contiene ' | '
    if "," in valor and " | " not in valor:
        # Separar por coma, limpiar espacios y reensamblar
        partes = [p.strip() for p in valor.split(",")]
    else:
        partes = [p.strip() for p in valor.split(" | ")]

    # Aplicar correcciones a cada parte individual
    partes_corregidas = [CORRECCIONES.get(p, p) for p in partes]

    # Eliminar duplicados manteniendo el orden
    vistas = set()
    partes_unicas = []
    for p in partes_corregidas:
        if p not in vistas:
            vistas.add(p)
            partes_unicas.append(p)

    # Filtrar etiquetas que no estén en el esquema válido
    partes_validas = [p for p in partes_unicas if p in ETIQUETAS_VALIDAS]

    # Si después de limpiar no queda ninguna etiqueta válida, asignar SIN CATEGORIA
    if not partes_validas:
        partes_validas = ["SIN CATEGORIA"]

    return " | ".join(partes_validas)


# ==========================================
# 4. APLICAR LIMPIEZA
# ==========================================
df["categories"] = df["categories"].apply(limpiar_categorias)

# ==========================================
# 5. VERIFICACIÓN POST-LIMPIEZA
# ==========================================
cats_post = df["categories"].str.split(" | ", regex=False)

counter_post = Counter()
for row in cats_post:
    for cat in row:
        counter_post[cat.strip()] += 1

print("\n=== Categorías después de la limpieza ===")
for cat, count in counter_post.most_common():
    print(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")

print(f"\n=== Etiquetas únicas encontradas ===")
todas = set(counter_post.keys())
print(todas)

inesperadas = todas - ETIQUETAS_VALIDAS
if inesperadas:
    print(f"\n⚠️  Etiquetas inesperadas aún presentes: {inesperadas}")
else:
    print("\n✅ Todas las etiquetas son válidas.")

print(f"\n=== Distribución de n° de etiquetas por comentario ===")
df["n_cats"] = cats_post.apply(len)
print(df["n_cats"].value_counts().sort_index())

# ==========================================
# 6. GUARDAR DATASET LIMPIO
# ==========================================
df = df.drop(columns=["n_cats"])
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"\n✅ Dataset limpio guardado en: {OUTPUT_PATH}")
print(f"   Filas finales: {len(df)}")