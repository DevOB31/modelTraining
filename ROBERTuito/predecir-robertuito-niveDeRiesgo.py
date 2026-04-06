import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 1. CARGAR EL MODELO
ruta_modelo = "./modelo_robertuito_base_final"
print("Cargando el cerebro de la IA...")

tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
modelo = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)
modelo.eval()

print("\n" + "="*50)
print("🤖 SISTEMA DE ALERTAS UFPS ACTIVADO")
print("Escribe un comentario o digita 'salir' para cerrar.")
print("="*50 + "\n")

# 2. BUCLE DE PREDICCIÓN
while True:
    texto = input("💬 Comentario del estudiante: ")
    
    if texto.lower() == 'salir':
        print("\n¡Cerrando sistema! Nos vemos.\n")
        break
        
    if not texto.strip():
        continue

    # Procesar
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = modelo(**inputs)
        
    probabilidades = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    prediccion_id = torch.argmax(outputs.logits, dim=-1).item()
    etiqueta = modelo.config.id2label[prediccion_id]
    
    # Colores simples para la terminal
    color = "🟢" if etiqueta == "BAJO" else "🟡" if etiqueta == "MEDIO" else "🔴"
    
    # Imprimir en formato Cuadro
    print("\n" + "╔" + "═"*48 + "╗")
    print(f"║ RESULTADO DEL ANÁLISIS {' ' * 23} ║")
    print("╠" + "═"*48 + "╣")
    print(f"║ {color} NIVEL DE ALERTA: {etiqueta:<28} ║")
    print("╠" + "═"*48 + "╣")
    print(f"║ DISTRIBUCIÓN DE PROBABILIDADES:{' '*16} ║")
    print(f"║   BAJO:  {probabilidades[0] * 100:>6.2f}%{' '*27} ║")
    print(f"║   MEDIO: {probabilidades[1] * 100:>6.2f}%{' '*27} ║")
    print(f"║   ALTO:  {probabilidades[2] * 100:>6.2f}%{' '*27} ║")
    print("╚" + "═"*48 + "╝\n")