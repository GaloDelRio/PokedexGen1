# PokedexGen1

<img width="736" height="531" alt="image" src="https://github.com/user-attachments/assets/ed90b1a4-1c35-4a35-b76b-4636d204aa79" />


##  ¿Qué hace? `modelo.py`

Este script entrena un modelo que reconoce Pokémon a partir de imágenes.

- Usa **ResNet18** (preentrenado en ImageNet).
- Carga las imágenes desde carpetas `train`, `validation` y `test`.
- Aplica aumentación de imágenes (rotación, recorte, etc.).
- Entrena el modelo y guarda el mejor resultado.
- Evalúa el modelo final en el conjunto de prueba.
- Genera gráficas de entrenamiento.

---

##  Salidas importantes
| Archivo | Descripción |
|--------|-------------|
| `best_model.pth` | Modelo final guardado |
| `class_to_idx.json` | Nombre de clase → índice |
| `loss_curve.png` | Curva de pérdida |
| `accuracy_curve.png` | Curva de precisión |
| `precision_macro_curve.png` | Curva de precisión macro |
| `test_metrics.json` | Resultados finales del modelo |

---
## Cómo correr

```bash
.\.venv\Scripts\python.exe .\modelo.py
```
---

## ¿Qué hace `predict_one.py`?

Carga un modelo entrenado (`best_model.pth`) y predice qué Pokémon aparece en una sola imagen.

- Restaura la ResNet18 y sus pesos desde el checkpoint.
- Redimensiona y normaliza la imagen igual que en entrenamiento.
- Calcula `softmax` para obtener la clase y su confianza.
- Imprime algo como: `Predicción: pikachu (confianza: 0.974)`

---

## Cómo correr

```bash
python predict_one.py --img ruta/a/tu_imagen.png \
  --ckpt runs/pokemon_resnet18/best_model.pth
```

---

## ¿Qué hace `app.py`?

Interfaz web con **Gradio** para clasificar imágenes de Pokémon usando tu **ResNet18** entrenada.

- Carga el checkpoint `best_model.pth` y el mapa de clases `class_to_idx.json`.
- Preprocesa la imagen como en entrenamiento (resize + normalize).
- Hace inferencia (usa **GPU** si hay CUDA, con AMP y `channels_last`).
- Muestra **Top-5** clases con barras de confianza y un resumen con la predicción.

---

## Cómo ejecutar

1) Asegúrate de tener estos archivos generados por `modelo.py`:


2) Instala dependencias (dentro de tu venv):
```bash
pip install torch torchvision pillow gradio
```

3) Ejecuta la app:
```bash
python app.py
```

Abre el navegador en:
```cpp
http://127.0.0.1:7860
```

---
### Recursos

[Pokemon Images, First Generation(17000 files)](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files?utm_source=chatgpt.com)


