# PokedexGen1

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


O utiliza el link que te da local, donde se presenta una interfaz
