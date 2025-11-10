## Reporte de Entrenamiento — ResNet18 (20 épocas)

Resumen: el modelo converge de forma estable y alcanza **desempeño alto** en validación y prueba.  
Se observan mejoras consistentes en `loss`, `accuracy` y `precision (macro)`.

---

### Métricas en TEST
- `accuracy`: 0.984
- `loss`: 0.057
- `precision_macro`: 0.985

---

### Mejor época (val)
- Época: 20
- `val_accuracy`: 0.985
- `val_loss`: 0.049
- `val_precision_macro`: 0.986

---

### Duración (aprox.)
- Tiempo promedio/época: ~136 s
- Tiempo total (20 épocas): ~45.3 min

---

### Curvas (outdir)
- `loss_curve.png`
- <img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/810be845-2a61-4dfa-a3bb-a1794ffe9315" />

- `accuracy_curve.png`
- <img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/c02edd76-676a-4b8e-9e52-65a7dbb3cdf0" />

- `precision_macro_curve.png`
- <img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/db0a3879-b176-4745-a033-bc977d6d2e2c" />


> Historial completo en `metrics_history.json` y mejores pesos en `best_model.pth`.

---

El entrenamiento muestra una convergencia sólida y generalización muy buena: la pérdida cae de 0.336 a 0.049 en validación y la accuracy sube de 0.904 a 0.985, mientras que en test se confirma el desempeño con accuracy 0.984 y precision (macro) 0.985. Las curvas indican que el modelo aprende rápido en las primeras épocas y luego mejora de forma gradual, sin señales de sobreajuste (la validación se mantiene igual o mejor que entrenamiento). La precision macro alta sugiere buen equilibrio entre clases, y el hecho de que la validación supere levemente al train es coherente con los aumentos y el dropout que introducen ruido en entrenamiento. En resumen, el modelo está bien calibrado, estable y listo para inferencia; si se busca aún más eficiencia, podría detenerse alrededor de la época 15–20 sin perder rendimiento.
