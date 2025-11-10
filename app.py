# app.py
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ---------------- Config ----------------
CKPT_PATH = Path("runs/pokemon_resnet18/best_model.pth")
CLASS_MAP_PATH = Path("runs/pokemon_resnet18/class_to_idx.json")
TITLE = "Pokémon Classifier (ResNet18)"
SUBTITLE = "Sube una imagen y te digo qué Pokémon es 🐾⚡"

# -------------- Carga modelo -----------
def load_checkpoint_safely(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def build_model(num_classes, pretrained=False):
    weights = None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"No encontré {CKPT_PATH}. Ajusta CKPT_PATH en app.py.")
if not CLASS_MAP_PATH.exists():
    raise FileNotFoundError(f"No encontré {CLASS_MAP_PATH}. Ajusta CLASS_MAP_PATH en app.py.")

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)

# Invertimos para idx -> nombre
idx_to_class = {v: k for k, v in class_to_idx.items()}

ckpt = load_checkpoint_safely(CKPT_PATH, device)
num_classes = len(idx_to_class)
img_size = ckpt.get("img_size", 160)

model = build_model(num_classes=num_classes, pretrained=False)
model.load_state_dict(ckpt["model_state"])
model.eval().to(device)
model = model.to(memory_format=torch.channels_last)

# Transforms (mismo preprocesamiento que en entrenamiento)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preproc = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize,
])

# -------------- Inferencia -------------
@torch.inference_mode()
def predict(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")

    x = preproc(img).unsqueeze(0).to(device, non_blocking=True, memory_format=torch.channels_last)
    use_amp = (device == "cuda")
    try:
        from torch.amp import autocast
        ctx = autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16)
    except Exception:
        # Fallback a autocast antiguo / CPU
        class _DummyCtx:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        ctx = _DummyCtx()

    with ctx:
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].float().cpu()

    # Top-5
    topk = min(5, probs.numel())
    confs, indices = torch.topk(probs, k=topk)
    result_dict = {idx_to_class[int(i)]: float(c) for c, i in zip(confs, indices)}

    # El Label de Gradio acepta un dict para mostrar barras (confidence plot)
    pred_name = max(result_dict, key=result_dict.get)
    return {k: v for k, v in result_dict.items()}, f"Predicción: {pred_name}"

# -------------- UI Gradio --------------
examples = []  # puedes poner rutas locales si quieres ejemplos precargados

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}\n{SUBTITLE}")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Sube una imagen (PNG/JPG)")
            btn = gr.Button("🔮 Predecir", variant="primary")
        with gr.Column():
            pred_label = gr.Label(num_top_classes=5, label="Top-5 (confianzas)")
            pred_text = gr.Textbox(label="Resumen", interactive=False)

    btn.click(fn=predict, inputs=inp, outputs=[pred_label, pred_text])
    inp.change(fn=predict, inputs=inp, outputs=[pred_label, pred_text])  # predice al soltar imagen
    gr.Examples(examples=examples, inputs=inp)

    gr.Markdown(
        "Tips: usa imágenes 160×160 si puedes para máxima velocidad. "
        "Si tienes CUDA, ya se usa automáticamente AMP/bfloat16 y canales_last."
    )

if __name__ == "__main__":
    # Abre en el navegador automáticamente
    demo.launch(inline=False, show_error=True)
