# predict_one.py
import argparse, json
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    from torchvision import models
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features, len(ckpt["classes"]))
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, ckpt["classes"], ckpt.get("img_size", 160)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/pokemon_resnet18/best_model.pth")
    ap.add_argument("--img", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, img_size = load_model(args.ckpt, device)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    img = Image.open(args.img).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        pred_name = classes[pred_idx]
        conf = float(probs[pred_idx])

    print(f"Predicción: {pred_name}  (confianza: {conf:.3f})")

if __name__ == "__main__":
    main()
