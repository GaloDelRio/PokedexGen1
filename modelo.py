# modelo.py
import argparse, json, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # asegura guardado de figuras sin GUI
import matplotlib.pyplot as plt


# -------- Utils de transforms --------
def build_transforms(img_size=160):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


# -------- Cargar datasets con "intersección" de clases ---------
def _get_class_names(root):
    return sorted([p.name for p in Path(root).iterdir() if p.is_dir()])

def _filter_and_remap_imagefolder(ds, keep_classes):
    name_by_idx = ds.classes
    keep_set = set(keep_classes)
    new_class_to_idx = {name: i for i, name in enumerate(keep_classes)}

    new_samples = []
    new_targets = []
    for path, old_idx in ds.samples:
        name = name_by_idx[old_idx]
        if name in keep_set:
            new_idx = new_class_to_idx[name]
            new_samples.append((path, new_idx))
            new_targets.append(new_idx)

    ds.samples = new_samples
    ds.targets = new_targets
    ds.classes = keep_classes
    ds.class_to_idx = new_class_to_idx
    return ds

def build_datasets(train_dir, val_dir, test_dir, img_size=160):
    train_tf, eval_tf = build_transforms(img_size)

    d_train = datasets.ImageFolder(train_dir, transform=train_tf)
    d_val   = datasets.ImageFolder(val_dir,   transform=eval_tf)
    d_test  = datasets.ImageFolder(test_dir,  transform=eval_tf)

    train_names = set(_get_class_names(train_dir))
    val_names   = set(_get_class_names(val_dir))
    test_names  = set(_get_class_names(test_dir))

    common = sorted(list(train_names & val_names & test_names))
    if not common:
        raise RuntimeError("No hay clases en común entre train/val/test. Revisa tus carpetas o contenido.")

    missing_msg = []
    if train_names != set(common):
        missing = sorted(list(train_names - set(common)))
        if missing: missing_msg.append(f"Se excluyen en train: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if val_names != set(common):
        missing = sorted(list(val_names - set(common)))
        if missing: missing_msg.append(f"Se excluyen en val: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if test_names != set(common):
        missing = sorted(list(test_names - set(common)))
        if missing: missing_msg.append(f"Se excluyen en test: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if missing_msg:
        print("[AVISO] No todas las clases coinciden. Usaré la intersección común.")
        for m in missing_msg:
            print("  -", m)

    d_train = _filter_and_remap_imagefolder(d_train, common)
    d_val   = _filter_and_remap_imagefolder(d_val,   common)
    d_test  = _filter_and_remap_imagefolder(d_test,  common)

    if len(d_train.samples) == 0 or len(d_val.samples) == 0 or len(d_test.samples) == 0:
        raise RuntimeError("Después de filtrar, algún split quedó vacío. Verifica que haya imágenes en esas clases.")

    return d_train, d_val, d_test


# --------- DataLoaders --------
def make_loaders(d_train, d_val, d_test, batch_size=64, num_workers=4, use_sampler_if_imbalance=True, device="cpu"):
    targets = [y for _, y in d_train.samples]
    class_counts = np.bincount(targets) if len(targets) > 0 else np.array([1])
    class_counts = class_counts[class_counts > 0]
    sampler = None
    if use_sampler_if_imbalance and len(class_counts) > 0 and (class_counts.max() / class_counts.min() >= 2):
        class_sample_count = np.bincount(targets, minlength=len(d_train.classes))
        class_sample_count[class_sample_count == 0] = 1
        weights = 1.0 / class_sample_count
        sample_weights = [weights[t] for t in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    pin = (device == "cuda")
    dl_train = DataLoader(d_train, batch_size=batch_size,
                          shuffle=(sampler is None), sampler=sampler,
                          num_workers=num_workers, pin_memory=pin)
    dl_val   = DataLoader(d_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    dl_test  = DataLoader(d_test,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    return dl_train, dl_val, dl_test


# ------- Modelo --------
def build_model(num_classes, pretrained=True):
    try:
        # TorchVision 0.14+: API con Weights
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        # Compatibilidad con versiones viejas
        model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


# -------- Evaluación --------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    return avg_loss, acc, y_true, y_pred


# -------- Graficado --------
def _plot_and_save(x, y_series, labels, title, ylabel, outpath):
    plt.figure(figsize=(8, 5))
    for y in y_series:
        plt.plot(x, y, linewidth=2)
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(labels)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  -> Gráfica guardada: {outpath}")

def save_curves(metrics, outdir):
    epochs = list(range(1, len(metrics["train_loss"]) + 1))

    _plot_and_save(
        epochs,
        [metrics["train_loss"], metrics["val_loss"]],
        ["Train", "Val"],
        "Curva de Loss",
        "Loss",
        Path(outdir) / "loss_curve.png",
    )

    _plot_and_save(
        epochs,
        [metrics["train_acc"], metrics["val_acc"]],
        ["Train", "Val"],
        "Curva de Accuracy",
        "Accuracy",
        Path(outdir) / "accuracy_curve.png",
    )

    _plot_and_save(
        epochs,
        [metrics["train_precision_macro"], metrics["val_precision_macro"]],
        ["Train", "Val"],
        "Curva de Precision (macro)",
        "Precision (macro)",
        Path(outdir) / "precision_macro_curve.png",
    )

    # Exporta JSON con el histórico también
    with open(Path(outdir) / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  -> Historial de métricas: {Path(outdir) / 'metrics_history.json'}")


# -------- Entrenamiento ----------
def train(train_dir, val_dir, test_dir, outdir, img_size=160,
          batch_size=64, epochs=20, lr=1e-3, wd=1e-4,
          pretrained=True, num_workers=4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    d_train, d_val, d_test = build_datasets(train_dir, val_dir, test_dir, img_size)
    dl_train, dl_val, dl_test = make_loaders(d_train, d_val, d_test, batch_size, num_workers, True, device)

    model = build_model(num_classes=len(d_train.classes), pretrained=pretrained).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(d_train.class_to_idx, f, ensure_ascii=False, indent=2)

    best_val = float("inf")
    best_path = outdir / "best_model.pth"

    # buffers para curvas
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_precision_macro": [],
        "val_precision_macro": [],
        "lr": [],
        "epoch_time_sec": [],
    }

    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0

        # Para precision (macro) en train
        train_preds_epoch = []
        train_targets_epoch = []

        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # métricas de entrenamiento
            preds = logits.argmax(1)
            running_loss += loss.item() * y.size(0)
            running_correct += (preds == y).sum().item()
            total += y.size(0)

            # guarda para precision macro
            train_preds_epoch.append(preds.detach().cpu().numpy())
            train_targets_epoch.append(y.detach().cpu().numpy())

        train_loss = running_loss / max(1, total)
        train_acc  = running_correct / max(1, total)
        y_true_train = np.concatenate(train_targets_epoch) if train_targets_epoch else np.array([])
        y_pred_train = np.concatenate(train_preds_epoch) if train_preds_epoch else np.array([])
        train_precision_macro = precision_score(
            y_true_train, y_pred_train, average="macro", zero_division=0
        ) if y_true_train.size > 0 else 0.0

        val_loss, val_acc, y_true_val, y_pred_val = evaluate(model, dl_val, device)
        val_precision_macro = precision_score(
            y_true_val, y_pred_val, average="macro", zero_division=0
        ) if y_true_val.size > 0 else 0.0

        scheduler.step(val_loss)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        print(f"[{epoch:03d}/{epochs}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} precM={train_precision_macro:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} precM={val_precision_macro:.4f} | "
              f"lr={lr_now:.5f} | {dt:.1f}s")

        # guarda histórico
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
        metrics["train_precision_macro"].append(train_precision_macro)
        metrics["val_precision_macro"].append(val_precision_macro)
        metrics["lr"].append(lr_now)
        metrics["epoch_time_sec"].append(dt)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "classes": d_train.classes,
                        "img_size": img_size}, best_path)
            print(f"  -> Guardado: {best_path}")

    # Guardar curvas al finalizar entrenamiento
    save_curves(metrics, outdir)

    # Evaluación en test con el mejor modelo
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, y_true, y_pred = evaluate(model, dl_test, device)
    test_precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) if y_true.size > 0 else 0.0

    print("\n=== Resultados en TEST ===")
    print(f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  test_precision_macro={test_precision_macro:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de confusión (filas=verdadero, cols=predicho):")
    print(cm)
    print("\nReporte por clase:")
    print(classification_report(y_true, y_pred, target_names=d_train.classes, digits=4))

    # guarda métricas finales de test
    with open(outdir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_precision_macro": float(test_precision_macro)
        }, f, ensure_ascii=False, indent=2)
    print(f"  -> Métricas de test: {outdir / 'test_metrics.json'}")


# -------- CLI --------
def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento clasificador Pokémon (ResNet18)")
    p.add_argument("--train_dir", default="split/train", type=str)
    p.add_argument("--val_dir",   default="split/validation",   type=str)
    p.add_argument("--test_dir",  default="split/test",         type=str)
    p.add_argument("--outdir",    default="runs/pokemon_resnet18", type=str)
    p.add_argument("--img_size",  default=160, type=int)
    p.add_argument("--batch_size",default=64, type=int)
    p.add_argument("--epochs",    default=20, type=int)
    p.add_argument("--lr",        default=1e-3, type=float)
    p.add_argument("--wd",        default=1e-4, type=float)
    p.add_argument("--no_pretrained", action="store_true", help="No usar pesos ImageNet")
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        outdir=args.outdir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        pretrained=not args.no_pretrained,
        num_workers=args.num_workers,
        seed=args.seed,
    )
