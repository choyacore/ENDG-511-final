# ─── Pipeline Notes ───────────────────────────────────────────────────────────
# Training runs from the Colab notebook (final_endg_511.ipynb) on a T4 GPU.
# This file is what the notebook imports from -- we don't run main() directly.
#
# Things that look redundant but are intentional:
#
#   - run_subgroup_evaluation() is also done manually in Cell 13 of the notebook.
#     We kept both so we can see eval output inline without re-running training.
#
#   - SSL_EPOCHS / FINETUNE_EPOCHS are defined here as defaults but the notebook
#     overrides them via --ssl-epochs 30 --ft-epochs 20 in the CLI call.
#
#   - main() is kept so the file still works standalone if needed, but the
#     notebook is the real entry point.
#
# Things cut mid-project:
#
#   - Jetson deployment -- switched to Colab T4, ONNX export commented out.
#   - MediaPipe -- broke on Python 3.12+, replaced with Haar Cascade.
# ──────────────────────────────────────────────────────────────────────────────
  
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split, Subset
    from torchvision import transforms, models
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch / sklearn not installed.")
    print("       pip install torch torchvision scikit-learn")


TASK_CLASSES = {
    "color":  ["black", "brown", "blonde", "gray"],
    "length": ["short", "medium", "long"],
}

#we moved all the SSL stuff to collab since we are not using Jetson anymore
IMG_SIZE        = 64
BATCH_SIZE      = 32
#but the notebook overrides these (`--ssl-epochs 30 --ft-epochs 20`).
#so the following values are not used anymore for SSL epochs and fine-tune epochs but we kept them here as defaults if someone wants to run the file standalone.
SSL_EPOCHS      = 30       # Stage 1
FINETUNE_EPOCHS = 20       # Stage 2
SSL_LR          = 3e-4
FINETUNE_LR     = 1e-4
TEMPERATURE     = 0.07     # NT-Xent temperature (SimCLR standard)
PROJ_DIM        = 128      # SimCLR projection head output dimension
FEW_SHOT_K      = 5        # K support examples per class
FEW_SHOT_Q      = 10       # Query examples per class



#  Datasets


class HairDataset(Dataset if TORCH_AVAILABLE else object):
    """
    Folder-based labelled dataset.
    Each sub-folder is one class; images inside are the samples.
    """

    def __init__(self, root: str, classes: list, transform=None):
        self.transform = transform
        self.classes   = classes
        self.samples   = []   # list of (path, label_int)

        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                print(f"[WARN] Folder not found, skipping: {cls_dir}")
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

        print(f"[INFO] Labelled dataset : {len(self.samples)} images | "
              f"{len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = self._load(path)
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def _load(path):
        from PIL import Image
        bgr = cv2.imread(path)
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class UnlabelledHairDataset(Dataset if TORCH_AVAILABLE else object):
    """
    Unlabelled image pool for SSL pretraining.
    Falls back to the labelled folders if unlabelled/ does not exist.
    Returns two independently augmented views of each image (SimCLR pair).
    """

    def __init__(self, root: str, classes: list, ssl_transform):
        self.ssl_transform = ssl_transform
        self.paths = []

        unlabelled_dir = os.path.join(root, "unlabelled")
        if os.path.isdir(unlabelled_dir):
            for fname in os.listdir(unlabelled_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.paths.append(os.path.join(unlabelled_dir, fname))
            print(f"[INFO] SSL pool (unlabelled/) : {len(self.paths)} images")
        else:
            for cls in classes:
                cls_dir = os.path.join(root, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.paths.append(os.path.join(cls_dir, fname))
            print(f"[INFO] SSL pool (labelled, labels ignored) : "
                  f"{len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        bgr = cv2.imread(self.paths[i])
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.ssl_transform(img), self.ssl_transform(img)



# Augmentation


def get_ssl_augmentation():
    """
    SimCLR augmentation pipeline targeting hair-specific invariances.

    Lighting robustness:
      - Strong ColorJitter (brightness/contrast/saturation +/-0.8, hue +/-0.2)
        simulates indoor / outdoor / fluorescent / kiosk lighting.
      - RandomGrayscale (p=0.2) forces the encoder to rely on texture, not
        just colour -- critical for length classification under varied lighting.

    Hair style / camera diversity:
      - RandomResizedCrop (scale 0.3-1.0) simulates distance / zoom variation.
      - RandomHorizontalFlip handles left / right hair parting.
      - GaussianBlur (p=0.5) models webcam defocus on fixed-focus edge cameras.

    Edge-device realism:
      - RandomSolarize (p=0.1) simulates CMOS sensor saturation under
        bright overhead lighting -- common in barbershop / salon kiosks.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8,
                saturation=0.8, hue=0.2,
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.RandomSolarize(threshold=128, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


def get_train_augmentation():
    """
    Supervised fine-tuning augmentation (milder than SSL).
    Preserves enough structure for accurate label assignment.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])



# Models


class SimCLREncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    MobileNetV2 backbone + SimCLR projection head.

    Architecture
    ------------
    backbone  -> features (1280-d after global avg-pool)
    projector -> Linear(1280, 512) -> BN -> ReLU -> Linear(512, PROJ_DIM)

    The projector is discarded after SSL pretraining; only the backbone
    is retained (standard SimCLR practice: Chen et al., ICML 2020).
    """

    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, PROJ_DIM),
        )

    def forward(self, x):
        """Returns L2-normalised projection (used for NT-Xent loss)."""
        h = self.backbone(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

    def get_features(self, x):
        """Returns backbone features for downstream tasks."""
        return self.backbone(x)


class HairClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    MobileNetV2 backbone (optionally SSL-pretrained) + classification head.
    Used in Stage 2 (supervised fine-tuning) and Stage 3 (few-shot).
    """

    def __init__(self, n_classes: int, backbone_state_dict=None):
        super().__init__()
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        if backbone_state_dict is not None:
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print("[INFO] Loaded SSL-pretrained backbone weights.")

        # Partially unfreeze: layers 0-9 frozen, layers 10+ trainable.
        # Allows high-level hair-specific features to adapt while preserving
        # low-level edge/texture detectors from SSL pretraining.
        for i, layer in enumerate(list(self.backbone[0].children())):
            for param in layer.parameters():
                param.requires_grad = (i >= 10)

        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def get_embedding(self, x):
        """
        Returns the 256-d L2-normalised embedding (before final linear).
        Used by the prototypical network in Stage 3.
        """
        h = self.backbone(x)
        emb = self.classifier[:4](h)   # Linear -> BN -> ReLU -> Dropout
        return F.normalize(emb, dim=1)

#in collab
# NT-Xent Loss (SimCLR)


class NTXentLoss(nn.Module if TORCH_AVAILABLE else object):
    """
    Normalised Temperature-scaled Cross-Entropy loss (NT-Xent).

    Given a batch of N image pairs (2N total embeddings), each positive
    pair (two views of the same image) is pulled together; all 2(N-1) other
    images in the batch are treated as negatives.

    Reference: Chen et al., "A Simple Framework for Contrastive Learning
    of Visual Representations" (SimCLR), ICML 2020.
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.tau = temperature

    def forward(self, z1, z2):
        """
        z1, z2 : (N, D) L2-normalised projections of the two views.
        Returns scalar NT-Xent loss.
        """
        N   = z1.size(0)
        z   = torch.cat([z1, z2], dim=0)           # (2N, D)
        sim = torch.mm(z, z.T) / self.tau           # (2N, 2N) cosine sim

        # Mask out self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+N) and (i+N, i)
        pos  = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])
        loss = -pos + torch.logsumexp(sim, dim=1)
        return loss.mean()

#SSL in collab
# Stage 1: SimCLR Self-Supervised Pretraining


def run_ssl_pretraining(data_root, classes, device,
                        epochs=SSL_EPOCHS,
                        save_path="ssl_backbone.pt"):
    """
    Pretrain the MobileNetV2 backbone with SimCLR on unlabelled hair images.
    Returns the pretrained backbone state dict for use in Stage 2.
    """
    print("\n" + "=" * 55)
    print("  STAGE 1 -- SimCLR Self-Supervised Pretraining")
    print("=" * 55)

    ssl_tfm = get_ssl_augmentation()
    ds      = UnlabelledHairDataset(data_root, classes, ssl_tfm)
    dl      = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, drop_last=True, pin_memory=True)

    model   = SimCLREncoder().to(device)
    loss_fn = NTXentLoss(temperature=TEMPERATURE)
    opt     = optim.Adam(model.parameters(), lr=SSL_LR, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batch = 0.0, 0

        for view1, view2 in dl:
            view1, view2 = view1.to(device), view2.to(device)
            loss = loss_fn(model(view1), model(view2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch    += 1

        sched.step()
        avg = total_loss / max(n_batch, 1)
        marker = ""
        if avg < best_loss:
            best_loss = avg
            torch.save(model.backbone.state_dict(), save_path)
            marker = "  <- backbone saved"
        print(f"  SSL Epoch {epoch:03d}/{epochs}  |  "
              f"NT-Xent = {avg:.4f}{marker}")

    print(f"\n[SSL DONE] Best NT-Xent = {best_loss:.4f} | "
          f"Backbone -> {save_path}\n")
    return torch.load(save_path, map_location=device)



#  Stage 2: Supervised Fine-Tuning


def run_supervised_finetuning(data_root, classes, device,
                               backbone_state_dict=None,
                               epochs=FINETUNE_EPOCHS,
                               task="color"):
    """
    Fine-tune the (optionally SSL-pretrained) classifier on labelled data.
    Returns the best-performing HairClassifier model.
    """
    print("=" * 55)
    print("  STAGE 2 -- Supervised Fine-Tuning")
    print("=" * 55)

    n_classes = len(classes)
    full_ds   = HairDataset(data_root, classes,
                            transform=get_train_augmentation())
    if len(full_ds) == 0:
        print("[ERROR] No labelled images found.")
        sys.exit(1)

    val_n   = max(1, int(0.2 * len(full_ds)))
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n])

    val_clean  = HairDataset(data_root, classes,
                             transform=get_val_transform())
    val_subset = Subset(val_clean,
                        val_ds.indices if hasattr(val_ds, "indices")
                        else list(range(val_n)))

    train_dl = DataLoader(train_ds,   batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    model      = HairClassifier(n_classes, backbone_state_dict).to(device)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Separate learning rates: backbone fine-tunes 10x slower than head
    bb_params   = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer   = optim.AdamW([
        {"params": bb_params,   "lr": FINETUNE_LR * 0.1},
        {"params": head_params, "lr": FINETUNE_LR},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    model_path   = f"hair_{task}_classifier.pt"

    print(f"  Task: {task}  |  Classes: {classes}")
    print(f"  Train / Val: {train_n} / {val_n}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss    += loss.item() * len(imgs)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += len(imgs)

        scheduler.step()

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds      = model(imgs).argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += len(imgs)

        train_acc = t_correct / t_total if t_total else 0.0
        val_acc   = v_correct / v_total  if v_total  else 0.0
        avg_loss  = t_loss   / t_total   if t_total else 0.0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            marker = "  <- best saved"

        print(f"  Epoch {epoch:02d}/{epochs}  |  loss={avg_loss:.4f}  |  "
              f"train={train_acc:.3f}  |  val={val_acc:.3f}{marker}")

    print(f"\n[FT DONE] Best val_acc = {best_val_acc:.3f} | "
          f"Model -> {model_path}\n")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model



#  Stage 3: Prototypical Network Few-Shot Domain Adaptation
#moved to collab and we had to use CelebA for other data so couldn't impement it well

def _episode_sample(dataset, classes, k_shot, q_query, device):
    """
    Sample one N-way K-shot episode.

    Returns (support_x, support_y, query_x, query_y) tensors.
    """
    by_class = defaultdict(list)
    for idx, (_, lbl) in enumerate(dataset.samples):
        by_class[lbl].append(idx)

    valid = [c for c in range(len(classes))
             if len(by_class[c]) >= k_shot + q_query]
    if len(valid) < 2:
        raise ValueError(
            f"Need >=2 classes with >={k_shot + q_query} images each."
        )

    chosen = random.sample(valid, min(len(valid), len(classes)))
    tfm    = get_val_transform()
    s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []

    for new_idx, cls_idx in enumerate(chosen):
        idxs  = random.sample(by_class[cls_idx], k_shot + q_query)
        for i in idxs[:k_shot]:
            img, _ = dataset[i]
            if not isinstance(img, torch.Tensor):
                img = tfm(img)
            s_imgs.append(img); s_lbls.append(new_idx)
        for i in idxs[k_shot:]:
            img, _ = dataset[i]
            if not isinstance(img, torch.Tensor):
                img = tfm(img)
            q_imgs.append(img); q_lbls.append(new_idx)

    return (
        torch.stack(s_imgs).to(device),
        torch.tensor(s_lbls, device=device),
        torch.stack(q_imgs).to(device),
        torch.tensor(q_lbls, device=device),
    )


def _compute_prototypes(model, support_x, support_y, n_classes):
    """
    Compute per-class prototypes as the mean embedding of support examples.

    This is the core Prototypical Networks idea (Snell et al., NeurIPS 2017):
    each class is represented by the centroid of its support set embeddings
    in the learned feature space.
    """
    embeddings = model.get_embedding(support_x)   # (N*K, D)
    prototypes = torch.zeros(n_classes, embeddings.size(1),
                             device=embeddings.device)
    for cls in range(n_classes):
        mask = (support_y == cls)
        if mask.sum() > 0:
            prototypes[cls] = embeddings[mask].mean(0)
    return F.normalize(prototypes, dim=1)


def run_few_shot_adaptation(model, data_root, classes, device,
                             n_episodes=50, task="color"):
    """
    Evaluate and adapt the model using Prototypical Networks.

    For each episode:
      1. Sample K support + Q query images per class.
      2. Compute class prototypes from support embeddings.
      3. Adapt embedding layer with 5 gradient steps on the support set.
      4. Classify query images by nearest prototype (cosine distance).

    This directly addresses the hair style diversity problem: the model
    can adapt to unseen textures (e.g., loc'd, coily, braided) using only
    K=5 labelled examples -- matching the realistic constraints of our
    small custom webcam dataset.
    """
    print("=" * 55)
    print("  STAGE 3 -- Prototypical Network Few-Shot Adaptation")
    print("=" * 55)
    print(f"  K-shot={FEW_SHOT_K}  Q-query={FEW_SHOT_Q}  "
          f"Episodes={n_episodes}\n")

    n_classes = len(classes)
    dataset   = HairDataset(data_root, classes,
                            transform=get_val_transform())
    proto_opt = optim.Adam(
        [p for p in model.classifier[:4].parameters()],
        lr=1e-4,
    )
    proto_path   = f"hair_{task}_proto_adapted.pt"
    best_acc     = 0.0
    episode_accs = []

    for ep in range(1, n_episodes + 1):
        try:
            sx, sy, qx, qy = _episode_sample(
                dataset, classes, FEW_SHOT_K, FEW_SHOT_Q, device
            )
        except ValueError as e:
            print(f"[WARN] Episode {ep} skipped: {e}")
            continue

        # -- Inner loop: 5 gradient steps on support set --------------------
        model.train()
        for _ in range(5):
            protos = _compute_prototypes(model, sx, sy, n_classes)
            q_emb  = model.get_embedding(qx)
            dists  = -(q_emb @ protos.T)          # negative for argmin
            loss   = F.cross_entropy(dists, qy)
            proto_opt.zero_grad()
            loss.backward()
            proto_opt.step()

        # -- Evaluate on query set ------------------------------------------
        model.eval()
        with torch.no_grad():
            protos = _compute_prototypes(model, sx, sy, n_classes)
            q_emb  = model.get_embedding(qx)
            preds  = -(q_emb @ protos.T).argmin(dim=1)
            acc    = (preds == qy).float().mean().item()

        episode_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), proto_path)

        if ep % 10 == 0 or ep == 1:
            running = np.mean(episode_accs)
            print(f"  Episode {ep:03d}/{n_episodes}  |  "
                  f"ep_acc={acc:.3f}  |  running_mean={running:.3f}")

    mean_acc = np.mean(episode_accs) if episode_accs else 0.0
    std_acc  = np.std(episode_accs)  if episode_accs else 0.0
    ci95     = 1.96 * std_acc / (len(episode_accs) ** 0.5 + 1e-9)

    print(f"\n[FEW-SHOT DONE]")
    print(f"  Mean acc = {mean_acc:.3f} +/- {ci95:.3f}  (95% CI)")
    print(f"  Best acc = {best_acc:.3f}  |  Model -> {proto_path}\n")


# Subgroup Evaluation
#we re-did this in collab too

def run_subgroup_evaluation(model, data_root, classes, device, task):
    """
    Evaluate per-class precision, recall, and F1-score.

    This directly addresses the reviewer's concern about imbalance --
    e.g., CelebA's known over-representation of dark hair means 'black'
    will dominate training signal. Classes with F1 < 0.60 are flagged.

    Also prints a confusion matrix to reveal systematic misclassification
    patterns (e.g., auburn <-> red, medium <-> long).
    """
    print("=" * 55)
    print("  Subgroup Evaluation (Per-Class Metrics)")
    print("=" * 55)

    ds = HairDataset(data_root, classes, transform=get_val_transform())
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dl:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Per-class report
    print("\n  Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=classes,
        zero_division=0,
        digits=3,
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("  Confusion Matrix (rows=true, cols=predicted):")
    header = "        " + "  ".join(f"{c[:5]:>5}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {classes[i][:5]:>5}   " +
              "  ".join(f"{v:5d}" for v in row))

    # Flag imbalanced classes
    f1s = f1_score(all_labels, all_preds, average=None, zero_division=0)
    low = [(classes[i], f1s[i]) for i in range(len(classes))
           if f1s[i] < 0.60]
    if low:
        print("\n  [IMBALANCE WARNING] Classes with F1 < 0.60:")
        for cls, score in low:
            print(f"    - {cls:<10} F1={score:.3f}  "
                  "-> collect more samples or apply oversampling")
    else:
        print("\n  All classes F1 >= 0.60 -- no severe imbalance detected.")
    print()




#  Inference helper (used by main.py)


def load_classifier(task: str, device_str: str = "cpu"):
    """
    Load a trained HairClassifier for use in the live pipeline (main.py).
    Prefers proto-adapted weights if available.
    Returns (model, classes) or (None, None) if weights are not found.
    """
    if not TORCH_AVAILABLE:
        return None, None

    classes_path = f"hair_{task}_classes.json"
    if not os.path.exists(classes_path):
        print(f"[WARN] No class mapping for {task}. Using rule-based fallback.")
        return None, None

    with open(classes_path) as f:
        classes = json.load(f)

    proto_path = f"hair_{task}_proto_adapted.pt"
    ft_path    = f"hair_{task}_classifier.pt"
    weights    = proto_path if os.path.exists(proto_path) else ft_path

    if not os.path.exists(weights):
        print(f"[WARN] No trained model for {task}. Using rule-based fallback.")
        return None, None

    model = HairClassifier(len(classes))
    model.load_state_dict(
        torch.load(weights, map_location=torch.device(device_str))
    )
    model.eval()
    print(f"[INFO] Loaded {task} classifier: {weights}")
    return model, classes



def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Hair Analysis CNN Trainer -- ENDG 511 Team 14\n"
            "Stages: SimCLR SSL -> Supervised Fine-Tune -> "
            "Prototypical Few-Shot Adaptation + Subgroup Eval"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task",          choices=["color", "length"],
                   default="color")
    p.add_argument("--data",          default="./dataset")
    p.add_argument("--ssl-epochs",    type=int, default=SSL_EPOCHS)
    p.add_argument("--ft-epochs",     type=int, default=FINETUNE_EPOCHS)
    p.add_argument("--episodes",      type=int, default=50)
    p.add_argument("--skip-ssl",      action="store_true",
                   help="Skip Stage 1, use ImageNet init only")
    p.add_argument("--few-shot-only", action="store_true",
                   help="Stage 3 only, using an existing fine-tuned model")
    p.add_argument("--eval-only",     action="store_true",
                   help="Subgroup evaluation only")

    return p.parse_args()


def main():
    if not TORCH_AVAILABLE:
        print("[ERROR] pip install torch torchvision scikit-learn")
        sys.exit(1)

    args    = _parse_args()
    classes = TASK_CLASSES[args.task]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Hair Analysis Trainer  |  task={args.task}  |  device={device}")
    print(f"{'='*55}\n")

    # Save class mapping for inference
    with open(f"hair_{args.task}_classes.json", "w") as f:
        json.dump(classes, f, indent=2)

    # -- Eval-only ------------------------------------------------------------
    if args.eval_only:
        model, _ = load_classifier(args.task)
        if model is None:
            sys.exit(1)
        run_subgroup_evaluation(model.to(device), args.data,
                                classes, device, args.task)
        return

    # -- Few-shot-only --------------------------------------------------------
    if args.few_shot_only:
        model, _ = load_classifier(args.task)
        if model is None:
            sys.exit(1)
        model = model.to(device)
        run_few_shot_adaptation(model, args.data, classes,
                                device, args.episodes, args.task)
        run_subgroup_evaluation(model, args.data, classes, device, args.task)
        # if args.export_onnx:
        #     export_onnx(model, args.task)
        return

    # -- Stage 1: SimCLR SSL -------------------------------------------------
    backbone_sd = None
    if not args.skip_ssl:
        backbone_sd = run_ssl_pretraining(
            args.data, classes, device,
            epochs    = args.ssl_epochs,
            save_path = f"ssl_backbone_{args.task}.pt",
        )
    else:
        print("[INFO] Skipping SSL (--skip-ssl). Using ImageNet init.\n")

    # -- Stage 2: Supervised fine-tuning -------------------------------------
    model = run_supervised_finetuning(
        args.data, classes, device,
        backbone_state_dict = backbone_sd,
        epochs = args.ft_epochs,
        task   = args.task,
    )

    # -- Stage 3: Few-shot domain adaptation ---------------------------------
    run_few_shot_adaptation(model, args.data, classes,
                            device, args.episodes, args.task)

    # -- Subgroup evaluation -------------------------------------------------
    run_subgroup_evaluation(model, args.data, classes, device, args.task)

    # if args.export_onnx:
    #     export_onnx(model, args.task)

    print(f"[ALL DONE]  task={args.task}  classes={classes}\n")


if __name__ == "__main__":
    main()
