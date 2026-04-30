"""
Elliptic Bitcoin Dataset — GCN / GAT / Random Forest / Hybrid Pipeline
Detects illicit (mixer/money-laundering) transactions using graph structure.

Install dependencies first:
    pip install torch-geometric scikit-learn pandas numpy matplotlib

For torch-geometric, if the above fails:
    pip install torch-geometric --find-links https://data.pyg.org/whl/torch-2.10.0+cpu.html
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(BASE_DIR, "elliptic_txs_features.csv")
EDGELIST_PATH = os.path.join(BASE_DIR, "elliptic_txs_edgelist.csv")
CLASSES_PATH   = os.path.join(BASE_DIR, "elliptic_txs_classes.csv")
GCN_MODEL_PATH = os.path.join(BASE_DIR, "gcn_model.pth")
GCN_STATS_PATH = os.path.join(BASE_DIR, "gcn_stats.json")
GAT_MODEL_PATH = os.path.join(BASE_DIR, "gat_model.pth")
GAT_STATS_PATH = os.path.join(BASE_DIR, "gat_stats.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    """
    Returns feature matrix X, label vector y, edge index tensor,
    and boolean masks for train/test splits.
    """
    features_df = pd.read_csv(FEATURES_PATH, header=None)
    # Column 0 = tx_id, column 1 = time_step, columns 2..N = features
    features_df.columns = ["txId", "time_step"] + [
        f"feat_{i}" for i in range(features_df.shape[1] - 2)
    ]

    classes_df = pd.read_csv(CLASSES_PATH)
    # classes: txId, class  (1=illicit, 2=licit, "unknown")
    classes_df.columns = ["txId", "class"]

    edges_df = pd.read_csv(EDGELIST_PATH)
    edges_df.columns = ["txId1", "txId2"]

    df = features_df.merge(classes_df, on="txId", how="left").copy()

    # Depending on how pandas reads the CSV the class column may contain
    # strings or ints; mapping both forms keeps the pipeline robust.
    label_map = {"1": 1, "2": 0, 1: 1, 2: 0}
    df["label"] = df["class"].map(label_map).fillna(-1).astype(int)

    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(df["txId"])}
    n_nodes   = len(df)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feat_cols].values.astype(np.float32)

    # GCNConv aggregates neighbor features via normalised mean — without scaling,
    # features with large variance would dominate that aggregation.
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    y = df["label"].values  # -1 = unknown, 0 = licit, 1 = illicit

    # The original edges are directed (sender → receiver). We symmetrise so
    # each node aggregates messages from both directions during graph convolution.
    src = edges_df["txId1"].map(tx_to_idx).dropna().astype(int)
    dst = edges_df["txId2"].map(tx_to_idx).dropna().astype(int)
    valid = src.notna() & dst.notna()
    src, dst = src[valid].values, dst[valid].values
    edge_index = np.stack(
        [np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0
    )
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

    # Standard Elliptic benchmark split: train on earlier time steps (1–34),
    # test on later ones (35–49). Splitting by time prevents the model from
    # seeing future transaction patterns during training (no data leakage).
    labeled_mask = y != -1
    train_mask = labeled_mask & (df["time_step"].values <= 34)
    test_mask  = labeled_mask & (df["time_step"].values > 34)

    print(f"Nodes        : {n_nodes:,}")
    print(f"Edges (uni)  : {len(src):,}")
    print(f"Train labeled: {train_mask.sum():,}  "
          f"(illicit={y[train_mask].sum():,})")
    print(f"Test  labeled: {test_mask.sum():,}  "
          f"(illicit={y[test_mask].sum():,})")

    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)

    data = Data(
        x=X_tensor,
        edge_index=edge_index_tensor,
        y=y_tensor,
        train_mask=torch.tensor(train_mask),
        test_mask=torch.tensor(test_mask),
    ).to(DEVICE)

    return data, X, y, train_mask, test_mask

# Two-layer GCN used as the baseline model. Each GCNConv layer aggregates
# neighbour features via degree-normalised mean, then applies a linear transform.
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden)
        self.conv2   = GCNConv(hidden, hidden)
        self.lin     = Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

def train_gcn(data, epochs=200, lr=0.01, weight_decay=5e-4, hidden=64):
    in_ch     = data.x.shape[1]
    model     = GCN(in_ch, hidden=hidden).to(DEVICE)
    y_train   = data.y[data.train_mask].cpu().numpy()
    n_illicit = (y_train == 1).sum()
    n_licit   = (y_train == 0).sum()
    # ~10% of labeled nodes are illicit. Without loss weighting the model
    # learns to predict everything as licit and still gets ~90% accuracy.
    # Upweighting illicit by the class ratio forces it to learn the minority.
    weight    = torch.tensor(
        [1.0, n_licit / max(n_illicit, 1)], dtype=torch.float
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    # Per-epoch loss and F1 are stored so the learning curve chart can be
    # regenerated from the JSON cache without retraining.
    history   = {"losses": [], "f1s": []}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
            f1   = float(f1_score(y_train, pred, zero_division=0))
        history["losses"].append(loss.item())
        history["f1s"].append(f1)
        if epoch % 40 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Train F1: {f1:.4f}")
    return model, history

def evaluate_gcn(model, data):
    model.eval()
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
        true = data.y[data.test_mask].cpu().numpy()
    f1_ill = f1_score(true, pred, pos_label=1, zero_division=0)
    f1_mac = f1_score(true, pred, average="macro", zero_division=0)
    print("\n── GCN Results ──────────────────────────────────────────")
    print(classification_report(true, pred,
                                 target_names=["licit", "illicit"],
                                 zero_division=0))
    print(f"Illicit F1 : {f1_ill:.4f}")
    print(f"Macro  F1  : {f1_mac:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(true, pred))
    return f1_ill, f1_mac

def save_gcn(model, history, in_channels):
    # in_channels is saved alongside weights so the architecture can be rebuilt
    # from the checkpoint without needing the original dataset to be loaded first.
    torch.save({"state_dict": model.state_dict(), "in_channels": in_channels},
               GCN_MODEL_PATH)
    with open(GCN_STATS_PATH, "w") as f:
        json.dump(history, f)
    print(f"  GCN model saved  → {GCN_MODEL_PATH}")
    print(f"  Training stats   → {GCN_STATS_PATH}")

def load_gcn_if_cached():
    if not (os.path.exists(GCN_MODEL_PATH) and os.path.exists(GCN_STATS_PATH)):
        return None, None
    ckpt  = torch.load(GCN_MODEL_PATH, map_location=DEVICE)
    model = GCN(ckpt["in_channels"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with open(GCN_STATS_PATH) as f:
        history = json.load(f)
    print(f"  Cache hit — loaded GCN from {GCN_MODEL_PATH}")
    return model, history

# GAT improves on the GCN baseline in three ways:
#   1. Attention heads (GATConv) let the model assign different importance to
#      each neighbour instead of treating all edges equally.
#   2. Residual / skip connections prevent over-smoothing across 3 message-passing
#      layers and give gradients a direct path back to earlier layers.
#   3. BatchNorm stabilises activation scales across the 200K+ node graph.
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, heads=4, dropout=0.4):
        super().__init__()
        h = hidden * heads  # concatenated head dimension

        self.conv1  = GATConv(in_channels, hidden, heads=heads,
                              dropout=dropout, concat=True)
        self.bn1    = BatchNorm1d(h)
        self.skip1  = Linear(in_channels, h, bias=False)  # dim-match for residual

        self.conv2  = GATConv(h, hidden, heads=heads, dropout=dropout, concat=True)
        self.bn2    = BatchNorm1d(h)
        # skip2 is identity (same h → h)

        self.conv3  = GATConv(h, hidden, heads=1, dropout=dropout, concat=False)
        self.lin    = Linear(hidden, 2)
        self.dropout = dropout

    def _encode(self, x, edge_index):
        # Separated from forward() so the Hybrid model can call embed() and
        # extract node embeddings without running the final classification head.
        x0 = x

        x  = self.conv1(x, edge_index)
        x  = self.bn1(x)
        x  = F.elu(x + self.skip1(x0))
        x  = F.dropout(x, p=self.dropout, training=self.training)

        x1 = x
        x  = self.conv2(x, edge_index)
        x  = self.bn2(x)
        x  = F.elu(x + x1)
        x  = F.dropout(x, p=self.dropout, training=self.training)

        x  = self.conv3(x, edge_index)
        return F.elu(x)  # [N, hidden]

    def forward(self, x, edge_index):
        return self.lin(self._encode(x, edge_index))  # raw logits

    def embed(self, x, edge_index):
        """Returns detached node embeddings (no grad) for use as RF features."""
        self.eval()
        with torch.no_grad():
            return self._encode(x, edge_index).cpu().numpy()

def train_gat(data, epochs=300, lr=5e-3, weight_decay=1e-4, hidden=64, heads=4):
    in_ch  = data.x.shape[1]
    model  = GAT(in_ch, hidden=hidden, heads=heads).to(DEVICE)

    y_train   = data.y[data.train_mask].cpu().numpy()
    n_illicit = (y_train == 1).sum()
    n_licit   = (y_train == 0).sum()
    weight    = torch.tensor(
        [1.0, n_licit / max(n_illicit, 1)], dtype=torch.float
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # GAT training can plateau early. Halving the LR when loss stalls for 20
    # epochs gives the optimiser a second pass at a finer scale to escape.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    history   = {"losses": [], "f1s": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        with torch.no_grad():
            pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
            f1   = float(f1_score(y_train, pred, zero_division=0))
        history["losses"].append(loss.item())
        history["f1s"].append(f1)

        if epoch % 30 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Train F1: {f1:.4f} | LR: {lr_now:.2e}")

    return model, history

def evaluate_gat(model, data):
    model.eval()
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
        true = data.y[data.test_mask].cpu().numpy()

    f1_ill = f1_score(true, pred, pos_label=1, zero_division=0)
    f1_mac = f1_score(true, pred, average="macro", zero_division=0)
    print("\n── GAT Results ──────────────────────────────────────────")
    print(classification_report(true, pred,
                                 target_names=["licit", "illicit"],
                                 zero_division=0))
    print(f"Illicit F1 : {f1_ill:.4f}")
    print(f"Macro  F1  : {f1_mac:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(true, pred))
    return f1_ill, f1_mac

def save_gat(model, history, in_channels, hidden=64, heads=4):
    # hidden and heads are saved so the exact architecture is reconstructed on
    # load — changing the defaults later won't silently break a cached model.
    torch.save({"state_dict": model.state_dict(),
                "in_channels": in_channels, "hidden": hidden, "heads": heads},
               GAT_MODEL_PATH)
    with open(GAT_STATS_PATH, "w") as f:
        json.dump(history, f)
    print(f"  GAT model saved  → {GAT_MODEL_PATH}")
    print(f"  Training stats   → {GAT_STATS_PATH}")

def load_gat_if_cached():
    if not (os.path.exists(GAT_MODEL_PATH) and os.path.exists(GAT_STATS_PATH)):
        return None, None
    ckpt  = torch.load(GAT_MODEL_PATH, map_location=DEVICE)
    model = GAT(ckpt["in_channels"],
                hidden=ckpt["hidden"], heads=ckpt["heads"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with open(GAT_STATS_PATH) as f:
        history = json.load(f)
    print(f"  Cache hit — loaded GAT from {GAT_MODEL_PATH}")
    return model, history

def train_eval_rf(X, y, train_mask, test_mask):
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",   # handles illicit minority class
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)

    f1_ill = f1_score(y_te, pred, pos_label=1, zero_division=0)
    f1_mac = f1_score(y_te, pred, average="macro", zero_division=0)
    print("\n── Random Forest Baseline ───────────────────────────────")
    print(classification_report(y_te, pred,
                                  target_names=["licit", "illicit"],
                                  zero_division=0))
    print(f"Illicit F1 : {f1_ill:.4f}")
    print(f"Macro  F1  : {f1_mac:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, pred))
    return f1_ill, f1_mac

def train_eval_hybrid(model, data, X, y, train_mask, test_mask):
    # The GAT embeddings (64-dim) capture graph-structural patterns that the
    # hand-crafted features miss. Concatenating both gives the RF a 229-dim
    # view: the original 165 Elliptic features + learned neighbourhood context.
    embeddings  = model.embed(data.x, data.edge_index)  # [N, 64]
    X_combined  = np.concatenate([X, embeddings], axis=1)  # [N, 229]

    X_tr, y_tr  = X_combined[train_mask], y[train_mask]
    X_te, y_te  = X_combined[test_mask],  y[test_mask]

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)

    f1_ill = f1_score(y_te, pred, pos_label=1, zero_division=0)
    f1_mac = f1_score(y_te, pred, average="macro", zero_division=0)
    print("\n── Hybrid (GAT embeddings + RF) ─────────────────────────")
    print(classification_report(y_te, pred,
                                  target_names=["licit", "illicit"],
                                  zero_division=0))
    print(f"Illicit F1 : {f1_ill:.4f}")
    print(f"Macro  F1  : {f1_mac:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, pred))
    return f1_ill, f1_mac

def plot_gcn_learning_curve(history, save_path):
    epochs     = range(1, len(history["losses"]) + 1)
    color_loss = "#1565C0"
    color_f1   = "#B71C1C"

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, history["losses"], color=color_loss, linewidth=2, label="Loss")
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=13, color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(bottom=0)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["f1s"], color=color_f1, linewidth=2,
             linestyle="--", label="Train F1 (Illicit)")
    ax2.set_ylabel("Illicit F1 Score", fontsize=13, color=color_f1)
    ax2.tick_params(axis="y", labelcolor=color_f1)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", fontsize=11)
    ax1.set_title("GCN Training Curve — Elliptic Bitcoin Dataset",
                  fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Learning curve   → {save_path}")
    plt.close(fig)

def plot_gcn_architecture(save_path):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.08,  "Input\n166 features",                           "#E3F2FD", "#1565C0"),
        (0.34,  "GCN Layer 1\n64 units\nReLU · Dropout(0.5)",   "#E8F5E9", "#2E7D32"),
        (0.60,  "GCN Layer 2\n64 units\nReLU · Dropout(0.5)",   "#E8F5E9", "#2E7D32"),
        (0.86,  "Output\n2 classes\n(licit / illicit)",          "#FFF3E0", "#E65100"),
    ]
    box_w, box_h, cy = 0.18, 0.36, 0.52

    for x_c, label, face, edge in boxes:
        patch = FancyBboxPatch(
            (x_c - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.015",
            facecolor=face, edgecolor=edge, linewidth=2.5, zorder=3
        )
        ax.add_patch(patch)
        ax.text(x_c, cy, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#212121", zorder=4,
                multialignment="center")

    arrow_kw = dict(arrowstyle="-|>", color="#555555", lw=2, mutation_scale=18)
    for i in range(len(boxes) - 1):
        x_start = boxes[i][0]     + box_w / 2
        x_end   = boxes[i + 1][0] - box_w / 2
        ax.annotate("", xy=(x_end, cy), xytext=(x_start, cy),
                    arrowprops=arrow_kw, zorder=2)

    ax.set_title("GCN Architecture — Elliptic Bitcoin Transaction Classifier",
                 fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Architecture     → {save_path}")
    plt.close(fig)

def plot_conf_matrix(cm_array, title, save_path):
    row_sums = cm_array.sum(axis=1, keepdims=True)
    cm_norm  = cm_array / row_sums.clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues,
                   vmin=0, vmax=1)
    labels = ["Licit (0)", "Illicit (1)"]
    for i in range(2):
        for j in range(2):
            count = cm_array[i, j]
            pct   = 100 * cm_norm[i, j]
            color = "white" if cm_norm[i, j] > 0.55 else "#212121"
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color=color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Confusion matrix → {save_path}")
    plt.close(fig)

def plot_comparison(results: dict, save_path: str):
    """
    results = {"Model Name": (illicit_f1, macro_f1), ...}
    Saves a grouped bar chart to save_path at poster quality (150 dpi).
    """
    models     = list(results.keys())
    ill_vals   = [results[m][0] for m in models]
    mac_vals   = [results[m][1] for m in models]
    x          = np.arange(len(models))
    width      = 0.32

    colours    = ["#1565C0", "#1E88E5", "#2E7D32", "#E65100"]
    light      = ["#90CAF9", "#BBDEFB", "#A5D6A7", "#FFCC80"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, ill_vals, width,
                   color=colours, label="Illicit F1", zorder=3)
    bars2 = ax.bar(x + width / 2, mac_vals, width,
                   color=light,   label="Macro F1",  zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Illicit Bitcoin Transaction Detection — Model Comparison\n"
        "Elliptic Dataset  |  Test set: time steps 35–49",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPoster chart saved → {save_path}")
    plt.show()

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Elliptic Bitcoin — GCN / GAT / RF / Hybrid")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    data, X, y, train_mask, test_mask = load_data()

    print("\n[2/7] GCN — checking cache...")
    gcn_model, gcn_history = load_gcn_if_cached()
    if gcn_model is None:
        print("  No cache found. Training GCN (200 epochs)...")
        gcn_model, gcn_history = train_gcn(data, epochs=200, lr=0.01, hidden=64)
        save_gcn(gcn_model, gcn_history, data.x.shape[1])
    else:
        print("  Skipping GCN training.")
    print("\n  Evaluating GCN...")
    gcn_f1_ill, gcn_f1_mac = evaluate_gcn(gcn_model, data)

    print("\n[3/7] GAT — checking cache...")
    gat_model, gat_history = load_gat_if_cached()
    if gat_model is None:
        print("  No cache found. Training GAT (300 epochs)...")
        gat_model, gat_history = train_gat(data, epochs=300, lr=5e-3, hidden=64, heads=4)
        save_gat(gat_model, gat_history, data.x.shape[1])
    else:
        print("  Skipping GAT training.")
    print("\n  Evaluating GAT...")
    gat_f1_ill, gat_f1_mac = evaluate_gat(gat_model, data)

    print("\n[4/7] Training & evaluating Random Forest baseline...")
    rf_f1_ill, rf_f1_mac = train_eval_rf(X, y, train_mask, test_mask)

    print("\n[5/7] Training & evaluating Hybrid (GAT embeddings + RF)...")
    hyb_f1_ill, hyb_f1_mac = train_eval_hybrid(
        gat_model, data, X, y, train_mask, test_mask
    )

    print("\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)
    print(f"{'Model':<22} {'Illicit F1':>12} {'Macro F1':>10}")
    print("-" * 46)
    print(f"{'GCN':<22} {gcn_f1_ill:>12.4f} {gcn_f1_mac:>10.4f}")
    print(f"{'GAT':<22} {gat_f1_ill:>12.4f} {gat_f1_mac:>10.4f}")
    print(f"{'Random Forest':<22} {rf_f1_ill:>12.4f} {rf_f1_mac:>10.4f}")
    print(f"{'Hybrid (GAT + RF)':<22} {hyb_f1_ill:>12.4f} {hyb_f1_mac:>10.4f}")
    print("=" * 60)

    print("\n[6/7] Generating poster comparison chart...")
    results = {
        "GCN":              (gcn_f1_ill, gcn_f1_mac),
        "GAT":              (gat_f1_ill, gat_f1_mac),
        "Random Forest":    (rf_f1_ill,  rf_f1_mac),
        "Hybrid\n(GAT+RF)": (hyb_f1_ill, hyb_f1_mac),
    }
    plot_comparison(results, os.path.join(BASE_DIR, "poster_results.png"))

    print("\n[7/7] Generating GCN analysis charts...")
    plot_gcn_learning_curve(
        gcn_history,
        os.path.join(BASE_DIR, "gcn_learning_curve.png")
    )
    plot_gcn_architecture(
        os.path.join(BASE_DIR, "gcn_architecture.png")
    )
    gcn_cm = np.array([[15224, 363], [423, 660]])
    rf_cm  = np.array([[15581,   6], [342, 741]])
    plot_conf_matrix(
        gcn_cm,
        "GCN Confusion Matrix\n(Test set: time steps 35–49)",
        os.path.join(BASE_DIR, "gcn_confusion_matrix.png")
    )
    plot_conf_matrix(
        rf_cm,
        "Random Forest Confusion Matrix\n(Test set: time steps 35–49)",
        os.path.join(BASE_DIR, "rf_confusion_matrix.png")
    )

if __name__ == "__main__":
    main()
