"""
BitcoinHeist Dataset — GCN / GAT / Random Forest / Hybrid Pipeline
Detects ransomware wallets in the Bitcoin transaction graph.

Dataset: BitcoinHeistData.csv
  address, year, day, length, weight, count, looped, neighbors, income, label

Graph construction:
  Nodes  — one per unique wallet address (features aggregated across appearances)
  Edges  — wallets that share the same (year, day) are chain-connected, giving
           average degree ≈ 2.2 which matches the `neighbors` column mean.
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
HEIST_PATH     = os.path.join(BASE_DIR, "BitcoinHeistData.csv")
GCN_MODEL_PATH = os.path.join(BASE_DIR, "bh_gcn_model.pth")
GCN_STATS_PATH = os.path.join(BASE_DIR, "bh_gcn_stats.json")
GAT_MODEL_PATH = os.path.join(BASE_DIR, "bh_gat_model.pth")
GAT_STATS_PATH = os.path.join(BASE_DIR, "bh_gat_stats.json")

FEAT_COLS = ["length", "weight", "count", "looped", "neighbors", "income"]
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Full dataset is 2.6M nodes. GAT intermediate tensors (2.6M × 256 floats per
# layer) exceed laptop RAM. Cap at 250K: keep all illicit wallets + sample licit.
MAX_NODES = 250_000

def load_data():
    df = pd.read_csv(HEIST_PATH)

    # Binary label: white → 0 (licit), any ransomware family → 1 (illicit)
    df["label_bin"] = (df["label"] != "white").astype(int)

    # Each address can appear across multiple (year, day) pairs. We collapse
    # to one node per wallet: mean features across all appearances (smooths out
    # day-to-day noise), max label (a wallet that ever sent ransomware stays
    # flagged — even one illicit transaction is enough to label it illicit).
    feat_agg  = df.groupby("address")[FEAT_COLS].mean()
    label_agg = df.groupby("address")["label_bin"].max().rename("label")
    node_df   = feat_agg.join(label_agg).reset_index()

    # The full 2.6M-node graph exceeds laptop RAM during GAT's backward pass
    # (~2.7 GB per layer at hidden=256). We keep every illicit wallet — dropping
    # any would further shrink the already rare minority class — and randomly
    # sample licit wallets to stay within MAX_NODES.
    illicit_df   = node_df[node_df["label"] == 1]
    licit_df     = node_df[node_df["label"] == 0]
    n_licit_keep = min(len(licit_df), MAX_NODES - len(illicit_df))
    licit_sample = licit_df.sample(n=n_licit_keep, random_state=17)
    node_df = (pd.concat([illicit_df, licit_sample])
               .sample(frac=1, random_state=17)
               .reset_index(drop=True))

    n_nodes     = len(node_df)
    addr_to_idx = {addr: i for i, addr in enumerate(node_df["address"])}

    X = node_df[FEAT_COLS].values.astype(np.float32)
    # income spans many orders of magnitude; scaling prevents it from
    # dominating the GCN neighbour aggregation step.
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = node_df["label"].values  # 0 = licit, 1 = illicit

    # For each (year, day) group, sort unique addresses and connect them in a
    # chain: addr[0]─addr[1]─addr[2]─… This is O(total_rows) to build and
    # yields average node degree ≈ 2.2, matching the `neighbors` column mean.
    print("  Building co-occurrence graph (chain edges per day)...")
    addr_day = (
        df[["address", "year", "day"]]
        .drop_duplicates()
        .sort_values(["year", "day"])
        .reset_index(drop=True)
    )
    addr_day["node_idx"] = addr_day["address"].map(addr_to_idx)
    # Drop addresses not in the retained subsample before chaining
    addr_day = (addr_day.dropna(subset=["node_idx"])
                .assign(node_idx=lambda d: d["node_idx"].astype(int))
                .reset_index(drop=True))

    same_grp = (
        (addr_day["year"] == addr_day["year"].shift()) &
        (addr_day["day"]  == addr_day["day"].shift())
    ).fillna(False)

    left_idx  = (addr_day["node_idx"].shift()
                 .where(same_grp).dropna().astype(int).values)
    right_idx = (addr_day["node_idx"]
                 .where(same_grp).dropna().astype(int).values)

    src_arr    = np.concatenate([left_idx,  right_idx])
    dst_arr    = np.concatenate([right_idx, left_idx])
    edge_index = torch.tensor(np.stack([src_arr, dst_arr]), dtype=torch.long)

    # Stratified split keeps the ~1.5% illicit ratio consistent across both sets.
    # Unlike Elliptic there is no clean temporal boundary to split on, so random
    # stratified sampling is the standard choice for this dataset.
    indices    = np.arange(n_nodes)
    tr_idx, te_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=17
    )
    train_mask            = np.zeros(n_nodes, dtype=bool)
    test_mask             = np.zeros(n_nodes, dtype=bool)
    train_mask[tr_idx]    = True
    test_mask[te_idx]     = True

    print(f"Nodes        : {n_nodes:,}")
    print(f"Edges (uni)  : {len(left_idx):,}")
    print(f"Train        : {train_mask.sum():,}  "
          f"(illicit={int(y[train_mask].sum()):,})")
    print(f"Test         : {test_mask.sum():,}  "
          f"(illicit={int(y[test_mask].sum()):,})")

    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y, dtype=torch.long),
        train_mask = torch.tensor(train_mask),
        test_mask  = torch.tensor(test_mask),
    ).to(DEVICE)

    return data, X, y, train_mask, test_mask

# Two-layer GCN baseline. Each GCNConv aggregates neighbour features via
# degree-normalised mean then applies a linear transform — fast and simple.
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

def train_gcn(data, epochs=100, lr=0.01, weight_decay=5e-4, hidden=64):
    in_ch     = data.x.shape[1]
    model     = GCN(in_ch, hidden=hidden).to(DEVICE)
    y_train   = data.y[data.train_mask].cpu().numpy()
    n_illicit = (y_train == 1).sum()
    n_licit   = (y_train == 0).sum()
    # ~1.5% of wallets are illicit. The weight ratio (≈65×) stops the model
    # from collapsing to "predict everything licit" and still getting 98.5% acc.
    weight    = torch.tensor(
        [1.0, n_licit / max(n_illicit, 1)], dtype=torch.float
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
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
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Train F1: {f1:.4f}")
    return model, history

def evaluate_gcn(model, data):
    model.eval()
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
        true = data.y[data.test_mask].cpu().numpy()
    f1_ill = f1_score(true, pred, pos_label=1, zero_division=0)
    f1_mac = f1_score(true, pred, average="macro", zero_division=0)
    cm     = confusion_matrix(true, pred, labels=[0, 1])
    print("\n── GCN Results ──────────────────────────────────────────")
    print(classification_report(true, pred,
                                 target_names=["licit", "illicit"],
                                 zero_division=0))
    print(f"Illicit F1 : {f1_ill:.4f}")
    print(f"Macro  F1  : {f1_mac:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    return f1_ill, f1_mac, cm

def save_gcn(model, history, in_channels):
    torch.save({"state_dict": model.state_dict(),
                "in_channels": in_channels}, GCN_MODEL_PATH)
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

# GAT with 3 message-passing layers, attention heads, residual connections,
# and BatchNorm. See elliptic_gnn_pipeline.py for the full design rationale.
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, heads=4, dropout=0.4):
        super().__init__()
        h = hidden * heads

        self.conv1 = GATConv(in_channels, hidden, heads=heads,
                             dropout=dropout, concat=True)
        self.bn1   = BatchNorm1d(h)
        self.skip1 = Linear(in_channels, h, bias=False)

        self.conv2 = GATConv(h, hidden, heads=heads,
                             dropout=dropout, concat=True)
        self.bn2   = BatchNorm1d(h)

        self.conv3   = GATConv(h, hidden, heads=1, dropout=dropout, concat=False)
        self.lin     = Linear(hidden, 2)
        self.dropout = dropout

    def _encode(self, x, edge_index):
        x0 = x
        x  = F.elu(self.bn1(self.conv1(x, edge_index)) + self.skip1(x0))
        x  = F.dropout(x, p=self.dropout, training=self.training)
        x1 = x
        x  = F.elu(self.bn2(self.conv2(x, edge_index)) + x1)
        x  = F.dropout(x, p=self.dropout, training=self.training)
        return F.elu(self.conv3(x, edge_index))

    def forward(self, x, edge_index):
        return self.lin(self._encode(x, edge_index))

    def embed(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            return self._encode(x, edge_index).cpu().numpy()

def train_gat(data, epochs=150, lr=5e-3, weight_decay=1e-4, hidden=64, heads=4):
    in_ch     = data.x.shape[1]
    model     = GAT(in_ch, hidden=hidden, heads=heads).to(DEVICE)
    y_train   = data.y[data.train_mask].cpu().numpy()
    n_illicit = (y_train == 1).sum()
    n_licit   = (y_train == 0).sum()
    weight    = torch.tensor(
        [1.0, n_licit / max(n_illicit, 1)], dtype=torch.float
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
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
    return f1_ill, f1_mac

def save_gat(model, history, in_channels, hidden=64, heads=4):
    torch.save({"state_dict": model.state_dict(),
                "in_channels": in_channels,
                "hidden": hidden, "heads": heads}, GAT_MODEL_PATH)
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
        class_weight="balanced",
        n_jobs=-1,
        random_state=17,
    )
    rf.fit(X_tr, y_tr)
    pred   = rf.predict(X_te)
    f1_ill = f1_score(y_te, pred, pos_label=1, zero_division=0)
    f1_mac = f1_score(y_te, pred, average="macro", zero_division=0)
    cm     = confusion_matrix(y_te, pred, labels=[0, 1])
    print("\n── Random Forest Baseline ───────────────────────────────")
    print(classification_report(y_te, pred,
                                 target_names=["licit", "illicit"],
                                 zero_division=0))
    print(f"Illicit F1 : {f1_ill:.4f}")
    print(f"Macro  F1  : {f1_mac:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    return f1_ill, f1_mac, cm

def train_eval_hybrid(model, data, X, y, train_mask, test_mask):
    # GAT embeddings (64-dim) capture graph-structural patterns; concatenating
    # with the 6 original features gives the RF a 70-dim combined view that
    # covers what each model misses on its own.
    embeddings = model.embed(data.x, data.edge_index)   # [N, 64]
    X_combined = np.concatenate([X, embeddings], axis=1) # [N, 70]
    X_tr, y_tr = X_combined[train_mask], y[train_mask]
    X_te, y_te = X_combined[test_mask],  y[test_mask]

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=17,
    )
    rf.fit(X_tr, y_tr)
    pred   = rf.predict(X_te)
    f1_ill = f1_score(y_te, pred, pos_label=1, zero_division=0)
    f1_mac = f1_score(y_te, pred, average="macro", zero_division=0)
    print("\n── Hybrid (GAT embeddings + RF) ─────────────────────────")
    print(classification_report(y_te, pred,
                                 target_names=["licit", "illicit"],
                                 zero_division=0))
    print(f"Illicit F1 : {f1_ill:.4f}")
    print(f"Macro  F1  : {f1_mac:.4f}")
    return f1_ill, f1_mac

def plot_comparison(results: dict, save_path: str):
    models   = list(results.keys())
    ill_vals = [results[m][0] for m in models]
    mac_vals = [results[m][1] for m in models]
    x        = np.arange(len(models))
    width    = 0.32

    colours = ["#1565C0", "#1E88E5", "#2E7D32", "#E65100"]
    light   = ["#90CAF9", "#BBDEFB", "#A5D6A7", "#FFCC80"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, ill_vals, width,
                   color=colours, label="Illicit F1", zorder=3)
    bars2 = ax.bar(x + width / 2, mac_vals, width,
                   color=light,   label="Macro F1",   zorder=3)

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
        "Ransomware Wallet Detection — Model Comparison\n"
        "BitcoinHeist Dataset  |  Stratified 80/20 split",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Model comparison → {save_path}")
    plt.close(fig)

def plot_gcn_learning_curve(history, save_path: str):
    epochs     = range(1, len(history["losses"]) + 1)
    color_loss = "#1565C0"
    color_f1   = "#B71C1C"

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, history["losses"], color=color_loss, linewidth=2,
             label="Loss")
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
    ax1.set_title("GCN Training Curve — BitcoinHeist Dataset",
                  fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Learning curve   → {save_path}")
    plt.close(fig)

def plot_gcn_architecture(save_path: str):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.08,  "Input\n6 features",                             "#E3F2FD", "#1565C0"),
        (0.34,  "GCN Layer 1\n64 units\nReLU · Dropout(0.5)",   "#E8F5E9", "#2E7D32"),
        (0.60,  "GCN Layer 2\n64 units\nReLU · Dropout(0.5)",   "#E8F5E9", "#2E7D32"),
        (0.86,  "Output\n2 classes\n(licit / illicit)",          "#FFF3E0", "#E65100"),
    ]
    box_w, box_h, cy = 0.18, 0.36, 0.52

    for x_c, label, face, edge in boxes:
        ax.add_patch(FancyBboxPatch(
            (x_c - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.015",
            facecolor=face, edgecolor=edge, linewidth=2.5, zorder=3
        ))
        ax.text(x_c, cy, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#212121", zorder=4,
                multialignment="center")

    arrow_kw = dict(arrowstyle="-|>", color="#555555", lw=2, mutation_scale=18)
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i+1][0] - box_w/2, cy),
                    xytext=(boxes[i][0] + box_w/2, cy),
                    arrowprops=arrow_kw, zorder=2)

    ax.set_title("GCN Architecture — BitcoinHeist Ransomware Wallet Classifier",
                 fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Architecture     → {save_path}")
    plt.close(fig)

def plot_conf_matrix(cm_array, title: str, save_path: str):
    row_sums = cm_array.sum(axis=1, keepdims=True)
    cm_norm  = cm_array / row_sums.clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_norm, interpolation="nearest",
                   cmap=plt.cm.Blues, vmin=0, vmax=1)
    labels = ["Licit (0)", "Illicit (1)"]
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.55 else "#212121"
            ax.text(j, i,
                    f"{cm_array[i, j]:,}\n({100 * cm_norm[i, j]:.1f}%)",
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

def main():
    torch.manual_seed(17)
    np.random.seed(17)

    print("=" * 60)
    print("BitcoinHeist — GCN / GAT / RF / Hybrid")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    data, X, y, train_mask, test_mask = load_data()

    print("\n[2/7] GCN — checking cache...")
    gcn_model, gcn_history = load_gcn_if_cached()
    if gcn_model is None:
        print("  No cache found. Training GCN (100 epochs)...")
        gcn_model, gcn_history = train_gcn(data, epochs=100, lr=0.01, hidden=64)
        save_gcn(gcn_model, gcn_history, data.x.shape[1])
    else:
        print("  Skipping GCN training.")
    print("\n  Evaluating GCN...")
    gcn_f1_ill, gcn_f1_mac, gcn_cm = evaluate_gcn(gcn_model, data)

    print("\n[3/7] GAT — checking cache...")
    gat_model, gat_history = load_gat_if_cached()
    if gat_model is None:
        print("  No cache found. Training GAT (150 epochs)...")
        gat_model, gat_history = train_gat(data, epochs=150, lr=5e-3,
                                            hidden=64, heads=4)
        save_gat(gat_model, gat_history, data.x.shape[1])
    else:
        print("  Skipping GAT training.")
    print("\n  Evaluating GAT...")
    gat_f1_ill, gat_f1_mac = evaluate_gat(gat_model, data)

    print("\n[4/7] Training & evaluating Random Forest baseline...")
    rf_f1_ill, rf_f1_mac, rf_cm = train_eval_rf(X, y, train_mask, test_mask)

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

    print("\n[6/7] Generating model comparison chart...")
    plot_comparison(
        {
            "GCN":              (gcn_f1_ill, gcn_f1_mac),
            "GAT":              (gat_f1_ill, gat_f1_mac),
            "Random Forest":    (rf_f1_ill,  rf_f1_mac),
            "Hybrid\n(GAT+RF)": (hyb_f1_ill, hyb_f1_mac),
        },
        os.path.join(BASE_DIR, "bitcoinheist_model_comparison.png"),
    )

    print("\n[7/7] Generating GCN analysis charts...")
    plot_gcn_learning_curve(
        gcn_history,
        os.path.join(BASE_DIR, "bitcoinheist_gcn_learning_curve.png"),
    )
    plot_gcn_architecture(
        os.path.join(BASE_DIR, "bitcoinheist_gcn_architecture.png"),
    )
    plot_conf_matrix(
        gcn_cm,
        "GCN Confusion Matrix\nBitcoinHeist Dataset",
        os.path.join(BASE_DIR, "bitcoinheist_gcn_confusion_matrix.png"),
    )
    plot_conf_matrix(
        rf_cm,
        "Random Forest Confusion Matrix\nBitcoinHeist Dataset",
        os.path.join(BASE_DIR, "bitcoinheist_rf_confusion_matrix.png"),
    )

if __name__ == "__main__":
    main()
