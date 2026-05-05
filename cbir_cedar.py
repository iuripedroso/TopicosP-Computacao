"""
CBIR System - Content-Based Image Retrieval
Dataset: CEDAR (Signatures/Handwriting)
Authors: [Adicionar nomes]

Pipeline:
1. Gerar dataset sintético simulando CEDAR (assinaturas binárias)
2. Selective Search / Sliding Window para propostas de regiões
3. Extração de descritores (HOG + momentos de Hu)
4. Indexação por KD-Tree / BallTree
5. Ranking por similaridade visual + IoU espacial
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.draw import bezier_curve
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GERAÇÃO DO DATASET SINTÉTICO (simula CEDAR)
# ─────────────────────────────────────────────

np.random.seed(42)

IMG_H, IMG_W = 128, 256
N_DOCS   = 30   # documentos no índice
N_QUERIES = 5   # queries

CLASSES = ['assinatura_A', 'assinatura_B', 'assinatura_C',
           'numeral_1',    'numeral_2']

def draw_stroke(img, x0, y0, length=40, thickness=2, seed=0):
    rng = np.random.RandomState(seed)
    pts = [(x0, y0)]
    x, y = x0, y0
    for _ in range(length // 5):
        dx = rng.randint(-6, 7)
        dy = rng.randint(-4, 5)
        x = np.clip(x + dx, 2, img.shape[1] - 3)
        y = np.clip(y + dy, 2, img.shape[0] - 3)
        pts.append((x, y))
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, 0, thickness)
    return img

def generate_signature(class_idx, variation=0, size=(128, 256)):
    """Gera uma assinatura/numeral sintético com variações controladas."""
    img = np.ones(size, dtype=np.uint8) * 255
    rng = np.random.RandomState(class_idx * 100 + variation)

    if class_idx == 0:   # assinatura_A: loop + traço
        cx, cy = size[1]//3 + rng.randint(-10, 10), size[0]//2 + rng.randint(-5, 5)
        axes = (30 + rng.randint(-5, 5), 18 + rng.randint(-3, 3))
        cv2.ellipse(img, (cx, cy), axes, rng.randint(0, 30), 0, 360, 0, 2)
        draw_stroke(img, cx + axes[0], cy, length=60, seed=variation)

    elif class_idx == 1:  # assinatura_B: duas curvas paralelas
        for i in range(2):
            y_off = size[0]//3 + i * 20 + rng.randint(-4, 4)
            pts = np.array([[20 + j*10 + rng.randint(-3, 3),
                             y_off + int(15*np.sin(j*0.6 + variation*0.2))]
                            for j in range(20)], dtype=np.int32)
            cv2.polylines(img, [pts], False, 0, 2)

    elif class_idx == 2:  # assinatura_C: zigue-zague
        y = size[0]//2 + rng.randint(-8, 8)
        for j in range(0, size[1]-20, 15):
            dy = 20 * (1 if j % 30 == 0 else -1) + rng.randint(-3, 3)
            cv2.line(img, (j+10, y), (j+25, y+dy), 0, 2)
            y = np.clip(y + dy, 10, size[0]-10)

    elif class_idx == 3:  # numeral_1: linha vertical + base
        cx = size[1]//2 + rng.randint(-10, 10)
        cv2.line(img, (cx, 20+rng.randint(-3,3)), (cx, 100+rng.randint(-3,3)), 0, 3)
        cv2.line(img, (cx-15, 100), (cx+15, 100), 0, 2)

    elif class_idx == 4:  # numeral_2: curva + base
        cx, cy = size[1]//2 + rng.randint(-8,8), size[0]//3
        cv2.ellipse(img, (cx, cy), (25+rng.randint(-3,3), 18), 0, 200, 360, 0, 2)
        cv2.ellipse(img, (cx, cy), (25+rng.randint(-3,3), 18), 0, 0, 160, 0, 2)
        pts = np.array([[cx+20, cy+18],
                        [cx+5+rng.randint(-3,3), cy+40+rng.randint(-5,5)],
                        [cx-20+rng.randint(-3,3), cy+60]], dtype=np.int32)
        cv2.polylines(img, [pts], False, 0, 2)
        cv2.line(img, (cx-25, cy+60), (cx+25, cy+60), 0, 2)

    return img


def build_dataset():
    docs = []
    for i in range(N_DOCS):
        class_idx = i % len(CLASSES)
        variation = i // len(CLASSES)
        img = generate_signature(class_idx, variation)
        docs.append({'id': i, 'class': CLASSES[class_idx],
                     'class_idx': class_idx, 'img': img})

    queries = []
    for qi in range(N_QUERIES):
        class_idx = qi % len(CLASSES)
        # query é uma nova variação da mesma classe
        img = generate_signature(class_idx, variation=50 + qi)
        queries.append({'id': qi, 'class': CLASSES[class_idx],
                        'class_idx': class_idx, 'img': img})
    return docs, queries


# ─────────────────────────────────────────────
# 2. PROPOSTAS DE REGIÕES (Sliding Window)
# ─────────────────────────────────────────────

def sliding_window_proposals(img, scales=None, step=16):
    """
    Gera propostas de regiões via sliding window multi-escala.
    Retorna lista de (x, y, w, h) para regiões com conteúdo.
    """
    if scales is None:
        scales = [(64, 128), (96, 192), (128, 256)]
    h, w = img.shape[:2]
    proposals = []
    for (ww, wh) in scales:
        for y in range(0, h - wh + 1, step):
            for x in range(0, w - ww + 1, step):
                roi = img[y:y+wh, x:x+ww]
                # Mantém apenas regiões com pixels escuros (conteúdo)
                dark_ratio = np.sum(roi < 128) / roi.size
                if dark_ratio > 0.01:
                    proposals.append((x, y, ww, wh))
    return proposals


def nms_proposals(proposals, iou_thresh=0.5):
    """Non-Maximum Suppression sobre propostas."""
    if not proposals:
        return []
    boxes = np.array(proposals, dtype=float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return [proposals[k] for k in keep]


# ─────────────────────────────────────────────
# 3. EXTRAÇÃO DE DESCRITORES
# ─────────────────────────────────────────────

HOG_SIZE = (64, 128)

def extract_hog(roi):
    """HOG descriptor de uma ROI redimensionada."""
    roi_resized = cv2.resize(roi, HOG_SIZE)
    fd = hog(roi_resized, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), feature_vector=True)
    return fd.astype(np.float32)

def extract_hu(roi):
    """Momentos de Hu (7 valores) de uma ROI."""
    _, bw = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
    m = cv2.moments(bw)
    hu = cv2.HuMoments(m).flatten()
    # Log-transform para estabilidade
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.astype(np.float32)

def extract_descriptor(roi):
    """Concatena HOG + Hu Moments."""
    hog_feat = extract_hog(roi)
    hu_feat  = extract_hu(roi)
    return np.concatenate([hog_feat, hu_feat])


def get_best_proposal(img, proposals):
    """Escolhe a proposta com maior área (região mais representativa)."""
    if not proposals:
        h, w = img.shape[:2]
        return (0, 0, w, h)
    areas = [p[2]*p[3] for p in proposals]
    return proposals[int(np.argmax(areas))]


# ─────────────────────────────────────────────
# 4. INDEXAÇÃO
# ─────────────────────────────────────────────

class CBIRIndex:
    def __init__(self, n_pca=64):
        self.n_pca = n_pca
        self.descriptors = []   # raw
        self.proposals   = []   # (x, y, w, h) da melhor região de cada doc
        self.meta        = []   # info do documento
        self.tree        = None
        self.pca         = None
        self._built      = False

    def add(self, doc_id, doc_class, descriptor, proposal):
        self.descriptors.append(descriptor)
        self.proposals.append(proposal)
        self.meta.append({'id': doc_id, 'class': doc_class})

    def build(self):
        X = np.array(self.descriptors, dtype=np.float32)
        X = normalize(X)
        # PCA para redução de dimensionalidade
        n_comp = min(self.n_pca, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=42)
        X_red = self.pca.fit_transform(X)
        X_red = normalize(X_red)
        self.tree = BallTree(X_red, metric='euclidean')
        self._X = X_red
        self._built = True
        print(f"[Index] {len(X)} documentos indexados | dim original={X.shape[1]} | dim PCA={n_comp}")

    def query(self, descriptor, k=10):
        assert self._built
        x = normalize(descriptor.reshape(1, -1))
        x_red = self.pca.transform(x)
        x_red = normalize(x_red)
        dists, idxs = self.tree.query(x_red, k=min(k, len(self.meta)))
        return dists[0], idxs[0]


# ─────────────────────────────────────────────
# 5. RANQUEAMENTO COM IoU ESPACIAL
# ─────────────────────────────────────────────

def compute_iou(boxA, boxB):
    """IoU entre dois bounding boxes (x, y, w, h)."""
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = ax1 + boxA[2], ay1 + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = bx1 + boxB[2], by1 + boxB[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-6)


def rank_results(visual_dists, idxs, query_proposal, index,
                 alpha=0.7, beta=0.3):
    """
    Score final = alpha * visual_sim + beta * iou_espacial
    visual_sim = 1 / (1 + dist)
    """
    results = []
    for dist, idx in zip(visual_dists, idxs):
        visual_sim = 1.0 / (1.0 + dist)
        doc_proposal = index.proposals[idx]
        iou = compute_iou(query_proposal, doc_proposal)
        score = alpha * visual_sim + beta * iou
        results.append({
            'idx':        idx,
            'meta':       index.meta[idx],
            'proposal':   doc_proposal,
            'visual_sim': visual_sim,
            'iou':        iou,
            'score':      score,
        })
    results.sort(key=lambda r: r['score'], reverse=True)
    return results


# ─────────────────────────────────────────────
# 6. AVALIAÇÃO (Precision@K, MAP)
# ─────────────────────────────────────────────

def precision_at_k(results, query_class, k):
    top_k = results[:k]
    hits = sum(1 for r in top_k if r['meta']['class'] == query_class)
    return hits / k

def average_precision(results, query_class):
    hits, ap, n_rel = 0, 0.0, 0
    for i, r in enumerate(results):
        if r['meta']['class'] == query_class:
            hits += 1
            n_rel += 1
            ap += hits / (i + 1)
    return ap / n_rel if n_rel > 0 else 0.0


# ─────────────────────────────────────────────
# 7. VISUALIZAÇÃO
# ─────────────────────────────────────────────

def plot_query_results(query, results, save_path, top_k=5):
    fig = plt.figure(figsize=(18, 4))
    fig.patch.set_facecolor('#F7F9FC')
    gs = GridSpec(1, top_k + 1, figure=fig, wspace=0.05)

    # Query
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(query['img'], cmap='gray', vmin=0, vmax=255)
    qp = query['proposal']
    rect = patches.Rectangle((qp[0], qp[1]), qp[2], qp[3],
                               linewidth=2, edgecolor='#E74C3C', facecolor='none')
    ax0.add_patch(rect)
    ax0.set_title(f"Query\n{query['class']}", fontsize=8, color='#E74C3C', fontweight='bold')
    ax0.axis('off')

    for i, r in enumerate(results[:top_k]):
        ax = fig.add_subplot(gs[0, i + 1])
        doc_img = docs[r['idx']]['img']
        ax.imshow(doc_img, cmap='gray', vmin=0, vmax=255)
        rp = r['proposal']
        color = '#27AE60' if r['meta']['class'] == query['class'] else '#E74C3C'
        rect = patches.Rectangle((rp[0], rp[1]), rp[2], rp[3],
                                   linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.set_title(
            f"#{i+1} {r['meta']['class']}\nScore:{r['score']:.3f} IoU:{r['iou']:.2f}",
            fontsize=7, color=color
        )
        ax.axis('off')

    plt.suptitle(f"CBIR — Query {query['id']} ({query['class']})",
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


def plot_metrics(map_scores, p_at_k_list, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#F7F9FC')

    # MAP por query
    q_labels = [f"Q{i}" for i in range(len(map_scores))]
    bars = axes[0].bar(q_labels, map_scores, color='#3498DB', edgecolor='white', width=0.5)
    axes[0].axhline(np.mean(map_scores), color='#E74C3C', linestyle='--', linewidth=1.5,
                    label=f"MAP={np.mean(map_scores):.3f}")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Average Precision por Query", fontweight='bold')
    axes[0].set_ylabel("AP")
    axes[0].legend()
    axes[0].set_facecolor('#F7F9FC')
    for bar, val in zip(bars, map_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
                     ha='center', fontsize=9)

    # P@K por query
    ks = [1, 3, 5]
    colors = ['#3498DB', '#27AE60', '#E67E22']
    x = np.arange(len(q_labels))
    width = 0.22
    for ki, (k, col) in enumerate(zip(ks, colors)):
        vals = [p_at_k_list[qi][ki] for qi in range(len(q_labels))]
        axes[1].bar(x + ki*width, vals, width, label=f"P@{k}", color=col, edgecolor='white')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(q_labels)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Precision@K por Query", fontweight='bold')
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].set_facecolor('#F7F9FC')

    plt.suptitle("Métricas de Avaliação — CBIR CEDAR", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def plot_dataset_overview(docs, save_path, n=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    fig.patch.set_facecolor('#F7F9FC')
    for i, ax in enumerate(axes.flat):
        if i < n:
            d = docs[i]
            ax.imshow(d['img'], cmap='gray')
            ax.set_title(f"Doc {d['id']}\n{d['class']}", fontsize=8)
        ax.axis('off')
    plt.suptitle("Amostra do Dataset (10 primeiros documentos)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def plot_proposals_example(doc, save_path):
    img = doc['img']
    all_props = sliding_window_proposals(img)
    nms_props  = nms_proposals(all_props, iou_thresh=0.4)
    best_prop  = get_best_proposal(img, nms_props)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#F7F9FC')

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Original — {doc['class']}", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    for p in all_props[:40]:
        rect = patches.Rectangle((p[0], p[1]), p[2], p[3],
                                   linewidth=0.5, edgecolor='#3498DB', facecolor='none', alpha=0.4)
        axes[1].add_patch(rect)
    axes[1].set_title(f"Propostas (sliding window)\n{len(all_props)} total → 40 exibidas", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray')
    for p in nms_props:
        rect = patches.Rectangle((p[0], p[1]), p[2], p[3],
                                   linewidth=1, edgecolor='#27AE60', facecolor='none', alpha=0.6)
        axes[2].add_patch(rect)
    bp = best_prop
    rect = patches.Rectangle((bp[0], bp[1]), bp[2], bp[3],
                               linewidth=2.5, edgecolor='#E74C3C', facecolor='none')
    axes[2].add_patch(rect)
    axes[2].set_title(f"Após NMS ({len(nms_props)} regiões)\nVermelho = melhor proposta", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Geração de Propostas de Regiões Candidatas", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('/home/claude/output', exist_ok=True)
    print("=" * 60)
    print("  CBIR — Content-Based Image Retrieval | CEDAR Dataset")
    print("=" * 60)

    # 1. Dataset
    print("\n[1] Gerando dataset sintético CEDAR...")
    docs, queries = build_dataset()
    print(f"    {len(docs)} documentos | {len(queries)} queries")

    # Visão geral do dataset
    plot_dataset_overview(docs, '/home/claude/output/fig1_dataset_overview.png')
    print("    → fig1_dataset_overview.png salvo")

    # 2. Propostas de regiões
    print("\n[2] Gerando propostas de regiões...")
    for d in docs:
        props = sliding_window_proposals(d['img'])
        props = nms_proposals(props, iou_thresh=0.4)
        d['proposals'] = props
        d['best_proposal'] = get_best_proposal(d['img'], props)

    for q in queries:
        props = sliding_window_proposals(q['img'])
        props = nms_proposals(props, iou_thresh=0.4)
        q['proposals'] = props
        q['proposal'] = get_best_proposal(q['img'], props)

    # Figura de exemplo de propostas
    plot_proposals_example(docs[0], '/home/claude/output/fig2_proposals.png')
    print("    → fig2_proposals.png salvo")

    # 3. Extração de descritores e indexação
    print("\n[3] Extraindo descritores e indexando...")
    index = CBIRIndex(n_pca=64)
    for d in docs:
        x, y, w, h = d['best_proposal']
        roi = d['img'][y:y+h, x:x+w]
        desc = extract_descriptor(roi)
        index.add(d['id'], d['class'], desc, d['best_proposal'])
    index.build()

    # 4. Queries e ranking
    print("\n[4] Executando queries e ranqueando resultados...")
    all_ap, all_p_at_k = [], []
    all_results = []

    for q in queries:
        x, y, w, h = q['proposal']
        roi = q['img'][y:y+h, x:x+w]
        desc = extract_descriptor(roi)

        dists, idxs = index.query(desc, k=N_DOCS)
        results = rank_results(dists, idxs, q['proposal'], index)
        all_results.append(results)

        ap = average_precision(results, q['class'])
        p1 = precision_at_k(results, q['class'], 1)
        p3 = precision_at_k(results, q['class'], 3)
        p5 = precision_at_k(results, q['class'], 5)
        all_ap.append(ap)
        all_p_at_k.append([p1, p3, p5])

        print(f"    Query {q['id']} ({q['class']:15s}) | AP={ap:.3f} | P@1={p1:.2f} P@3={p3:.2f} P@5={p5:.2f}")

        save = f"/home/claude/output/fig3_query_{q['id']}.png"
        plot_query_results(q, results, save)

    print(f"\n    MAP = {np.mean(all_ap):.4f}")

    # 5. Métricas
    plot_metrics(all_ap, all_p_at_k, '/home/claude/output/fig4_metrics.png')
    print("\n    → fig4_metrics.png salvo")

    # Exportar sumário
    with open('/home/claude/output/summary.txt', 'w') as f:
        f.write("CBIR CEDAR — Sumário de Avaliação\n")
        f.write("=" * 40 + "\n")
        for i, (q, ap, pks) in enumerate(zip(queries, all_ap, all_p_at_k)):
            f.write(f"Query {i} ({q['class']}): AP={ap:.3f} P@1={pks[0]:.2f} P@3={pks[1]:.2f} P@5={pks[2]:.2f}\n")
        f.write(f"\nMAP = {np.mean(all_ap):.4f}\n")

    print("\n[✓] Pipeline completo!")
