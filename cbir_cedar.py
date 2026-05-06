"""
CBIR System - Content-Based Image Retrieval
Dataset: CEDAR (assinaturas reais)
Pastas esperadas:
    signatures/
        full_org/   -> assinaturas originais
        full_forg/  -> assinaturas falsificadas

Como rodar:
    python cbir_cedar_real.py --dataset "C:/caminho/para/signatures"
"""

import os
import sys
import argparse
import random
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
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1. CARREGAR DATASET REAL
# ─────────────────────────────────────────────────────────────

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def load_images_from_folder(folder, label, max_images=None):
    """Carrega todas as imagens de uma pasta com um rótulo."""
    entries = []
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(EXTENSIONS)
    ])
    if max_images:
        files = files[:max_images]
    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        entries.append({
            'filename': fname,
            'path':     path,
            'label':    label,   # 'original' ou 'falsificada'
            'img':      img,
        })
    return entries


def build_dataset(dataset_root, n_docs=25, n_queries=5):
    """
    Monta dataset com pelo menos 25 documentos e 5 queries.
    Usa imagens de full_org e full_forg.
    As queries são escolhidas do próprio banco (enunciado pede que
    as queries estejam presentes nos documentos).
    """
    org_folder  = os.path.join(dataset_root, 'full_org')
    forg_folder = os.path.join(dataset_root, 'full_forg')

    if not os.path.exists(org_folder):
        raise FileNotFoundError(f"Pasta não encontrada: {org_folder}")
    if not os.path.exists(forg_folder):
        raise FileNotFoundError(f"Pasta não encontrada: {forg_folder}")

    org_imgs  = load_images_from_folder(org_folder,  'original')
    forg_imgs = load_images_from_folder(forg_folder, 'falsificada')

    print(f"[Dataset] {len(org_imgs)} originais | {len(forg_imgs)} falsificadas encontradas")

    # Mescla e limita ao necessário
    all_imgs = org_imgs + forg_imgs
    random.shuffle(all_imgs)

    # Garante pelo menos n_docs + n_queries imagens
    needed = n_docs + n_queries
    if len(all_imgs) < needed:
        raise ValueError(f"Dataset tem apenas {len(all_imgs)} imagens, precisa de pelo menos {needed}")

    # Seleciona documentos e queries
    # As queries ESTÃO dentro do banco (conforme enunciado)
    selected = all_imgs[:needed]
    for i, entry in enumerate(selected):
        entry['id'] = i

    docs    = selected[:n_docs]
    queries = selected[n_docs:n_docs + n_queries]

    print(f"[Dataset] {len(docs)} documentos indexados | {len(queries)} queries")
    return docs, queries


# ─────────────────────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────────────────────

def preprocess(img, target_size=(128, 256)):
    """
    Normaliza a imagem para tamanho padrão mantendo aspecto,
    aplica binarização adaptativa para realçar os traços.
    """
    # Redimensiona mantendo proporção
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Canvas branco
    canvas = np.ones((th, tw), dtype=np.uint8) * 255
    y_off = (th - new_h) // 2
    x_off = (tw - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # Binarização adaptativa
    binarized = cv2.adaptiveThreshold(
        canvas, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )
    return binarized


# ─────────────────────────────────────────────────────────────
# 3. PROPOSTAS DE REGIÕES (Sliding Window + NMS)
# ─────────────────────────────────────────────────────────────

def sliding_window_proposals(img, scales=None, step=16, min_dark_ratio=0.01):
    """Gera propostas de regiões via sliding window multi-escala."""
    if scales is None:
        scales = [(64, 128), (96, 192), (128, 256)]
    h, w = img.shape[:2]
    proposals = []
    for (ww, wh) in scales:
        if ww > w or wh > h:
            continue
        for y in range(0, h - wh + 1, step):
            for x in range(0, w - ww + 1, step):
                roi = img[y:y+wh, x:x+ww]
                dark_ratio = np.sum(roi < 128) / roi.size
                if dark_ratio > min_dark_ratio:
                    proposals.append((x, y, ww, wh))
    return proposals


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


def nms_proposals(proposals, iou_thresh=0.5):
    """Non-Maximum Suppression: remove propostas redundantes."""
    if not proposals:
        return []
    boxes  = np.array(proposals, dtype=float)
    areas  = boxes[:, 2] * boxes[:, 3]
    order  = areas.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        x1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        y1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        x2 = np.minimum(boxes[i,0]+boxes[i,2], boxes[order[1:],0]+boxes[order[1:],2])
        y2 = np.minimum(boxes[i,1]+boxes[i,3], boxes[order[1:],1]+boxes[order[1:],3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return [proposals[k] for k in keep]


def get_best_proposal(img, proposals):
    """Retorna a proposta com maior área (região mais representativa)."""
    if not proposals:
        h, w = img.shape[:2]
        return (0, 0, w, h)
    areas = [p[2]*p[3] for p in proposals]
    return proposals[int(np.argmax(areas))]


# ─────────────────────────────────────────────────────────────
# 4. EXTRAÇÃO DE DESCRITORES
# ─────────────────────────────────────────────────────────────

HOG_SIZE = (64, 128)   # tamanho padrão de entrada do HOG

def extract_hog(roi):
    """HOG descriptor — captura gradientes locais (forma dos traços)."""
    roi_r = cv2.resize(roi, HOG_SIZE)
    fd = hog(roi_r, orientations=9, pixels_per_cell=(8,8),
             cells_per_block=(2,2), feature_vector=True)
    return fd.astype(np.float32)


def extract_hu_moments(roi):
    """Momentos de Hu — invariantes a escala/rotação/translação."""
    _, bw = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
    m  = cv2.moments(bw)
    hu = cv2.HuMoments(m).flatten()
    # Log-transform para estabilidade numérica
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.astype(np.float32)


def extract_lbp_hist(roi, radius=2, n_points=16):
    """
    LBP (Local Binary Pattern) — textura dos traços manuscritos.
    Implementação manual para não depender de biblioteca extra.
    """
    roi_r = cv2.resize(roi, (64, 64))
    h, w  = roi_r.shape
    lbp   = np.zeros((h, w), dtype=np.uint8)
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = roi_r[i, j]
            code = 0
            neighbors = [
                roi_r[i-radius, j], roi_r[i-radius, j+radius],
                roi_r[i, j+radius], roi_r[i+radius, j+radius],
                roi_r[i+radius, j], roi_r[i+radius, j-radius],
                roi_r[i, j-radius], roi_r[i-radius, j-radius],
            ]
            for k, nb in enumerate(neighbors):
                if nb >= center:
                    code |= (1 << k)
            lbp[i, j] = code
    hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
    return hist.astype(np.float32)


def extract_descriptor(roi):
    """
    Descritor final = HOG + Momentos de Hu + LBP
    Combina forma global, invariantes geométricos e textura.
    """
    hog_feat = extract_hog(roi)
    hu_feat  = extract_hu_moments(roi)
    lbp_feat = extract_lbp_hist(roi)
    return np.concatenate([hog_feat, hu_feat, lbp_feat])


# ─────────────────────────────────────────────────────────────
# 5. INDEXAÇÃO
# ─────────────────────────────────────────────────────────────

class CBIRIndex:
    def __init__(self, n_pca=64):
        self.n_pca       = n_pca
        self.descriptors = []
        self.proposals   = []
        self.meta        = []
        self.pca         = None
        self.tree        = None
        self._X          = None

    def add(self, doc_id, label, filename, descriptor, proposal):
        self.descriptors.append(descriptor)
        self.proposals.append(proposal)
        self.meta.append({'id': doc_id, 'label': label, 'filename': filename})

    def build(self):
        X = normalize(np.array(self.descriptors, dtype=np.float32))
        n_comp = min(self.n_pca, X.shape[0]-1, X.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=42)
        X_red = normalize(self.pca.fit_transform(X))
        self.tree = BallTree(X_red, metric='euclidean')
        self._X   = X_red
        print(f"[Index] {X.shape[0]} docs | dim original={X.shape[1]} | dim PCA={n_comp}")

    def query(self, descriptor, k=10):
        x     = normalize(descriptor.reshape(1, -1))
        x_red = normalize(self.pca.transform(x))
        dists, idxs = self.tree.query(x_red, k=min(k, len(self.meta)))
        return dists[0], idxs[0]


# ─────────────────────────────────────────────────────────────
# 6. RANQUEAMENTO (Visual + IoU espacial)
# ─────────────────────────────────────────────────────────────

def rank_results(visual_dists, idxs, query_proposal, index, alpha=0.7, beta=0.3):
    """
    Score final = alpha × visual_sim + beta × IoU_espacial
    alpha=0.7 → prioriza similaridade visual
    beta=0.3  → considera posição geográfica da região
    """
    results = []
    for dist, idx in zip(visual_dists, idxs):
        visual_sim = 1.0 / (1.0 + float(dist))
        iou        = compute_iou(query_proposal, index.proposals[idx])
        score      = alpha * visual_sim + beta * iou
        results.append({
            'idx':        int(idx),
            'meta':       index.meta[idx],
            'proposal':   index.proposals[idx],
            'visual_sim': visual_sim,
            'iou':        iou,
            'score':      score,
        })
    return sorted(results, key=lambda r: r['score'], reverse=True)


# ─────────────────────────────────────────────────────────────
# 7. AVALIAÇÃO
# ─────────────────────────────────────────────────────────────

def precision_at_k(results, query_label, k):
    hits = sum(1 for r in results[:k] if r['meta']['label'] == query_label)
    return hits / k

def average_precision(results, query_label):
    hits, ap = 0, 0.0
    n_rel = sum(1 for r in results if r['meta']['label'] == query_label)
    if n_rel == 0:
        return 0.0
    for i, r in enumerate(results):
        if r['meta']['label'] == query_label:
            hits += 1
            ap   += hits / (i + 1)
    return ap / n_rel


# ─────────────────────────────────────────────────────────────
# 8. VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────────

def plot_dataset_overview(docs, save_path, n=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    fig.patch.set_facecolor('#F7F9FC')
    for i, ax in enumerate(axes.flat):
        if i < min(n, len(docs)):
            d = docs[i]
            ax.imshow(d['proc'], cmap='gray')
            ax.set_title(f"{d['label']}\n{d['filename'][:15]}", fontsize=7)
        ax.axis('off')
    plt.suptitle("Amostra do Dataset CEDAR (10 documentos)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_proposals_example(doc, save_path):
    img = doc['proc']
    all_props = sliding_window_proposals(img)
    nms_props = nms_proposals(all_props, iou_thresh=0.4)
    best      = get_best_proposal(img, nms_props)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#F7F9FC')

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Original — {doc['label']}", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    for p in all_props[:50]:
        axes[1].add_patch(patches.Rectangle(
            (p[0],p[1]), p[2], p[3],
            lw=0.4, edgecolor='#3498DB', facecolor='none', alpha=0.4))
    axes[1].set_title(f"Sliding Window\n{len(all_props)} propostas (50 exibidas)", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray')
    for p in nms_props:
        axes[2].add_patch(patches.Rectangle(
            (p[0],p[1]), p[2], p[3],
            lw=1, edgecolor='#27AE60', facecolor='none', alpha=0.5))
    axes[2].add_patch(patches.Rectangle(
        (best[0],best[1]), best[2], best[3],
        lw=2.5, edgecolor='#E74C3C', facecolor='none'))
    axes[2].set_title(f"Após NMS ({len(nms_props)} regiões)\nVermelho = melhor proposta", fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Geração de Regiões Candidatas", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_query_results(query, results, docs_list, save_path, top_k=5):
    fig = plt.figure(figsize=(18, 4))
    fig.patch.set_facecolor('#F7F9FC')
    gs = GridSpec(1, top_k + 1, figure=fig, wspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(query['proc'], cmap='gray')
    qp = query['proposal']
    ax0.add_patch(patches.Rectangle((qp[0],qp[1]), qp[2], qp[3],
                                     lw=2, edgecolor='#E74C3C', facecolor='none'))
    ax0.set_title(f"QUERY\n{query['label']}\n{query['filename'][:12]}", fontsize=8,
                  color='#E74C3C', fontweight='bold')
    ax0.axis('off')

    for i, r in enumerate(results[:top_k]):
        ax = fig.add_subplot(gs[0, i+1])
        doc_img = docs_list[r['idx']]['proc']
        ax.imshow(doc_img, cmap='gray')
        rp    = r['proposal']
        color = '#27AE60' if r['meta']['label'] == query['label'] else '#E74C3C'
        ax.add_patch(patches.Rectangle((rp[0],rp[1]), rp[2], rp[3],
                                        lw=2, edgecolor=color, facecolor='none'))
        ax.set_title(
            f"#{i+1} {r['meta']['label']}\nScore:{r['score']:.3f} IoU:{r['iou']:.2f}",
            fontsize=7, color=color)
        ax.axis('off')

    plt.suptitle(f"Query {query['id']} — {query['label']} ({query['filename']})",
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_metrics(ap_scores, pk_list, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#F7F9FC')
    q_labels = [f"Q{i}" for i in range(len(ap_scores))]

    bars = axes[0].bar(q_labels, ap_scores, color='#3498DB', edgecolor='white', width=0.5)
    axes[0].axhline(np.mean(ap_scores), color='#E74C3C', ls='--', lw=1.5,
                    label=f"MAP={np.mean(ap_scores):.3f}")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Average Precision por Query", fontweight='bold')
    axes[0].set_ylabel("AP")
    axes[0].legend()
    axes[0].set_facecolor('#F7F9FC')
    for bar, val in zip(bars, ap_scores):
        axes[0].text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.2f}",
                     ha='center', fontsize=9)

    ks = [1, 3, 5]
    colors_bar = ['#3498DB','#27AE60','#E67E22']
    x = np.arange(len(q_labels))
    for ki, (k, c) in enumerate(zip(ks, colors_bar)):
        vals = [pk_list[qi][ki] for qi in range(len(q_labels))]
        axes[1].bar(x + ki*0.22, vals, 0.22, label=f"P@{k}", color=c, edgecolor='white')
    axes[1].set_xticks(x + 0.22)
    axes[1].set_xticklabels(q_labels)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Precision@K por Query", fontweight='bold')
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].set_facecolor('#F7F9FC')

    plt.suptitle("Métricas de Avaliação — CBIR CEDAR", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CBIR CEDAR')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Caminho para a pasta "signatures" (contém full_org e full_forg)')
    parser.add_argument('--output', type=str, default='output_cbir',
                        help='Pasta de saída para figuras e PDF')
    parser.add_argument('--n_docs', type=int, default=25)
    parser.add_argument('--n_queries', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  CBIR — Content-Based Image Retrieval | CEDAR Dataset")
    print("=" * 60)

    # 1. Carregar dataset
    print("\n[1] Carregando imagens reais do CEDAR...")
    docs, queries = build_dataset(args.dataset, args.n_docs, args.n_queries)

    # Pré-processar todas as imagens
    for entry in docs + queries:
        entry['proc'] = preprocess(entry['img'])

    plot_dataset_overview(docs, os.path.join(args.output, 'fig1_dataset_overview.png'))

    # 2. Propostas de regiões
    print("\n[2] Gerando propostas de regiões (Sliding Window + NMS)...")
    for entry in docs + queries:
        props = sliding_window_proposals(entry['proc'])
        props = nms_proposals(props, iou_thresh=0.4)
        entry['proposals'] = props
        entry['proposal']  = get_best_proposal(entry['proc'], props)
    print(f"  Exemplo: doc[0] → {len(docs[0]['proposals'])} regiões após NMS")

    plot_proposals_example(docs[0], os.path.join(args.output, 'fig2_proposals.png'))

    # 3. Extrair descritores e indexar
    print("\n[3] Extraindo descritores HOG + Hu + LBP e indexando...")
    index = CBIRIndex(n_pca=64)
    for d in docs:
        x, y, w, h = d['proposal']
        roi  = d['proc'][y:y+h, x:x+w]
        desc = extract_descriptor(roi)
        index.add(d['id'], d['label'], d['filename'], desc, d['proposal'])
    index.build()

    # 4. Queries e ranqueamento
    print("\n[4] Executando queries e ranqueando resultados...")
    all_ap, all_pk = [], []

    for q in queries:
        x, y, w, h = q['proposal']
        roi  = q['proc'][y:y+h, x:x+w]
        desc = extract_descriptor(roi)

        dists, idxs = index.query(desc, k=args.n_docs)
        results = rank_results(dists, idxs, q['proposal'], index)

        ap = average_precision(results, q['label'])
        p1 = precision_at_k(results, q['label'], 1)
        p3 = precision_at_k(results, q['label'], min(3, len(results)))
        p5 = precision_at_k(results, q['label'], min(5, len(results)))
        all_ap.append(ap)
        all_pk.append([p1, p3, p5])

        print(f"  Query {q['id']} ({q['label']:12s}) | AP={ap:.3f} | P@1={p1:.2f} P@3={p3:.2f} P@5={p5:.2f}")

        fig_path = os.path.join(args.output, f'fig3_query_{q["id"]}.png')
        plot_query_results(q, results, docs, fig_path)

    map_score = np.mean(all_ap)
    print(f"\n  MAP = {map_score:.4f}")

    plot_metrics(all_ap, all_pk, os.path.join(args.output, 'fig4_metrics.png'))

    # Salva sumário
    summary_path = os.path.join(args.output, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CBIR CEDAR — Sumário\n" + "="*40 + "\n")
        for q, ap, pk in zip(queries, all_ap, all_pk):
            f.write(f"Query {q['id']} ({q['label']}): AP={ap:.3f} P@1={pk[0]:.2f} P@3={pk[1]:.2f} P@5={pk[2]:.2f}\n")
        f.write(f"\nMAP = {map_score:.4f}\n")

    print(f"\n[✓] Concluído! Resultados em: {args.output}/")
    print(f"    MAP final = {map_score:.4f}")

if __name__ == '__main__':
    main()