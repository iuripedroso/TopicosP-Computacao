import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
from .proposals import sliding_window_proposals, nms_proposals, get_best_proposal

def plot_dataset_overview(docs, save_path, n=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    fig.patch.set_facecolor('#F7F9FC')
    for i, ax in enumerate(axes.flat):
        if i < min(n, len(docs)):
            d = docs[i]
            ax.imshow(d['proc'], cmap='gray')
            ax.set_title(f"{d['label']}\n{d['filename'][:15]}", fontsize=7)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_proposals_example(doc, save_path):
    img = doc['proc']
    all_props = sliding_window_proposals(img)
    nms_props = nms_proposals(all_props, iou_thresh=0.4)
    best      = get_best_proposal(img, nms_props)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    for p in all_props[:50]:
        axes[1].add_patch(patches.Rectangle((p[0],p[1]), p[2], p[3], lw=0.4, edgecolor='#3498DB', facecolor='none', alpha=0.4))
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray')
    for p in nms_props:
        axes[2].add_patch(patches.Rectangle((p[0],p[1]), p[2], p[3], lw=1, edgecolor='#27AE60', facecolor='none', alpha=0.5))
    axes[2].add_patch(patches.Rectangle((best[0],best[1]), best[2], best[3], lw=2.5, edgecolor='#E74C3C', facecolor='none'))
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def plot_query_results(query, results, docs_list, save_path, top_k=5):
    fig = plt.figure(figsize=(18, 4))
    gs = GridSpec(1, top_k + 1, figure=fig, wspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(query['proc'], cmap='gray')
    qp = query['proposal']
    ax0.add_patch(patches.Rectangle((qp[0],qp[1]), qp[2], qp[3], lw=2, edgecolor='#E74C3C', facecolor='none'))
    ax0.axis('off')

    for i, r in enumerate(results[:top_k]):
        ax = fig.add_subplot(gs[0, i+1])
        doc_img = docs_list[r['idx']]['proc']
        ax.imshow(doc_img, cmap='gray')
        rp    = r['proposal']
        color = '#27AE60' if r['meta']['label'] == query['label'] else '#E74C3C'
        ax.add_patch(patches.Rectangle((rp[0],rp[1]), rp[2], rp[3], lw=2, edgecolor=color, facecolor='none'))
        ax.set_title(f"Score:{r['score']:.3f}", fontsize=7, color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def plot_metrics(ap_scores, pk_list, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    q_labels = [f"Q{i}" for i in range(len(ap_scores))]

    axes[0].bar(q_labels, ap_scores, color='#3498DB', edgecolor='white', width=0.5)
    axes[0].axhline(np.mean(ap_scores), color='#E74C3C', ls='--', lw=1.5)
    axes[0].set_ylim(0, 1.1)

    ks = [1, 3, 5]
    colors_bar = ['#3498DB','#27AE60','#E67E22']
    x = np.arange(len(q_labels))
    for ki, (k, c) in enumerate(zip(ks, colors_bar)):
        vals = [pk_list[qi][ki] for qi in range(len(q_labels))]
        axes[1].bar(x + ki*0.22, vals, 0.22, label=f"P@{k}", color=c)
    
    axes[1].set_xticks(x + 0.22)
    axes[1].set_xticklabels(q_labels)
    axes[1].set_ylim(0, 1.15)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()