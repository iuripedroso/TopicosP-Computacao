import os
import argparse
import random
import numpy as np

from src.dataset import build_dataset, preprocess
from src.proposals import sliding_window_proposals, nms_proposals, get_best_proposal
from src.features import extract_descriptor
from src.index import CBIRIndex, rank_results
from src.evaluation import average_precision, precision_at_k
from src.visualization import (plot_dataset_overview, plot_proposals_example, 
                               plot_query_results, plot_metrics)

def main():
    random.seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description='CBIR CEDAR')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_cbir')
    parser.add_argument('--n_docs', type=int, default=25)
    parser.add_argument('--n_queries', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("\n[1] Carregando imagens reais do CEDAR...")
    docs, queries = build_dataset(args.dataset, args.n_docs, args.n_queries)

    for entry in docs + queries:
        entry['proc'] = preprocess(entry['img'])

    plot_dataset_overview(docs, os.path.join(args.output, 'fig1_dataset_overview.png'))

    print("\n[2] Gerando propostas de regiões (Sliding Window + NMS)...")
    for entry in docs + queries:
        props = sliding_window_proposals(entry['proc'])
        props = nms_proposals(props, iou_thresh=0.4)
        entry['proposals'] = props
        entry['proposal']  = get_best_proposal(entry['proc'], props)

    plot_proposals_example(docs[0], os.path.join(args.output, 'fig2_proposals.png'))

    print("\n[3] Extraindo descritores HOG + Hu + LBP e indexando...")
    index = CBIRIndex(n_pca=64)
    for d in docs:
        x, y, w, h = d['proposal']
        roi  = d['proc'][y:y+h, x:x+w]
        desc = extract_descriptor(roi)
        index.add(d['id'], d['label'], d['filename'], desc, d['proposal'])
    index.build()

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

    summary_path = os.path.join(args.output, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CBIR CEDAR — Sumário\n" + "="*40 + "\n")
        for q, ap, pk in zip(queries, all_ap, all_pk):
            f.write(f"Query {q['id']} ({q['label']}): AP={ap:.3f} P@1={pk[0]:.2f} P@3={pk[1]:.2f} P@5={pk[2]:.2f}\n")
        f.write(f"\nMAP = {map_score:.4f}\n")

if __name__ == '__main__':
    main()