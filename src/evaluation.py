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