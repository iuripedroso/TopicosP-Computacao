import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from .proposals import compute_iou

class CBIRIndex:
    def __init__(self, n_pca=64):
        self.n_pca       = n_pca
        self.descriptors = []
        self.proposals   = []
        self.meta        = []
        self.pca         = None
        self.tree        = None

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
        print(f"[Index] {X.shape[0]} docs | dim original={X.shape[1]} | dim PCA={n_comp}")

    def query(self, descriptor, k=10):
        x     = normalize(descriptor.reshape(1, -1))
        x_red = normalize(self.pca.transform(x))
        dists, idxs = self.tree.query(x_red, k=min(k, len(self.meta)))
        return dists[0], idxs[0]

def rank_results(visual_dists, idxs, query_proposal, index, alpha=0.7, beta=0.3):
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