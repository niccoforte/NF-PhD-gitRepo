from resources.imports import *

from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors


@dataclass
class TokenizationConfig:
    n_tokens: int = 32
    k_neighbors: int = 6
    elite_quantile: float = 0.90
    embedding_dim: int = 8
    random_state: int = 42


class OutputInformedEmbedder:
    """
    Lightweight supervised embedder:
    - computes feature-score correlation weights (output-informed),
    - applies PCA on weighted patch features.
    """
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.w = None
        self.pca = None

    def fit(self, X, y):
        y = y.reshape(-1)
        Xc = X - np.mean(X, axis=0, keepdims=True)
        yc = y - np.mean(y)
        denom = (np.std(Xc, axis=0) * (np.std(yc) + 1e-12)) + 1e-12
        corr = np.mean(Xc * yc[:, None], axis=0) / denom
        self.w = np.abs(corr) + 1e-3

        Xw = X * self.w[None, :]
        n_comp = min(self.n_components, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=max(1, n_comp))
        self.pca.fit(Xw)
        return self

    def transform(self, X):
        Xw = X * self.w[None, :]
        return self.pca.transform(Xw)


def _normalized_score(props_df, objectives, weights=None, baseline_index=0):
    if weights is None:
        weights = np.ones(len(objectives), dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    base = props_df.iloc[baseline_index][objectives].to_numpy(dtype=float)
    vals = props_df[objectives].to_numpy(dtype=float)
    norm = vals / (base + 1e-12)
    return np.dot(norm, weights)


def _build_edge_index(base_nodes, k_neighbors=6):
    n_nodes = base_nodes.shape[0]
    k = max(1, min(k_neighbors + 1, n_nodes))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(base_nodes)
    knn = nbrs.kneighbors(return_distance=False)

    edges = set()
    for i in range(n_nodes):
        for j in knn[i]:
            if i == j:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))
    return np.array(sorted(edges), dtype=int)


def _build_adjacency(n_nodes, edges):
    adj = [[] for _ in range(n_nodes)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _extract_patch_features(X_nodes, adj):
    n_samples, n_nodes, _ = X_nodes.shape
    feats = np.zeros((n_samples, n_nodes, 8), dtype=float)

    mags = np.linalg.norm(X_nodes, axis=-1)
    angs = np.arctan2(X_nodes[..., 1], X_nodes[..., 0] + 1e-12)

    for n in range(n_nodes):
        neigh = adj[n]
        if len(neigh) == 0:
            neigh_mean = np.zeros((n_samples, 2), dtype=float)
            neigh_mag_mean = np.zeros((n_samples,), dtype=float)
            neigh_mag_std = np.zeros((n_samples,), dtype=float)
        else:
            neigh_xy = X_nodes[:, neigh, :]
            neigh_mag = mags[:, neigh]
            neigh_mean = np.mean(neigh_xy, axis=1)
            neigh_mag_mean = np.mean(neigh_mag, axis=1)
            neigh_mag_std = np.std(neigh_mag, axis=1)

        feats[:, n, 0:2] = X_nodes[:, n, :]
        feats[:, n, 2] = mags[:, n]
        feats[:, n, 3] = angs[:, n]
        feats[:, n, 4:6] = neigh_mean
        feats[:, n, 6] = neigh_mag_mean
        feats[:, n, 7] = neigh_mag_std
    return feats


class OutputInformedTokenizer:
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.embedding_model = None
        self.codebook = None
        self.edges = None
        self.adj = None

    def fit(self, X_nodes, base_nodes, score):
        self.edges = _build_edge_index(base_nodes, k_neighbors=self.config.k_neighbors)
        self.adj = _build_adjacency(base_nodes.shape[0], self.edges)
        patch_feats = _extract_patch_features(X_nodes, self.adj)

        Xp = patch_feats.reshape(-1, patch_feats.shape[-1])
        yp = np.repeat(score, X_nodes.shape[1]).reshape(-1, 1)

        emb_dim = min(self.config.embedding_dim, Xp.shape[1], Xp.shape[0])
        self.embedding_model = OutputInformedEmbedder(n_components=emb_dim)
        self.embedding_model.fit(Xp, yp)
        Z = self.embedding_model.transform(Xp)

        self.codebook = KMeans(
            n_clusters=self.config.n_tokens,
            n_init=20,
            random_state=self.config.random_state
        )
        self.codebook.fit(Z)
        return self

    def tokenize(self, X_nodes):
        patch_feats = _extract_patch_features(X_nodes, self.adj)
        Xp = patch_feats.reshape(-1, patch_feats.shape[-1])
        Z = self.embedding_model.transform(Xp)
        tok = self.codebook.predict(Z)
        return tok.reshape(X_nodes.shape[0], X_nodes.shape[1])

    def token_hist(self, token_ids):
        h = np.zeros((token_ids.shape[0], self.config.n_tokens), dtype=float)
        for i in range(token_ids.shape[0]):
            vals, cnts = np.unique(token_ids[i], return_counts=True)
            h[i, vals] = cnts
        h = h / np.maximum(1.0, np.sum(h, axis=1, keepdims=True))
        return h

    def diagnostics(self, token_ids, score):
        hist = self.token_hist(token_ids)
        freq = np.mean(hist, axis=0)
        p = freq / (np.sum(freq) + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        entropy_norm = entropy / np.log(len(p) + 1e-12)

        q = np.quantile(score, self.config.elite_quantile)
        elite_mask = score >= q
        non_mask = ~elite_mask

        elite_freq = np.mean(hist[elite_mask], axis=0) + 1e-12
        non_freq = np.mean(hist[non_mask], axis=0) + 1e-12
        enrich = np.log2(elite_freq / non_freq)

        mi = mutual_info_regression(hist, score, random_state=self.config.random_state)
        return {
            "entropy_norm": float(entropy_norm),
            "elite_threshold": float(q),
            "token_enrichment_log2": enrich,
            "token_mi_proxy": mi,
            "global_token_freq": p,
        }


def save_tokenization_artifacts(path, token_ids, diagnostics, config: TokenizationConfig):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "token_ids.npy"), token_ids)
    np.save(os.path.join(path, "token_enrichment_log2.npy"), diagnostics["token_enrichment_log2"])
    np.save(os.path.join(path, "token_mi_proxy.npy"), diagnostics["token_mi_proxy"])
    np.save(os.path.join(path, "global_token_freq.npy"), diagnostics["global_token_freq"])
    with open(os.path.join(path, "tokenizer_config.txt"), "w", encoding="utf-8") as f:
        f.write(str(config))
    with open(os.path.join(path, "diagnostics_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"entropy_norm={diagnostics['entropy_norm']}\n")
        f.write(f"elite_threshold={diagnostics['elite_threshold']}\n")


def prepare_xy_from_data_object(data, objectives, weights=None, use_ft_inputs=True):
    props_df = data.common_allProps_df if hasattr(data, "common_allProps_df") else data.UT_allProps_df
    score = _normalized_score(props_df, objectives=objectives, weights=weights, baseline_index=0)

    if use_ft_inputs and hasattr(data, "FT_all_in"):
        X = data.FT_all_in
    else:
        X = data.UT_all_in
    n_samples = X.shape[0]
    X_nodes = X.reshape(n_samples, X.shape[1] // 2, 2)
    base_nodes = data.UT_perIN_df.to_numpy().reshape(-1, 2)
    return X_nodes, base_nodes, score, props_df
