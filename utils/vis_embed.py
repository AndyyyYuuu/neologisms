from matplotlib import pyplot as plt
import torch
from neologisms.lm import LMBackend
from faiss import IndexFlatL2
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

def visualize(embeds: list[torch.Tensor], cust_embed_labels: list[str], ref_model: LMBackend = None, pca_dims: int = 2, k_neighbors: int = 10, title: str = "Embeddings", embed_name: str = "<neo>", n_components: int = 2):

    if len(embeds) < len(cust_embed_labels):
        raise ValueError("embeds must have at least as many elements as cust_embed_labels")
    elif len(embeds) > len(cust_embed_labels):
        cust_embed_labels.extend([""] * (len(embeds) - len(cust_embed_labels)))
    
    cust_embeds = np.stack([embed.detach().float().cpu().numpy() for embed in embeds])
    pca = PCA(n_components=pca_dims)
    if ref_model: 
        ref_matrix = ref_model.embedding_matrix().float().cpu().numpy()
        index = IndexFlatL2(ref_matrix.shape[1])
        index.add(ref_matrix)
        indices: set[int] = set()
        for row in cust_embeds:
            q = row.reshape(1, -1).astype(np.float32)
            _, near_indices = index.search(q, k=k_neighbors)
            indices.update(near_indices[0].tolist())
        ref_embeds: list[np.ndarray] = ref_matrix[list(indices)]
        ref_embed_labels: list[str] = [
            f"`{ref_model.ids_to_str(torch.tensor([tid], device=ref_model.device))}`" for tid in indices
        ]
        all_embeds: np.ndarray = np.concatenate((cust_embeds, ref_embeds), axis=0)
        all_labels = cust_embed_labels + ref_embed_labels
        all_classes = ['custom'] * len(cust_embeds) + ['reference'] * len(ref_embeds)

        show_legend = True
    else: 
        all_embeds = cust_embeds
        all_labels = cust_embed_labels
        all_classes = ['custom'] * len(cust_embeds)
        show_legend = False

    pca_embeds: np.ndarray = pca.fit_transform(all_embeds)
    evr = pca.explained_variance_ratio_
    cum = float(np.sum(evr))
    print("PCA explained variance ratio per component:")
    for i, r in enumerate(evr):
        print(f"\tPC{i + 1}: {r:.6f}")
    print(f"Cumulative variance (PC1–PC{pca_dims}): {cum:.6f}")

    df = pd.DataFrame({
        "x": pca_embeds[:, 0],
        "y": pca_embeds[:, 1],
        "label": all_labels,
        "class": all_classes,
    })

    sns.set_theme(context="notebook", style="darkgrid")
    _, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=df, x="x", y="y", hue="class", ax=ax, legend=show_legend)
    for i, row in df.iterrows():
        ax.annotate(row["label"], (row["x"], row["y"]), fontsize=9, alpha=0.5)
    plt.title(title)
    plt.show()
