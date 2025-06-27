#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corex Topic Discovery with Word Cloud Grid and Correlated Synthetic Data
Updated: Jun 23, 2025
"""

import sys
sys.path.append('/home/u1876024/bio_corex/')  # Adjust if needed

import numpy as np
import corex as ce
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.stats import kendalltau

import numpy as np
import corex as ce

def run_corex_with_filtering(X, gene_names, layers=[10], dim_hidden=2, tc_threshold=1e-3,
                             marginal='gaussian', filtering=True, verbose=False, **kwargs):
    """
    Trains Corex model(s) layer-wise on data X.

    Parameters
    ----------
    X : array-like
        Input data matrix (samples × genes), dense or sparse.
    gene_names : list
        List of gene/feature names.
    layers : list of int
        Number of topics per layer.
    dim_hidden : int
        Latent states per topic (usually 2).
    tc_threshold : float
        Minimum total correlation to retain topic.
    marginal : str
        Type of marginal (e.g., 'gaussian').
    filtering : bool
        Whether to remove low-TC topics.
    verbose : bool
        If True, prints Corex messages.
    kwargs : dict
        Additional arguments to Corex constructor.

    Returns
    -------
    valid_topics_list : list of lists
        Retained topic indices for each layer.
    corexes : list
        Trained Corex models for each layer.
    """
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    corexes = []
    current_input = X
    for i, n_hidden in enumerate(layers):
        marginal_type = marginal if i == 0 else 'discrete'
        model = ce.Corex(n_hidden=n_hidden, dim_hidden=dim_hidden,
                         marginal_description=marginal_type,
                         verbose=verbose, **kwargs)
        model.fit(current_input)
        corexes.append(model)
        current_input = model.labels

    valid_topics_list = []
    for model in corexes:
        if filtering:
            valid = [i for i, tc in enumerate(model.tcs) if tc > tc_threshold]
        else:
            valid = list(range(model.n_hidden))
        valid_topics_list.append(valid)

    return valid_topics_list, corexes

from scipy.stats import kendalltau

def flip_topic_signs_by_correlation(corex, X, gene_names=None, verbose=False):
    """
    Flips topic directions to ensure positive Kendall tau correlation with most contributing gene.

    Parameters
    ----------
    corex : Corex or FakeCorex object
        Trained model with `.labels`, `.alpha`, `.mis`.
    X : np.ndarray
        Expression matrix (samples × genes).
    gene_names : list of str, optional
        Gene names for logging.
    verbose : bool
        If True, prints which topics were flipped.

    Returns
    -------
    corex : Corex object (modified in-place)
    """
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    for j in range(corex.n_hidden):
        contrib = corex.alpha[j, :, 0] * corex.mis[j]
        top_gene_idx = np.argmax(contrib)
        tau, _ = kendalltau(X[:, top_gene_idx], corex.labels[:, j])
        if tau < 0:
            corex.labels[:, j] = 1 - corex.labels[:, j]  # flip 0↔1
            if hasattr(corex, "p_y_given_x"):  # flip probability matrix if exists
                corex.p_y_given_x[j] = corex.p_y_given_x[j][:, ::-1]
            if verbose:
                gname = gene_names[top_gene_idx] if gene_names else f"G{top_gene_idx}"
                print(f"Flipped topic {j} for positive correlation with top gene {gname}")
    return corex


import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from scipy.stats import kendalltau
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

class ColorByKendallTau:
    """
    Callable class that maps a gene name to an RGB color based on its Kendall tau correlation.

    Parameters
    ----------
    word_to_corr : dict
        Dictionary mapping words (gene names) to Kendall's tau values (range: -1 to 1).
    threshold : float
        Tau values below this absolute threshold are treated as zero (white).

    Usage
    -----
    color_func = ColorByKendallTau(word_to_corr)
    wordcloud.recolor(color_func=color_func)
    """
    def __init__(self, word_to_corr, threshold=0.05):
        """
        Parameters:
            word_to_corr (dict): maps words to Kendall's tau values in [-1, 1]
            threshold (float): absolute tau values below this are considered "neutral" (white)
        """
        self.word_to_corr = word_to_corr
        self.threshold = threshold

        self.norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        self.cmap = plt.get_cmap("bwr")

    def __call__(self, word, **kwargs):
        tau = self.word_to_corr.get(word, 0.0)
        if abs(tau) < self.threshold:
            tau = 0.0
        r, g, b, _ = self.cmap(self.norm(tau))
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"


def plot_corex_wordclouds_grid(corex, X, gene_names, valid_topics=None, max_genes=100, n_cols=3):    
    """
    Plots a grid of word clouds for Corex topics, with color encoding of gene-topic correlations.

    Parameters
    ----------
    corex : Corex or Corex-like object
        Trained Corex model or simulated version with attributes .labels, .alpha, .mis.
    X : np.ndarray
        Expression data (samples × genes).
    gene_names : list of str
        Gene names corresponding to columns of X.
    valid_topics : list of int or None
        Indices of topics to include. If None, uses all topics.
    max_genes : int
        Max number of genes to include in each word cloud.
    n_cols : int
        Number of columns in the word cloud grid.

    Notes
    -----
    - The title of each subplot includes the topic’s total correlation (TC).
    - A horizontal colorbar indicates Kendall's tau values (blue = negative, red = positive).
    """

    X = X.toarray() if not isinstance(X, np.ndarray) else X
    gene_names = np.array(gene_names)
    valid_topics = valid_topics or list(range(corex.n_hidden))
    labels = corex.labels
    n_topics = len(valid_topics)
    n_rows = int(np.ceil(n_topics / n_cols))

    # Prepare grid with space below for colorbar
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows + 1))
    spec = gridspec.GridSpec(n_rows + 1, n_cols, height_ratios=[4]*n_rows + [0.2])
    axs = [fig.add_subplot(spec[i, j]) for i in range(n_rows) for j in range(n_cols)]

    # Compute Kendall tau
    correlations = np.zeros((corex.n_hidden, X.shape[1]))
    for i in valid_topics:
        for j in range(X.shape[1]):
            tau, _ = kendalltau(X[:, j], labels[:, i])
            correlations[i, j] = tau if not np.isnan(tau) else 0.0

    for plot_idx, i in enumerate(valid_topics):
        weights = (corex.alpha[i, :, 0] * corex.mis[i]).clip(min=0)
        top = np.argsort(weights)[-max_genes:]
        word_freq = {gene_names[j]: weights[j] for j in top if weights[j] > 0}
        word_corr = {gene_names[j]: correlations[i, j] for j in top}

        wc = WordCloud(width=400, height=300, background_color='white')
        wc.generate_from_frequencies(word_freq)
        wc.recolor(color_func=ColorByKendallTau(word_corr))

        axs[plot_idx].imshow(wc, interpolation='bilinear')
        tc_value = corex.tcs[i] if hasattr(corex, "tcs") else np.nan
        axs[plot_idx].set_title(f"Topic {i} (TC={tc_value:.2f})")
        axs[plot_idx].axis('off')

    # Hide unused subplots
    for j in range(len(valid_topics), len(axs)):
        axs[j].axis('off')

    # Add horizontal colorbar across entire bottom row
    cax = fig.add_subplot(spec[-1, :])
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sm = ScalarMappable(cmap="bwr", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label("Kendall's Tau Correlation (Gene ↔ Topic Status)")

    fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom
    plt.show()



def plot_sample_topic_bicluster(corex, valid_topics=None, method='average'):    
    """
    Visualizes patient-topic relationships as a heatmap, clustering patients by topic profiles.

    Parameters
    ----------
    corex : Corex or Corex-like object
        Trained model with .labels attribute (samples × topics).
    valid_topics : list of int, optional
        Topics to include (default: all).
    method : str
        Linkage method for hierarchical clustering of rows ('average', 'ward', etc.).

    Notes
    -----
    - Topics are kept in original order; samples are hierarchically clustered.
    - Useful for visualizing topic activation patterns across patients.
    """

    from scipy.cluster.hierarchy import linkage, leaves_list
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    labels = corex.labels  # shape: (n_samples, n_hidden)
    if valid_topics is None:
        valid_topics = list(range(corex.n_hidden))

    topic_matrix = labels[:, valid_topics]  # binary or categorical (e.g., 0/1)
    topic_matrix = np.asarray(topic_matrix)

    # Cluster rows (samples), but keep columns in numerical order
    row_order = leaves_list(linkage(topic_matrix, method=method))
    col_order = np.arange(len(valid_topics))  # keep original order

    # Reorder matrix
    X_reordered = topic_matrix[row_order][:, col_order]

    plt.figure(figsize=(1 + 0.5 * len(valid_topics), 6))
    sns.heatmap(X_reordered, cmap='Purples', cbar=False, linewidths=0.5,
                xticklabels=[f"T{valid_topics[i]}" for i in col_order], yticklabels=False)
    plt.title("Patient-Clustered Heatmap: Corex Topic Status")
    plt.xlabel("Latent Topics")
    plt.ylabel("Samples (clustered)")
    plt.tight_layout()
    plt.show()


def plot_expression_bicluster_heatmap(X, gene_names, corex, valid_topics,
                                      max_genes=10, tau_threshold=0.2, p_threshold=0.01):    
    """
    Visualizes a clustered gene expression heatmap, filtered by correlation with Corex topics.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (samples × genes).
    gene_names : list of str
        Gene names.
    corex : Corex
        Trained model with .labels attribute.
    valid_topics : list of int
        List of topic indices to analyze.
    max_genes : int
        Maximum number of correlated genes per topic to show.
    tau_threshold : float
        Minimum absolute Kendall's tau value for inclusion.
    p_threshold : float
        Maximum p-value for significance of tau.

    Notes
    -----
    - Selects only genes that are significantly correlated with at least one topic.
    - Clusters both genes and samples using hierarchical clustering.
    """

    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list
    from matplotlib.colors import Normalize
    from scipy.stats import kendalltau

    X = X.toarray() if not isinstance(X, np.ndarray) else X
    gene_names = np.array(gene_names)
    selected_gene_indices = []
    column_labels = []

    labels = corex.labels

    for topic_idx in valid_topics:
        gene_corrs = []
        for gene_idx in range(X.shape[1]):
            tau, p = kendalltau(X[:, gene_idx], labels[:, topic_idx])
            if not np.isnan(tau) and abs(tau) >= tau_threshold and p < p_threshold:
                gene_corrs.append((gene_idx, tau))

        # Sort by absolute correlation and take top genes
        gene_corrs.sort(key=lambda x: -abs(x[1]))
        top_genes = gene_corrs[:max_genes]

        for gene_idx, tau in top_genes:
            if gene_idx not in selected_gene_indices:
                selected_gene_indices.append(gene_idx)
                column_labels.append(f"T{topic_idx}:{gene_names[gene_idx]}")

    if not selected_gene_indices:
        print("No genes passed the correlation and significance filters.")
        return

    X_subset = X[:, selected_gene_indices]

    # Cluster samples and genes
    row_order = leaves_list(linkage(X_subset, method='average'))
    col_order = leaves_list(linkage(X_subset.T, method='average'))

    X_ordered = X_subset[row_order][:, col_order]
    reordered_labels = [column_labels[i] for i in col_order]

    plt.figure(figsize=(12, 8))
    sns.heatmap(X_ordered, cmap='vlag', xticklabels=reordered_labels, yticklabels=False,
                linewidths=0.2,
                norm=Normalize(vmin=np.percentile(X_ordered, 5), vmax=np.percentile(X_ordered, 95)))
    plt.title("Bicluster Heatmap of Expression (Significant Correlations Only)")
    plt.xlabel("Genes (grouped by topic)")
    plt.ylabel("Samples (clustered)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


import numpy as np

class FakeCorex:    
    """
    Corex-like container for ground truth topic-gene and topic-sample matrices.

    Parameters
    ----------
    G : np.ndarray
        Matrix (n_topics × n_genes) with ±1 or 0 for gene-topic membership.
    A : np.ndarray
        Matrix (n_samples × n_topics), binary topic activation per sample.

    Attributes
    ----------
    n_hidden : int
        Number of latent topics.
    alpha : np.ndarray
        Topic-gene weights (for compatibility with Corex outputs).
    mis : np.ndarray
        Used as a placeholder for TC; set to the number of genes per topic.
    labels : np.ndarray
        Topic activation matrix A (samples × topics).
    """

    def __init__(self, G, A):
        """
        Simulated Corex-like object with ground truth topic-gene and topic-sample relationships.

        Parameters
        ----------
        G : np.ndarray
            Binary matrix (n_topics × n_genes), 1 if gene j belongs to topic i.
        A : np.ndarray
            Matrix (n_samples × n_topics), topic activations per sample.
        """
        self.n_hidden = G.shape[0]
        self.alpha = G[:, :, None].astype(float)
        self.mis = np.sum(G, axis=1).astype(float)  # Use number of genes per topic as pseudo-TC
        self.labels = A

class CorexDataGenerator:
    """
    Generator for synthetic data with true underlying topic structure (positive and negative associations).

    Parameters
    ----------
    n_samples : int
        Number of synthetic samples (patients).
    n_topics : int
        Number of ground truth latent topics.
    genes_per_topic : int
        Number of genes associated with each topic.
    noise_genes : int
        Number of unrelated noisy genes.
    noise : float
        Standard deviation of Gaussian noise added to all gene values.
    seed : int
        Random seed for reproducibility.
    prob_pos : float
        Probability of positive association for each gene-topic pair.
    prob_neg : float
        Probability of negative association for each gene-topic pair.

    Methods
    -------
    generate()
        Returns expression matrix, gene names, topic-gene map, topic activation matrix, and a FakeCorex model.

    Returns (from generate)
    -------
    X : np.ndarray
        Expression data (samples × genes).
    gene_names : list of str
        Names of genes.
    G : np.ndarray
        True topic-gene association matrix (±1, 0).
    A : np.ndarray
        Topic activations per sample.
    true_corex : FakeCorex
        Ground truth Corex-like object for evaluation.
    """

    def __init__(self, n_samples=150, n_topics=3, genes_per_topic=15, noise_genes=20,
                 noise=0.2, seed=42, prob_pos=0.4, prob_neg=0.4):
        """
        Generator for synthetic datasets with mixed topic-gene associations.

        Parameters
        ----------
        n_samples : int
            Number of samples (patients).
        n_topics : int
            Number of latent topics.
        genes_per_topic : int
            Number of genes associated with each topic.
        noise_genes : int
            Number of genes unassociated with any topic.
        noise : float
            Amount of Gaussian noise added to all genes.
        seed : int
            Random seed for reproducibility.
        prob_pos : float
            Probability that a gene is positively associated with the topic.
        prob_neg : float
            Probability that a gene is negatively associated with the topic.
        """
        self.n_samples = n_samples
        self.n_topics = n_topics
        self.genes_per_topic = genes_per_topic
        self.noise_genes = noise_genes
        self.noise = noise
        self.seed = seed
        self.prob_pos = prob_pos
        self.prob_neg = prob_neg
        assert prob_pos + prob_neg <= 1.0, "prob_pos + prob_neg must be ≤ 1"

    def generate(self):
        np.random.seed(self.seed)
        total_genes = self.n_topics * self.genes_per_topic + self.noise_genes
        X = np.zeros((self.n_samples, total_genes))

        G = np.zeros((self.n_topics, total_genes))  # topic-to-gene (±1 for pos/neg association)
        A = np.zeros((self.n_samples, self.n_topics))  # sample-to-topic activation

        for t in range(self.n_topics):
            topic_activity = np.random.binomial(1, 0.5, size=(self.n_samples, 1))
            A[:, t] = topic_activity[:, 0]

            gene_signs = np.random.choice([1, -1, 0], size=self.genes_per_topic,
                                          p=[self.prob_pos,
                                             self.prob_neg,
                                             1 - self.prob_pos - self.prob_neg])

            gene_block = np.zeros((self.n_samples, self.genes_per_topic))
            for j, sign in enumerate(gene_signs):
                if sign != 0:
                    gene_block[:, j] = topic_activity[:, 0] * sign * (1 + 0.5 * np.random.randn())
                gene_block[:, j] += self.noise * np.random.randn(self.n_samples)

            gene_indices = slice(t * self.genes_per_topic, (t + 1) * self.genes_per_topic)
            X[:, gene_indices] = gene_block
            G[t, gene_indices] = gene_signs  # ±1 for pos/neg association, 0 for none

        # Add unrelated noisy genes
        X[:, -self.noise_genes:] = np.random.randn(self.n_samples, self.noise_genes)

        gene_names = [f"G{j}" for j in range(total_genes)]
        true_corex = FakeCorex(G=G, A=A)
        return X, gene_names, G, A, true_corex


# === QUICK TESTING BLOCK ===
if __name__ == "__main__":
    generator = CorexDataGenerator()
    X, gene_names, G, A, true_corex = generator.generate()

    valid_topics_list, corexes = run_corex_with_filtering(
        X, gene_names, layers=[3], filtering=True, verbose=True,
        marginal='gaussian', dim_hidden=2, n_cpu=4
    )
    corexes[0] = flip_topic_signs_by_correlation(corexes[0], X, gene_names, verbose=True)

    plot_corex_wordclouds_grid(corexes[0], X, gene_names, valid_topics=valid_topics_list[0])
    plot_corex_wordclouds_grid(true_corex, X, gene_names); plt.title("True")
    # Show topic-patient bicluster heatmap
    plot_sample_topic_bicluster(corexes[0], valid_topics_list[0])
    plot_sample_topic_bicluster(true_corex)
    #plot_expression_bicluster_heatmap(X, gene_names, corexes[0], valid_topics_list[0])


