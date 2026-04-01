import matplotlib.pyplot as plt

def plot_metric_scores(results):
    """
    Simple bar plot of metric scores.

    Parameters:
    - results: dict of metric_name -> score
    """
    names = list(results.keys())
    scores = list(results.values())
    
    plt.figure(figsize=(8, 4))
    plt.bar(names, scores, color='skyblue')
    plt.ylabel('Score')
    plt.title('Benchmark Metric Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
