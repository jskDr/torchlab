import matplotlib.pyplot as plt

def plot_scatter(X, fig, ax, annotate:bool=True, symbol='x'):
    ax.scatter(X[:,0], X[:,1])
    if annotate:
        # Add labels for each point
        for i, (x, y) in enumerate(zip(X[:,0], X[:,1])):
            ax.annotate(f'${symbol}_{i}$({x:.2f}, {y:.2f})', (x, y), xytext=(5, 5), textcoords='offset points')