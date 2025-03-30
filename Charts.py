import matplotlib
matplotlib.use('Agg')  # Przełącz na backend nieinteraktywny
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import io
import base64

def generate_charts(allocation, costs):
    allocation = np.array(allocation)
    costs = np.array(costs)

    # 1. Macierz ciepła (Heatmap)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    heatmap = ax1.imshow(allocation, cmap='Blues', interpolation='nearest')
    ax1.set_xticks(np.arange(allocation.shape[1]))
    ax1.set_yticks(np.arange(allocation.shape[0]))
    ax1.set_xticklabels([f"O{j+1}" for j in range(allocation.shape[1])], color='white')
    ax1.set_yticklabels([f"D{i+1}" for i in range(allocation.shape[0])], color='white')
    ax1.set_xlabel("Odbiorcy", color='white')
    ax1.set_ylabel("Dostawcy", color='white')
    fig1.patch.set_facecolor('#1F2937')
    ax1.set_facecolor('#1F2937')
    for i in range(allocation.shape[0]):
        for j in range(allocation.shape[1]):
            ax1.text(j, i, f"{allocation[i, j]}\n{costs[i, j]}", 
                     ha="center", va="center", color="black" if allocation[i, j] > allocation.max()/2 else "white")
    cbar = plt.colorbar(heatmap)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.set_ylabel('Alokacja', color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    plt.title("Macierz ciepła alokacji", color='white')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig1.get_facecolor())
    buf.seek(0)
    heatmap_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig1)

    # 2. Graf sieciowy (Network Graph)
    G = nx.DiGraph()
    nodes = []
    for i in range(allocation.shape[0]):
        nodes.append(f"D{i+1}")
        G.add_node(f"D{i+1}", group="dostawca")
    for j in range(allocation.shape[1]):
        nodes.append(f"O{j+1}")
        G.add_node(f"O{j+1}", group="odbiorca")
    links = []
    for i in range(allocation.shape[0]):
        for j in range(allocation.shape[1]):
            if allocation[i, j] > 0:
                G.add_edge(f"D{i+1}", f"O{j+1}", weight=allocation[i, j], cost=costs[i, j])
                links.append({"source": f"D{i+1}", "target": f"O{j+1}", "value": allocation[i, j], "cost": costs[i, j]})
    
    pos = {node: (np.cos(2 * np.pi * idx / len(nodes)) * 10, np.sin(2 * np.pi * idx / len(nodes)) * 10) for idx, node in enumerate(nodes)}
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors = ['#FF6384' if G.nodes[n]['group'] == 'dostawca' else '#36A2EB' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax2)
    nx.draw_networkx_labels(G, pos, font_color='white', ax=ax2)
    edge_widths = [G[u][v]['weight'] / max([link['value'] for link in links]) * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='#4BC0C0', ax=ax2)
    ax2.set_axis_off()
    fig2.patch.set_facecolor('#1F2937')
    ax2.set_facecolor('#1F2937')
    plt.title("Graf sieciowy przepływów", color='white')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig2.get_facecolor())
    buf.seek(0)
    network_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig2)

    return heatmap_image, network_image

def generate_chart_images(costs, steps):
    allocation = steps['allocation']
    heatmap_image, network_image = generate_charts(allocation, costs)
    return {"heatmap_image": heatmap_image, "network_image": network_image}

if __name__ == "__main__":
    from Logika import logika
    costs = [[4, 6], [5, 7]]
    supply = [50, 60]
    demand = [60, 50]
    allocation, total_cost, steps = logika(costs, supply, demand)
    charts = generate_chart_images(costs, steps)
    print("Wykresy wygenerowane:", charts.keys())