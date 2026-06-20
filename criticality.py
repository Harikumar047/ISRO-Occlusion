import networkx as nx
import numpy as np
from occlusion import haversine_distance

def compute_criticality(G, weights=None, poi_data=None):
    """
    Computes node and edge centrality metrics, articulation points, and
    calculates a Composite Criticality Index (CCI) for both nodes and edges.
    
    Parameters:
      G: networkx.Graph - The graph to analyze (typically the reconstructed graph)
      weights: dict - w1 (betweenness), w2 (degree), w3 (hospital access), w4 (poi density)
      poi_data: dict - from download_poi_data()
      
    Returns:
      node_criticality: dict mapping node -> CCI (0.0 to 1.0)
      edge_criticality: dict mapping edge tuple (u, v) -> CCI (0.0 to 1.0)
      articulation_points: set - Set of articulation point node IDs
      centrality_data: dict - Raw metrics for mapping and dashboards
    """
    # 1. Edge and Node Betweenness Centrality
    print("Computing betweenness centralities...")
    node_betweenness = nx.betweenness_centrality(G, normalized=True)
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)
    
    # 2. Articulation Points (critical nodes whose removal disconnects the graph)
    print("Finding articulation points...")
    articulation_pts = set(nx.articulation_points(G))
    
    # 3. Calculate Node Composite Criticality Index (CCI)
    node_cci = {}
    max_deg = max([G.degree(n) for n in G.nodes()]) if G.number_of_nodes() > 0 else 1.0
    
    if weights is None:
        weights = {"w1": 0.5, "w2": 0.3, "w3": 0.1, "w4": 0.1}
    w1, w2, w3, w4 = weights.get("w1", 0.5), weights.get("w2", 0.3), weights.get("w3", 0.0), weights.get("w4", 0.0)

    # Calculate hospital access and POI density if required
    node_hospital_score = {}
    node_poi_score = {}
    
    if poi_data and w3 > 0:
        hospitals = poi_data.get("hospitals", [])
        if hospitals:
            node_dists = {}
            for n, data in G.nodes(data=True):
                min_dist = min([haversine_distance(data['y'], data['x'], hy, hx) for hy, hx in hospitals])
                node_dists[n] = min_dist
            max_dist = max(node_dists.values()) if len(node_dists) > 0 and max(node_dists.values()) > 0 else 1.0
            for n, d in node_dists.items():
                node_hospital_score[n] = 1.0 - (d / max_dist) # Closer = higher score
        else:
            for n in G.nodes(): node_hospital_score[n] = 0.0
    else:
        for n in G.nodes(): node_hospital_score[n] = 0.0
        
    if poi_data and w4 > 0:
        all_pois = poi_data.get("all_pois", [])
        buffer_m = 500
        if all_pois:
            node_counts = {}
            for n, data in G.nodes(data=True):
                # Using a fast bounding box filter would be better, but we do simple haversine for demo
                count = sum(1 for py, px in all_pois if haversine_distance(data['y'], data['x'], py, px) <= buffer_m)
                node_counts[n] = count
            max_count = max(node_counts.values()) if len(node_counts) > 0 and max(node_counts.values()) > 0 else 1.0
            for n, c in node_counts.items():
                node_poi_score[n] = c / max_count
        else:
            for n in G.nodes(): node_poi_score[n] = 0.0
    else:
        for n in G.nodes(): node_poi_score[n] = 0.0
    
    for node in G.nodes():
        norm_between = node_betweenness.get(node, 0.0)
        norm_deg = G.degree(node) / max_deg
        
        # Weighted combination
        cci = w1 * norm_between + w2 * norm_deg + w3 * node_hospital_score.get(node, 0.0) + w4 * node_poi_score.get(node, 0.0)
        node_cci[node] = float(cci)
        
    # Normalize Node CCI to [0, 1] range
    node_cci_vals = list(node_cci.values())
    max_node_cci = max(node_cci_vals) if len(node_cci_vals) > 0 else 1.0
    min_node_cci = min(node_cci_vals) if len(node_cci_vals) > 0 else 0.0
    cci_range_node = max_node_cci - min_node_cci
    if cci_range_node > 0:
        for node in node_cci:
            node_cci[node] = (node_cci[node] - min_node_cci) / cci_range_node
            
    # 4. Calculate Edge Composite Criticality Index (CCI)
    edge_cci = {}
    
    # Find max edge betweenness for normalization
    edge_bet_vals = list(edge_betweenness.values())
    max_edge_bet = max(edge_bet_vals) if len(edge_bet_vals) > 0 else 1.0
    
    for u, v in G.edges():
        norm_bet = edge_betweenness.get((u, v), edge_betweenness.get((v, u), 0.0))
        # Sum of end node degrees normalized by twice the max degree
        deg_sum_factor = (G.degree(u) + G.degree(v)) / (2 * max_deg)
        hosp_factor = (node_hospital_score.get(u, 0.0) + node_hospital_score.get(v, 0.0)) / 2.0
        poi_factor = (node_poi_score.get(u, 0.0) + node_poi_score.get(v, 0.0)) / 2.0
        
        cci = w1 * norm_bet + w2 * deg_sum_factor + w3 * hosp_factor + w4 * poi_factor
        edge_cci[(min(u, v), max(u, v))] = float(cci)
        
    # Normalize Edge CCI to [0, 1] range
    edge_cci_vals = list(edge_cci.values())
    max_edge_cci = max(edge_cci_vals) if len(edge_cci_vals) > 0 else 1.0
    min_edge_cci = min(edge_cci_vals) if len(edge_cci_vals) > 0 else 0.0
    cci_range_edge = max_edge_cci - min_edge_cci
    if cci_range_edge > 0:
        for edge in edge_cci:
            edge_cci[edge] = (edge_cci[edge] - min_edge_cci) / cci_range_edge

    centrality_data = {
        "node_betweenness": node_betweenness,
        "edge_betweenness": edge_betweenness,
        "articulation_points": articulation_pts,
        "node_cci": node_cci,
        "edge_cci": edge_cci
    }
    
    return node_cci, edge_cci, articulation_pts, centrality_data

def compute_connectivity_ratio(G_before, G_after):
    """
    Calculates percentage increase in largest connected component after healing.
    Connectivity Ratio = (LCC_size_after - LCC_size_before) / LCC_size_before * 100
    """
    if len(G_before) == 0: return 0.0
    lcc_before = len(max(nx.connected_components(G_before), key=len)) if len(G_before) > 0 else 0
    lcc_after = len(max(nx.connected_components(G_after), key=len)) if len(G_after) > 0 else 0
    return ((lcc_after - lcc_before) / lcc_before) * 100 if lcc_before > 0 else 0.0

def run_node_ablation_test(G, top_k=5, weights=None, poi_data=None, node_cci=None):
    """
    Simulates targeted attack/removal of top-k NODES with highest Node CCI.
    Computes connectivity metrics before and after the removal.
    """
    N = G.number_of_nodes()
    if N == 0:
        return {}
        
    # 1. Compute baseline criticality
    if node_cci is None:
        node_cci, _, _, _ = compute_criticality(G, weights=weights, poi_data=poi_data)
        
    # Sort nodes by criticality descending
    sorted_nodes = sorted(node_cci.items(), key=lambda x: x[1], reverse=True)
    nodes_to_remove = [node for node, cci in sorted_nodes[:top_k]]
    
    # 2. Compute baseline metrics
    baseline_lcc = max(nx.connected_components(G), key=len)
    baseline_lcc_fraction = len(baseline_lcc) / N
    
    G_baseline_lcc = G.subgraph(baseline_lcc)
    baseline_aspl = nx.average_shortest_path_length(G_baseline_lcc, weight="length") if len(G_baseline_lcc) > 1 else 0.0
    baseline_eff = nx.global_efficiency(G)
    
    # 3. Create stressed graph and remove nodes
    G_stressed = G.copy()
    G_stressed.remove_nodes_from(nodes_to_remove)
    
    # 4. Compute stressed metrics
    if len(G_stressed) > 0:
        stressed_lcc = max(nx.connected_components(G_stressed), key=len)
        stressed_lcc_fraction = len(stressed_lcc) / N
        G_stressed_lcc = G_stressed.subgraph(stressed_lcc)
        stressed_aspl = nx.average_shortest_path_length(G_stressed_lcc, weight="length") if len(G_stressed_lcc) > 1 else 0.0
        stressed_eff = nx.global_efficiency(G_stressed)
    else:
        stressed_lcc_fraction = 0.0
        stressed_aspl = 0.0
        stressed_eff = 0.0
        
    resilience_index = baseline_aspl / stressed_aspl if stressed_aspl > 0 else 0.0
    
    lcc_change_pct = ((stressed_lcc_fraction - baseline_lcc_fraction) / baseline_lcc_fraction) * 100 if baseline_lcc_fraction > 0 else 0
    aspl_change_pct = ((stressed_aspl - baseline_aspl) / baseline_aspl) * 100 if baseline_aspl > 0 else 0
    eff_change_pct = ((stressed_eff - baseline_eff) / baseline_eff) * 100 if baseline_eff > 0 else 0
    
    results = {
        "top_k": top_k,
        "removed_nodes": nodes_to_remove,
        "baseline": {
            "lcc_fraction": baseline_lcc_fraction,
            "aspl": baseline_aspl,
            "efficiency": baseline_eff
        },
        "stressed": {
            "lcc_fraction": stressed_lcc_fraction,
            "aspl": stressed_aspl,
            "efficiency": stressed_eff
        },
        "changes_pct": {
            "lcc": lcc_change_pct,
            "aspl": aspl_change_pct,
            "efficiency": eff_change_pct
        },
        "resilience_index": resilience_index
    }
    
    return results

def run_stress_test(G, top_k=5, weights=None, poi_data=None, edge_cci=None):
    """
    Simulates targeted attack/removal of top-k roads with highest Edge CCI.
    Computes connectivity metrics before and after the removal.
    """
    N = G.number_of_nodes()
    if N == 0:
        return {}
        
    # 1. Compute baseline criticality
    if edge_cci is None:
        _, edge_cci, _, _ = compute_criticality(G, weights=weights, poi_data=poi_data)
    
    # Sort edges by criticality descending
    sorted_edges = sorted(edge_cci.items(), key=lambda x: x[1], reverse=True)
    edges_to_remove = [edge for edge, cci in sorted_edges[:top_k]]
    
    # 2. Compute baseline metrics
    baseline_lcc = max(nx.connected_components(G), key=len)
    baseline_lcc_fraction = len(baseline_lcc) / N
    
    # Average shortest path on Largest Connected Component (LCC)
    G_baseline_lcc = G.subgraph(baseline_lcc)
    if len(G_baseline_lcc) > 1:
        baseline_aspl = nx.average_shortest_path_length(G_baseline_lcc, weight="length")
    else:
        baseline_aspl = 0.0
        
    # Global Efficiency (computationally intensive, so let's do it efficiently)
    # To speed up, we can approximate or compute directly. On 500 nodes nx.global_efficiency is fast.
    baseline_eff = nx.global_efficiency(G)
    
    # 3. Create stressed graph and remove edges
    G_stressed = G.copy()
    actual_removed = []
    for u, v in edges_to_remove:
        if G_stressed.has_edge(u, v):
            G_stressed.remove_edge(u, v)
            actual_removed.append((u, v))
            
    # 4. Compute stressed metrics
    stressed_lcc = max(nx.connected_components(G_stressed), key=len)
    stressed_lcc_fraction = len(stressed_lcc) / N
    
    G_stressed_lcc = G_stressed.subgraph(stressed_lcc)
    if len(G_stressed_lcc) > 1:
        stressed_aspl = nx.average_shortest_path_length(G_stressed_lcc, weight="length")
    else:
        stressed_aspl = 0.0
        
    stressed_eff = nx.global_efficiency(G_stressed)
    
    # Calculate percentage changes
    lcc_change_pct = ((stressed_lcc_fraction - baseline_lcc_fraction) / baseline_lcc_fraction) * 100 if baseline_lcc_fraction > 0 else 0
    aspl_change_pct = ((stressed_aspl - baseline_aspl) / baseline_aspl) * 100 if baseline_aspl > 0 else 0
    eff_change_pct = ((stressed_eff - baseline_eff) / baseline_eff) * 100 if baseline_eff > 0 else 0
    
    results = {
        "top_k": top_k,
        "removed_edges": actual_removed,
        "baseline": {
            "lcc_fraction": baseline_lcc_fraction,
            "aspl": baseline_aspl,
            "efficiency": baseline_eff
        },
        "stressed": {
            "lcc_fraction": stressed_lcc_fraction,
            "aspl": stressed_aspl,
            "efficiency": stressed_eff
        },
        "changes_pct": {
            "lcc": lcc_change_pct,
            "aspl": aspl_change_pct,
            "efficiency": eff_change_pct
        }
    }
    
    return results

if __name__ == "__main__":
    from data_loader import download_road_network
    
    # Test criticality calculations
    place_name = "Panaji, Goa, India"
    G = download_road_network(place_name)
    
    node_cci, edge_cci, articulation_pts, data = compute_criticality(G)
    print(f"Computed criticality for {len(node_cci)} nodes and {len(edge_cci)} edges.")
    print(f"Found {len(articulation_pts)} articulation points.")
    
    # Test stress test
    print("\nRunning stress test (removing top 5 roads)...")
    results = run_stress_test(G, top_k=5)
    print(f"Baseline LCC fraction: {results['baseline']['lcc_fraction']:.4f}")
    print(f"Stressed LCC fraction: {results['stressed']['lcc_fraction']:.4f}")
    print(f"Efficiency drop: {results['changes_pct']['efficiency']:.2f}%")
    print(f"Removed roads: {results['removed_edges']}")
