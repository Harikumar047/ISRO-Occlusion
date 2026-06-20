import random
import math
import networkx as nx

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points on the Earth's surface
    using the Haversine formula. Returns distance in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def simulate_occlusion(G, num_centers=2, radius_meters=400, occlusion_ratio=0.8, seed=42):
    """
    Simulates satellite occlusion by removing a percentage of edges within 
    circular regions on the graph.
    
    Parameters:
      G: networkx.Graph - Ground truth graph
      num_centers: int - Number of circular occlusion zones
      radius_meters: float - Radius of each occlusion zone in meters
      occlusion_ratio: float - Fraction of edges within the zones to remove (0.0 to 1.0)
      seed: int - Random seed for reproducibility
      
    Returns:
      G_observed: networkx.Graph - The observed graph (with edges removed)
      removed_edges: list of tuples (u, v, data) - The ground truth missing edges
      occlusion_centers: list of dicts - The coordinates and radii of occlusion zones
    """
    if seed is not None:
        random.seed(seed)
        
    G_observed = G.copy()
    
    # 1. Select random nodes as center points of occlusion
    nodes_list = list(G.nodes())
    if len(nodes_list) < num_centers:
        num_centers = len(nodes_list)
        
    center_nodes = random.sample(nodes_list, num_centers)
    occlusion_centers = []
    
    for c_node in center_nodes:
        occlusion_centers.append({
            "node_id": c_node,
            "y": G.nodes[c_node]["y"],
            "x": G.nodes[c_node]["x"],
            "radius": radius_meters
        })
        
    # 2. Identify nodes inside the circular occlusion zones
    occluded_nodes = set()
    for center in occlusion_centers:
        c_lat, c_lon = center["y"], center["x"]
        for node, data in G.nodes(data=True):
            dist = haversine_distance(c_lat, c_lon, data["y"], data["x"])
            if dist <= radius_meters:
                occluded_nodes.add(node)
                
    # 3. Find edges where both endpoints are inside the occluded nodes
    # (or at least one endpoint, depending on design. Both endpoints gives a cleaner circle)
    candidate_edges = []
    for u, v, data in G.edges(data=True):
        if u in occluded_nodes and v in occluded_nodes:
            candidate_edges.append((u, v, data))
            
    # 4. Randomly remove a fraction of the candidate edges
    num_to_remove = int(len(candidate_edges) * occlusion_ratio)
    edges_to_remove = random.sample(candidate_edges, min(num_to_remove, len(candidate_edges)))
    
    # Track the removed edges
    removed_edges = []
    for u, v, data in edges_to_remove:
        if G_observed.has_edge(u, v):
            G_observed.remove_edge(u, v)
            removed_edges.append((u, v, data))
            
    print(f"Occlusion Simulation:")
    print(f"  Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")
    print(f"  Nodes in occlusion zones: {len(occluded_nodes)}")
    print(f"  Edges inside occlusion zones: {len(candidate_edges)}")
    print(f"  Edges removed: {len(removed_edges)}")
    
    return G_observed, removed_edges, occlusion_centers

if __name__ == "__main__":
    from data_loader import download_road_network
    
    # Test simulation
    place_name = "Panaji, Goa, India"
    G = download_road_network(place_name)
    G_obs, rem_edges, centers = simulate_occlusion(G, num_centers=2, radius_meters=300, occlusion_ratio=0.7)
    
    print(f"Original edges: {G.number_of_edges()}")
    print(f"Observed edges: {G_obs.number_of_edges()}")
    print(f"Removed edges: {len(rem_edges)}")
    print(f"Occlusion centers: {centers}")
