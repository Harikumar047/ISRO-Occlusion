import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv

from occlusion import haversine_distance

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class SAGEEncoder(nn.Module):
    """2-layer GraphSAGE Encoder."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GATEncoder(nn.Module):
    """2-layer Graph Attention Network (GAT) Encoder."""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    """MLP Decoder to predict link existence from node embeddings."""
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_u, z_v):
        x = torch.cat([z_u, z_v], dim=-1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return torch.sigmoid(x)

def prepare_pyg_data(G_observed):
    """
    Converts a networkx.Graph G_observed to a PyTorch Geometric Data object.
    Creates 4 node features:
      1. Normalized Lat (y)
      2. Normalized Lon (x)
      3. Normalized Node Degree
      4. PageRank Score
    """
    num_nodes = G_observed.number_of_nodes()
    
    # Extract node coordinates
    lats = np.array([G_observed.nodes[i]["y"] for i in range(num_nodes)])
    lons = np.array([G_observed.nodes[i]["x"] for i in range(num_nodes)])
    
    # Normalize coordinates (Min-Max)
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    norm_lats = (lats - lat_min) / (lat_max - lat_min + 1e-8)
    norm_lons = (lons - lon_min) / (lon_max - lon_min + 1e-8)
    
    # Degrees
    degrees = np.array([G_observed.degree(i) for i in range(num_nodes)])
    max_degree = degrees.max() if degrees.max() > 0 else 1.0
    norm_degrees = degrees / max_degree
    
    # PageRank
    try:
        pagerank_dict = nx.pagerank(G_observed, alpha=0.85)
        pageranks = np.array([pagerank_dict.get(i, 0.0) for i in range(num_nodes)])
        # Scale PageRank to [0, 1] for normalization consistency
        if pageranks.max() > 0:
            norm_pageranks = pageranks / pageranks.max()
        else:
            norm_pageranks = pageranks
    except Exception:
        norm_pageranks = np.ones(num_nodes) / num_nodes
        
    # Combine features: shape (N, 4)
    x = np.stack([norm_lats, norm_lons, norm_degrees, norm_pageranks], axis=1)
    x_tensor = torch.tensor(x, dtype=torch.float)
    
    # Convert networkx edges to bidirectional edge_index for GNN message passing
    edges = list(G_observed.edges())
    edge_index_list = []
    for u, v in edges:
        edge_index_list.append([u, v])
        edge_index_list.append([v, u])
        
    if len(edge_index_list) > 0:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    data = Data(x=x_tensor, edge_index=edge_index)
    return data

def sample_negative_edges(G_observed, num_samples, exclude_edges=None):
    """
    Randomly samples node pairs (u, v) that are NOT connected in G_observed.
    """
    neg_edges = set()
    num_nodes = G_observed.number_of_nodes()
    
    # Compile set of existing edges (including duplicates and order-independent)
    existing_edges = set()
    for u, v in G_observed.edges():
        existing_edges.add((min(u, v), max(u, v)))
        
    if exclude_edges:
        for u, v in exclude_edges:
            existing_edges.add((min(u, v), max(u, v)))
            
    max_possible_neg = (num_nodes * (num_nodes - 1)) // 2 - len(existing_edges)
    num_samples = min(num_samples, max_possible_neg)
    
    attempts = 0
    max_attempts = num_samples * 100
    
    while len(neg_edges) < num_samples and attempts < max_attempts:
        attempts += 1
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u == v:
            continue
        edge = (min(u, v), max(u, v))
        if edge not in existing_edges and edge not in neg_edges:
            neg_edges.add(edge)
            
    return list(neg_edges)

def compute_metrics(pos_preds, neg_preds):
    """Computes AUC and AP from positive and negative predictions."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_true = [1] * len(pos_preds) + [0] * len(neg_preds)
    y_pred = pos_preds + neg_preds
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return auc, ap

def train_link_predictor(G_observed, removed_edges, occlusion_centers, model_type="SAGE", epochs=80, lr=0.01, seed=42):
    """
    Trains the encoder GNN and decoder LinkPredictor to perform link prediction.
    Uses the removed_edges as positive validation data.
    """
    set_seed(seed)
    pyg_data = prepare_pyg_data(G_observed)
    
    in_channels = pyg_data.x.size(1)
    hidden_channels = 32
    out_channels = 16
    
    # Initialize GNN encoder based on type
    if model_type.upper() == "GAT":
        encoder = GATEncoder(in_channels, hidden_channels, out_channels)
    else:
        encoder = SAGEEncoder(in_channels, hidden_channels, out_channels)
        
    predictor = LinkPredictor(out_channels, hidden_channels)
    
    # Training positive edges: all edges in G_observed
    train_pos_edges = list(G_observed.edges())
    train_pos_u = torch.tensor([u for u, v in train_pos_edges], dtype=torch.long)
    train_pos_v = torch.tensor([v for u, v in train_pos_edges], dtype=torch.long)
    
    # Create test validation edges
    test_pos_edges = [(u, v) for u, v, _ in removed_edges]
    # Sample test negatives (not in original graph G_observed, nor in removed_edges)
    test_neg_edges = sample_negative_edges(G_observed, len(test_pos_edges), exclude_edges=test_pos_edges)
    
    optimizer = Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    criterion = nn.BCELoss()
    
    history = []
    
    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()
        
        # Get node embeddings from encoder GNN
        z = encoder(pyg_data.x, pyg_data.edge_index)
        
        # Sample negative training edges dynamically per epoch
        train_neg_edges = sample_negative_edges(G_observed, len(train_pos_edges))
        train_neg_u = torch.tensor([u for u, v in train_neg_edges], dtype=torch.long)
        train_neg_v = torch.tensor([v for u, v in train_neg_edges], dtype=torch.long)
        
        # Forward pass on training pairs
        pos_out = predictor(z[train_pos_u], z[train_pos_v]).squeeze()
        neg_out = predictor(z[train_neg_u], z[train_neg_v]).squeeze()
        
        loss = criterion(pos_out, torch.ones_like(pos_out)) + criterion(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation/test set
        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            z = encoder(pyg_data.x, pyg_data.edge_index)
            
            # Val pos predictions
            if len(test_pos_edges) > 0:
                val_pos_u = torch.tensor([u for u, v in test_pos_edges], dtype=torch.long)
                val_pos_v = torch.tensor([v for u, v in test_pos_edges], dtype=torch.long)
                val_pos_out = predictor(z[val_pos_u], z[val_pos_v]).squeeze().tolist()
                # Wrap scalar to list if only 1 item
                if isinstance(val_pos_out, float): val_pos_out = [val_pos_out]
            else:
                val_pos_out = []
                
            # Val neg predictions
            if len(test_neg_edges) > 0:
                val_neg_u = torch.tensor([u for u, v in test_neg_edges], dtype=torch.long)
                val_neg_v = torch.tensor([v for u, v in test_neg_edges], dtype=torch.long)
                val_neg_out = predictor(z[val_neg_u], z[val_neg_v]).squeeze().tolist()
                if isinstance(val_neg_out, float): val_neg_out = [val_neg_out]
            else:
                val_neg_out = []
                
            val_auc, val_ap = compute_metrics(val_pos_out, val_neg_out)
            
        history.append({
            "epoch": epoch,
            "loss": loss.item(),
            "val_auc": val_auc,
            "val_ap": val_ap
        })
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")
            
    return encoder, predictor, history

def generate_candidate_edges(G_observed, occlusion_centers, max_dist=250):
    """
    Generates candidate edges to reconstruct.
    Candidates are node pairs (u, v) that:
      1. Are NOT connected in G_observed
      2. Have distance <= max_dist meters
      3. At least one endpoint is within the occlusion zones.
    """
    candidates = []
    num_nodes = G_observed.number_of_nodes()
    
    # 1. Identify which nodes are in occlusion zones
    occluded_nodes = set()
    for center in occlusion_centers:
        c_lat, c_lon = center["y"], center["x"]
        radius = center["radius"]
        for node, data in G_observed.nodes(data=True):
            dist = haversine_distance(c_lat, c_lon, data["y"], data["x"])
            if dist <= radius:
                occluded_nodes.add(node)
                
    # 2. Iterate through all node pairs (restricted to local vicinity)
    # To be efficient, we check nodes that are in the occlusion zone, and search neighbors within max_dist
    occluded_nodes_list = list(occluded_nodes)
    checked_pairs = set()
    
    for u in occluded_nodes_list:
        u_lat, u_lon = G_observed.nodes[u]["y"], G_observed.nodes[u]["x"]
        for v in range(num_nodes):
            if u == v:
                continue
            
            pair = (min(u, v), max(u, v))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            # Check if edge already exists in observed
            if G_observed.has_edge(u, v):
                continue
                
            # Check spatial distance
            v_lat, v_lon = G_observed.nodes[v]["y"], G_observed.nodes[v]["x"]
            dist = haversine_distance(u_lat, u_lon, v_lat, v_lon)
            if dist <= max_dist:
                candidates.append((u, v, dist))
                
    print(f"Generated {len(candidates)} candidate edges within {max_dist}m for reconstruction.")
    return candidates

def predict_reconstruction_links(encoder, predictor, G_observed, candidate_edges):
    """
    Predicts existence probability for each candidate edge.
    Returns list of dicts: [{'u': u, 'v': v, 'dist': dist, 'prob': probability}]
    """
    if len(candidate_edges) == 0:
        return []
        
    pyg_data = prepare_pyg_data(G_observed)
    
    encoder.eval()
    predictor.eval()
    
    with torch.no_grad():
        z = encoder(pyg_data.x, pyg_data.edge_index)
        
        u_indices = torch.tensor([u for u, v, _ in candidate_edges], dtype=torch.long)
        v_indices = torch.tensor([v for u, v, _ in candidate_edges], dtype=torch.long)
        
        probs = predictor(z[u_indices], z[v_indices]).squeeze().tolist()
        # Wrap if single prediction
        if isinstance(probs, float):
            probs = [probs]
            
    reconstructed = []
    for i, (u, v, dist) in enumerate(candidate_edges):
        reconstructed.append({
            "u": u,
            "v": v,
            "dist": dist,
            "prob": probs[i]
        })
        
    # Sort by probability descending
    reconstructed = sorted(reconstructed, key=lambda x: x["prob"], reverse=True)
    return reconstructed

def mst_disjoint_set_healing(G_observed, candidate_edges):
    """
    Implements geometric topological healing using MST and Disjoint Set (Union-Find)
    as a baseline reconstruction method per ISRO specification.
    """
    if len(candidate_edges) == 0:
        return []
        
    num_nodes = G_observed.number_of_nodes()
    
    # 1. Identify connected components (Union-Find / Disjoint Set)
    components = list(nx.connected_components(G_observed))
    
    # Map each node to its component ID
    node_to_comp = {}
    for comp_id, comp_nodes in enumerate(components):
        for node in comp_nodes:
            node_to_comp[node] = comp_id
            
    # 2. Score candidate edges based on Euclidean distance and Angular alignment
    scored_candidates = []
    max_dist = max([dist for _, _, dist in candidate_edges]) if candidate_edges else 1.0
    if max_dist == 0: max_dist = 1.0
    
    for u, v, dist in candidate_edges:
        # Only care about connecting different components
        if node_to_comp.get(u) == node_to_comp.get(v):
            continue
            
        norm_dist = dist / max_dist
        
        # Angular alignment penalty
        u_neighbors = list(G_observed.neighbors(u))
        v_neighbors = list(G_observed.neighbors(v))
        
        angle_penalty = 0.0
        
        u_y, u_x = G_observed.nodes[u]["y"], G_observed.nodes[u]["x"]
        v_y, v_x = G_observed.nodes[v]["y"], G_observed.nodes[v]["x"]
        
        def bearing(lat1, lon1, lat2, lon2):
            dLon = lon2 - lon1
            y = np.sin(dLon) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
            return np.arctan2(y, x)
            
        bearing_uv = bearing(u_y, u_x, v_y, v_x)
        bearing_vu = bearing(v_y, v_x, u_y, u_x)
        
        if u_neighbors:
            nu_y, nu_x = G_observed.nodes[u_neighbors[0]]["y"], G_observed.nodes[u_neighbors[0]]["x"]
            bearing_u_nu = bearing(u_y, u_x, nu_y, nu_x)
            diff = abs(bearing_uv - bearing_u_nu)
            diff = min(diff, 2*np.pi - diff)
            angle_penalty += abs(np.pi - diff) / np.pi
            
        if v_neighbors:
            nv_y, nv_x = G_observed.nodes[v_neighbors[0]]["y"], G_observed.nodes[v_neighbors[0]]["x"]
            bearing_v_nv = bearing(v_y, v_x, nv_y, nv_x)
            diff = abs(bearing_vu - bearing_v_nv)
            diff = min(diff, 2*np.pi - diff)
            angle_penalty += abs(np.pi - diff) / np.pi
            
        cost = norm_dist + 0.5 * angle_penalty
        prob = max(0.1, 1.0 - (cost / 2.0))
        
        scored_candidates.append({
            "u": u,
            "v": v,
            "dist": dist,
            "cost": cost,
            "prob": prob,
            "comp_u": node_to_comp.get(u),
            "comp_v": node_to_comp.get(v)
        })
        
    # 3. Kruskal's algorithm to find MST
    scored_candidates.sort(key=lambda x: x["cost"])
    
    parent = {i: i for i in range(len(components))}
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False
        
    reconstructed = []
    for edge in scored_candidates:
        if union(edge["comp_u"], edge["comp_v"]):
            reconstructed.append({
                "u": edge["u"],
                "v": edge["v"],
                "dist": edge["dist"],
                "prob": edge["prob"]
            })
            
    reconstructed.sort(key=lambda x: x["prob"], reverse=True)
    return reconstructed

def heuristic_reconstruction(G_observed, candidate_edges):
    """
    Purely geometric heuristic stitching.
    Connects candidates based on distance. Shorter distance gets higher probability.
    """
    if len(candidate_edges) == 0:
        return []
        
    reconstructed = []
    # Find max dist for normalization
    max_dist = max([dist for u, v, dist in candidate_edges]) if candidate_edges else 1.0
    if max_dist == 0: max_dist = 1.0
    
    for i, (u, v, dist) in enumerate(candidate_edges):
        # simple heuristic: closer is higher probability
        # Optional: could incorporate angle here, but distance is primary
        prob = 1.0 - (dist / max_dist)
        # Ensure a minimum probability so it shows up in UI if threshold is low
        prob = max(0.1, prob)
        
        reconstructed.append({
            "u": u,
            "v": v,
            "dist": dist,
            "prob": prob
        })
        
    # Sort by probability descending
    reconstructed = sorted(reconstructed, key=lambda x: x["prob"], reverse=True)
    return reconstructed

if __name__ == "__main__":
    from data_loader import download_road_network
    from occlusion import simulate_occlusion
    
    # End-to-end GNN model check
    place_name = "Panaji, Goa, India"
    G = download_road_network(place_name)
    G_obs, rem_edges, centers = simulate_occlusion(G, num_centers=2, radius_meters=300, occlusion_ratio=0.7)
    
    print("\nTraining GraphSAGE...")
    encoder, predictor, history = train_link_predictor(
        G_obs, rem_edges, centers, model_type="SAGE", epochs=30
    )
    
    candidates = generate_candidate_edges(G_obs, centers, max_dist=200)
    preds = predict_reconstruction_links(encoder, predictor, G_obs, candidates)
    if len(preds) > 0:
        print(f"Top 5 predictions: {preds[:5]}")
