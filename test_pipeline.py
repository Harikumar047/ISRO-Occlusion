import time
from data_loader import download_road_network
from occlusion import simulate_occlusion
from gnn_model import (
    train_link_predictor,
    generate_candidate_edges,
    predict_reconstruction_links,
    mst_disjoint_set_healing,
    heuristic_reconstruction
)
from criticality import compute_criticality, run_stress_test, compute_connectivity_ratio, run_node_ablation_test

def run_pipeline_test():
    print("==================================================")
    print("=== Starting End-to-End Pipeline Verification Test ===")
    print("==================================================")
    
    start_time = time.time()
    
    # 1. Test Data Setup
    print("\n[Step 1/9] Testing Data Setup...")
    place_name = "Panaji, Goa, India"
    G = download_road_network(place_name)
    assert G is not None, "Failed to download graph"
    assert G.number_of_nodes() > 0, "Graph has 0 nodes"
    assert G.number_of_edges() > 0, "Graph has 0 edges"
    print(f"[PASS] Data Setup passed. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # 2. Test Occlusion Simulation
    print("\n[Step 2/9] Testing Occlusion Simulation...")
    num_centers = 2
    radius = 300
    ratio = 0.7
    G_obs, rem_edges, centers = simulate_occlusion(G, num_centers, radius, ratio, seed=42)
    assert G_obs is not None, "Observed graph is None"
    assert len(rem_edges) > 0, "No edges were removed"
    assert len(centers) == num_centers, "Incorrect number of occlusion centers"
    print(f"[PASS] Occlusion simulation passed. Removed: {len(rem_edges)} edges")
    
    # 3. Test GNN Training
    print("\n[Step 3/9] Testing GNN Model Training (5 Fast Epochs)...")
    encoder, predictor, history = train_link_predictor(
        G_obs, rem_edges, centers, model_type="SAGE", epochs=5, seed=42
    )
    assert len(history) == 5, "Training history size mismatch"
    assert history[0]["loss"] > 0, "Training loss is invalid"
    print(f"[PASS] GNN training passed. Final Epoch AUC: {history[-1]['val_auc']:.4f}")
    
    # 4. Test Candidate Generation & Link Prediction
    print("\n[Step 4/9] Testing Candidate Generation & Link Prediction...")
    candidates = generate_candidate_edges(G_obs, centers, max_dist=200)
    assert len(candidates) > 0, "No candidates generated"
    preds = predict_reconstruction_links(encoder, predictor, G_obs, candidates)
    assert len(preds) == len(candidates), "Prediction count mismatch"
    assert "prob" in preds[0], "Probability field missing from prediction dict"
    print(f"[PASS] Link prediction passed. Top prediction prob: {preds[0]['prob']:.4f}")
    
    # 5. Test MST + Disjoint Set Healing
    print("\n[Step 5/9] Testing MST + Disjoint Set Healing...")
    mst_preds = mst_disjoint_set_healing(G_obs, candidates)
    assert isinstance(mst_preds, list), "MST healing did not return a list"
    if len(mst_preds) > 0:
        assert "prob" in mst_preds[0], "Probability field missing from MST healing dict"
        assert "u" in mst_preds[0], "Node u field missing from MST healing dict"
        assert "v" in mst_preds[0], "Node v field missing from MST healing dict"
    print(f"[PASS] MST healing passed. Edges proposed: {len(mst_preds)}")
    
    # 6. Test Heuristic Reconstruction
    print("\n[Step 6/9] Testing Heuristic Reconstruction...")
    heuristic_preds = heuristic_reconstruction(G_obs, candidates)
    assert isinstance(heuristic_preds, list), "Heuristic did not return a list"
    if len(heuristic_preds) > 0:
        assert "prob" in heuristic_preds[0], "Probability field missing from heuristic dict"
    print(f"[PASS] Heuristic reconstruction passed. Edges proposed: {len(heuristic_preds)}")
    
    # 7. Test Criticality Analysis
    print("\n[Step 7/9] Testing Criticality Calculations...")
    G_recon = G_obs.copy()
    for edge in preds[:10]:
        G_recon.add_edge(edge["u"], edge["v"], length=edge["dist"])
        
    node_cci, edge_cci, articulation_pts, data = compute_criticality(G_recon)
    assert len(node_cci) == G_recon.number_of_nodes(), "Node CCI count mismatch"
    assert len(edge_cci) == G_recon.number_of_edges(), "Edge CCI count mismatch"
    print(f"[PASS] Criticality calculations passed. Articulation points found: {len(articulation_pts)}")
    
    # 8. Test Connectivity Ratio
    print("\n[Step 8/9] Testing Connectivity Ratio Metric...")
    conn_ratio = compute_connectivity_ratio(G_obs, G_recon)
    assert isinstance(conn_ratio, float), "Connectivity ratio is not a float"
    print(f"[PASS] Connectivity Ratio: {conn_ratio:+.2f}%")
    
    # 9. Test Stress Test + Node Ablation
    print("\n[Step 9/9] Testing Stress-Test & Node Ablation...")
    stress_k = 3
    
    # Edge removal test
    stress_results = run_stress_test(G_recon, top_k=stress_k, edge_cci=edge_cci)
    assert "baseline" in stress_results, "Baseline metrics missing from edge stress test"
    assert "stressed" in stress_results, "Stressed metrics missing from edge stress test"
    assert len(stress_results["removed_edges"]) <= stress_k, "Too many edges removed"
    print(f"[PASS] Edge stress-test passed. Efficiency change: {stress_results['changes_pct']['efficiency']:.2f}%")
    
    # Node ablation test
    ablation_results = run_node_ablation_test(G_recon, top_k=stress_k, node_cci=node_cci)
    assert "baseline" in ablation_results, "Baseline metrics missing from node ablation test"
    assert "stressed" in ablation_results, "Stressed metrics missing from node ablation test"
    assert "resilience_index" in ablation_results, "Resilience index missing from node ablation"
    assert len(ablation_results["removed_nodes"]) <= stress_k, "Too many nodes removed"
    print(f"[PASS] Node ablation test passed. Resilience Index: {ablation_results['resilience_index']:.4f}")
    
    total_time = time.time() - start_time
    print("\n==================================================")
    print(f"[SUCCESS] ALL {9} PIPELINE TESTS PASSED IN {total_time:.2f}s!")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline_test()
