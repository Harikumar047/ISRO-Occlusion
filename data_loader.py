import os
import pickle
import re
import networkx as nx
import osmnx as ox

# Configure OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = False

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def sanitize_filename(name):
    """Sanitizes a string to make it a valid filename."""
    return re.sub(r"[^\w\-_]", "_", name).strip("_")

def download_road_network(place_name, network_type="drive", force_reload=False):
    """
    Downloads a road network from OpenStreetMap using OSMnx,
    converts it to a simple undirected networkx.Graph with 0-indexed integer labels,
    and caches the result.
    """
    safe_name = sanitize_filename(place_name)
    cache_path = os.path.join(CACHE_DIR, f"{safe_name}_{network_type}.gpickle")

    if not force_reload and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                G = pickle.load(f)
            print(f"Loaded road network for '{place_name}' from cache.")
            return G
        except Exception as e:
            print(f"Error loading cache: {e}. Downloading from scratch.")

    print(f"Downloading road network for '{place_name}' ({network_type})...")
    # Download as MultiDiGraph
    try:
        G_raw = ox.graph_from_place(place_name, network_type=network_type)
    except Exception as e:
        print(f"Error downloading for place '{place_name}' using place boundary: {e}")
        print("Retrying download using graph_from_address with 1.5km buffer...")
        try:
            G_raw = ox.graph_from_address(place_name, dist=1500, network_type=network_type)
        except Exception as e2:
            print(f"Fallback download also failed: {e2}")
            raise e2

    print(f"Preprocessing graph: converting to simple undirected Graph...")
    # Convert MultiDiGraph to simple undirected Graph
    # 1. Convert to undirected MultiGraph
    G_multi = G_raw.to_undirected()
    
    # 2. Convert MultiGraph to simple Graph (combining parallel edges by choosing the shortest length)
    G_simple = nx.Graph()
    for u, v, data in G_multi.edges(data=True):
        length = data.get("length", 1.0)
        # If edge already exists, keep the shorter one
        if G_simple.has_edge(u, v):
            if length < G_simple[u][v]["length"]:
                G_simple[u][v]["length"] = length
                if "geometry" in data:
                    G_simple[u][v]["geometry"] = data["geometry"]
        else:
            edge_data = {"length": length}
            if "geometry" in data:
                edge_data["geometry"] = data["geometry"]
            G_simple.add_edge(u, v, **edge_data)

    # Copy node attributes (x, y coordinates are critical)
    for node, data in G_multi.nodes(data=True):
        if node in G_simple:
            G_simple.nodes[node]["x"] = data.get("x")
            G_simple.nodes[node]["y"] = data.get("y")

    # Filter out isolated nodes or nodes missing coordinates
    to_remove = [n for n, d in G_simple.nodes(data=True) if d.get("x") is None or d.get("y") is None]
    G_simple.remove_nodes_from(to_remove)

    # Simplify to largest connected component to make analysis clean
    if len(G_simple) > 0:
        largest_cc = max(nx.connected_components(G_simple), key=len)
        G_simple = G_simple.subgraph(largest_cc).copy()

    # Convert node labels to sequential 0-indexed integers
    G = nx.convert_node_labels_to_integers(G_simple, label_attribute="original_id")
    
    # Cache node degrees
    for node in G.nodes():
        G.nodes[node]["degree"] = G.degree(node)

    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Downloaded and cached road network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    return G

def download_poi_data(place_name, force_reload=False):
    """
    Downloads POIs (hospitals, clinics, and general amenities for density) 
    and caches them.
    """
    safe_name = sanitize_filename(place_name)
    cache_path = os.path.join(CACHE_DIR, f"{safe_name}_pois.gpickle")

    if not force_reload and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                poi_data = pickle.load(f)
            print(f"Loaded POI data for '{place_name}' from cache.")
            return poi_data
        except Exception as e:
            print(f"Error loading POI cache: {e}. Downloading from scratch.")

    print(f"Downloading POI data for '{place_name}'...")
    poi_data = {"hospitals": [], "all_pois": []}
    
    try:
        # Fetch hospitals and clinics
        tags_medical = {"amenity": ["hospital", "clinic"]}
        gdf_medical = ox.features_from_place(place_name, tags=tags_medical)
        if not gdf_medical.empty:
            # We just need lat/lon
            for _, row in gdf_medical.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Point':
                    poi_data["hospitals"].append((geom.y, geom.x))
                elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                    centroid = geom.centroid
                    poi_data["hospitals"].append((centroid.y, centroid.x))
    except Exception as e:
        print(f"Error downloading medical POIs: {e}.")

    try:
        # Fetch general amenities for density (population proxy)
        ox.settings.timeout = 60
        tags_all = {"amenity": True}
        gdf_all = ox.features_from_place(place_name, tags=tags_all)
        if not gdf_all.empty:
            for _, row in gdf_all.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Point':
                    poi_data["all_pois"].append((geom.y, geom.x))
                elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                    centroid = geom.centroid
                    poi_data["all_pois"].append((centroid.y, centroid.x))
                    
    except Exception as e:
        print(f"POI density data unavailable for this region — falling back to betweenness/degree/hospital only for CCI. Error: {e}")
        poi_data["all_pois"] = []
        poi_data["missing_density"] = True
        
    with open(cache_path, "wb") as f:
        pickle.dump(poi_data, f)
        
    print(f"Cached {len(poi_data['hospitals'])} medical POIs and {len(poi_data['all_pois'])} total amenities.")
    return poi_data

if __name__ == "__main__":
    # Test download for a small Indian town/area (e.g., Panaji historic area, or a small town like Madikeri)
    # We will use "Panaji, Goa, India" which is relatively small and fast.
    test_place = "Panaji, Goa, India"
    try:
        G = download_road_network(test_place)
        print(f"Success! Graph nodes: {len(G)}, edges: {G.number_of_edges()}")
        pois = download_poi_data(test_place)
        print(f"Success! POIs: {len(pois['hospitals'])} medical, {len(pois['all_pois'])} total.")
    except Exception as e:
        print(f"Test failed: {e}")
