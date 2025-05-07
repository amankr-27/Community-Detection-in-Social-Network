from fastapi import FastAPI, File, UploadFile, Response, Query, HTTPException
import pandas as pd
import networkx as nx
import leidenalg as la
import igraph as ig
import io
import warnings
import sys
import scipy
import random
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import base64

warnings.simplefilter("ignore")

app = FastAPI()

# # Global storage for clustering results
global_graph = None
global_partition = None
global_communities = {}

# Visualization of searched Function
def visualize_communities(communities, G, num1, num2, partition):
    try:
        num1_community = None
        num2_community = None
        #print(f"num1_community : {num1_community}, num2_community : {num2_community}")

        # Ensure num1 and num2 exist in the graph
        if num1 not in G.nodes() or num2 not in G.nodes():
            print(f"Error: One or both nodes ({num1}, {num2}) do not exist in the graph.")
            return None
            # sys.exit(1)
        # print(f"num1 : {num1}, num2: {num2}")
        # print(f"num1_community : {num1_community}, num2_community : {num2_community}")

        # Identify which communities num1 and num2 belong to
        for community_id, members in communities.items():
            if num1 in members:
                num1_community = community_id
            if num2 in members:
                num2_community = community_id
            if num1_community is not None and num2_community is not None:
                break  # Stop early if both are found
        
        #print(f"num1_community : {num1_community}, num2_community : {num2_community}")
        #print(f"num1 : {num1}, num2: {num2}")

        if num1_community is None or num2_community is None:
            print(f"Error: One or both nodes ({num1}, {num2}) do not belong to any community.")
            return None            
            #sys.exit(1)

        selected_communities_id = {num1_community, num2_community}
        print(f"Selected Communities: {selected_communities_id}")

        # Extract nodes from the selected communities
        selected_nodes = set()
        for community_id in selected_communities_id:
            if community_id is not None:
                selected_nodes.update(communities[community_id])

        # Ensure there are selected nodes to form a subgraph
        if not selected_nodes:
            print("Error: No nodes found for the selected communities.")
            return None
            # sys.exit(1)

        # Create a subgraph containing only the selected nodes
        subgraph = G.subgraph(selected_nodes)

        # Filter edges to include only those within or between selected communities
        filtered_edges = []
        for u, v in subgraph.edges():
            u_community = None
            v_community = None
            for community_id, members in communities.items():
                if u in members:
                    u_community = community_id
                if v in members:
                    v_community = community_id
                if u_community is not None and v_community is not None:
                    break
            if u_community in selected_communities_id and v_community in selected_communities_id:
                filtered_edges.append((u, v))

        plt.figure(figsize=(10, 8))

        # Compute separate layouts for each community
        community_positions = {}
        base_x = 0.5  # Offset for community separation
        spacing = 2.5  # Distance between communities

        for i, community_id in enumerate(selected_communities_id):
            if community_id is None:
                continue

            community_nodes = [node for node in communities.get(community_id, []) if node in subgraph.nodes()]
            if not community_nodes:
                continue

            # Generate positions only for this community
            sub_pos = nx.spring_layout(G.subgraph(community_nodes), seed=11, scale=1.0)

            # Offset community positions
            for node, (x, y) in sub_pos.items():
                community_positions[node] = (x + base_x, y)

            base_x += spacing  # Move the next community further right

        # Node colors based on partition
        unique_community_ids = set(partition[node] for node in subgraph.nodes() if node in partition)
        cmap = plt.cm.get_cmap('jet', len(unique_community_ids))
        community_colors = {comm_id: cmap(i) for i, comm_id in enumerate(sorted(unique_community_ids))}
        node_colors = {node: community_colors[partition[node]] for node in subgraph.nodes()}

        # Draw nodes, edges, and labels
        nx.draw_networkx_nodes(subgraph, community_positions, node_color=[node_colors[node] for node in subgraph.nodes()], node_size=200)
        nx.draw_networkx_labels(subgraph, community_positions, font_size=8, font_color="black")

        # Draw edges (using filtered_edges)
        nx.draw_networkx_edges(subgraph, community_positions, alpha=0.5, edgelist=filtered_edges, arrows=True, edge_color="grey")

        # Highlight num1 and num2
        if num1 in community_positions:
            nx.draw_networkx_nodes(subgraph, community_positions, nodelist=[num1], node_color='red', node_size=300)
        if num2 in community_positions:
            nx.draw_networkx_nodes(subgraph, community_positions, nodelist=[num2], node_color='green', node_size=300)

        # Draw ellipses around communities with matching colors
        ax = plt.gca()
        for community_id in selected_communities_id:
            if community_id is None:
                continue

            community_nodes = [node for node in communities.get(community_id, []) if node in community_positions]
            if not community_nodes:
                continue

            community_pos = [community_positions[node] for node in community_nodes]
            x_coords, y_coords = zip(*community_pos)
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            width = max(x_coords) - min(x_coords) + 0.5
            height = max(y_coords) - min(y_coords) + 0.5

            representative_node = community_nodes[0]
            community_color = node_colors.get(representative_node, "black")

            ellipse = Ellipse(
                xy=(center_x, center_y), width=width, height=height,
                edgecolor=community_color, facecolor='none', linestyle='--', lw=2
            )
            ax.add_patch(ellipse)

        # Add edge weight labels (using filtered_edges)
        edge_weights = nx.get_edge_attributes(subgraph, 'weight')
        filtered_edge_weights = {edge: edge_weights[edge] for edge in filtered_edges if edge in edge_weights}  # Filter edge weights
        nx.draw_networkx_edge_labels(subgraph, community_positions, edge_labels=filtered_edge_weights, font_color='blue', font_size=8)

        plt.title(f"Communities of {num1} and {num2}")
        plt.savefig("selected_Numbers.png")
        plt.show()
        plt.close()
        with open("selected_Numbers.png", "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        # sys.exit(1)


@app.post("/leiden-clustering-post/")
async def leiden_clustering(file: UploadFile = File(...)):
    """Processes a CSV file, applies Leiden clustering, computes frequency, and returns results."""
    global global_graph, global_partition, global_communities

    try:
        # Read CSV and compute frequency
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if not {'Calling Party', 'Called Party', 'Call Date', 'Call Type', 'Start Time', 'End Time', 'Duration (seconds)' }.issubset(df.columns):
            return {"error": "Missing required columns: 'Calling Party', 'Called Party', 'Call Date', 'Call Type', 'Start Time', 'End Time', 'Duration (seconds)'"}
        else:
            print("_" * 150)        
            print(df.head())
            print("_" * 150)        


                
        columns_to_convert = ["Calling Party", "Called Party", "IMEI", "IMSI", "Duration (seconds)"]
        df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors="coerce").astype("Int64")
        df['Call Date'] = pd.to_datetime(df['Call Date'])

        print("_" * 150)
        print(df.head())
        print("_" * 150)

        # Creating a Directed Graph (weighted by frequency of calls or SMS)
        G = nx.DiGraph()

        try:
            # Compute frequency of interactions (calls + SMS) between each pair
            interaction_counts = df.groupby(['Calling Party', 'Called Party']).size().reset_index(name='Frequency')

            # Adding directed edges with frequency as weights
            for _, row in interaction_counts.iterrows():
                G.add_edge(row['Calling Party'], row['Called Party'], weight=row['Frequency'])

            # Removing self-loops
            G.remove_edges_from(nx.selfloop_edges(G))

            # full_graph_vis = visualize_full_graph(G)

            # Visualizing the graph
            plt.figure(figsize=(10, 8))
            random.seed(50)
            pos = nx.kamada_kawai_layout(G)  # Position nodes using a force-directed algorithm
            weights = nx.get_edge_attributes(G, 'weight')

            # Create a list of edge widths based on weights
            widths = [weights[edge] for edge in G.edges()]

            # Draw the graph with varying edge widths
            nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue", edge_color="gray", font_size=7, width=widths)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_color="red")
            plt.title("Directed and Weighted Graph (Frequency of Interactions)")
            plt.tight_layout()
            plt.savefig("Directed_Graph.png")
            plt.show()
            with open("Directed_Graph.png", "rb") as image_file:
                full_graph_vis = base64.b64encode(image_file.read()).decode()
            plt.close()
            

        except KeyError as e:
            print(f"Error: Missing column in your CSV file: {e}. Please ensure 'Calling Party', 'Called Party', and 'Frequency' columns are present.")
            sys.exit(1)
        except (TypeError, ValueError) as e:
            print(f"Error: Data type mismatch or unexpected value: {e}. Please check the data in your CSV file.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during graph creation: {e}")
            sys.exit(1)

        # Converting NetworkX graph to igraph, including weights
        try:
            edge_list = [(u, v, G[u][v]['weight']) for u, v in G.edges()]  # Include weights in edge list
        except KeyError:
            print("Error: 'weight' attribute not found on all edges. Ensure all edges have weights.")
            sys.exit(1)  # Exit with error code 1
        except Exception as e:
            print(f"An unexpected error occurred during edge list creation: {e}")
            sys.exit(1)

        # Creating the igraph graph
        try:
            ig_graph = ig.Graph.TupleList(edge_list, directed=True, edge_attrs=['weight'])  # Specify edge attribute
        except Exception as e:
            print(f"Error creating igraph graph: {e}")
            sys.exit(1)        

        
        # Optimizing the graph with Leiden algorithm (using weights)
        try:
            partition_leiden = la.find_partition(ig_graph, la.ModularityVertexPartition, weights='weight')  # Use weights
        except Exception as e:
            print(f"Error during Leiden algorithm execution: {e}")
            sys.exit(1)

        # Getting the communities from the Leiden algorithm
        try:
            partition = {}
            for i, node in enumerate(ig_graph.vs):
                partition[node["name"]] = partition_leiden.membership[i]
        except Exception as e:
            print(f"Error extracting communities from Leiden results: {e}")
            sys.exit(1)

        # Computing the modularity score
        try:
            modularity_score = partition_leiden.modularity
        except AttributeError:
            print("Error: 'modularity' attribute not found in Leiden results.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while computing modularity: {e}")
            sys.exit(1)

        # Store computed results globally
        global_graph = G
        global_partition = partition
        global_communities = {i: [node for node, comm in partition.items() if comm == i] for i in set(partition.values())}

        # # Visualize the communities with polygon layout
        # try:
        #     plt.figure(figsize=(10, 8))

        #     # Group nodes by community, starting from Community 1
        #     communities = {}
        #     for node, community_id in partition.items():
        #         # Adding 1 to the community_id to start from 1
        #         adjusted_community_id = community_id + 1

        #         if adjusted_community_id not in communities:
        #             communities[adjusted_community_id] = []
        #         communities[adjusted_community_id].append(node)

        #     # Adjusting positions within communities to gather nodes together
        #     adjusted_pos = {}

        #     # Increasing the proximity between nodes of the same community
        #     for community_id, nodes in communities.items():
        #         x_pos = [random.uniform(-1, 1) for _ in nodes]
        #         y_pos = [random.uniform(-1, 1) for _ in nodes]
        #         for node, x, y in zip(nodes, x_pos, y_pos):
        #             adjusted_pos[node] = (x + community_id * 1.5, y)

        #     # Nodes colored by community
        #     num_communities = len(communities)
        #     cmap = plt.cm.get_cmap('jet', num_communities)
        #     colors = [cmap(partition[node]) for node in G.nodes()]

        #     nx.draw_networkx_nodes(G, adjusted_pos, node_color=colors, cmap=plt.cm.jet, node_size=100)

        #     # Draw directed edges
        #     nx.draw_networkx_edges(G, adjusted_pos, alpha=0.5, edgelist=G.edges(), arrows=True, edge_color="grey")

        #     # Draw node labels
        #     nx.draw_networkx_labels(G, adjusted_pos, font_size=7, font_color="black")

        #     # Add community borders
        #     for community_id, nodes in communities.items():
        #         community_pos = [adjusted_pos[node] for node in nodes]
        #         x_coords, y_coords = zip(*community_pos)

        #         if nodes:
        #             try:
        #                 node_index = list(G.nodes()).index(nodes[0])
        #                 community_color = colors[node_index]
        #             except ValueError:
        #                 community_color = 'black'
        #         else:
        #             community_color = 'black'

        #         line_style = '--'

        #         plt.gca().add_patch(plt.Polygon(list(zip(x_coords, y_coords)),
        #                                     fill=None,
        #                                     edgecolor=community_color,
        #                                     linestyle=line_style,
        #                                     lw=2))

        #     plt.title("Community Detection in Social Network (Call Data) - Leiden")
        #     plt.savefig("community_detection_leiden_polygon.png")
        #     plt.show()
        #     with open("community_detection_leiden_polygon.png", "rb") as image_file:
        #         clustering_vis_polygon = base64.b64encode(image_file.read()).decode()
        #     plt.close()
            

        # except Exception as e:
        #     print(f"An error occurred during visualization: {e}")
        #     sys.exit(1)

        # Visualizing the communities with ellipse layout
        try:
            plt.figure(figsize=(10, 8))

            # Group nodes by community, starting from Community 1
            communities = {}
            for node, community_id in partition.items():
                # Add 1 to the community_id to start from 1 instead of 0
                adjusted_community_id = community_id + 1

                if adjusted_community_id not in communities:
                    communities[adjusted_community_id] = []
                communities[adjusted_community_id].append(node)

            # Adjust positions within communities to gather nodes together
            adjusted_pos = {}

            # Increase the proximity between nodes of the same community
            for community_id, nodes in communities.items():
                x_pos = [random.uniform(-1, 1) for _ in nodes]
                y_pos = [random.uniform(-1, 1) for _ in nodes]
                for node, x, y in zip(nodes, x_pos, y_pos):
                    adjusted_pos[node] = (x + community_id * 2, y)

            # Nodes colored by community
            num_communities = len(communities)
            cmap = plt.cm.get_cmap('jet', num_communities)
            colors = [cmap(partition[node]) for node in G.nodes()]

            nx.draw_networkx_nodes(G, adjusted_pos, node_color=colors, cmap=plt.cm.jet, node_size=100)

            # Draw directed edges
            nx.draw_networkx_edges(G, adjusted_pos, alpha=0.5, edgelist=G.edges(), arrows=True, edge_color="grey")

            # Draw node labels
            nx.draw_networkx_labels(G, adjusted_pos, font_size=7, font_color="black")

            # Add community borders (using Ellipse for oval/circular shapes)
            for community_id, nodes in communities.items():
                # Get positions of nodes in this community
                community_pos = [adjusted_pos[node] for node in nodes]
                x_coords, y_coords = zip(*community_pos)

                # Calculate center and size of the ellipse
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                width = max(x_coords) - min(x_coords) + 0.5  # Adjust +0.5 for padding
                height = max(y_coords) - min(y_coords) + 0.5  # Adjust +0.5 for padding

                # Get community color (same logic as before)
                if nodes:
                    try:
                        node_index = list(G.nodes()).index(nodes[0])
                        community_color = colors[node_index]
                    except ValueError:
                        community_color = 'black'
                else:
                    community_color = 'black'

                # Create an Ellipse patch
                ellipse = Ellipse(xy=(center_x, center_y), width=width, height=height,
                                edgecolor=community_color, facecolor='none', linestyle='--', lw=2)

                # Add the ellipse to the plot
                plt.gca().add_patch(ellipse)

            plt.title("Community Detection in Call Data by Leiden Algorithm (Ellipse Shaped)")
            plt.savefig("community_detection_leiden_ellipse.png")            
            plt.show()
            with open("community_detection_leiden_ellipse.png", "rb") as image_file:
                clustering_vis_ellipse = base64.b64encode(image_file.read()).decode()
            plt.close()

        except Exception as e:
            print(f"An error occurred during visualization: {e}")
            sys.exit(1)

        # Displaying the detected communities in sorted order
        print(f"\nModularity Score : {modularity_score}")
        print(f"\nDetected Communities :")
        print("-" * 150)

        try:
            for community_id in sorted(communities.keys()):
                members = sorted(communities[community_id])
                print(f"Community {community_id}: {members}")
                print("-" * 150)
        except KeyError as e:
            print(f"Error: Community ID not found: {e}")
        except TypeError as e:
            print(f"Error: Invalid community structure: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return {
            "message": "Leiden clustering completed successfully!",
            "modularity_score": modularity_score,
            "Communities": list(global_communities.items()),
            "Full_graph_visualization": full_graph_vis,
            # "clustering_visualization_polygon": clustering_vis_polygon,
            "clustering_visualization_ellipse": clustering_vis_ellipse
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/check-community-get/")
async def check_community(num1: str = Query(...), num2: str = Query(...)):
    num1 = int(num1)
    num2 = int(num2)
    """Checks if two nodes belong to the same community and returns visualization."""
    if global_graph is None or global_partition is None:
        return {"error": "Run /leiden-clustering/ first to compute communities!"}

    num1_community = global_partition.get(num1)
    num2_community = global_partition.get(num2)

    # Printing the Values and types of the two nodes and their communities to the terminal
    print("-" * 150)
    print(f"num1 : {num1} , type : {type(num1)} - community : {num1_community} , type : {type(num1_community)}")
    print(f"num2 : {num2} , type : {type(num2)} - community : {num2_community} , type : {type(num2_community)}")
    print("-" * 150)

    if num1_community == num2_community and num1_community is not None:
        vis = visualize_communities(global_communities, global_graph, num1, num2, global_partition)
        return {"same_community": True, "community_id": num1_community, "visualization": vis}
    else:
        #error
        vis = visualize_communities(global_communities, global_graph, num1, num2, global_partition)
        return {
            "same_community": False,
            "num1_community": num1_community,
            "num2_community": num2_community,
            "visualization": vis
        }
