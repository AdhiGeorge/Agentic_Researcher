
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json
from datetime import datetime

def render_swarm_visualizer(visualization_data: Dict[str, Any]):
    """Render a visualization of the swarm's agents and their interactions"""
    st.subheader("Swarm Visualization")
    
    # Check if we have visualization data
    if not visualization_data or 'nodes' not in visualization_data or 'edges' not in visualization_data:
        st.warning("No swarm visualization data available")
        return
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes with status as attributes
    for node in visualization_data.get('nodes', []):
        node_id = node.get('id', '')
        status = node.get('status', 'unknown')
        G.add_node(node_id, status=status)
    
    # Add edges
    for edge in visualization_data.get('edges', []):
        from_node = edge.get('from', '')
        to_node = edge.get('to', '')
        if from_node and to_node and from_node in G.nodes and to_node in G.nodes:
            G.add_edge(from_node, to_node)
    
    # Create the plot
    if G.number_of_nodes() > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Define colors for different statuses
        status_colors = {
            'idle': 'lightgray',
            'running': 'lightblue',
            'completed': 'lightgreen',
            'error': 'salmon',
            'waiting': 'lightyellow',
            'unknown': 'white'
        }
        
        # Get node colors based on status
        node_colors = [status_colors.get(G.nodes[node].get('status', 'unknown'), 'white') for node in G.nodes]
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=500)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title("Agent Interaction Graph")
        plt.axis('off')
        
        # Display the plot
        st.pyplot(fig)
        
        # Display stats
        st.markdown("### Swarm Statistics")
        st.markdown(f"- **Agents**: {G.number_of_nodes()}")
        st.markdown(f"- **Interactions**: {G.number_of_edges()}")
        
        # Status counts
        status_counts = {}
        for node in G.nodes:
            status = G.nodes[node].get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Display status counts
        for status, count in status_counts.items():
            color = status_colors.get(status, 'gray')
            st.markdown(f"- <span style='color:{color};'>**{status.capitalize()}**</span>: {count}", unsafe_allow_html=True)
    else:
        st.warning("No agents to display in the swarm visualization")
    
    # Add timestamps
    st.markdown(f"_Visualization generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
