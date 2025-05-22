"""
Knowledge Graph Builder for Agentic Researcher

This module implements a knowledge graph builder that extracts entities from research content
and builds a semantic graph to represent relationships between concepts.
"""

import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
import spacy
from spacy.tokens import Doc
import json

# For visualization if needed
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class KnowledgeGraphBuilder:
    """
    Knowledge Graph Builder for constructing and managing semantic knowledge graphs.
    
    This class extracts named entities and key concepts from text, builds relationships
    between them, and provides methods for graph pruning and retrieval.
    
    Attributes:
        logger (logging.Logger): Logger for the graph builder
        nlp (spacy.Language): SpaCy NLP model for entity recognition
        qdrant_manager: Qdrant database manager for vector storage
        graph (nx.Graph): NetworkX graph for representing the knowledge structure
        entities (Dict[str, Dict]): Dictionary of extracted entities with metadata
    """
    
    def __init__(self, qdrant_manager=None):
        """Initialize the KnowledgeGraphBuilder.
        
        Args:
            qdrant_manager: Qdrant database manager for storing vectors
        """
        self.logger = logging.getLogger("utils.graph_builder")
        
        # Initialize SpaCy for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_md")
            self.logger.info("Loaded SpaCy model: en_core_web_md")
        except Exception as e:
            self.logger.warning(f"SpaCy model not found: {str(e)}")
            try:
                # Download a smaller model as fallback
                self.logger.info("Downloading SpaCy model en_core_web_sm...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded SpaCy model: en_core_web_sm")
            except Exception as e2:
                self.logger.error(f"Failed to download SpaCy model: {str(e2)}")
                raise ImportError("SpaCy models are required. Please install with: pip install spacy && python -m spacy download en_core_web_sm")
        
        # Reference to the Qdrant manager
        self.qdrant_manager = qdrant_manager
        
        # Initialize graph data structures
        self.graph = nx.Graph()
        self.entities = {}
        
        self.logger.info("KnowledgeGraphBuilder initialized")
    
    def process_text(self, text: str, source_id: str, metadata: Dict = None) -> List[Dict]:
        """Process text to extract entities and relationships.
        
        Args:
            text (str): The text to process
            source_id (str): Identifier for the source document
            metadata (Dict, optional): Additional metadata. Defaults to None.
            
        Returns:
            List[Dict]: List of extracted entities with metadata
        """
        if not text or len(text) < 10:
            self.logger.warning(f"Text too short to process: {len(text) if text else 0} chars")
            return []
        
        metadata = metadata or {}
        self.logger.info(f"Processing text from source {source_id}: {len(text)} chars")
        
        # Process with SpaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = self._extract_entities(doc, source_id, metadata)
        
        # Build relationships between entities
        self._build_relationships(entities, doc)
        
        # Store in Qdrant if available
        if self.qdrant_manager:
            self._store_entities_in_qdrant(entities, source_id)
        
        return entities
    
    def _extract_entities(self, doc: Doc, source_id: str, metadata: Dict) -> List[Dict]:
        """Extract named entities and key concepts from the document.
        
        Args:
            doc (Doc): SpaCy document
            source_id (str): Source identifier
            metadata (Dict): Additional metadata
            
        Returns:
            List[Dict]: Extracted entities
        """
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entity_id = f"{ent.text.lower().replace(' ', '_')}_{ent.label_}"
            
            # Create or update entity
            if entity_id not in self.entities:
                entity = {
                    "id": entity_id,
                    "text": ent.text,
                    "type": ent.label_,
                    "sources": [source_id],
                    "metadata": metadata.copy(),
                    "count": 1
                }
                self.entities[entity_id] = entity
                
                # Add to graph
                self.graph.add_node(entity_id, **entity)
            else:
                # Update existing entity
                entity = self.entities[entity_id]
                entity["count"] += 1
                if source_id not in entity["sources"]:
                    entity["sources"].append(source_id)
                
                # Update graph node
                self.graph.nodes[entity_id]["count"] = entity["count"]
                self.graph.nodes[entity_id]["sources"] = entity["sources"]
            
            entities.append(entity)
        
        # Extract key noun phrases as concepts
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3 and chunk.root.pos_ in ("NOUN", "PROPN"):
                concept_id = f"concept_{chunk.text.lower().replace(' ', '_')}"
                
                # Create or update concept
                if concept_id not in self.entities:
                    concept = {
                        "id": concept_id,
                        "text": chunk.text,
                        "type": "CONCEPT",
                        "sources": [source_id],
                        "metadata": metadata.copy(),
                        "count": 1
                    }
                    self.entities[concept_id] = concept
                    
                    # Add to graph
                    self.graph.add_node(concept_id, **concept)
                else:
                    # Update existing concept
                    concept = self.entities[concept_id]
                    concept["count"] += 1
                    if source_id not in concept["sources"]:
                        concept["sources"].append(source_id)
                    
                    # Update graph node
                    self.graph.nodes[concept_id]["count"] = concept["count"]
                    self.graph.nodes[concept_id]["sources"] = concept["sources"]
                
                entities.append(concept)
        
        self.logger.debug(f"Extracted {len(entities)} entities from source {source_id}")
        return entities
    
    def _build_relationships(self, entities: List[Dict], doc: Doc) -> None:
        """Build relationships between entities based on co-occurrence and syntax.
        
        Args:
            entities (List[Dict]): List of extracted entities
            doc (Doc): SpaCy document
        """
        # If no entities, nothing to do
        if not entities:
            return
        
        # Map entities to their sentence indices
        entity_sentences = {}
        for sent_idx, sent in enumerate(doc.sents):
            for ent in sent.ents:
                entity_id = f"{ent.text.lower().replace(' ', '_')}_{ent.label_}"
                if entity_id not in entity_sentences:
                    entity_sentences[entity_id] = []
                entity_sentences[entity_id].append(sent_idx)
        
        # Connect entities that appear in the same sentence
        entity_ids = [entity["id"] for entity in entities]
        for i, entity1_id in enumerate(entity_ids):
            if entity1_id not in entity_sentences:
                continue
                
            sent_indices1 = entity_sentences[entity1_id]
            
            for j in range(i+1, len(entity_ids)):
                entity2_id = entity_ids[j]
                if entity2_id not in entity_sentences:
                    continue
                    
                sent_indices2 = entity_sentences[entity2_id]
                
                # Find common sentences
                common_sents = set(sent_indices1).intersection(set(sent_indices2))
                
                if common_sents:
                    # Create or update edge
                    if not self.graph.has_edge(entity1_id, entity2_id):
                        self.graph.add_edge(
                            entity1_id, 
                            entity2_id, 
                            weight=len(common_sents),
                            type="co-occurrence"
                        )
                    else:
                        # Increase weight for existing relationship
                        self.graph[entity1_id][entity2_id]["weight"] += len(common_sents)
    
    def _store_entities_in_qdrant(self, entities: List[Dict], source_id: str) -> None:
        """Store entities in Qdrant with relationships as metadata.
        
        Args:
            entities (List[Dict]): Entities to store
            source_id (str): Source identifier
        """
        if not self.qdrant_manager or not entities:
            return
        
        try:
            # For each entity, store its connections as metadata
            for entity in entities:
                entity_id = entity["id"]
                
                # Get all neighbors
                if entity_id in self.graph:
                    neighbors = list(self.graph.neighbors(entity_id))
                    
                    # Add neighbor data to metadata
                    entity_metadata = entity.copy()
                    entity_metadata["connected_entities"] = neighbors
                    entity_metadata["source_id"] = source_id
                    
                    # Store in Qdrant
                    # This assumes text_embedding method exists in qdrant_manager
                    if hasattr(self.qdrant_manager, "text_embedding"):
                        self.qdrant_manager.text_embedding(
                            text=entity["text"],
                            metadata=entity_metadata,
                            collection="entities"
                        )
        except Exception as e:
            self.logger.error(f"Error storing entities in Qdrant: {str(e)}")
    
    def prune_graph_for_query(self, query: str, max_nodes: int = 50) -> nx.Graph:
        """Prune the knowledge graph to include only nodes relevant to the query.
        
        Args:
            query (str): The query to use for pruning
            max_nodes (int, optional): Maximum number of nodes to include. Defaults to 50.
            
        Returns:
            nx.Graph: Pruned graph
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            self.logger.warning("Graph is empty, nothing to prune")
            return nx.Graph()
        
        # Process query with SpaCy
        query_doc = self.nlp(query)
        
        # Extract key entities and concepts from the query
        query_entities = []
        for ent in query_doc.ents:
            query_entities.append(ent.text.lower())
        
        for chunk in query_doc.noun_chunks:
            if chunk.root.pos_ in ("NOUN", "PROPN"):
                query_entities.append(chunk.text.lower())
        
        # Score nodes by relevance to query
        node_scores = {}
        for node_id, node_data in self.graph.nodes(data=True):
            score = 0
            node_text = node_data.get("text", "").lower()
            
            # Check for exact matches
            for query_entity in query_entities:
                if query_entity in node_text or node_text in query_entity:
                    score += 5
            
            # Add score based on node centrality
            score += self.graph.degree(node_id)
            
            # Add score based on frequency
            score += node_data.get("count", 0)
            
            node_scores[node_id] = score
        
        # Sort nodes by score
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N nodes
        top_nodes = [node_id for node_id, _ in sorted_nodes[:max_nodes]]
        
        # Create subgraph with these nodes
        pruned_graph = self.graph.subgraph(top_nodes).copy()
        
        self.logger.info(f"Pruned graph from {self.graph.number_of_nodes()} to {pruned_graph.number_of_nodes()} nodes")
        return pruned_graph
    
    def get_relevant_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """Get entities most relevant to the query.
        
        Args:
            query (str): The query text
            limit (int, optional): Maximum number of entities to return. Defaults to 10.
            
        Returns:
            List[Dict]: List of relevant entities with metadata
        """
        # Process query
        query_doc = self.nlp(query)
        
        # Extract query entities and keywords
        query_terms = set()
        for ent in query_doc.ents:
            query_terms.add(ent.text.lower())
        
        for token in query_doc:
            if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ"):
                query_terms.add(token.lemma_.lower())
        
        # Score all entities
        entity_scores = []
        for entity_id, entity_data in self.entities.items():
            score = 0
            entity_text = entity_data["text"].lower()
            
            # Score exact matches
            for term in query_terms:
                if term in entity_text or entity_text in term:
                    score += 3
            
            # Add centrality score
            if entity_id in self.graph:
                score += len(list(self.graph.neighbors(entity_id)))
            
            # Add frequency score
            score += entity_data.get("count", 0)
            
            entity_scores.append((entity_id, score))
        
        # Sort by score
        sorted_entities = sorted(entity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top entities
        result = []
        for entity_id, score in sorted_entities[:limit]:
            entity = self.entities[entity_id].copy()
            entity["relevance_score"] = score
            result.append(entity)
        
        return result
    
    def visualize_graph(self, graph=None, output_file=None):
        """Visualize the knowledge graph.
        
        Args:
            graph (nx.Graph, optional): Graph to visualize. If None, uses the full graph.
            output_file (str, optional): Output file path. Defaults to None.
        """
        if plt is None:
            self.logger.warning("Matplotlib not available for visualization")
            return
        
        # Use provided graph or full graph
        g = graph if graph is not None else self.graph
        
        if g.number_of_nodes() == 0:
            self.logger.warning("Graph is empty, nothing to visualize")
            return
        
        # Set up colors based on entity types
        colors = []
        node_types = nx.get_node_attributes(g, "type")
        for node in g.nodes():
            node_type = node_types.get(node, "")
            if "PERSON" in node_type:
                colors.append("lightblue")
            elif "ORG" in node_type:
                colors.append("lightgreen")
            elif "GPE" in node_type or "LOC" in node_type:
                colors.append("lightcoral")
            elif "CONCEPT" in node_type:
                colors.append("lightyellow")
            else:
                colors.append("gray")
        
        # Set up node sizes based on centrality
        sizes = []
        for node in g.nodes():
            degree = g.degree(node)
            sizes.append(100 + (degree * 20))
        
        # Create plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(g, seed=42)
        nx.draw_networkx(
            g, pos, 
            with_labels=True, 
            node_color=colors,
            node_size=sizes,
            font_size=8,
            width=0.5,
            alpha=0.8
        )
        
        # Save or show
        if output_file:
            plt.savefig(output_file)
            self.logger.info(f"Graph visualization saved to {output_file}")
        else:
            plt.show()
    
    def serialize(self, path: str = None) -> Dict:
        """Serialize the graph to JSON format.
        
        Args:
            path (str, optional): Path to save the serialized graph. Defaults to None.
            
        Returns:
            Dict: Serialized graph data
        """
        # Convert graph to serializable format
        nodes = [
            {**self.graph.nodes[node]} for node in self.graph.nodes()
        ]
        
        edges = [
            {
                "source": u,
                "target": v,
                **self.graph[u][v]
            } for u, v in self.graph.edges()
        ]
        
        data = {
            "nodes": nodes,
            "edges": edges,
            "entity_count": len(self.entities)
        }
        
        # Save to file if path provided
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Graph serialized to {path}")
        
        return data
    
    def deserialize(self, data: Dict = None, path: str = None) -> None:
        """Deserialize a graph from JSON format.
        
        Args:
            data (Dict, optional): Serialized graph data. Defaults to None.
            path (str, optional): Path to load serialized graph from. Defaults to None.
        """
        if path and not data:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        if not data:
            self.logger.error("No data provided for deserialization")
            return
        
        # Create new graph
        self.graph = nx.Graph()
        self.entities = {}
        
        # Add nodes
        for node_data in data["nodes"]:
            node_id = node_data.pop("id", None)
            if node_id:
                self.graph.add_node(node_id, **node_data)
                self.entities[node_id] = node_data
        
        # Add edges
        for edge_data in data["edges"]:
            source = edge_data.pop("source", None)
            target = edge_data.pop("target", None)
            if source and target:
                self.graph.add_edge(source, target, **edge_data)
        
        self.logger.info(f"Graph deserialized with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")


# Example usage with real-world scenarios
if __name__ == "__main__":
    import os
    import sys
    import time
    from pathlib import Path
    import tempfile
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("===== Knowledge Graph Builder Example Usage =====")
    print("This example demonstrates knowledge graph construction")
    print("and entity extraction for research content analysis.")
    
    # Fix imports for running directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp(prefix="graph_builder_example_")
    print(f"\nCreated temporary directory for outputs: {temp_dir}")
    
    # Create graph builder
    graph_builder = KnowledgeGraphBuilder()
    
    print("\nExample 1: Basic Entity Extraction and Graph Building")
    print("-" * 60)
    
    # Example text about VIX
    text1 = """
    The Volatility Index (VIX) is a real-time market index that represents the market's expectation 
    of 30-day forward-looking volatility. Derived from the price inputs of the S&P 500 index options, 
    it provides a measure of market risk and investors' sentiments. It is also known as the 'Fear Index' 
    or 'Fear Gauge'. The VIX was created by the Chicago Board Options Exchange (CBOE) and is maintained 
    by Cboe Global Markets. It is an important measure of market risk and is often used to gauge the 
    level of fear, stress, or risk in the market.
    """
    
    print("Processing text about the VIX index...")
    start_time = time.time()
    entities = graph_builder.process_text(text1, "vix_intro")
    processing_time = time.time() - start_time
    
    # Print extracted entities
    print(f"\nExtracted {len(entities)} entities in {processing_time:.2f}s:")
    for entity in entities[:5]:  # Show first 5
        print(f"  - {entity['text']} ({entity['type']})")
    
    # Show graph stats
    print(f"\nInitial graph has {graph_builder.graph.number_of_nodes()} nodes and {graph_builder.graph.number_of_edges()} edges")
    
    print("\nExample 2: Multi-Document Knowledge Integration")
    print("-" * 60)
    
    # Add a second text about options trading
    text2 = """
    Options trading involves buying and selling options contracts on financial securities. Options give 
    the buyer the right, but not the obligation, to buy or sell an underlying asset at a specified price 
    (strike price) on or before a certain date (expiration date). Call options give the holder the right 
    to buy the underlying asset, while put options give the holder the right to sell the underlying asset. 
    Options are often used for hedging, income generation, and speculative purposes. The Chicago Board 
    Options Exchange (CBOE) is the largest U.S. options exchange where these contracts are traded.
    """
    
    print("Processing text about options trading...")
    entities2 = graph_builder.process_text(text2, "options_basics", metadata={"category": "finance"})
    
    # Add a third text about machine learning
    text3 = """
    Machine learning is a branch of artificial intelligence focused on building systems that learn from data. 
    These algorithms identify patterns and make decisions with minimal human intervention. In finance, 
    machine learning is used for market prediction, risk assessment, and algorithmic trading. 
    Neural networks, a popular machine learning technique, are particularly effective for analyzing complex 
    financial time series data such as stock prices and volatility indices.
    """
    
    print("Processing text about machine learning in finance...")
    entities3 = graph_builder.process_text(text3, "ml_finance", metadata={"category": "technology"})
    
    # Show updated graph stats
    print(f"\nGraph now has {graph_builder.graph.number_of_nodes()} nodes and {graph_builder.graph.number_of_edges()} edges")
    
    # Identify common entities/concepts across documents
    # This would normally use more sophisticated methods
    nodes = list(graph_builder.graph.nodes(data=True))
    
    # Find nodes with multiple sources
    cross_document_nodes = []
    for node_id, data in nodes:
        sources = set()
        if 'sources' in data:
            if isinstance(data['sources'], list):
                sources = set(data['sources'])
            elif isinstance(data['sources'], str):
                sources = {data['sources']}
        
        if len(sources) > 1:
            cross_document_nodes.append((node_id, data, sources))
    
    print(f"\nFound {len(cross_document_nodes)} concepts that appear across multiple documents:")
    for node_id, data, sources in cross_document_nodes[:3]:  # Show top 3
        text = data.get('text', node_id)
        print(f"  - {text} (appears in: {', '.join(sources)})")
    
    print("\nExample 3: Query-Based Graph Exploration")
    print("-" * 60)
    
    # Try different queries to explore the graph
    test_queries = [
        "How is the VIX calculated?",
        "What are options trading strategies?",
        "How is machine learning used in finance?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Get relevant entities
        relevant = graph_builder.get_relevant_entities(query, limit=3)
        
        print(f"Top relevant entities:")
        for entity in relevant:
            print(f"  - {entity['text']} (Score: {entity.get('relevance_score', 'N/A')})")
        
        # Get pruned graph for the query
        pruned_graph = graph_builder.prune_graph_for_query(query, max_nodes=10)
        print(f"Pruned graph has {pruned_graph.number_of_nodes()} nodes and {pruned_graph.number_of_edges()} edges")
    
    print("\nExample 4: Graph Visualization")
    print("-" * 60)
    
    # Check if visualization is available
    if plt:
        print("Matplotlib is available for visualization")
        
        # Visualize the full graph (uncomment for interactive visualization)
        # graph_builder.visualize_graph()
        
        # Visualize and save pruned graphs for each query
        for i, query in enumerate(test_queries):
            pruned_graph = graph_builder.prune_graph_for_query(query, max_nodes=8)
            output_file = os.path.join(temp_dir, f"query_{i+1}_graph.png")
            
            try:
                # Visualize and save to file
                graph_builder.visualize_graph(pruned_graph, output_file)
                print(f"Saved visualization for '{query}' to {output_file}")
            except Exception as e:
                print(f"Error visualizing graph: {str(e)}")
    else:
        print("Matplotlib not available for visualization")
    
    print("\nExample 5: Graph Serialization and Persistence")
    print("-" * 60)
    
    # Serialize the graph
    graph_file = os.path.join(temp_dir, "knowledge_graph.json")
    
    print(f"Serializing graph with {graph_builder.graph.number_of_nodes()} nodes...")
    start_time = time.time()
    graph_data = graph_builder.serialize(graph_file)
    serialization_time = time.time() - start_time
    
    print(f"Graph serialized in {serialization_time:.2f}s to {graph_file}")
    print(f"Serialized data size: {os.path.getsize(graph_file) / 1024:.1f} KB")
    
    # Deserialize to a new graph builder
    print("\nDeserializing graph to a new instance...")
    new_graph_builder = KnowledgeGraphBuilder()
    new_graph_builder.deserialize(path=graph_file)
    
    print(f"Deserialized graph has {new_graph_builder.graph.number_of_nodes()} nodes and {new_graph_builder.graph.number_of_edges()} edges")
    
    print("\nExample 6: Integration with Research Pipeline")
    print("-" * 60)
    
    # Simulate integration with other components
    print("This example shows how the graph builder integrates with other components")
    print("of the Agentic Researcher system (simulation only).")
    
    # Simulate a search and extract pipeline
    def simulate_research_pipeline(query):
        print(f"\nResearching: '{query}'")
        
        # Step 1: Simulate search using DuckDuckGo (in real usage, would call actual search)
        print("1. Searching web sources...")
        search_results = [
            {"title": "Introduction to Financial Volatility", "url": "https://example.com/volatility"},
            {"title": "The VIX Index Explained", "url": "https://example.com/vix-explained"},
            {"title": "Options Trading for Beginners", "url": "https://example.com/options-trading"}
        ]
        print(f"   Found {len(search_results)} relevant results")
        
        # Step 2: Simulate content extraction (in real usage, would use unified_scraper)
        print("2. Extracting content from sources...")
        contents = [
            "The VIX index measures market volatility using options pricing data...",
            "Options have intrinsic and time value components that affect pricing..."
        ]
        print(f"   Extracted content from {len(contents)} sources")
        
        # Step 3: Process content with graph builder
        print("3. Building knowledge graph from extracted content...")
        for i, content in enumerate(contents):
            source_id = f"research_{int(time.time())}_{i}"
            graph_builder.process_text(content, source_id, metadata={"query": query})
        
        # Step 4: Get insights from the graph
        print("4. Extracting key insights from knowledge graph...")
        relevant_entities = graph_builder.get_relevant_entities(query, limit=5)
        
        insights = []
        for entity in relevant_entities:
            # Find connected concepts (in a real system would be more sophisticated)
            neighbors = list(graph_builder.graph.neighbors(entity.get('id', '')))
            neighbor_data = [graph_builder.graph.nodes[n] for n in neighbors if n in graph_builder.graph.nodes]
            neighbor_texts = [n.get('text', '') for n in neighbor_data if 'text' in n][:3]
            
            insights.append({
                "concept": entity.get('text', ''),
                "type": entity.get('type', ''),
                "related_concepts": neighbor_texts,
                "relevance": entity.get('relevance_score', 0)
            })
        
        return insights
    
    # Run the simulated pipeline
    research_query = "How do options prices affect the VIX index calculation?"
    insights = simulate_research_pipeline(research_query)
    
    print("\nKey insights extracted:")
    for i, insight in enumerate(insights, 1):
        print(f"Insight {i}: {insight['concept']} ({insight['type']})")
        if insight['related_concepts']:
            print(f"  Related concepts: {', '.join(insight['related_concepts'])}")
    
    print("\n" + "=" * 80)
    print("Knowledge Graph Builder examples completed!")
    print("This utility enables semantic understanding of research content")
    print("and forms the foundation for knowledge-based reasoning.")
    print("=" * 80)
    print(f"\nExample files are available in: {temp_dir}")
    print("You can delete this directory when you're finished exploring the examples.")

