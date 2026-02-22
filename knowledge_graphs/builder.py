"""
Main module for the Knowledge Graph component of the extended hybrid framework.

This module implements an advanced knowledge graph construction and management system,
incorporating community-based summarization, graph neural networks, and hypergraph structures.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import KNOWLEDGE_GRAPH
from utils.helpers import timer_decorator, save_json, load_json

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Advanced knowledge graph builder for constructing and managing knowledge graphs.
    
    This class implements a multi-stage process for knowledge graph construction:
    1. Entity extraction and enhancement
    2. Relationship identification
    3. Graph construction
    4. Community detection and summarization
    5. Graph embedding generation
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the knowledge graph builder.
        
        Args:
            config: Configuration parameters (defaults to KNOWLEDGE_GRAPH from config)
        """
        self.config = config or KNOWLEDGE_GRAPH
        self.entities = {}
        self.relationships = {}
        self.graph = nx.DiGraph()
        self.communities = {}
        self.embeddings = {}
        
        logger.info("Initialized KnowledgeGraphBuilder")
    
    @timer_decorator
    def extract_entities(self, ontology: Dict, llm_client) -> Dict:
        """
        Extract and enhance entities from ontology and text data.
        
        Args:
            ontology: Ontology dictionary
            llm_client: LLM client for entity enhancement
            
        Returns:
            Dict: Extracted entities
        """
        logger.info("Extracting entities from ontology")
        
        entities = {}
        
        # Extract entities from ontology classes
        if "classes" in ontology:
            for class_name, class_data in ontology["classes"].items():
                entity_id = f"class_{class_name}"
                entities[entity_id] = {
                    "id": entity_id,
                    "name": class_name,
                    "type": "Class",
                    "description": class_data.get("description", ""),
                    "synonyms": class_data.get("synonyms", []),
                    "properties": {},
                    "metadata": {
                        "source": "ontology",
                        "confidence": 1.0
                    }
                }
                
                # Add properties if available
                if "data_properties" in ontology:
                    for prop_id, prop_data in ontology["data_properties"].items():
                        if prop_data.get("domain") == class_name:
                            entities[entity_id]["properties"][prop_id] = {
                                "name": prop_data.get("label", prop_id),
                                "type": prop_data.get("range", "xsd:string"),
                                "description": prop_data.get("description", "")
                            }
        
        # Extract entities from ontology individuals (if any)
        if "individuals" in ontology:
            for indiv_id, indiv_data in ontology["individuals"].items():
                entity_id = f"individual_{indiv_id}"
                entities[entity_id] = {
                    "id": entity_id,
                    "name": indiv_data.get("label", indiv_id),
                    "type": indiv_data.get("type", "Individual"),
                    "description": indiv_data.get("description", ""),
                    "properties": {},
                    "metadata": {
                        "source": "ontology",
                        "confidence": 1.0
                    }
                }
                
                # Add property values if available
                if "property_values" in indiv_data:
                    for prop_id, value in indiv_data["property_values"].items():
                        entities[entity_id]["properties"][prop_id] = {
                            "value": value
                        }
        
        # Enhance entities with LLM
        enhanced_entities = self._enhance_entities(entities, llm_client)
        
        # Filter entities by confidence threshold
        confidence_threshold = self.config["entity_extraction"]["confidence_threshold"]
        filtered_entities = {
            entity_id: entity for entity_id, entity in enhanced_entities.items()
            if entity["metadata"].get("confidence", 0) >= confidence_threshold
        }
        
        # Limit to max entities if specified
        max_entities = self.config["entity_extraction"]["max_entities"]
        if max_entities and len(filtered_entities) > max_entities:
            # Sort by confidence
            sorted_entities = sorted(
                filtered_entities.items(),
                key=lambda x: x[1]["metadata"].get("confidence", 0),
                reverse=True
            )
            filtered_entities = dict(sorted_entities[:max_entities])
        
        self.entities = filtered_entities
        logger.info(f"Entity extraction complete. Extracted {len(filtered_entities)} entities")
        
        return filtered_entities
    
    def _enhance_entities(self, entities: Dict, llm_client) -> Dict:
        """
        Enhance entities with additional information using LLM.
        
        Args:
            entities: Dictionary of entities
            llm_client: LLM client for entity enhancement
            
        Returns:
            Dict: Enhanced entities
        """
        logger.info(f"Enhancing {len(entities)} entities with LLM")
        
        enhanced_entities = entities.copy()
        
        # In a real implementation, this would use LLM to enhance entities
        # For this example, we'll simulate the enhancement
        
        for entity_id, entity in enhanced_entities.items():
            # Simulate enhancement by adding additional fields
            if "description" not in entity or not entity["description"]:
                entity["description"] = f"Enhanced description for {entity['name']}"
            
            if "synonyms" not in entity or not entity["synonyms"]:
                entity["synonyms"] = [f"{entity['name']} variant", f"Alternative {entity['name']}"]
            
            # Add confidence score if not present
            if "metadata" not in entity:
                entity["metadata"] = {}
            
            if "confidence" not in entity["metadata"]:
                entity["metadata"]["confidence"] = 0.9  # High confidence for ontology-derived entities
        
        return enhanced_entities
    
    @timer_decorator
    def extract_relationships(self, ontology: Dict, llm_client) -> Dict:
        """
        Extract relationships from ontology and enhance with LLM.
        
        Args:
            ontology: Ontology dictionary
            llm_client: LLM client for relationship enhancement
            
        Returns:
            Dict: Extracted relationships
        """
        logger.info("Extracting relationships from ontology")
        
        if not self.entities:
            logger.error("No entities available. Run extract_entities first.")
            return {}
        
        relationships = {}
        
        # Extract relationships from ontology object properties
        if "object_properties" in ontology:
            for prop_id, prop_data in ontology["object_properties"].items():
                # Get domains and ranges
                domains = prop_data.get("domain", [])
                if not isinstance(domains, list):
                    domains = [domains]
                
                ranges = prop_data.get("range", [])
                if not isinstance(ranges, list):
                    ranges = [ranges]
                
                # Create relationships for each domain-range pair
                for domain in domains:
                    for range_class in ranges:
                        # Find corresponding entities
                        domain_entity_id = f"class_{domain}"
                        range_entity_id = f"class_{range_class}"
                        
                        if domain_entity_id in self.entities and range_entity_id in self.entities:
                            rel_id = f"rel_{domain}_{prop_id}_{range_class}"
                            relationships[rel_id] = {
                                "id": rel_id,
                                "type": prop_id,
                                "source": domain_entity_id,
                                "target": range_entity_id,
                                "label": prop_data.get("label", prop_id),
                                "description": prop_data.get("description", ""),
                                "metadata": {
                                    "source": "ontology",
                                    "confidence": 1.0
                                }
                            }
        
        # Extract relationships from axioms
        if "axioms" in ontology:
            for axiom in ontology["axioms"]:
                if axiom.get("type") == "SubClassOf":
                    subclass = axiom.get("subClass")
                    superclass = axiom.get("superClass")
                    
                    subclass_entity_id = f"class_{subclass}"
                    superclass_entity_id = f"class_{superclass}"
                    
                    if subclass_entity_id in self.entities and superclass_entity_id in self.entities:
                        rel_id = f"rel_{subclass}_subClassOf_{superclass}"
                        relationships[rel_id] = {
                            "id": rel_id,
                            "type": "subClassOf",
                            "source": subclass_entity_id,
                            "target": superclass_entity_id,
                            "label": "is a",
                            "description": f"{subclass} is a subclass of {superclass}",
                            "metadata": {
                                "source": "ontology",
                                "confidence": 1.0
                            }
                        }
        
        # Enhance relationships with LLM
        enhanced_relationships = self._enhance_relationships(relationships, llm_client)
        
        # Filter relationships by confidence threshold
        confidence_threshold = self.config["relationship_extraction"]["confidence_threshold"]
        filtered_relationships = {
            rel_id: rel for rel_id, rel in enhanced_relationships.items()
            if rel["metadata"].get("confidence", 0) >= confidence_threshold
        }
        
        # Limit to max relationships if specified
        max_relationships = self.config["relationship_extraction"]["max_relationships"]
        if max_relationships and len(filtered_relationships) > max_relationships:
            # Sort by confidence
            sorted_relationships = sorted(
                filtered_relationships.items(),
                key=lambda x: x[1]["metadata"].get("confidence", 0),
                reverse=True
            )
            filtered_relationships = dict(sorted_relationships[:max_relationships])
        
        self.relationships = filtered_relationships
        logger.info(f"Relationship extraction complete. Extracted {len(filtered_relationships)} relationships")
        
        return filtered_relationships
    
    def _enhance_relationships(self, relationships: Dict, llm_client) -> Dict:
        """
        Enhance relationships with additional information using LLM.
        
        Args:
            relationships: Dictionary of relationships
            llm_client: LLM client for relationship enhancement
            
        Returns:
            Dict: Enhanced relationships
        """
        logger.info(f"Enhancing {len(relationships)} relationships with LLM")
        
        enhanced_relationships = relationships.copy()
        
        # In a real implementation, this would use LLM to enhance relationships
        # For this example, we'll simulate the enhancement
        
        for rel_id, rel in enhanced_relationships.items():
            # Simulate enhancement by adding additional fields
            if "description" not in rel or not rel["description"]:
                source_name = self.entities[rel["source"]]["name"] if rel["source"] in self.entities else "Unknown"
                target_name = self.entities[rel["target"]]["name"] if rel["target"] in self.entities else "Unknown"
                rel["description"] = f"Relationship of type {rel['type']} from {source_name} to {target_name}"
            
            # Add confidence score if not present
            if "metadata" not in rel:
                rel["metadata"] = {}
            
            if "confidence" not in rel["metadata"]:
                rel["metadata"]["confidence"] = 0.9  # High confidence for ontology-derived relationships
        
        return enhanced_relationships
    
    @timer_decorator
    def build_graph(self) -> nx.DiGraph:
        """
        Build a graph from extracted entities and relationships.
        
        Returns:
            nx.DiGraph: Constructed graph
        """
        logger.info("Building knowledge graph from entities and relationships")
        
        if not self.entities:
            logger.error("No entities available. Run extract_entities first.")
            return nx.DiGraph()
        
        if not self.relationships:
            logger.warning("No relationships available. Graph will only contain isolated nodes.")
        
        # Create a new directed graph
        graph = nx.DiGraph()
        
        # Add entities as nodes
        for entity_id, entity in self.entities.items():
            # Add node with all entity attributes
            graph.add_node(entity_id, **entity)
        
        # Add relationships as edges
        for rel_id, rel in self.relationships.items():
            source = rel["source"]
            target = rel["target"]
            
            # Skip if source or target not in graph
            if source not in graph or target not in graph:
                continue
            
            # Add edge with all relationship attributes
            graph.add_edge(source, target, id=rel_id, **rel)
        
        self.graph = graph
        logger.info(f"Graph building complete. Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        return graph
    
    @timer_decorator
    def detect_communities(self) -> Dict:
        """
        Detect communities in the knowledge graph using the Leiden algorithm.
        
        Returns:
            Dict: Detected communities
        """
        logger.info("Detecting communities in knowledge graph")
        
        if not self.graph or len(self.graph.nodes) == 0:
            logger.error("No graph available or graph is empty. Run build_graph first.")
            return {}
        
        # In a real implementation, this would use the Leiden algorithm
        # For this example, we'll simulate community detection using NetworkX
        
        # Convert directed graph to undirected for community detection
        undirected_graph = self.graph.to_undirected()
        
        # Use connected components as a simple community detection method
        # In a real implementation, we would use more sophisticated algorithms
        connected_components = list(nx.connected_components(undirected_graph))
        
        # Filter communities by minimum size
        min_community_size = self.config["community_detection"]["min_community_size"]
        filtered_communities = [
            component for component in connected_components
            if len(component) >= min_community_size
        ]
        
        # Create community dictionary
        communities = {}
        for i, community in enumerate(filtered_communities):
            community_id = f"community_{i}"
            
            # Get community nodes
            community_nodes = list(community)
            
            # Get community subgraph
            community_subgraph = self.graph.subgraph(community_nodes)
            
            # Generate community summary
            summary = self._generate_community_summary(community_subgraph)
            
            communities[community_id] = {
                "id": community_id,
                "nodes": community_nodes,
                "size": len(community_nodes),
                "summary": summary,
                "metadata": {
                    "density": nx.density(community_subgraph),
                    "diameter": self._calculate_diameter(community_subgraph)
                }
            }
        
        self.communities = communities
        logger.info(f"Community detection complete. Detected {len(communities)} communities")
        
        return communities
    
    def _calculate_diameter(self, graph: nx.Graph) -> float:
        """
        Calculate the diameter of a graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            float: Diameter of the graph
        """
        try:
            # For disconnected graphs, calculate the maximum eccentricity of each component
            if not nx.is_connected(graph.to_undirected()):
                diameter = 0
                for component in nx.connected_components(graph.to_undirected()):
                    subgraph = graph.subgraph(component)
                    if len(subgraph) > 1:
                        try:
                            component_diameter = nx.diameter(subgraph)
                            diameter = max(diameter, component_diameter)
                        except:
                            pass
                return diameter
            else:
                return nx.diameter(graph)
        except:
            # Return -1 for empty graphs or graphs with a single node
            return -1
    
    def _generate_community_summary(self, community_graph: nx.Graph) -> Dict:
        """
        Generate a summary for a community.
        
        Args:
            community_graph: NetworkX graph of the community
            
        Returns:
            Dict: Community summary
        """
        # In a real implementation, this would use LLM to generate summaries
        # For this example, we'll create a simple statistical summary
        
        # Get node types
        node_types = {}
        for node, data in community_graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            if node_type in node_types:
                node_types[node_type] += 1
            else:
                node_types[node_type] = 1
        
        # Get edge types
        edge_types = {}
        for _, _, data in community_graph.edges(data=True):
            edge_type = data.get("type", "Unknown")
            if edge_type in edge_types:
                edge_types[edge_type] += 1
            else:
                edge_types[edge_type] = 1
        
        # Get central nodes (by degree centrality)
        centrality = nx.degree_centrality(community_graph)
        central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create summary
        summary = {
            "title": f"Community of {len(community_graph.nodes)} nodes",
            "description": f"This community contains {len(community_graph.nodes)} nodes and {len(community_graph.edges)} edges.",
            "node_types": node_types,
            "edge_types": edge_types,
            "central_nodes": [
                {
                    "id": node,
                    "name": community_graph.nodes[node].get("name", node),
                    "centrality": round(score, 3)
                }
                for node, score in central_nodes
            ]
        }
        
        return summary
    
    @timer_decorator
    def generate_embeddings(self) -> Dict:
        """
        Generate embeddings for nodes in the knowledge graph.
        
        Returns:
            Dict: Node embeddings
        """
        logger.info("Generating embeddings for knowledge graph nodes")
        
        if not self.graph or len(self.graph.nodes) == 0:
            logger.error("No graph available or graph is empty. Run build_graph first.")
            return {}
        
        # In a real implementation, this would use node2vec or similar algorithms
        # For this example, we'll generate random embeddings
        
        embedding_dim = self.config["graph_embeddings"]["dimension"]
        embeddings = {}
        
        for node in self.graph.nodes:
            # Generate random embedding vector
            embedding = np.random.randn(embedding_dim)
            # Normalize to unit length
            embedding = embedding / np.linalg.norm(embedding)
            embeddings[node] = embedding.tolist()
        
        self.embeddings = embeddings
        logger.info(f"Embedding generation complete. Generated embeddings for {len(embeddings)} nodes")
        
        return embeddings
    
    @timer_decorator
    def save_graph(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save the knowledge graph and related data to files.
        
        Args:
            output_dir: Directory to save files
            
        Returns:
            Dict[str, str]: Paths to saved files
        """
        logger.info(f"Saving knowledge graph to {output_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        saved_files = {}
        
        # Save entities
        if self.entities:
            entities_path = output_dir / "entities.json"
            save_json(self.entities, entities_path)
            saved_files["entities"] = str(entities_path)
        
        # Save relationships
        if self.relationships:
            relationships_path = output_dir / "relationships.json"
            save_json(self.relationships, relationships_path)
            saved_files["relationships"] = str(relationships_path)
        
        # Save graph as adjacency list
        if self.graph and len(self.graph.nodes) > 0:
            # Convert graph to dictionary
            graph_dict = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes with attributes
            for node, attrs in self.graph.nodes(data=True):
                node_data = {"id": node}
                node_data.update(attrs)
                graph_dict["nodes"].append(node_data)
            
            # Add edges with attributes
            for source, target, attrs in self.graph.edges(data=True):
                edge_data = {
                    "source": source,
                    "target": target
                }
                edge_data.update(attrs)
                graph_dict["edges"].append(edge_data)
            
            graph_path = output_dir / "graph.json"
            save_json(graph_dict, graph_path)
            saved_files["graph"] = str(graph_path)
        
        # Save communities
        if self.communities:
            communities_path = output_dir / "communities.json"
            save_json(self.communities, communities_path)
            saved_files["communities"] = str(communities_path)
        
        # Save embeddings
        if self.embeddings:
            embeddings_path = output_dir / "embeddings.json"
            save_json(self.embeddings, embeddings_path)
            saved_files["embeddings"] = str(embeddings_path)
        
        logger.info(f"Knowledge graph saved successfully. Files: {list(saved_files.keys())}")
        
        return saved_files
    
    @timer_decorator
    def load_graph(self, input_dir: Union[str, Path]) -> bool:
        """
        Load the knowledge graph and related data from files.
        
        Args:
            input_dir: Directory containing saved files
            
        Returns:
            bool: Whether loading was successful
        """
        logger.info(f"Loading knowledge graph from {input_dir}")
        
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            logger.error(f"Input directory {input_dir} does not exist")
            return False
        
        try:
            # Load entities
            entities_path = input_dir / "entities.json"
            if entities_path.exists():
                self.entities = load_json(entities_path)
                logger.debug(f"Loaded {len(self.entities)} entities")
            
            # Load relationships
            relationships_path = input_dir / "relationships.json"
            if relationships_path.exists():
                self.relationships = load_json(relationships_path)
                logger.debug(f"Loaded {len(self.relationships)} relationships")
            
            # Load graph
            graph_path = input_dir / "graph.json"
            if graph_path.exists():
                graph_dict = load_json(graph_path)
                
                # Create new graph
                graph = nx.DiGraph()
                
                # Add nodes with attributes
                for node_data in graph_dict["nodes"]:
                    node_id = node_data.pop("id")
                    graph.add_node(node_id, **node_data)
                
                # Add edges with attributes
                for edge_data in graph_dict["edges"]:
                    source = edge_data.pop("source")
                    target = edge_data.pop("target")
                    graph.add_edge(source, target, **edge_data)
                
                self.graph = graph
                logger.debug(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            
            # Load communities
            communities_path = input_dir / "communities.json"
            if communities_path.exists():
                self.communities = load_json(communities_path)
                logger.debug(f"Loaded {len(self.communities)} communities")
            
            # Load embeddings
            embeddings_path = input_dir / "embeddings.json"
            if embeddings_path.exists():
                self.embeddings = load_json(embeddings_path)
                logger.debug(f"Loaded embeddings for {len(self.embeddings)} nodes")
            
            logger.info("Knowledge graph loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {str(e)}")
            return False
    
    def run_pipeline(self, ontology: Dict, llm_client, output_dir: Optional[Union[str, Path]] = None) -> nx.DiGraph:
        """
        Run the complete knowledge graph pipeline.
        
        Args:
            ontology: Ontology dictionary
            llm_client: LLM client for entity and relationship enhancement
            output_dir: Directory to save output files
            
        Returns:
            nx.DiGraph: Final knowledge graph
        """
        logger.info("Running complete knowledge graph pipeline")
        
        # Step 1: Extract entities
        self.extract_entities(ontology, llm_client)
        
        # Step 2: Extract relationships
        self.extract_relationships(ontology, llm_client)
        
        # Step 3: Build graph
        self.build_graph()
        
        # Step 4: Detect communities
        self.detect_communities()
        
        # Step 5: Generate embeddings
        self.generate_embeddings()
        
        # Save results if output directory provided
        if output_dir:
            self.save_graph(output_dir)
        
        logger.info("Knowledge graph pipeline complete")
        
        return self.graph
