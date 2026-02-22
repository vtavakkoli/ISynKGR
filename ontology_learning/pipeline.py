"""
Main module for the Ontology Learning component of the extended hybrid framework.

This module implements an enhanced ontology learning pipeline inspired by OntoGenix,
incorporating deep learning techniques and automated validation mechanisms.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import ONTOLOGY_LEARNING
from utils.helpers import clean_text, timer_decorator, load_text, save_json

logger = logging.getLogger(__name__)


class OntologyLearningPipeline:
    """
    Enhanced ontology learning pipeline for extracting and structuring semantic information.
    
    This pipeline implements a multi-stage process for ontology development:
    1. Data preprocessing
    2. Ontology planning
    3. Concept extraction
    4. Relation extraction
    5. Ontology building
    6. Validation and refinement
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ontology learning pipeline.
        
        Args:
            config: Configuration parameters (defaults to ONTOLOGY_LEARNING from config)
        """
        self.config = config or ONTOLOGY_LEARNING
        self.preprocessed_data = {}
        self.ontology_plan = {}
        self.concepts = {}
        self.relations = {}
        self.ontology = {}
        
        logger.info("Initialized OntologyLearningPipeline")
    
    @timer_decorator
    def preprocess_data(self, data_files: List[Union[str, Path]]) -> Dict:
        """
        Preprocess input data files for ontology learning.
        
        Args:
            data_files: List of paths to data files
            
        Returns:
            Dict: Preprocessed data
        """
        logger.info(f"Preprocessing {len(data_files)} data files")
        
        preprocessed_data = {}
        
        for file_path in tqdm(data_files, desc="Preprocessing files"):
            try:
                # Load and clean text
                text = load_text(file_path)
                cleaned_text = clean_text(
                    text,
                    min_length=self.config["preprocessing"]["min_token_length"],
                    remove_stopwords=self.config["preprocessing"]["stopwords_removal"]
                )
                
                # Tokenize and segment
                segments = self._segment_text(cleaned_text)
                
                # Store preprocessed data
                file_name = Path(file_path).name
                preprocessed_data[file_name] = {
                    "original_text": text,
                    "cleaned_text": cleaned_text,
                    "segments": segments,
                    "metadata": {
                        "file_path": str(file_path),
                        "token_count": len(cleaned_text.split()),
                        "segment_count": len(segments)
                    }
                }
                
                logger.debug(f"Successfully preprocessed {file_name}")
                
            except Exception as e:
                logger.error(f"Error preprocessing {file_path}: {str(e)}")
                continue
        
        self.preprocessed_data = preprocessed_data
        logger.info(f"Preprocessing complete. Processed {len(preprocessed_data)} files")
        
        return preprocessed_data
    
    def _segment_text(self, text: str, max_segment_length: int = 1000) -> List[str]:
        """
        Segment text into manageable chunks.
        
        Args:
            text: Text to segment
            max_segment_length: Maximum length of each segment
            
        Returns:
            List[str]: List of text segments
        """
        # Simple segmentation by paragraphs
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        segments = []
        current_segment = ""
        
        for paragraph in paragraphs:
            if len(current_segment) + len(paragraph) <= max_segment_length:
                current_segment += paragraph + "\n"
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = paragraph + "\n"
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    @timer_decorator
    def plan_ontology(self, llm_client) -> Dict:
        """
        Generate an ontology plan using LLM guidance.
        
        Args:
            llm_client: LLM client for generating the ontology plan
            
        Returns:
            Dict: Ontology plan
        """
        logger.info("Planning ontology structure using LLM")
        
        if not self.preprocessed_data:
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return {}
        
        # Prepare input for LLM
        domain_text = ""
        for file_data in self.preprocessed_data.values():
            # Use a sample of segments to avoid token limits
            sample_segments = file_data["segments"][:5]
            domain_text += "\n\n".join(sample_segments)
        
        # Truncate if too long
        if len(domain_text) > 10000:
            domain_text = domain_text[:10000]
        
        # Create prompt for ontology planning
        prompt = f"""
        You are an expert ontology engineer. Based on the following domain text, 
        create a comprehensive ontology plan that includes:
        
        1. Main classes/concepts that should be included in the ontology
        2. Properties for each class
        3. Relationships between classes
        4. Hierarchical structure (taxonomy)
        5. Key axioms or constraints
        
        Format your response as a structured JSON with these sections.
        
        Domain text:
        {domain_text}
        """
        
        try:
            # Get response from LLM
            response = llm_client.generate_text(prompt)
            
            # Parse the ontology plan from the response
            # In a real implementation, this would include proper JSON extraction
            # For this example, we'll simulate the structure
            ontology_plan = {
                "classes": [
                    {"name": "Standard", "description": "A technical standard document"},
                    {"name": "Sensor", "description": "A device that detects events or changes"},
                    {"name": "Actuator", "description": "A component that controls a mechanism"},
                    {"name": "Protocol", "description": "A set of rules for data exchange"},
                    {"name": "DataModel", "description": "Structure of data representation"}
                ],
                "properties": {
                    "Standard": ["id", "name", "version", "publication_date", "organization"],
                    "Sensor": ["id", "type", "measurement_unit", "accuracy", "range"],
                    "Actuator": ["id", "type", "control_mechanism", "response_time"],
                    "Protocol": ["id", "name", "version", "encoding"],
                    "DataModel": ["id", "name", "schema", "format"]
                },
                "relationships": [
                    {"source": "Standard", "target": "Protocol", "type": "defines"},
                    {"source": "Standard", "target": "DataModel", "type": "specifies"},
                    {"source": "Sensor", "target": "Protocol", "type": "uses"},
                    {"source": "Actuator", "target": "Protocol", "type": "uses"},
                    {"source": "DataModel", "target": "Sensor", "type": "represents"},
                    {"source": "DataModel", "target": "Actuator", "type": "controls"}
                ],
                "taxonomy": {
                    "Device": ["Sensor", "Actuator"],
                    "Specification": ["Standard", "Protocol", "DataModel"]
                },
                "axioms": [
                    "Every Sensor must use at least one Protocol",
                    "Every Actuator must use at least one Protocol",
                    "Every DataModel must be specified by at least one Standard"
                ]
            }
            
            self.ontology_plan = ontology_plan
            logger.info("Ontology planning complete")
            
            return ontology_plan
            
        except Exception as e:
            logger.error(f"Error in ontology planning: {str(e)}")
            return {}
    
    @timer_decorator
    def extract_concepts(self) -> Dict:
        """
        Extract concepts from preprocessed data.
        
        Returns:
            Dict: Extracted concepts with metadata
        """
        logger.info("Extracting concepts from preprocessed data")
        
        if not self.preprocessed_data:
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return {}
        
        # In a real implementation, this would use NLP techniques like:
        # - Named Entity Recognition
        # - Noun phrase extraction
        # - Term frequency analysis
        # - Word embeddings clustering
        
        # For this example, we'll simulate the concept extraction process
        all_concepts = {}
        concept_frequency = {}
        
        # Process each file
        for file_name, file_data in self.preprocessed_data.items():
            for segment in file_data["segments"]:
                # Extract candidate concepts (in a real implementation, this would use NLP)
                candidate_concepts = self._extract_candidate_concepts(segment)
                
                # Update frequency counts
                for concept in candidate_concepts:
                    if concept in concept_frequency:
                        concept_frequency[concept] += 1
                    else:
                        concept_frequency[concept] = 1
        
        # Filter concepts by frequency
        min_frequency = self.config["concept_extraction"]["min_concept_frequency"]
        filtered_concepts = {
            concept: freq for concept, freq in concept_frequency.items() 
            if freq >= min_frequency
        }
        
        # Limit to max concepts if specified
        max_concepts = self.config["concept_extraction"]["max_concepts"]
        if max_concepts and len(filtered_concepts) > max_concepts:
            sorted_concepts = sorted(filtered_concepts.items(), key=lambda x: x[1], reverse=True)
            filtered_concepts = dict(sorted_concepts[:max_concepts])
        
        # Create concept dictionary with metadata
        for concept, frequency in filtered_concepts.items():
            all_concepts[concept] = {
                "frequency": frequency,
                "sources": [],  # In a real implementation, this would track source files
                "definition": "",  # In a real implementation, this would extract definitions
                "synonyms": [],  # In a real implementation, this would identify synonyms
                "category": ""  # In a real implementation, this would classify concepts
            }
        
        self.concepts = all_concepts
        logger.info(f"Concept extraction complete. Extracted {len(all_concepts)} concepts")
        
        return all_concepts
    
    def _extract_candidate_concepts(self, text: str) -> List[str]:
        """
        Extract candidate concepts from text.
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            List[str]: List of candidate concepts
        """
        # In a real implementation, this would use NLP techniques
        # For this example, we'll return a simulated list of concepts
        
        # Simulated concepts related to standards and data models
        simulated_concepts = [
            "IEEE 1451", "ISO 15926", "IEC 61499",
            "sensor", "actuator", "protocol",
            "data model", "smart manufacturing", "transportation",
            "interoperability", "semantic mapping", "ontology",
            "knowledge graph", "large language model",
            "data translation", "cross-standard integration"
        ]
        
        # Randomly select a subset of concepts that might appear in this text
        import random
        num_concepts = random.randint(3, 8)
        selected_concepts = random.sample(simulated_concepts, min(num_concepts, len(simulated_concepts)))
        
        return selected_concepts
    
    @timer_decorator
    def extract_relations(self) -> Dict:
        """
        Extract relations between concepts.
        
        Returns:
            Dict: Extracted relations with metadata
        """
        logger.info("Extracting relations between concepts")
        
        if not self.concepts:
            logger.error("No concepts available. Run extract_concepts first.")
            return {}
        
        # In a real implementation, this would use techniques like:
        # - Dependency parsing
        # - Relation extraction models
        # - Pattern matching
        # - Co-occurrence analysis
        
        # For this example, we'll simulate the relation extraction process
        all_relations = {}
        relation_id = 0
        
        # Use the ontology plan if available
        if self.ontology_plan and "relationships" in self.ontology_plan:
            for rel in self.ontology_plan["relationships"]:
                source = rel["source"]
                target = rel["target"]
                rel_type = rel["type"]
                
                if source in self.concepts and target in self.concepts:
                    relation_id += 1
                    relation_key = f"relation_{relation_id}"
                    all_relations[relation_key] = {
                        "source": source,
                        "target": target,
                        "type": rel_type,
                        "confidence": 0.9,  # High confidence since from ontology plan
                        "evidence": [],  # In a real implementation, this would include text evidence
                        "metadata": {
                            "source": "ontology_plan",
                            "extraction_method": "llm_guided"
                        }
                    }
        
        # Add additional relations based on simulated extraction
        concept_list = list(self.concepts.keys())
        relation_types = ["uses", "defines", "contains", "specifies", "implements", "relatesTo"]
        
        # Simulate additional relations
        for i in range(len(concept_list)):
            for j in range(len(concept_list)):
                if i != j:
                    # Only create some relations, not all possible combinations
                    if (i + j) % 5 == 0:  # Arbitrary condition to limit relations
                        source = concept_list[i]
                        target = concept_list[j]
                        rel_type = relation_types[(i + j) % len(relation_types)]
                        confidence = 0.6 + (0.1 * ((i + j) % 4))  # Simulated confidence between 0.6-0.9
                        
                        relation_id += 1
                        relation_key = f"relation_{relation_id}"
                        all_relations[relation_key] = {
                            "source": source,
                            "target": target,
                            "type": rel_type,
                            "confidence": confidence,
                            "evidence": [],
                            "metadata": {
                                "source": "extracted",
                                "extraction_method": "simulated"
                            }
                        }
        
        # Filter relations by confidence threshold
        min_confidence = self.config["relation_extraction"]["min_relation_confidence"]
        filtered_relations = {
            rel_id: rel for rel_id, rel in all_relations.items() 
            if rel["confidence"] >= min_confidence
        }
        
        # Limit relations per concept if specified
        max_relations_per_concept = self.config["relation_extraction"]["max_relations_per_concept"]
        if max_relations_per_concept:
            concept_relation_count = {}
            final_relations = {}
            
            # Count relations per concept
            for rel_id, rel in filtered_relations.items():
                source = rel["source"]
                if source in concept_relation_count:
                    concept_relation_count[source] += 1
                else:
                    concept_relation_count[source] = 1
                
                # Keep relation if under the limit
                if concept_relation_count[source] <= max_relations_per_concept:
                    final_relations[rel_id] = rel
            
            filtered_relations = final_relations
        
        self.relations = filtered_relations
        logger.info(f"Relation extraction complete. Extracted {len(filtered_relations)} relations")
        
        return filtered_relations
    
    @timer_decorator
    def build_ontology(self, output_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Build the ontology from extracted concepts and relations.
        
        Args:
            output_path: Path to save the ontology
            
        Returns:
            Dict: Built ontology
        """
        logger.info("Building ontology from extracted concepts and relations")
        
        if not self.concepts:
            logger.error("No concepts available. Run extract_concepts first.")
            return {}
        
        if not self.relations:
            logger.warning("No relations available. Ontology will only contain concepts.")
        
        # Create ontology structure
        ontology = {
            "metadata": {
                "name": "Extended Hybrid Framework Ontology",
                "description": "Ontology for data translation across standards",
                "version": "1.0.0",
                "created": pd.Timestamp.now().isoformat(),
                "creator": "Extended Hybrid Framework"
            },
            "classes": {},
            "object_properties": {},
            "data_properties": {},
            "individuals": {},
            "axioms": []
        }
        
        # Add classes from concepts
        for concept_name, concept_data in self.concepts.items():
            ontology["classes"][concept_name] = {
                "label": concept_name,
                "description": concept_data.get("definition", ""),
                "synonyms": concept_data.get("synonyms", []),
                "metadata": {
                    "frequency": concept_data.get("frequency", 0),
                    "category": concept_data.get("category", "")
                }
            }
        
        # Add properties from ontology plan if available
        if self.ontology_plan and "properties" in self.ontology_plan:
            for class_name, properties in self.ontology_plan["properties"].items():
                if class_name in ontology["classes"]:
                    for prop in properties:
                        property_id = f"{class_name}_{prop}"
                        ontology["data_properties"][property_id] = {
                            "label": prop,
                            "domain": class_name,
                            "range": "xsd:string",  # Default range
                            "description": f"Property {prop} of {class_name}"
                        }
        
        # Add object properties from relations
        relation_types = {}
        for relation_id, relation_data in self.relations.items():
            rel_type = relation_data["type"]
            source = relation_data["source"]
            target = relation_data["target"]
            
            # Create object property if not exists
            if rel_type not in relation_types:
                relation_types[rel_type] = {
                    "label": rel_type,
                    "domain": [],
                    "range": [],
                    "description": f"Relation of type {rel_type}"
                }
            
            # Add domain and range
            if source not in relation_types[rel_type]["domain"]:
                relation_types[rel_type]["domain"].append(source)
            
            if target not in relation_types[rel_type]["range"]:
                relation_types[rel_type]["range"].append(target)
        
        # Add relation types to ontology
        for rel_type, rel_data in relation_types.items():
            ontology["object_properties"][rel_type] = rel_data
        
        # Add taxonomy if available in ontology plan
        if self.ontology_plan and "taxonomy" in self.ontology_plan:
            for parent, children in self.ontology_plan["taxonomy"].items():
                # Add parent class if not exists
                if parent not in ontology["classes"]:
                    ontology["classes"][parent] = {
                        "label": parent,
                        "description": f"Parent class for {', '.join(children)}",
                        "synonyms": [],
                        "metadata": {}
                    }
                
                # Add subclass relationships
                for child in children:
                    if child in ontology["classes"]:
                        axiom = {
                            "type": "SubClassOf",
                            "subClass": child,
                            "superClass": parent
                        }
                        ontology["axioms"].append(axiom)
        
        # Add axioms from ontology plan
        if self.ontology_plan and "axioms" in self.ontology_plan:
            for axiom_text in self.ontology_plan["axioms"]:
                ontology["axioms"].append({
                    "type": "TextualAxiom",
                    "text": axiom_text
                })
        
        self.ontology = ontology
        logger.info("Ontology building complete")
        
        # Save ontology if output path provided
        if output_path:
            try:
                save_json(ontology, output_path)
                logger.info(f"Ontology saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving ontology to {output_path}: {str(e)}")
        
        return ontology
    
    @timer_decorator
    def validate_ontology(self) -> Dict:
        """
        Validate the built ontology for consistency and completeness.
        
        Returns:
            Dict: Validation results
        """
        logger.info("Validating ontology")
        
        if not self.ontology:
            logger.error("No ontology available. Run build_ontology first.")
            return {"valid": False, "errors": ["No ontology to validate"]}
        
        validation_results = {
            "valid": True,
            "consistency": {
                "valid": True,
                "errors": []
            },
            "completeness": {
                "valid": True,
                "score": 0.0,
                "missing_elements": []
            },
            "metrics": {
                "class_count": len(self.ontology.get("classes", {})),
                "object_property_count": len(self.ontology.get("object_properties", {})),
                "data_property_count": len(self.ontology.get("data_properties", {})),
                "individual_count": len(self.ontology.get("individuals", {})),
                "axiom_count": len(self.ontology.get("axioms", []))
            }
        }
        
        # Check consistency
        # In a real implementation, this would use an OWL reasoner
        # For this example, we'll perform basic checks
        
        # Check for classes without properties
        classes_without_properties = []
        for class_name in self.ontology.get("classes", {}):
            has_property = False
            
            # Check data properties
            for prop_id, prop_data in self.ontology.get("data_properties", {}).items():
                if prop_data.get("domain") == class_name:
                    has_property = True
                    break
            
            # Check object properties
            if not has_property:
                for prop_id, prop_data in self.ontology.get("object_properties", {}).items():
                    if class_name in prop_data.get("domain", []):
                        has_property = True
                        break
            
            if not has_property:
                classes_without_properties.append(class_name)
        
        if classes_without_properties:
            validation_results["completeness"]["missing_elements"].append({
                "type": "ClassesWithoutProperties",
                "classes": classes_without_properties
            })
        
        # Check for orphan classes (not in taxonomy)
        orphan_classes = []
        taxonomy_classes = set()
        
        # Collect all classes in taxonomy
        for axiom in self.ontology.get("axioms", []):
            if axiom.get("type") == "SubClassOf":
                taxonomy_classes.add(axiom.get("subClass"))
                taxonomy_classes.add(axiom.get("superClass"))
        
        # Find orphans
        for class_name in self.ontology.get("classes", {}):
            if class_name not in taxonomy_classes:
                orphan_classes.append(class_name)
        
        if orphan_classes:
            validation_results["completeness"]["missing_elements"].append({
                "type": "OrphanClasses",
                "classes": orphan_classes
            })
        
        # Calculate completeness score
        total_elements = validation_results["metrics"]["class_count"] + \
                         validation_results["metrics"]["object_property_count"] + \
                         validation_results["metrics"]["data_property_count"]
        
        missing_elements = len(classes_without_properties) + len(orphan_classes)
        
        if total_elements > 0:
            completeness_score = 1.0 - (missing_elements / total_elements)
        else:
            completeness_score = 0.0
        
        validation_results["completeness"]["score"] = completeness_score
        
        # Check if completeness meets threshold
        completeness_threshold = self.config["validation"]["completeness_threshold"]
        if completeness_score < completeness_threshold:
            validation_results["completeness"]["valid"] = False
            validation_results["valid"] = False
        
        logger.info(f"Ontology validation complete. Valid: {validation_results['valid']}")
        
        return validation_results
    
    @timer_decorator
    def refine_ontology(self, validation_results: Dict, llm_client) -> Dict:
        """
        Refine the ontology based on validation results.
        
        Args:
            validation_results: Results from validate_ontology
            llm_client: LLM client for refinement suggestions
            
        Returns:
            Dict: Refined ontology
        """
        logger.info("Refining ontology based on validation results")
        
        if not self.ontology:
            logger.error("No ontology available. Run build_ontology first.")
            return {}
        
        if validation_results.get("valid", True):
            logger.info("Ontology is already valid. No refinement needed.")
            return self.ontology
        
        # Create a copy of the ontology for refinement
        refined_ontology = self.ontology.copy()
        
        # Handle classes without properties
        if "missing_elements" in validation_results.get("completeness", {}):
            for missing_element in validation_results["completeness"]["missing_elements"]:
                if missing_element["type"] == "ClassesWithoutProperties":
                    for class_name in missing_element["classes"]:
                        # In a real implementation, this would use LLM to suggest properties
                        # For this example, we'll add a generic property
                        property_id = f"{class_name}_description"
                        refined_ontology["data_properties"][property_id] = {
                            "label": "description",
                            "domain": class_name,
                            "range": "xsd:string",
                            "description": f"Description of {class_name}"
                        }
                
                elif missing_element["type"] == "OrphanClasses":
                    # In a real implementation, this would use LLM to suggest taxonomy
                    # For this example, we'll add orphans to a generic parent
                    if "Thing" not in refined_ontology["classes"]:
                        refined_ontology["classes"]["Thing"] = {
                            "label": "Thing",
                            "description": "Root class for orphan classes",
                            "synonyms": [],
                            "metadata": {}
                        }
                    
                    for class_name in missing_element["classes"]:
                        axiom = {
                            "type": "SubClassOf",
                            "subClass": class_name,
                            "superClass": "Thing"
                        }
                        refined_ontology["axioms"].append(axiom)
        
        # In a real implementation, we would use LLM for more sophisticated refinements
        # For example, suggesting missing relationships, improving definitions, etc.
        
        self.ontology = refined_ontology
        logger.info("Ontology refinement complete")
        
        return refined_ontology
    
    def run_pipeline(self, data_files: List[Union[str, Path]], llm_client, 
                     output_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Run the complete ontology learning pipeline.
        
        Args:
            data_files: List of paths to data files
            llm_client: LLM client for ontology planning and refinement
            output_path: Path to save the final ontology
            
        Returns:
            Dict: Final ontology
        """
        logger.info("Running complete ontology learning pipeline")
        
        # Step 1: Preprocess data
        self.preprocess_data(data_files)
        
        # Step 2: Plan ontology using LLM
        self.plan_ontology(llm_client)
        
        # Step 3: Extract concepts
        self.extract_concepts()
        
        # Step 4: Extract relations
        self.extract_relations()
        
        # Step 5: Build ontology
        self.build_ontology()
        
        # Step 6: Validate ontology
        validation_results = self.validate_ontology()
        
        # Step 7: Refine ontology if needed
        if not validation_results.get("valid", True):
            self.refine_ontology(validation_results, llm_client)
        
        # Save final ontology if output path provided
        if output_path:
            try:
                save_json(self.ontology, output_path)
                logger.info(f"Final ontology saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving final ontology to {output_path}: {str(e)}")
        
        logger.info("Ontology learning pipeline complete")
        
        return self.ontology
