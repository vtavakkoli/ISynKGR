"""
Main module for the LLM Integration component of the extended hybrid framework.

This module implements state-of-the-art LLM integration with advanced prompting techniques,
multimodal processing, and model selection capabilities.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import LLM
from utils.helpers import timer_decorator, save_json, load_json

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Advanced LLM manager for integrating multiple state-of-the-art language models.
    
    This class provides a unified interface for:
    1. Multi-model selection and orchestration
    2. Advanced prompting techniques (chain-of-thought, few-shot learning)
    3. Multimodal processing (text, images)
    4. Caching and optimization
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the LLM manager.
        
        Args:
            config: Configuration parameters (defaults to LLM from config)
        """
        self.config = config or LLM
        self.models = {}
        self.default_model = self.config["default_model"]
        self.cache = {}
        
        logger.info("Initialized LLMManager")
    
    @timer_decorator
    def initialize_models(self) -> Dict:
        """
        Initialize LLM models based on configuration.
        
        Returns:
            Dict: Initialized models
        """
        logger.info("Initializing LLM models")
        
        initialized_models = {}
        
        for model_id, model_config in self.config["models"].items():
            try:
                # In a real implementation, this would initialize actual LLM clients
                # For this example, we'll create mock clients
                
                provider = model_config.get("provider", "unknown")
                model_name = model_config.get("name", model_id)
                
                # Check for API key in environment variables
                api_key_env = model_config.get("api_key_env")
                api_key = os.environ.get(api_key_env, "mock_api_key")
                
                # Create mock client
                client = self._create_mock_client(provider, model_name, api_key)
                
                initialized_models[model_id] = {
                    "client": client,
                    "config": model_config,
                    "metadata": {
                        "initialized_at": pd.Timestamp.now().isoformat(),
                        "status": "ready"
                    }
                }
                
                logger.debug(f"Initialized model {model_id} ({model_name})")
                
            except Exception as e:
                logger.error(f"Error initializing model {model_id}: {str(e)}")
                initialized_models[model_id] = {
                    "client": None,
                    "config": model_config,
                    "metadata": {
                        "initialized_at": pd.Timestamp.now().isoformat(),
                        "status": "error",
                        "error": str(e)
                    }
                }
        
        self.models = initialized_models
        logger.info(f"Model initialization complete. Initialized {len(initialized_models)} models")
        
        return initialized_models
    
    def _create_mock_client(self, provider: str, model_name: str, api_key: str) -> Any:
        """
        Create a mock LLM client for simulation purposes.
        
        Args:
            provider: LLM provider name
            model_name: Model name
            api_key: API key
            
        Returns:
            Any: Mock client
        """
        # In a real implementation, this would create actual clients
        # For this example, we'll create a simple mock object
        
        class MockLLMClient:
            def __init__(self, provider, model_name, api_key):
                self.provider = provider
                self.model_name = model_name
                self.api_key = api_key
            
            def generate(self, prompt, **kwargs):
                # Simulate generation delay
                time.sleep(0.1)
                
                # Return mock response based on prompt
                return f"Response from {self.model_name} to prompt: {prompt[:50]}..."
        
        return MockLLMClient(provider, model_name, api_key)
    
    @timer_decorator
    def generate_text(self, prompt: str, model_id: Optional[str] = None, 
                      use_chain_of_thought: Optional[bool] = None,
                      use_few_shot: Optional[bool] = None,
                      examples: Optional[List[Dict]] = None,
                      **kwargs) -> str:
        """
        Generate text using the specified LLM.
        
        Args:
            prompt: Input prompt
            model_id: Model ID to use (defaults to default_model)
            use_chain_of_thought: Whether to use chain-of-thought reasoning
            use_few_shot: Whether to use few-shot learning
            examples: Examples for few-shot learning
            **kwargs: Additional parameters for the model
            
        Returns:
            str: Generated text
        """
        # Select model
        model_id = model_id or self.default_model
        
        if model_id not in self.models:
            logger.error(f"Model {model_id} not initialized. Run initialize_models first.")
            return ""
        
        model_data = self.models[model_id]
        client = model_data["client"]
        
        if not client:
            logger.error(f"Model {model_id} client is not available")
            return ""
        
        # Set default values from config if not specified
        if use_chain_of_thought is None:
            use_chain_of_thought = self.config["prompting"].get("use_chain_of_thought", False)
        
        if use_few_shot is None:
            use_few_shot = self.config["prompting"].get("use_few_shot", False)
        
        # Prepare prompt with enhancements
        enhanced_prompt = self._enhance_prompt(
            prompt, 
            use_chain_of_thought=use_chain_of_thought,
            use_few_shot=use_few_shot,
            examples=examples
        )
        
        # Check cache
        cache_key = f"{model_id}_{hash(enhanced_prompt)}"
        if cache_key in self.cache:
            logger.debug(f"Using cached response for {model_id}")
            return self.cache[cache_key]
        
        try:
            # In a real implementation, this would call the actual LLM API
            # For this example, we'll use the mock client
            
            # Get model-specific parameters
            model_config = model_data["config"]
            max_tokens = kwargs.get("max_tokens", model_config.get("max_tokens", 1024))
            temperature = kwargs.get("temperature", model_config.get("temperature", 0.7))
            
            # Generate response
            response = client.generate(
                enhanced_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Cache response
            self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text with model {model_id}: {str(e)}")
            return f"Error: {str(e)}"
    
    def _enhance_prompt(self, prompt: str, use_chain_of_thought: bool = False,
                        use_few_shot: bool = False, examples: Optional[List[Dict]] = None) -> str:
        """
        Enhance prompt with advanced techniques like chain-of-thought and few-shot learning.
        
        Args:
            prompt: Original prompt
            use_chain_of_thought: Whether to use chain-of-thought reasoning
            use_few_shot: Whether to use few-shot learning
            examples: Examples for few-shot learning
            
        Returns:
            str: Enhanced prompt
        """
        enhanced_prompt = prompt
        
        # Add chain-of-thought instruction
        if use_chain_of_thought:
            cot_instruction = (
                "Think step-by-step to solve this problem. "
                "First, break down the task into smaller parts. "
                "Then, work through each part systematically. "
                "Finally, combine your insights to provide a comprehensive answer."
            )
            enhanced_prompt = f"{cot_instruction}\n\n{enhanced_prompt}"
        
        # Add few-shot examples
        if use_few_shot:
            if not examples:
                # Use default number of examples from config
                num_examples = self.config["prompting"].get("num_examples", 3)
                examples = self._get_default_examples(num_examples)
            
            few_shot_prefix = "Here are some examples to guide your response:\n\n"
            
            for i, example in enumerate(examples):
                few_shot_prefix += f"Example {i+1}:\n"
                few_shot_prefix += f"Input: {example.get('input', '')}\n"
                few_shot_prefix += f"Output: {example.get('output', '')}\n\n"
            
            few_shot_prefix += "Now, please respond to the following:\n\n"
            enhanced_prompt = f"{few_shot_prefix}{enhanced_prompt}"
        
        return enhanced_prompt
    
    def _get_default_examples(self, num_examples: int) -> List[Dict]:
        """
        Get default examples for few-shot learning.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            List[Dict]: List of example dictionaries
        """
        # In a real implementation, this would retrieve relevant examples
        # For this example, we'll return generic examples
        
        default_examples = [
            {
                "input": "Extract the main entities from this text about IEEE 1451.",
                "output": "Entities: IEEE 1451, Smart Transducer Interface, Sensor, Actuator, Network, Protocol"
            },
            {
                "input": "Identify relationships between ISO 15926 and IEC 61499.",
                "output": "Relationships: ISO 15926 defines data models that can be used by IEC 61499 systems, IEC 61499 implements control systems that can use ISO 15926 data models"
            },
            {
                "input": "Summarize the key features of knowledge graphs for data integration.",
                "output": "Key features: Graph-based data representation, Semantic relationships, Entity resolution, Flexible schema, Query capabilities, Inference support, Integration of heterogeneous data sources"
            }
        ]
        
        return default_examples[:min(num_examples, len(default_examples))]
    
    @timer_decorator
    def process_multimodal(self, text: str, image_paths: List[Union[str, Path]] = None,
                           model_id: Optional[str] = None, **kwargs) -> str:
        """
        Process multimodal inputs (text and images) using the specified LLM.
        
        Args:
            text: Input text
            image_paths: Paths to input images
            model_id: Model ID to use (defaults to default_model)
            **kwargs: Additional parameters for the model
            
        Returns:
            str: Generated response
        """
        # Select model
        model_id = model_id or self.default_model
        
        if model_id not in self.models:
            logger.error(f"Model {model_id} not initialized. Run initialize_models first.")
            return ""
        
        model_data = self.models[model_id]
        client = model_data["client"]
        
        if not client:
            logger.error(f"Model {model_id} client is not available")
            return ""
        
        # Check if model supports multimodal inputs
        model_config = model_data["config"]
        if not model_config.get("multimodal", False) and image_paths:
            logger.warning(f"Model {model_id} does not support multimodal inputs. Ignoring images.")
            return self.generate_text(text, model_id=model_id, **kwargs)
        
        try:
            # In a real implementation, this would process images and call the multimodal API
            # For this example, we'll simulate multimodal processing
            
            if image_paths:
                # Simulate image processing
                image_descriptions = []
                for i, image_path in enumerate(image_paths):
                    # In a real implementation, this would process the actual image
                    image_descriptions.append(f"[Image {i+1}: {os.path.basename(image_path)}]")
                
                # Add image descriptions to prompt
                multimodal_prompt = f"{text}\n\nImages: {', '.join(image_descriptions)}"
            else:
                multimodal_prompt = text
            
            # Generate response
            response = self.generate_text(multimodal_prompt, model_id=model_id, **kwargs)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing multimodal input with model {model_id}: {str(e)}")
            return f"Error: {str(e)}"
    
    @timer_decorator
    def run_parallel_models(self, prompt: str, model_ids: List[str] = None, **kwargs) -> Dict[str, str]:
        """
        Run multiple models in parallel and return all responses.
        
        Args:
            prompt: Input prompt
            model_ids: List of model IDs to use (defaults to all initialized models)
            **kwargs: Additional parameters for the models
            
        Returns:
            Dict[str, str]: Dictionary of model ID to response
        """
        logger.info("Running parallel models")
        
        if not self.models:
            logger.error("No models initialized. Run initialize_models first.")
            return {}
        
        # Use specified models or all initialized models
        if model_ids:
            target_models = [model_id for model_id in model_ids if model_id in self.models]
        else:
            target_models = list(self.models.keys())
        
        if not target_models:
            logger.error("No valid models specified")
            return {}
        
        # In a real implementation, this would use parallel processing
        # For this example, we'll process sequentially
        
        responses = {}
        for model_id in target_models:
            logger.debug(f"Generating response with model {model_id}")
            response = self.generate_text(prompt, model_id=model_id, **kwargs)
            responses[model_id] = response
        
        logger.info(f"Parallel model execution complete. Generated {len(responses)} responses")
        
        return responses
    
    @timer_decorator
    def evaluate_responses(self, responses: Dict[str, str], criteria: List[str] = None) -> Dict[str, Dict]:
        """
        Evaluate responses from multiple models based on specified criteria.
        
        Args:
            responses: Dictionary of model ID to response
            criteria: List of evaluation criteria (defaults to ["relevance", "accuracy", "completeness"])
            
        Returns:
            Dict[str, Dict]: Evaluation results
        """
        logger.info("Evaluating model responses")
        
        if not responses:
            logger.error("No responses to evaluate")
            return {}
        
        # Use specified criteria or defaults
        if not criteria:
            criteria = ["relevance", "accuracy", "completeness"]
        
        # In a real implementation, this would use sophisticated evaluation methods
        # For this example, we'll simulate evaluation with random scores
        
        evaluation_results = {}
        for model_id, response in responses.items():
            model_evaluation = {
                "response": response,
                "scores": {},
                "overall_score": 0.0
            }
            
            # Generate scores for each criterion
            total_score = 0.0
            for criterion in criteria:
                # Simulate score between 0.5 and 1.0
                score = 0.5 + (0.5 * np.random.random())
                model_evaluation["scores"][criterion] = round(score, 2)
                total_score += score
            
            # Calculate overall score
            model_evaluation["overall_score"] = round(total_score / len(criteria), 2)
            
            evaluation_results[model_id] = model_evaluation
        
        # Rank models by overall score
        ranked_models = sorted(
            evaluation_results.items(),
            key=lambda x: x[1]["overall_score"],
            reverse=True
        )
        
        # Add ranking to results
        for i, (model_id, _) in enumerate(ranked_models):
            evaluation_results[model_id]["rank"] = i + 1
        
        logger.info("Response evaluation complete")
        
        return evaluation_results
    
    @timer_decorator
    def save_cache(self, cache_path: Union[str, Path]) -> bool:
        """
        Save the response cache to a file.
        
        Args:
            cache_path: Path to save the cache
            
        Returns:
            bool: Whether saving was successful
        """
        logger.info(f"Saving LLM response cache to {cache_path}")
        
        try:
            save_json(self.cache, cache_path)
            logger.info(f"Cache saved successfully with {len(self.cache)} entries")
            return True
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
            return False
    
    @timer_decorator
    def load_cache(self, cache_path: Union[str, Path]) -> bool:
        """
        Load the response cache from a file.
        
        Args:
            cache_path: Path to load the cache from
            
        Returns:
            bool: Whether loading was successful
        """
        logger.info(f"Loading LLM response cache from {cache_path}")
        
        try:
            self.cache = load_json(cache_path)
            logger.info(f"Cache loaded successfully with {len(self.cache)} entries")
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return False
    
    def clear_cache(self) -> None:
        """
        Clear the response cache.
        """
        logger.info("Clearing LLM response cache")
        self.cache = {}
    
    def run_pipeline(self, prompt: str, use_parallel: bool = False, 
                     model_ids: List[str] = None, evaluate: bool = False,
                     **kwargs) -> Union[str, Dict]:
        """
        Run the complete LLM pipeline.
        
        Args:
            prompt: Input prompt
            use_parallel: Whether to use parallel models
            model_ids: List of model IDs to use
            evaluate: Whether to evaluate responses
            **kwargs: Additional parameters for the models
            
        Returns:
            Union[str, Dict]: Generated response or evaluation results
        """
        logger.info("Running complete LLM pipeline")
        
        # Initialize models if not already done
        if not self.models:
            self.initialize_models()
        
        # Run parallel models if requested
        if use_parallel:
            responses = self.run_parallel_models(prompt, model_ids=model_ids, **kwargs)
            
            # Evaluate responses if requested
            if evaluate:
                evaluation_results = self.evaluate_responses(responses)
                logger.info("LLM pipeline complete with evaluation")
                return evaluation_results
            else:
                logger.info("LLM pipeline complete with parallel responses")
                return responses
        else:
            # Use default or specified model
            model_id = model_ids[0] if model_ids and len(model_ids) > 0 else self.default_model
            response = self.generate_text(prompt, model_id=model_id, **kwargs)
            logger.info("LLM pipeline complete with single response")
            return response
