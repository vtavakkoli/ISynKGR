"""
Utility functions for the extended hybrid framework.

This module contains common utility functions used across different components
of the framework, including data processing, file handling, and logging.
"""

import logging
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from utils.config import LOGGING

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING["level"]),
    format=LOGGING["format"],
    filename=LOGGING["file"],
)
logger = logging.getLogger(__name__)


def setup_directory_structure() -> Dict[str, Path]:
    """
    Create the necessary directory structure for the framework.
    
    Returns:
        Dict[str, Path]: Dictionary of created directory paths
    """
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directories = {
        "data": base_dir.parent / "data",
        "models": base_dir.parent / "models",
        "output": base_dir.parent / "output",
        "logs": base_dir.parent / "logs",
    }
    
    # Create subdirectories
    for name, path in directories.items():
        path.mkdir(exist_ok=True, parents=True)
        if name == "data":
            (path / "standards").mkdir(exist_ok=True)
            (path / "processed").mkdir(exist_ok=True)
            (path / "embeddings").mkdir(exist_ok=True)
        elif name == "models":
            (path / "ontologies").mkdir(exist_ok=True)
            (path / "graphs").mkdir(exist_ok=True)
            (path / "llm_cache").mkdir(exist_ok=True)
        elif name == "output":
            (path / "visualizations").mkdir(exist_ok=True)
            (path / "translations").mkdir(exist_ok=True)
            (path / "evaluations").mkdir(exist_ok=True)
    
    logger.info(f"Directory structure set up successfully")
    return directories


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict[str, Any]: Loaded JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        raise


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        raise


def load_text(file_path: Union[str, Path]) -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        str: Loaded text
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.debug(f"Successfully loaded text from {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error loading text from {file_path}: {str(e)}")
        raise


def save_text(text: str, file_path: Union[str, Path]) -> None:
    """
    Save text to a file.
    
    Args:
        text: Text to save
        file_path: Path to save the text file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.debug(f"Successfully saved text to {file_path}")
    except Exception as e:
        logger.error(f"Error saving text to {file_path}: {str(e)}")
        raise


def timer_decorator(func):
    """
    Decorator to measure the execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def clean_text(text: str, min_length: int = 3, remove_stopwords: bool = True) -> str:
    """
    Clean and preprocess text.
    
    Args:
        text: Text to clean
        min_length: Minimum token length to keep
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove short tokens
    if min_length > 1:
        tokens = text.split()
        tokens = [token for token in tokens if len(token) >= min_length]
        text = ' '.join(tokens)
    
    # Remove stopwords if requested
    if remove_stopwords:
        try:
            import nltk
            from nltk.corpus import stopwords
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            stop_words = set(stopwords.words('english'))
            tokens = text.split()
            tokens = [token for token in tokens if token not in stop_words]
            text = ' '.join(tokens)
        except ImportError:
            logger.warning("NLTK not available, skipping stopwords removal")
    
    return text


def batch_process(items: List[Any], process_func, batch_size: int = 32, 
                  show_progress: bool = True, **kwargs) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each batch
        batch_size: Size of each batch
        show_progress: Whether to show a progress bar
        **kwargs: Additional arguments to pass to process_func
        
    Returns:
        List[Any]: List of processed items
    """
    results = []
    
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    # Process each batch
    iterator = tqdm(batches) if show_progress else batches
    for batch in iterator:
        batch_results = process_func(batch, **kwargs)
        results.extend(batch_results)
    
    return results


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        float: Cosine similarity
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries
        sep: Separator for keys
        
    Returns:
        Dict[str, Any]: Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File extension
    """
    return os.path.splitext(str(file_path))[1].lower()


def is_valid_file(file_path: Union[str, Path], allowed_extensions: List[str] = None) -> bool:
    """
    Check if a file exists and has a valid extension.
    
    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed extensions
        
    Returns:
        bool: Whether the file is valid
    """
    if not os.path.isfile(file_path):
        return False
    
    if allowed_extensions:
        extension = get_file_extension(file_path)
        return extension in allowed_extensions
    
    return True
