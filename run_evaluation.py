"""
Main script to run the extended hybrid framework evaluation.

This script executes the complete pipeline from data preparation to evaluation and visualization.
"""

import logging
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import setup_directory_structure
from evaluation.evaluator import FrameworkEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("framework_evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Run the complete framework evaluation pipeline.
    """
    logger.info("Starting framework evaluation")
    
    # Create the output directory relative to the project root.  This
    # ensures that results are written into the repository rather than
    # relying on an absolute path that may not exist on the target system.
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "evaluation"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize evaluator
    evaluator = FrameworkEvaluator()
    
    # Run evaluation pipeline
    results = evaluator.run_pipeline(output_dir=output_dir)
    
    # Log summary of results
    overall_metrics = results.get("overall_metrics", {})
    
    if overall_metrics:
        logger.info("Evaluation Results Summary:")
        
        extended = overall_metrics.get("extended_framework", {})
        baseline = overall_metrics.get("baseline_framework", {})
        
        logger.info(f"Extended Framework - Precision: {extended.get('precision', 0):.4f}, "
                   f"Recall: {extended.get('recall', 0):.4f}, "
                   f"F1 Score: {extended.get('f1_score', 0):.4f}, "
                   f"Accuracy: {extended.get('accuracy', 0):.4f}")
        
        logger.info(f"Baseline Framework - Precision: {baseline.get('precision', 0):.4f}, "
                   f"Recall: {baseline.get('recall', 0):.4f}, "
                   f"F1 Score: {baseline.get('f1_score', 0):.4f}, "
                   f"Accuracy: {baseline.get('accuracy', 0):.4f}")
        
        improvement = overall_metrics.get("improvement", {})
        logger.info(f"Improvement - Precision: {improvement.get('precision', 0):.4f}, "
                   f"Recall: {improvement.get('recall', 0):.4f}, "
                   f"F1 Score: {improvement.get('f1_score', 0):.4f}, "
                   f"Accuracy: {improvement.get('accuracy', 0):.4f}")
    
    # Log visualization paths
    visualizations = results.get("visualizations", {})
    
    if visualizations:
        logger.info("Generated Visualizations:")
        for name, path in visualizations.items():
            logger.info(f"- {name}: {path}")
    
    logger.info("Framework evaluation complete")
    
    return results


if __name__ == "__main__":
    main()
