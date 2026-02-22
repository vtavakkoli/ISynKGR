"""
Main module for running the extended hybrid framework.

This script demonstrates the complete pipeline from initialization to translation and querying.
"""

import logging
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import STANDARDS
from utils.helpers import setup_directory_structure
from hybrid_framework.framework import HybridFramework

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("framework_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def prepare_test_data():
    """
    Prepare test data for the framework.
    
    Returns:
        Dict: Standard files dictionary
    """
    logger.info("Preparing test data")
    
    # Create test directory
    test_dir = Path("/home/ubuntu/paper_project/data/test")
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Define standards
    standard_names = ["IEEE_1451", "ISO_15926", "IEC_61499"]
    standard_files = {}
    
    for standard_name in standard_names:
        standard_dir = test_dir / standard_name
        standard_dir.mkdir(exist_ok=True)
        
        files = []
        
        # Create sample files
        for i in range(2):
            file_path = standard_dir / f"{standard_name.lower()}_sample_{i+1}.txt"
            
            # Generate content based on standard
            if standard_name == "IEEE_1451":
                content = f"""
                IEEE 1451 Standard - Sample {i + 1}
                
                The IEEE 1451 family of standards defines a set of open, common, network-independent communication interfaces for connecting transducers (sensors or actuators) to microprocessors, instrumentation systems, and control/field networks.
                
                Key Components:
                
                1. Sensor: A device that detects events or changes in its environment and sends the information to other electronics, frequently a computer processor. A sensor is always used with other electronics.
                   - Properties: id, type, measurement_unit, accuracy, range
                
                2. Actuator: A component of a machine that is responsible for moving and controlling a mechanism or system, for example by opening a valve.
                   - Properties: id, type, control_mechanism, response_time
                
                3. Protocol: A set of rules for data exchange between devices.
                   - Properties: id, name, version, encoding
                
                4. DataModel: Structure of data representation.
                   - Properties: id, name, schema, format
                
                Relationships:
                - Sensor uses Protocol
                - Actuator uses Protocol
                - DataModel represents Sensor
                - DataModel controls Actuator
                
                Example Implementation:
                A temperature sensor (Sensor) uses MQTT (Protocol) to send data in JSON format (DataModel).
                """
            elif standard_name == "ISO_15926":
                content = f"""
                ISO 15926 Standard - Sample {i + 1}
                
                ISO 15926 is a standard for data integration, sharing, exchange, and hand-over between computer systems. It was developed for process plants, including oil and gas production facilities, but is equally applicable to other industries.
                
                Key Components:
                
                1. MeasurementDevice: A device that measures physical quantities.
                   - Properties: identifier, deviceType, unit, precision, measurementRange
                
                2. ControlDevice: A device that controls processes.
                   - Properties: id, category, mechanism, responseLatency
                
                3. CommunicationProtocol: A protocol for communication between devices.
                   - Properties: id, name, version, encodingMethod
                
                4. InformationModel: Model for representing information.
                   - Properties: id, name, structure, encoding
                
                Relationships:
                - MeasurementDevice communicatesWith CommunicationProtocol
                - ControlDevice communicatesWith CommunicationProtocol
                - InformationModel describes MeasurementDevice
                - InformationModel configures ControlDevice
                
                Example Implementation:
                A pressure transmitter (MeasurementDevice) communicates via OPC UA (CommunicationProtocol) using a specific data structure (InformationModel).
                """
            else:  # IEC_61499
                content = f"""
                IEC 61499 Standard - Sample {i + 1}
                
                IEC 61499 is an open standard for distributed control and automation. It defines a generic model for distributed control systems and is based on the IEC 61131 standard.
                
                Key Components:
                
                1. InputDevice: A device that provides input to the control system.
                   - Properties: deviceId, inputType, samplingRate, resolution
                
                2. OutputDevice: A device that receives output from the control system.
                   - Properties: deviceId, outputType, updateRate, range
                
                3. NetworkProtocol: A protocol for network communication.
                   - Properties: protocolId, name, version, parameters
                
                4. FunctionBlock: A software component that processes data.
                   - Properties: blockId, name, algorithm, interface
                
                Relationships:
                - InputDevice connectsTo NetworkProtocol
                - OutputDevice connectsTo NetworkProtocol
                - FunctionBlock processes InputDevice
                - FunctionBlock controls OutputDevice
                
                Example Implementation:
                A motion sensor (InputDevice) connects via Ethernet/IP (NetworkProtocol) to a control algorithm (FunctionBlock) that controls a motor (OutputDevice).
                """
            
            # Write content to file
            with open(file_path, 'w') as f:
                f.write(content)
            
            files.append(str(file_path))
        
        standard_files[standard_name] = files
    
    logger.info(f"Test data prepared with {len(standard_files)} standards")
    
    return standard_files


def main():
    """
    Run the complete hybrid framework pipeline.
    """
    logger.info("Starting hybrid framework demonstration")
    
    # Create output directory
    output_dir = Path("/home/ubuntu/paper_project/output/framework")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare test data
    standard_files = prepare_test_data()
    
    # Initialize framework
    framework = HybridFramework()
    framework.initialize_components()
    
    # Define source and target standards
    source_standard = "IEEE_1451"
    target_standard = "ISO_15926"
    
    # Process standards
    logger.info(f"Processing standards: {source_standard} and {target_standard}")
    processing_results = framework.process_standards({
        source_standard: standard_files[source_standard],
        target_standard: standard_files[target_standard]
    })
    
    # Translate between standards
    logger.info(f"Translating from {source_standard} to {target_standard}")
    translation_results = framework.translate_between_standards(source_standard, target_standard)
    
    # Query the framework
    query = f"Compare the representation of sensors in {source_standard} and {target_standard}"
    logger.info(f"Querying framework: {query}")
    query_results = framework.query_framework(query, use_parallel_retrievers=True)
    
    # Save results
    logger.info("Saving framework results")
    saved_files = framework.save_results(output_dir)
    
    # Log summary
    logger.info("Framework Demonstration Summary:")
    
    logger.info(f"Processed Standards: {', '.join(processing_results.keys())}")
    
    translation_status = translation_results.get("status", "unknown")
    logger.info(f"Translation Status: {translation_status}")
    
    validation_score = translation_results.get("validation", {}).get("overall_score", 0)
    logger.info(f"Translation Validation Score: {validation_score:.4f}")
    
    logger.info(f"Query Response Length: {len(query_results.get('response', ''))}")
    
    logger.info(f"Results saved to: {output_dir}")
    for file_type, file_path in saved_files.items():
        logger.info(f"- {file_type}: {file_path}")
    
    logger.info("Hybrid framework demonstration complete")
    
    return {
        "processing_results": processing_results,
        "translation_results": translation_results,
        "query_results": query_results,
        "saved_files": saved_files
    }


if __name__ == "__main__":
    main()
