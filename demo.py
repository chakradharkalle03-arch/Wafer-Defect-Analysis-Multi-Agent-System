"""
Demo script for Wafer Defect Analysis System
This script demonstrates how to use the system programmatically
"""
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.orchestrator import MultiAgentOrchestrator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_analysis(image_path: str):
    """
    Demonstrate wafer analysis workflow
    """
    print("=" * 60)
    print("Wafer Defect Analysis Multi-Agent System - Demo")
    print("=" * 60)
    print()
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        print("Please provide a valid wafer image path.")
        return
    
    print(f"Analyzing image: {image_path}")
    print()
    
    try:
        # Initialize orchestrator
        print("Initializing multi-agent system...")
        orchestrator = MultiAgentOrchestrator()
        print("✓ System initialized")
        print()
        
        # Run analysis
        print("Starting analysis workflow...")
        print("-" * 60)
        
        result = orchestrator.analyze_wafer(
            image_path=image_path,
            wafer_id="DEMO_WAFER_001",
            batch_id="DEMO_BATCH_001"
        )
        
        print("-" * 60)
        print()
        
        # Display results
        print("=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print()
        
        print(f"Analysis ID: {result.analysis_id}")
        print(f"Wafer ID: {result.wafer_id}")
        print(f"Batch ID: {result.batch_id}")
        print(f"Timestamp: {result.timestamp}")
        print()
        
        print(f"Total Defects Detected: {result.total_defects}")
        print(f"Severity Score: {result.severity_score:.2%}")
        print()
        
        # Defect summary
        print("Defect Summary:")
        print("-" * 60)
        for defect_type, count in result.defect_summary.items():
            print(f"  {defect_type.replace('_', ' ').title()}: {count}")
        print()
        
        # Top defects
        if result.defects:
            print("Top 5 Defects:")
            print("-" * 60)
            for i, (defect, classification) in enumerate(
                zip(result.defects[:5], result.classifications[:5])
            ):
                print(f"\n{i+1}. {defect.defect_id}")
                print(f"   Type: {classification.defect_type.value.replace('_', ' ').title()}")
                print(f"   Confidence: {classification.confidence:.2%}")
                print(f"   Location: ({defect.bbox.x_min:.1f}, {defect.bbox.y_min:.1f}) to ({defect.bbox.x_max:.1f}, {defect.bbox.y_max:.1f})")
                print(f"   Area: {defect.area:.2f} pixels²")
        
        print()
        
        # Root causes
        if result.root_causes:
            print("Root Cause Analysis:")
            print("-" * 60)
            process_summary = {}
            for rc in result.root_causes:
                process = rc.process_step.value
                if process not in process_summary:
                    process_summary[process] = []
                process_summary[process].append(rc.likely_cause)
            
            for process, causes in process_summary.items():
                print(f"\n{process}:")
                most_common = max(set(causes), key=causes.count)
                print(f"  Most likely cause: {most_common}")
                print(f"  Affected defects: {len(causes)}")
        
        print()
        
        # Report
        if result.report_path:
            print(f"Report generated: {result.report_path}")
            print("You can view the full report with visualizations in the PDF.")
        
        print()
        print("=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Please check the logs for more details.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default: look for sample images
        sample_images = [
            "data/raw/sample_wafer.jpg",
            "data/raw/sample_wafer.png",
            "data/raw/test_image.jpg",
            "test_wafer.jpg",
            "sample.jpg"
        ]
        
        image_path = None
        for path in sample_images:
            if Path(path).exists():
                image_path = path
                break
        
        if not image_path:
            print("Usage: python demo.py <path_to_wafer_image>")
            print("\nOr place a test image in one of these locations:")
            for path in sample_images:
                print(f"  - {path}")
            sys.exit(1)
    
    demo_analysis(image_path)

