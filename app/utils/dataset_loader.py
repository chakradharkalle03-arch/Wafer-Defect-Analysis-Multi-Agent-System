"""
Dataset Loader - Loads and manages wafer defect datasets from HuggingFace
"""
import logging
from typing import Dict, Optional, List, Any, TYPE_CHECKING
from pathlib import Path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = Any

try:
    from datasets import load_dataset, Dataset as _Dataset
    Dataset = _Dataset
    DATASETS_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    DATASETS_AVAILABLE = False
    logger.warning(f"datasets library not available: {e}")


class DatasetLoader:
    """
    Loads and manages wafer defect datasets from HuggingFace
    """
    
    def __init__(self, hf_api_key: str, cache_dir: str = "./models_cache"):
        """Initialize dataset loader"""
        self.hf_api_key = hf_api_key
        self.cache_dir = cache_dir
        self.datasets: Dict[str, Dataset] = {}
        self._available_datasets = [
            "lslattery/wafer-defect-detection",
            "sitloboi2012/semiconductor_scirepeval_v1"
        ]
    
    def load_all_datasets(self) -> Dict[str, Dataset]:
        """Load all available wafer defect datasets"""
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available")
            return {}
        
        loaded = {}
        
        for dataset_name in self._available_datasets:
            try:
                logger.info(f"Loading dataset: {dataset_name}")
                dataset = load_dataset(
                    dataset_name,
                    token=self.hf_api_key,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                loaded[dataset_name] = dataset
                logger.info(f"✓ Successfully loaded {dataset_name}")
            except Exception as e:
                logger.warning(f"Could not load {dataset_name}: {e}")
        
        self.datasets = loaded
        return loaded
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a dataset"""
        if dataset_name not in self.datasets:
            return None
        
        dataset = self.datasets[dataset_name]
        info = {
            "name": dataset_name,
            "splits": list(dataset.keys()) if isinstance(dataset, dict) else ["train"],
            "num_samples": {}
        }
        
        if isinstance(dataset, dict):
            for split_name, split_data in dataset.items():
                info["num_samples"][split_name] = len(split_data)
        else:
            info["num_samples"]["train"] = len(dataset)
        
        return info
    
    def get_sample_images(self, dataset_name: str, num_samples: int = 5) -> List:
        """Get sample images from a dataset"""
        if dataset_name not in self.datasets:
            return []
        
        dataset = self.datasets[dataset_name]
        samples = []
        
        try:
            # Get first split (usually 'train')
            if isinstance(dataset, dict):
                split_data = list(dataset.values())[0]
            else:
                split_data = dataset
            
            # Get samples
            for i in range(min(num_samples, len(split_data))):
                sample = split_data[i]
                samples.append(sample)
        
        except Exception as e:
            logger.error(f"Error getting samples from {dataset_name}: {e}")
        
        return samples

