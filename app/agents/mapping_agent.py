"""
Defect Mapping Agent - Creates spatial maps of defects on wafer
Analyzes defect distribution, clusters, and spatial patterns
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json

from app.models.schemas import DefectDetection, ClassificationResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class DefectMap:
    """Represents a spatial map of defects on a wafer"""
    
    def __init__(
        self,
        defects: List[DefectDetection],
        classifications: List[ClassificationResult],
        image_shape: Tuple[int, int],
        wafer_radius: Optional[float] = None
    ):
        self.defects = defects
        self.classifications = classifications
        self.image_shape = image_shape  # (height, width)
        self.wafer_radius = wafer_radius or min(image_shape) / 2
        
        # Calculate defect centers
        self.defect_centers = self._calculate_centers()
        
        # Spatial analysis
        self.density_map = None
        self.clusters = []
        self.spatial_statistics = {}
        
    def _calculate_centers(self) -> List[Tuple[float, float]]:
        """Calculate center coordinates for each defect"""
        centers = []
        for defect in self.defects:
            center_x = (defect.bbox.x_min + defect.bbox.x_max) / 2
            center_y = (defect.bbox.y_min + defect.bbox.y_max) / 2
            centers.append((center_x, center_y))
        return centers
    
    def analyze_spatial_distribution(self) -> Dict:
        """Analyze spatial distribution of defects"""
        if not self.defect_centers:
            return {}
        
        centers = np.array(self.defect_centers)
        
        # Calculate centroid
        centroid = np.mean(centers, axis=0)
        
        # Calculate spread
        distances_from_center = np.linalg.norm(centers - centroid, axis=1)
        mean_distance = np.mean(distances_from_center)
        std_distance = np.std(distances_from_center)
        
        # Calculate radial distribution
        image_center = np.array([self.image_shape[1] / 2, self.image_shape[0] / 2])
        radial_distances = np.linalg.norm(centers - image_center, axis=1)
        
        # Identify clusters using simple distance-based clustering
        clusters = self._identify_clusters(centers)
        
        # Calculate density map
        self.density_map = self._calculate_density_map(centers)
        
        self.spatial_statistics = {
            "centroid": {"x": float(centroid[0]), "y": float(centroid[1])},
            "mean_distance_from_centroid": float(mean_distance),
            "std_distance_from_centroid": float(std_distance),
            "radial_mean": float(np.mean(radial_distances)),
            "radial_std": float(np.std(radial_distances)),
            "num_clusters": len(clusters),
            "cluster_sizes": [len(c) for c in clusters],
            "defect_density": len(self.defects) / (np.pi * self.wafer_radius ** 2) if self.wafer_radius > 0 else 0,
            "spatial_uniformity": float(std_distance / mean_distance) if mean_distance > 0 else 0
        }
        
        self.clusters = clusters
        
        return self.spatial_statistics
    
    def _identify_clusters(self, centers: np.ndarray, distance_threshold: float = 50.0) -> List[List[int]]:
        """Identify defect clusters using distance-based clustering"""
        if len(centers) == 0:
            return []
        
        clusters = []
        assigned = set()
        
        for i, center in enumerate(centers):
            if i in assigned:
                continue
            
            cluster = [i]
            assigned.add(i)
            
            # Find nearby defects
            for j, other_center in enumerate(centers):
                if j in assigned or i == j:
                    continue
                
                distance = np.linalg.norm(center - other_center)
                if distance < distance_threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            if len(cluster) > 1:  # Only keep clusters with multiple defects
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_density_map(self, centers: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """Calculate 2D density map of defects"""
        h, w = self.image_shape
        density = np.zeros((grid_size, grid_size))
        
        if len(centers) == 0:
            return density
        
        # Normalize centers to grid coordinates
        x_coords = centers[:, 0] / w * grid_size
        y_coords = centers[:, 1] / h * grid_size
        
        # Create density map using Gaussian kernel
        for x, y in zip(x_coords, y_coords):
            x_idx = int(np.clip(x, 0, grid_size - 1))
            y_idx = int(np.clip(y, 0, grid_size - 1))
            
            # Add Gaussian contribution
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y_idx + dy, x_idx + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        distance = np.sqrt(dx**2 + dy**2)
                        density[ny, nx] += np.exp(-distance**2 / 2)
        
        return density


class MappingAgent:
    """
    Defect Mapping Agent - Creates spatial maps and analyzes defect distribution
    """
    
    def __init__(self):
        """Initialize the Mapping Agent"""
        logger.info("Initializing Defect Mapping Agent...")
        self.output_dir = Path("reports/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Defect Mapping Agent initialized")
    
    def create_defect_map(
        self,
        defects: List[DefectDetection],
        classifications: List[ClassificationResult],
        image_path: str,
        analysis_id: str
    ) -> Dict:
        """
        Create a comprehensive defect map with spatial analysis
        
        Returns:
            Dictionary containing:
            - map_image_path: Path to generated map image
            - spatial_statistics: Spatial analysis results
            - clusters: Identified defect clusters
            - density_map: 2D density array
        """
        try:
            logger.info(f"Mapping Agent: Creating defect map for {len(defects)} defects")
            
            # Handle empty defects case
            if not defects:
                logger.warning("Mapping Agent: No defects to map, returning empty map")
                return {
                    "map_image_path": None,
                    "spatial_statistics": {
                        "num_clusters": 0,
                        "defect_density": 0.0,
                        "mean_distance_from_centroid": 0.0,
                        "spatial_uniformity": 0.0
                    },
                    "clusters": [],
                    "density_map": [],
                    "defect_positions": []
                }
            
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_shape = image.shape[:2]  # (height, width)
            
            # Create defect map
            defect_map = DefectMap(
                defects=defects,
                classifications=classifications,
                image_shape=image_shape
            )
            
            # Analyze spatial distribution
            spatial_stats = defect_map.analyze_spatial_distribution()
            
            # Generate visualization
            map_image_path = self._generate_map_visualization(
                defect_map=defect_map,
                image_path=image_path,
                analysis_id=analysis_id,
                spatial_stats=spatial_stats
            )
            
            # Convert density map to list for JSON serialization
            density_map_list = defect_map.density_map.tolist() if defect_map.density_map is not None else []
            
            # Format clusters for response
            cluster_data = []
            for i, cluster in enumerate(defect_map.clusters):
                cluster_defects = []
                for idx in cluster:
                    if idx < len(defects):
                        defect = defects[idx]
                        classification = classifications[idx] if idx < len(classifications) else None
                        cluster_defects.append({
                            "defect_id": defect.defect_id,
                            "center": defect_map.defect_centers[idx],
                            "bbox": {
                                "x_min": defect.bbox.x_min,
                                "y_min": defect.bbox.y_min,
                                "x_max": defect.bbox.x_max,
                                "y_max": defect.bbox.y_max
                            },
                            "type": classification.defect_type.value if classification else "unknown",
                            "confidence": classification.confidence if classification else 0.0
                        })
                cluster_data.append({
                    "cluster_id": f"cluster_{i+1}",
                    "size": len(cluster),
                    "defects": cluster_defects
                })
            
            result = {
                "map_image_path": str(map_image_path),
                "spatial_statistics": spatial_stats,
                "clusters": cluster_data,
                "density_map": density_map_list,
                "defect_positions": [
                    {
                        "defect_id": d.defect_id,
                        "center": defect_map.defect_centers[i],
                        "bbox": {
                            "x_min": d.bbox.x_min,
                            "y_min": d.bbox.y_min,
                            "x_max": d.bbox.x_max,
                            "y_max": d.bbox.y_max
                        },
                        "type": classifications[i].defect_type.value if i < len(classifications) else "unknown",
                        "confidence": classifications[i].confidence if i < len(classifications) else 0.0
                    }
                    for i, d in enumerate(defects)
                ]
            }
            
            logger.info(f"Mapping Agent: Generated defect map with {len(cluster_data)} clusters")
            return result
            
        except Exception as e:
            logger.error(f"Error creating defect map: {e}")
            raise
    
    def _generate_map_visualization(
        self,
        defect_map: DefectMap,
        image_path: str,
        analysis_id: str,
        spatial_stats: Dict
    ) -> Path:
        """Generate visualization of defect map"""
        try:
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            
            # 1. Original image with defect overlays
            ax1 = plt.subplot(2, 3, 1)
            ax1.imshow(image_rgb)
            ax1.set_title("Original Image with Defects", fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Draw bounding boxes
            colors = plt.cm.tab10(np.linspace(0, 1, len(defect_map.defects)))
            for i, (defect, center) in enumerate(zip(defect_map.defects, defect_map.defect_centers)):
                bbox = defect.bbox
                rect = Rectangle(
                    (bbox.x_min, bbox.y_min),
                    bbox.x_max - bbox.x_min,
                    bbox.y_max - bbox.y_min,
                    linewidth=2,
                    edgecolor=colors[i % len(colors)],
                    facecolor='none',
                    alpha=0.7
                )
                ax1.add_patch(rect)
                # Add defect ID
                ax1.text(center[0], center[1], defect.defect_id, 
                        fontsize=8, color='white', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            # 2. Defect scatter plot
            ax2 = plt.subplot(2, 3, 2)
            if defect_map.defect_centers:
                centers = np.array(defect_map.defect_centers)
                # Color by defect type
                type_colors = {}
                for i, classification in enumerate(defect_map.classifications):
                    defect_type = classification.defect_type.value
                    if defect_type not in type_colors:
                        type_colors[defect_type] = colors[len(type_colors) % len(colors)]
                
                for i, (center, classification) in enumerate(zip(centers, defect_map.classifications)):
                    defect_type = classification.defect_type.value
                    color = type_colors.get(defect_type, 'gray')
                    ax2.scatter(center[0], center[1], c=[color], s=100, alpha=0.6, edgecolors='black', linewidths=1)
                
                ax2.set_xlim(0, defect_map.image_shape[1])
                ax2.set_ylim(defect_map.image_shape[0], 0)  # Invert y-axis
                ax2.set_xlabel("X Position (pixels)", fontsize=10)
                ax2.set_ylabel("Y Position (pixels)", fontsize=10)
                ax2.set_title("Defect Locations", fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Add legend
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=defect_type)
                                 for defect_type, color in type_colors.items()]
                ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            # 3. Density heatmap
            ax3 = plt.subplot(2, 3, 3)
            if defect_map.density_map is not None:
                im = ax3.imshow(defect_map.density_map, cmap='hot', interpolation='bilinear', origin='upper')
                ax3.set_title("Defect Density Heatmap", fontsize=12, fontweight='bold')
                ax3.set_xlabel("X Position", fontsize=10)
                ax3.set_ylabel("Y Position", fontsize=10)
                plt.colorbar(im, ax=ax3, label='Density')
            
            # 4. Radial distribution
            ax4 = plt.subplot(2, 3, 4)
            if defect_map.defect_centers:
                centers = np.array(defect_map.defect_centers)
                image_center = np.array([defect_map.image_shape[1] / 2, defect_map.image_shape[0] / 2])
                radial_distances = np.linalg.norm(centers - image_center, axis=1)
                
                ax4.hist(radial_distances, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
                ax4.axvline(np.mean(radial_distances), color='red', linestyle='--', linewidth=2, label='Mean')
                ax4.set_xlabel("Distance from Center (pixels)", fontsize=10)
                ax4.set_ylabel("Number of Defects", fontsize=10)
                ax4.set_title("Radial Distribution", fontsize=12, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            # 5. Cluster visualization
            ax5 = plt.subplot(2, 3, 5)
            ax5.imshow(image_rgb, alpha=0.3)
            if defect_map.clusters:
                cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(defect_map.clusters)))
                for cluster_idx, cluster in enumerate(defect_map.clusters):
                    cluster_centers = np.array([defect_map.defect_centers[i] for i in cluster])
                    ax5.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                              c=[cluster_colors[cluster_idx]], s=150, alpha=0.8, 
                              edgecolors='black', linewidths=2, label=f'Cluster {cluster_idx+1}')
                ax5.set_title(f"Defect Clusters ({len(defect_map.clusters)} clusters)", fontsize=12, fontweight='bold')
                ax5.set_xlabel("X Position (pixels)", fontsize=10)
                ax5.set_ylabel("Y Position (pixels)", fontsize=10)
                ax5.legend(loc='upper right', fontsize=8)
                ax5.axis('off')
            else:
                ax5.set_title("No Clusters Detected", fontsize=12, fontweight='bold')
                ax5.axis('off')
            
            # 6. Statistics summary
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            stats_text = f"""
SPATIAL ANALYSIS SUMMARY

Total Defects: {len(defect_map.defects)}
Clusters Found: {spatial_stats.get('num_clusters', 0)}

Centroid:
  X: {spatial_stats.get('centroid', {}).get('x', 0):.1f}
  Y: {spatial_stats.get('centroid', {}).get('y', 0):.1f}

Mean Distance from Centroid:
  {spatial_stats.get('mean_distance_from_centroid', 0):.1f} pixels
  (Std: {spatial_stats.get('std_distance_from_centroid', 0):.1f})

Radial Distribution:
  Mean: {spatial_stats.get('radial_mean', 0):.1f} pixels
  Std: {spatial_stats.get('radial_std', 0):.1f} pixels

Defect Density:
  {spatial_stats.get('defect_density', 0):.4f} defects/pixel²

Spatial Uniformity:
  {spatial_stats.get('spatial_uniformity', 0):.3f}
  (Lower = more uniform)
            """
            ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(f"Wafer Defect Map - Analysis ID: {analysis_id}", 
                        fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_filename = f"defect_map_{analysis_id}_{timestamp}.png"
            map_path = self.output_dir / map_filename
            
            plt.savefig(map_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Mapping Agent: Saved map visualization to {map_path}")
            return map_path
            
        except Exception as e:
            logger.error(f"Error generating map visualization: {e}")
            raise

