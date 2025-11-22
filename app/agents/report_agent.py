"""
Report Agent - Auto-generates QC reports with plots and summaries
Creates comprehensive quality control reports with visualizations
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import cv2
from PIL import Image as PILImage

from app.core.config import settings
from app.models.schemas import (
    ImageAnalysisResponse,
    ClassificationResult,
    RootCauseAnalysis
)

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    Report Generation Agent
    Creates comprehensive QC reports with visualizations
    """
    
    def __init__(self):
        """Initialize the Report Agent"""
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.plots_dir = self.reports_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def generate_report(
        self,
        analysis: ImageAnalysisResponse,
        image_path: str,
        format: str = "pdf"
    ) -> str:
        """
        Generate comprehensive QC report
        """
        try:
            report_id = f"report_{analysis.analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if format.lower() == "pdf":
                report_path = self._generate_pdf_report(analysis, image_path, report_id)
            elif format.lower() == "html":
                report_path = self._generate_html_report(analysis, image_path, report_id)
            else:
                report_path = self._generate_json_report(analysis, report_id)
            
            logger.info(f"Report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def _generate_pdf_report(
        self,
        analysis: ImageAnalysisResponse,
        image_path: str,
        report_id: str
    ) -> str:
        """Generate PDF report"""
        report_path = self.reports_dir / f"{report_id}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Title
        elements.append(Paragraph("Wafer Defect Analysis Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        metadata_data = [
            ['Analysis ID:', analysis.analysis_id],
            ['Wafer ID:', analysis.wafer_id or 'N/A'],
            ['Batch ID:', analysis.batch_id or 'N/A'],
            ['Date:', analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Defects:', str(analysis.total_defects)],
            ['Severity Score:', f"{analysis.severity_score:.2f}"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(metadata_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_text = self._generate_summary_text(analysis)
        elements.append(Paragraph(summary_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Defect Summary
        elements.append(Paragraph("Defect Summary", styles['Heading2']))
        defect_summary_data = [['Defect Type', 'Count']]
        for defect_type, count in analysis.defect_summary.items():
            defect_summary_data.append([defect_type.replace('_', ' ').title(), str(count)])
        
        defect_table = Table(defect_summary_data)
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (1, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(defect_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add plots
        plot_paths = self._generate_plots(analysis, image_path, report_id)
        for plot_path in plot_paths:
            if Path(plot_path).exists():
                img = Image(plot_path, width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
        
        elements.append(PageBreak())
        
        # Detailed Defect Analysis
        elements.append(Paragraph("Detailed Defect Analysis", styles['Heading2']))
        
        for i, (defect, classification, root_cause) in enumerate(
            zip(analysis.defects, analysis.classifications, analysis.root_causes)
        ):
            elements.append(Paragraph(
                f"Defect {i+1}: {defect.defect_id}",
                styles['Heading3']
            ))
            
            defect_details = [
                ['Type:', classification.defect_type.value.replace('_', ' ').title()],
                ['Confidence:', f"{classification.confidence:.2%}"],
                ['Location:', f"({defect.bbox.x_min:.1f}, {defect.bbox.y_min:.1f}) to ({defect.bbox.x_max:.1f}, {defect.bbox.y_max:.1f})"],
                ['Area:', f"{defect.area:.2f} pixels²"],
                ['Process Step:', root_cause.process_step.value],
                ['Likely Cause:', root_cause.likely_cause],
                ['Root Cause Confidence:', f"{root_cause.confidence:.2%}"]
            ]
            
            detail_table = Table(defect_details, colWidths=[2*inch, 4*inch])
            detail_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            elements.append(detail_table)
            
            if root_cause.recommendations:
                elements.append(Spacer(1, 0.1*inch))
                elements.append(Paragraph("Recommendations:", styles['Heading4']))
                for rec in root_cause.recommendations:
                    elements.append(Paragraph(f"• {rec}", styles['Normal']))
            
            elements.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(elements)
        return str(report_path)
    
    def _generate_plots(
        self,
        analysis: ImageAnalysisResponse,
        image_path: str,
        report_id: str
    ) -> List[str]:
        """Generate visualization plots"""
        plot_paths = []
        
        try:
            # 1. Defect type distribution pie chart
            pie_path = self.plots_dir / f"{report_id}_defect_distribution.png"
            self._plot_defect_distribution(analysis, str(pie_path))
            plot_paths.append(str(pie_path))
            
            # 2. Defect location scatter plot
            scatter_path = self.plots_dir / f"{report_id}_defect_locations.png"
            self._plot_defect_locations(analysis, image_path, str(scatter_path))
            plot_paths.append(str(scatter_path))
            
            # 3. Process step analysis
            process_path = self.plots_dir / f"{report_id}_process_analysis.png"
            self._plot_process_analysis(analysis, str(process_path))
            plot_paths.append(str(process_path))
            
            # 4. Confidence distribution
            conf_path = self.plots_dir / f"{report_id}_confidence_dist.png"
            self._plot_confidence_distribution(analysis, str(conf_path))
            plot_paths.append(str(conf_path))
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        return plot_paths
    
    def _plot_defect_distribution(self, analysis: ImageAnalysisResponse, output_path: str):
        """Plot defect type distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        defect_types = list(analysis.defect_summary.keys())
        counts = list(analysis.defect_summary.values())
        
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(defect_types)))
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=[dt.replace('_', ' ').title() for dt in defect_types],
            autopct='%1.1f%%',
            colors=colors_list,
            startangle=90
        )
        
        ax.set_title('Defect Type Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_defect_locations(self, analysis: ImageAnalysisResponse, image_path: str, output_path: str):
        """Plot defect locations on wafer image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                # Create blank image if can't load
                img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(img_rgb)
            
            # Plot defects
            colors_map = plt.cm.tab10(np.linspace(0, 1, len(analysis.defects)))
            for i, defect in enumerate(analysis.defects):
                bbox = defect.bbox
                rect = plt.Rectangle(
                    (bbox.x_min, bbox.y_min),
                    bbox.x_max - bbox.x_min,
                    bbox.y_max - bbox.y_min,
                    linewidth=2,
                    edgecolor=colors_map[i],
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    bbox.x_min, bbox.y_min - 5,
                    defect.defect_id,
                    color=colors_map[i],
                    fontsize=8,
                    fontweight='bold'
                )
            
            ax.set_title('Defect Locations on Wafer', fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting defect locations: {e}")
    
    def _plot_process_analysis(self, analysis: ImageAnalysisResponse, output_path: str):
        """Plot process step analysis"""
        # Count defects by process step
        process_counts = {}
        for root_cause in analysis.root_causes:
            process = root_cause.process_step.value
            process_counts[process] = process_counts.get(process, 0) + 1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        processes = list(process_counts.keys())
        counts = list(process_counts.values())
        
        bars = ax.bar(processes, counts, color=plt.cm.viridis(np.linspace(0, 1, len(processes))))
        ax.set_xlabel('Process Step', fontsize=12)
        ax.set_ylabel('Defect Count', fontsize=12)
        ax.set_title('Defects by Process Step', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, analysis: ImageAnalysisResponse, output_path: str):
        """Plot confidence distribution"""
        confidences = [c.confidence for c in analysis.classifications]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(confidences, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Classification Confidence Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.2f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_text(self, analysis: ImageAnalysisResponse) -> str:
        """Generate executive summary text"""
        total = analysis.total_defects
        severity = analysis.severity_score
        
        if severity > 0.8:
            severity_level = "CRITICAL"
        elif severity > 0.5:
            severity_level = "HIGH"
        elif severity > 0.3:
            severity_level = "MODERATE"
        else:
            severity_level = "LOW"
        
        most_common = max(analysis.defect_summary.items(), key=lambda x: x[1])[0] if analysis.defect_summary else "Unknown"
        
        summary = f"""
        This analysis identified {total} defects on the wafer with a severity score of {severity:.2f} ({severity_level}).
        The most common defect type is {most_common.replace('_', ' ').title()}.
        """
        
        return summary.strip()
    
    def _generate_html_report(
        self,
        analysis: ImageAnalysisResponse,
        image_path: str,
        report_id: str
    ) -> str:
        """Generate HTML report"""
        report_path = self.reports_dir / f"{report_id}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wafer Defect Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #1a1a1a; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Wafer Defect Analysis Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Analysis ID:</strong> {analysis.analysis_id}</p>
                <p><strong>Total Defects:</strong> {analysis.total_defects}</p>
                <p><strong>Severity Score:</strong> {analysis.severity_score:.2f}</p>
            </div>
            
            <h2>Defect Summary</h2>
            <table>
                <tr><th>Defect Type</th><th>Count</th></tr>
        """
        
        for defect_type, count in analysis.defect_summary.items():
            html_content += f"<tr><td>{defect_type.replace('_', ' ').title()}</td><td>{count}</td></tr>"
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_json_report(
        self,
        analysis: ImageAnalysisResponse,
        report_id: str
    ) -> str:
        """Generate JSON report"""
        report_path = self.reports_dir / f"{report_id}.json"
        
        import json
        report_data = {
            "analysis_id": analysis.analysis_id,
            "timestamp": analysis.timestamp.isoformat(),
            "total_defects": analysis.total_defects,
            "severity_score": analysis.severity_score,
            "defect_summary": analysis.defect_summary,
            "defects": [
                {
                    "defect_id": d.defect_id,
                    "bbox": {
                        "x_min": d.bbox.x_min,
                        "y_min": d.bbox.y_min,
                        "x_max": d.bbox.x_max,
                        "y_max": d.bbox.y_max
                    },
                    "area": d.area
                }
                for d in analysis.defects
            ],
            "classifications": [
                {
                    "defect_id": c.defect_id,
                    "defect_type": c.defect_type.value,
                    "confidence": c.confidence
                }
                for c in analysis.classifications
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(report_path)

