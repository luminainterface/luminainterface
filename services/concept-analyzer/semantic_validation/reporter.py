"""
Health reporter for semantic validation results.
Generates detailed reports in various formats.
"""

from typing import Dict, List
from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class HealthReporter:
    def __init__(self, results: Dict):
        """Initialize the health reporter with validation results"""
        self.results = results
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        
    def generate_markdown_report(self) -> str:
        """Generate a markdown report of validation results"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = [
            f"# Lumina Semantic Health Report\n",
            f"Generated at: {now}\n",
            "## Summary\n"
        ]
        
        # Process each scenario
        for scenario_name, results in self.results.items():
            report.extend([
                f"### {scenario_name}\n",
                f"- Total Events: {len(results)}",
                f"- Time Range: {results[0]['timestamp']} to {results[-1]['timestamp']}\n",
                "#### Key Metrics:\n"
            ])
            
            if scenario_name == 'DriftScenario':
                drift_data = pd.DataFrame(results)
                avg_drift = drift_data['drift_amount'].abs().mean()
                max_drift = drift_data['drift_amount'].abs().max()
                report.extend([
                    f"- Average Drift: {avg_drift:.3f}",
                    f"- Maximum Drift: {max_drift:.3f}",
                    "- Top Drifting Concepts:\n"
                ])
                
                # Get top drifting concepts
                top_drifts = drift_data.groupby('concept')['drift_amount'].agg(
                    lambda x: x.abs().mean()
                ).sort_values(ascending=False).head(3)
                
                for concept, drift in top_drifts.items():
                    report.append(f"  * {concept}: {drift:.3f}")
                    
            elif scenario_name == 'UsageScenario':
                usage_data = pd.DataFrame(results)
                total_increase = sum(r['new_usage'] - r['old_usage'] for r in results)
                avg_increase = total_increase / len(results)
                report.extend([
                    f"- Total Usage Increase: {total_increase}",
                    f"- Average Usage Increase: {avg_increase:.2f}",
                    "- Most Used Concepts:\n"
                ])
                
                # Get most used concepts
                top_usage = usage_data.groupby('concept')['new_usage'].max().sort_values(
                    ascending=False
                ).head(3)
                
                for concept, usage in top_usage.items():
                    report.append(f"  * {concept}: {usage}")
                    
            elif scenario_name == 'RelationshipScenario':
                rel_data = pd.DataFrame(results)
                avg_strength = rel_data['relationship_strength'].mean()
                report.extend([
                    f"- Average Relationship Strength: {avg_strength:.3f}",
                    "- Strongest Relationships:\n"
                ])
                
                # Get strongest relationships
                rel_data['relationship'] = rel_data.apply(
                    lambda x: f"{x['source_concept']} → {x['target_concept']}", axis=1
                )
                top_rels = rel_data.groupby('relationship')['relationship_strength'].mean().sort_values(
                    ascending=False
                ).head(3)
                
                for rel, strength in top_rels.items():
                    report.append(f"  * {rel}: {strength:.3f}")
                    
            report.append("\n")  # Add spacing between sections
            
        return "\n".join(report)
        
    def generate_html_report(self) -> str:
        """Generate an HTML report with interactive visualizations"""
        markdown_content = self.generate_markdown_report()
        
        # Convert markdown to HTML (basic conversion)
        html_content = f"""
        <html>
        <head>
            <title>Lumina Semantic Health Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                h3 {{ color: #7f8c8d; }}
                .metric {{ 
                    background: #f7f9fc;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            {markdown_content.replace('# ', '<h1>')
                           .replace('## ', '<h2>')
                           .replace('### ', '<h3>')
                           .replace('#### ', '<h4>')
                           .replace('- ', '<div class="metric">• ')
                           .replace('\n', '</div>\n')}
        </body>
        </html>
        """
        
        return html_content
        
    def save_reports(self, base_filename: str):
        """Save reports in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save markdown report
        md_path = self.report_dir / f"{base_filename}_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(self.generate_markdown_report())
            
        # Save HTML report
        html_path = self.report_dir / f"{base_filename}_{timestamp}.html"
        with open(html_path, 'w') as f:
            f.write(self.generate_html_report())
            
        # Save raw results as JSON
        json_path = self.report_dir / f"{base_filename}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        return {
            'markdown': str(md_path),
            'html': str(html_path),
            'json': str(json_path)
        }
        
    def plot_drift_trends(self, save_path: str = None):
        """Generate drift trend visualization"""
        if 'DriftScenario' not in self.results:
            return
            
        drift_data = pd.DataFrame(self.results['DriftScenario'])
        drift_data['timestamp'] = pd.to_datetime(drift_data['timestamp'])
        
        plt.figure(figsize=(12, 6))
        for concept in drift_data['concept'].unique():
            concept_data = drift_data[drift_data['concept'] == concept]
            plt.plot(concept_data['timestamp'], concept_data['new_quality'], 
                    label=concept, marker='o')
            
        plt.title('Concept Quality Over Time')
        plt.xlabel('Time')
        plt.ylabel('Quality Score')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()
        
    def plot_relationship_heatmap(self, save_path: str = None):
        """Generate relationship strength heatmap"""
        if 'RelationshipScenario' not in self.results:
            return
            
        rel_data = pd.DataFrame(self.results['RelationshipScenario'])
        
        # Create pivot table of relationship strengths
        strength_matrix = rel_data.pivot_table(
            values='relationship_strength',
            index='source_concept',
            columns='target_concept',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        plt.imshow(strength_matrix, cmap='YlOrRd')
        plt.colorbar(label='Relationship Strength')
        
        plt.xticks(range(len(strength_matrix.columns)), strength_matrix.columns, rotation=45)
        plt.yticks(range(len(strength_matrix.index)), strength_matrix.index)
        
        plt.title('Concept Relationship Strength Heatmap')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close() 