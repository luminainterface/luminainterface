import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from language.recursive_pattern_analyzer import RecursivePatternAnalyzer

class RecursivePatternVisualizer:
    """
    Visualization tools for recursive language patterns detected by the RecursivePatternAnalyzer.
    Provides various visualization methods to represent recursive patterns, their depths,
    time evolution, and network relationships.
    """
    
    def __init__(self, data_dir: str = "data/recursive_patterns", 
                 output_dir: str = "output/visualizations",
                 analyzer: Optional[RecursivePatternAnalyzer] = None):
        """
        Initialize the RecursivePatternVisualizer.
        
        Args:
            data_dir: Directory where pattern data is stored
            output_dir: Directory to save visualizations
            analyzer: Optional RecursivePatternAnalyzer instance
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.analyzer = analyzer or RecursivePatternAnalyzer(data_dir=data_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('RecursivePatternVisualizer')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Initialized RecursivePatternVisualizer with data_dir={data_dir} and output_dir={output_dir}")
    
    def load_patterns(self) -> Dict:
        """Load patterns from the pattern history file"""
        pattern_file = os.path.join(self.data_dir, "patterns.json")
        if not os.path.exists(pattern_file):
            self.logger.warning(f"Pattern file not found: {pattern_file}")
            return {}
        
        with open(pattern_file, 'r') as f:
            return json.load(f)
    
    def visualize_pattern_distribution(self, save: bool = True) -> None:
        """
        Create a bar chart showing the distribution of different pattern types.
        
        Args:
            save: Whether to save the visualization to a file
        """
        patterns = self.load_patterns()
        if not patterns or "pattern_counts" not in patterns:
            self.logger.warning("No pattern counts found in the loaded data")
            return
        
        pattern_counts = patterns.get("pattern_counts", {})
        if not pattern_counts:
            self.logger.warning("Pattern counts dictionary is empty")
            return
            
        # Prepare data for visualization
        pattern_types = list(pattern_counts.keys())
        counts = list(pattern_counts.values())
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(pattern_types, counts, color='skyblue')
        plt.xlabel('Pattern Type')
        plt.ylabel('Count')
        plt.title('Distribution of Recursive Pattern Types')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add counts above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        if save:
            output_path = os.path.join(self.output_dir, 'pattern_distribution.png')
            plt.savefig(output_path, dpi=300)
            self.logger.info(f"Saved pattern distribution visualization to {output_path}")
        
        plt.show()
    
    def visualize_pattern_network(self, text: str = None, max_nodes: int = 50, save: bool = True) -> None:
        """
        Create a network graph visualization of pattern relationships.
        
        Args:
            text: Optional text to analyze for patterns
            max_nodes: Maximum number of nodes to include in the visualization
            save: Whether to save the visualization to a file
        """
        # Get pattern data
        if text:
            result = self.analyzer.analyze_text(text)
            patterns = result.get("patterns", [])
        else:
            data = self.load_patterns()
            patterns = data.get("recent_patterns", [])[:max_nodes]
        
        if not patterns:
            self.logger.warning("No patterns available for network visualization")
            return
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes and edges
        for i, pattern in enumerate(patterns):
            pattern_type = pattern.get("type", "unknown")
            pattern_text = pattern.get("text", "")[:30] + "..." if len(pattern.get("text", "")) > 30 else pattern.get("text", "")
            node_id = f"{i}_{pattern_type}"
            G.add_node(node_id, type=pattern_type, text=pattern_text)
            
            # Connect patterns of the same type
            for j, other_pattern in enumerate(patterns):
                if i != j and pattern.get("type") == other_pattern.get("type"):
                    G.add_edge(node_id, f"{j}_{other_pattern.get('type', 'unknown')}")
            
            # Connect patterns that share words
            pattern_words = set(pattern_text.lower().split())
            for j, other_pattern in enumerate(patterns):
                if i != j:
                    other_text = other_pattern.get("text", "")[:30] + "..." if len(other_pattern.get("text", "")) > 30 else other_pattern.get("text", "")
                    other_words = set(other_text.lower().split())
                    if pattern_words.intersection(other_words):
                        G.add_edge(node_id, f"{j}_{other_pattern.get('type', 'unknown')}")
        
        # Limit number of nodes for clarity
        if len(G.nodes) > max_nodes:
            self.logger.info(f"Limiting network visualization to {max_nodes} nodes")
            G = nx.subgraph(G, list(G.nodes)[:max_nodes])
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Define node colors based on pattern type
        type_colors = {
            "self_reference": "red",
            "meta_linguistic": "blue",
            "linguistic_loop": "green",
            "recursive_structure": "purple",
            "consciousness_assertion": "orange",
            "unknown": "gray"
        }
        
        node_colors = [type_colors.get(G.nodes[node]['type'], 'gray') for node in G.nodes]
        
        # Define layout
        layout = nx.spring_layout(G, seed=42)
        
        # Draw graph
        nx.draw_networkx_nodes(G, layout, node_size=300, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, layout, width=1, alpha=0.5)
        nx.draw_networkx_labels(G, layout, labels={node: G.nodes[node]['text'][:10] for node in G.nodes}, 
                               font_size=8, font_family='sans-serif')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     label=pattern_type, markerfacecolor=color, markersize=10)
                          for pattern_type, color in type_colors.items() if pattern_type in [G.nodes[node]['type'] for node in G.nodes]]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('Recursive Pattern Relationship Network')
        plt.axis('off')
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.output_dir, 'pattern_network.png')
            plt.savefig(output_path, dpi=300)
            self.logger.info(f"Saved pattern network visualization to {output_path}")
        
        plt.show()
    
    def visualize_recursive_depth(self, time_series: bool = False, save: bool = True) -> None:
        """
        Visualize the recursive depth of patterns over time or as a distribution.
        
        Args:
            time_series: If True, show depth over time; otherwise, show depth distribution
            save: Whether to save the visualization to a file
        """
        patterns = self.load_patterns()
        if not patterns or "pattern_history" not in patterns:
            self.logger.warning("No pattern history found in the loaded data")
            return
        
        pattern_history = patterns.get("pattern_history", [])
        if not pattern_history:
            self.logger.warning("Pattern history is empty")
            return
        
        # Extract depths and timestamps
        depths = [entry.get("avg_depth", 0) for entry in pattern_history]
        timestamps = list(range(len(depths)))
        
        plt.figure(figsize=(12, 6))
        
        if time_series:
            # Time series visualization
            plt.plot(timestamps, depths, marker='o', linestyle='-', color='blue')
            plt.xlabel('Pattern Entry Index')
            plt.ylabel('Average Recursive Depth')
            plt.title('Evolution of Recursive Pattern Depth Over Time')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add trend line
            z = np.polyfit(timestamps, depths, 1)
            p = np.poly1d(z)
            plt.plot(timestamps, p(timestamps), "r--", alpha=0.7, 
                    label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
            plt.legend()
            
            filename = 'depth_time_series.png'
        else:
            # Distribution visualization
            plt.hist(depths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel('Recursive Depth')
            plt.ylabel('Frequency')
            plt.title('Distribution of Recursive Pattern Depths')
            plt.axvline(np.mean(depths), color='red', linestyle='dashed', linewidth=2, 
                       label=f'Mean: {np.mean(depths):.2f}')
            plt.axvline(np.median(depths), color='green', linestyle='dashed', linewidth=2, 
                       label=f'Median: {np.median(depths):.2f}')
            plt.legend()
            
            filename = 'depth_distribution.png'
        
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300)
            self.logger.info(f"Saved recursive depth visualization to {output_path}")
        
        plt.show()
    
    def visualize_llm_impact(self, text_samples: List[str] = None, save: bool = True) -> None:
        """
        Visualize the impact of LLM weight on pattern detection.
        
        Args:
            text_samples: List of text samples to analyze with different LLM weights
            save: Whether to save the visualization to a file
        """
        if not text_samples:
            text_samples = [
                "This sentence is referring to itself in a recursive manner.",
                "I am aware that I am using language to describe my own language processing capabilities.",
                "The following statement is true: the preceding statement refers to this statement recursively."
            ]
        
        llm_weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        results = []
        
        # Analyze each text sample with different LLM weights
        for text in text_samples:
            text_results = []
            for weight in llm_weights:
                # Create a temporary analyzer with the specific weight
                temp_analyzer = RecursivePatternAnalyzer(data_dir=self.data_dir, llm_weight=weight)
                result = temp_analyzer.analyze_text(text)
                text_results.append({
                    'llm_weight': weight,
                    'pattern_count': len(result.get('patterns', [])),
                    'avg_depth': result.get('avg_depth', 0),
                    'confidence': result.get('confidence', 0)
                })
            results.append((text, text_results))
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        for i, (text, text_results) in enumerate(results):
            plt.subplot(len(results), 1, i+1)
            
            # Get data for this text sample
            weights = [r['llm_weight'] for r in text_results]
            pattern_counts = [r['pattern_count'] for r in text_results]
            depths = [r['avg_depth'] for r in text_results]
            confidences = [r['confidence'] for r in text_results]
            
            # Plot metrics
            plt.plot(weights, pattern_counts, 'o-', label='Pattern Count', color='blue')
            plt.plot(weights, depths, 's-', label='Avg Depth', color='green')
            plt.plot(weights, confidences, '^-', label='Confidence', color='red')
            
            plt.xlabel('LLM Weight')
            plt.ylabel('Value')
            plt.title(f'Text Sample {i+1}: "{text[:50]}..."')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
        
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.output_dir, 'llm_impact.png')
            plt.savefig(output_path, dpi=300)
            self.logger.info(f"Saved LLM impact visualization to {output_path}")
        
        plt.show()
    
    def generate_report(self, text: str = None) -> str:
        """
        Generate a comprehensive report about the recursive patterns.
        
        Args:
            text: Optional text to analyze for patterns
            
        Returns:
            A formatted string report with visualization references
        """
        if text:
            result = self.analyzer.analyze_text(text)
            self.logger.info(f"Generated analysis for provided text: {text[:50]}...")
        else:
            data = self.load_patterns()
            result = {
                "pattern_counts": data.get("pattern_counts", {}),
                "avg_depth": data.get("avg_depth", 0),
                "confidence": data.get("confidence", 0),
                "patterns": data.get("recent_patterns", [])[:10]
            }
        
        # Create visualizations for the report
        self.visualize_pattern_distribution(save=True)
        self.visualize_pattern_network(text=text if text else None, save=True)
        self.visualize_recursive_depth(save=True)
        
        # Generate report text
        report = f"""
        ======================================
        RECURSIVE PATTERN ANALYSIS REPORT
        ======================================
        
        SUMMARY STATISTICS:
        -------------------
        Total patterns analyzed: {sum(result.get("pattern_counts", {}).values())}
        Average recursive depth: {result.get("avg_depth", 0):.2f}
        Overall confidence score: {result.get("confidence", 0):.2f}
        
        PATTERN DISTRIBUTION:
        --------------------
        {self._format_pattern_counts(result.get("pattern_counts", {}))}
        
        TOP PATTERNS DETECTED:
        --------------------
        {self._format_patterns(result.get("patterns", []))}
        
        VISUALIZATIONS:
        --------------
        Pattern distribution chart: {os.path.join(self.output_dir, 'pattern_distribution.png')}
        Pattern network graph: {os.path.join(self.output_dir, 'pattern_network.png')}
        Recursive depth analysis: {os.path.join(self.output_dir, 'depth_distribution.png')}
        
        ======================================
        """
        
        # Save report to file
        report_path = os.path.join(self.output_dir, 'recursive_pattern_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Generated comprehensive report at {report_path}")
        return report
    
    def _format_pattern_counts(self, pattern_counts: Dict) -> str:
        """Format pattern counts for the report"""
        if not pattern_counts:
            return "No pattern counts available."
        
        formatted = ""
        for pattern_type, count in pattern_counts.items():
            formatted += f"- {pattern_type}: {count}\n        "
        return formatted
    
    def _format_patterns(self, patterns: List[Dict]) -> str:
        """Format patterns for the report"""
        if not patterns:
            return "No patterns available."
        
        formatted = ""
        for i, pattern in enumerate(patterns[:10]):  # Limit to top 10
            pattern_text = pattern.get("text", "")[:50] + "..." if len(pattern.get("text", "")) > 50 else pattern.get("text", "")
            formatted += f"{i+1}. Type: {pattern.get('type', 'unknown')}\n           Text: \"{pattern_text}\"\n           Depth: {pattern.get('depth', 0)}\n           Confidence: {pattern.get('confidence', 0):.2f}\n        "
        return formatted

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Recursive Pattern Visualizer')
    parser.add_argument('--text', type=str, help='Text to analyze and visualize')
    parser.add_argument('--data_dir', type=str, default='data/recursive_patterns', 
                        help='Directory containing pattern data')
    parser.add_argument('--output_dir', type=str, default='output/visualizations', 
                        help='Directory to save visualizations')
    parser.add_argument('--distribution', action='store_true', 
                        help='Generate pattern distribution visualization')
    parser.add_argument('--network', action='store_true', 
                        help='Generate pattern network visualization')
    parser.add_argument('--depth', action='store_true', 
                        help='Generate recursive depth visualization')
    parser.add_argument('--time_series', action='store_true', 
                        help='Show depth as time series instead of distribution')
    parser.add_argument('--llm_impact', action='store_true', 
                        help='Generate LLM impact visualization')
    parser.add_argument('--report', action='store_true', 
                        help='Generate comprehensive report')
    parser.add_argument('--all', action='store_true', 
                        help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = RecursivePatternVisualizer(data_dir=args.data_dir, output_dir=args.output_dir)
    
    if args.all or (not any([args.distribution, args.network, args.depth, args.llm_impact, args.report])):
        print("Generating all visualizations...")
        visualizer.visualize_pattern_distribution()
        visualizer.visualize_pattern_network(text=args.text)
        visualizer.visualize_recursive_depth(time_series=args.time_series)
        visualizer.visualize_llm_impact()
        report = visualizer.generate_report(text=args.text)
        print("\nReport summary:")
        print(report[:500] + "...\n")
        
    else:
        if args.distribution:
            visualizer.visualize_pattern_distribution()
        
        if args.network:
            visualizer.visualize_pattern_network(text=args.text)
        
        if args.depth:
            visualizer.visualize_recursive_depth(time_series=args.time_series)
        
        if args.llm_impact:
            visualizer.visualize_llm_impact()
        
        if args.report:
            report = visualizer.generate_report(text=args.text)
            print("\nReport summary:")
            print(report[:500] + "...\n") 