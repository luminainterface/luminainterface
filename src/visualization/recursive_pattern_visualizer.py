import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from language.recursive_pattern_analyzer import RecursivePatternAnalyzer

class RecursivePatternVisualizer:
    """
    Visualizes recursive language patterns detected by the RecursivePatternAnalyzer.
    Creates visual representations of pattern distribution, network relationships,
    recursive depth, and LLM weight impact.
    """
    
    def __init__(self, 
                 data_dir: str = "data/recursive_patterns",
                 output_dir: str = "output/visualizations",
                 analyzer: Optional[RecursivePatternAnalyzer] = None):
        """
        Initialize the RecursivePatternVisualizer.
        
        Args:
            data_dir: Directory containing pattern data
            output_dir: Directory to save visualizations
            analyzer: Optional RecursivePatternAnalyzer instance
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.analyzer = analyzer
        
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
        
        # Initialize pattern data
        self.patterns = []
        self.pattern_history = []
        self.pattern_counts = {}
        
        self.logger.info(f"Initialized RecursivePatternVisualizer with data_dir={data_dir}")
    
    def load_patterns(self) -> bool:
        """
        Load pattern data from the data directory.
        
        Returns:
            True if data was successfully loaded, False otherwise
        """
        pattern_file = os.path.join(self.data_dir, "patterns.json")
        if not os.path.exists(pattern_file):
            self.logger.warning(f"Pattern data file not found: {pattern_file}")
            return False
        
        try:
            with open(pattern_file, 'r') as f:
                data = json.load(f)
                self.pattern_history = data.get("pattern_history", [])
                self.patterns = data.get("recent_patterns", [])
                self.pattern_counts = data.get("pattern_counts", {})
            
            self.logger.info(f"Loaded {len(self.patterns)} patterns and {len(self.pattern_history)} history entries")
            return True
        except Exception as e:
            self.logger.error(f"Error loading pattern data: {e}")
            return False
    
    def visualize_pattern_distribution(self) -> str:
        """
        Visualize the distribution of pattern types.
        
        Returns:
            Path to the generated visualization file
        """
        self.logger.info("Generating pattern distribution visualization")
        
        # Ensure pattern data is available
        if not self.pattern_counts and not self.load_patterns():
            self.logger.warning("No pattern data available for visualization")
            if self.analyzer:
                self.pattern_counts = self.analyzer.get_pattern_statistics()["pattern_counts"]
            else:
                return ""
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Extract pattern types and counts
        pattern_types = list(self.pattern_counts.keys())
        counts = [self.pattern_counts[t] for t in pattern_types]
        
        # Create bar chart
        bars = plt.bar(pattern_types, counts)
        
        # Add labels and formatting
        plt.title("Distribution of Recursive Pattern Types")
        plt.xlabel("Pattern Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        output_file = os.path.join(self.output_dir, "pattern_distribution.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        self.logger.info(f"Pattern distribution visualization saved to {output_file}")
        return output_file
    
    def visualize_pattern_network(self, text: Optional[str] = None) -> str:
        """
        Visualize the network of pattern relationships.
        
        Args:
            text: Optional text to analyze for patterns
            
        Returns:
            Path to the generated visualization file
        """
        self.logger.info("Generating pattern network visualization")
        
        # Ensure pattern data is available
        if not self.patterns and not self.load_patterns():
            self.logger.warning("No pattern data available for network visualization")
            if self.analyzer and text:
                result = self.analyzer.analyze_text(text)
                self.patterns = result.get("patterns", [])
            else:
                return ""
        
        # Limit the number of patterns to avoid overcrowded visualization
        max_patterns = 15
        patterns_to_visualize = self.patterns[:max_patterns] if len(self.patterns) > max_patterns else self.patterns
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes for pattern types
        pattern_types = set(p.get("type") for p in patterns_to_visualize if "type" in p)
        for ptype in pattern_types:
            G.add_node(ptype, node_type="pattern_type")
        
        # Add nodes for patterns and connect to their types
        for i, pattern in enumerate(patterns_to_visualize):
            if "type" in pattern:
                pattern_id = f"pattern_{i}"
                pattern_text = pattern.get("text", "")[:30] + "..." if len(pattern.get("text", "")) > 30 else pattern.get("text", "")
                G.add_node(pattern_id, 
                          node_type="pattern_instance", 
                          text=pattern_text,
                          depth=pattern.get("depth", 1))
                G.add_edge(pattern.get("type"), pattern_id)
        
        # Connect related patterns (those sharing text or having similar types)
        for i, p1 in enumerate(patterns_to_visualize):
            for j, p2 in enumerate(patterns_to_visualize):
                if i < j:  # Avoid duplicate connections
                    # Connect patterns of same type
                    if p1.get("type") == p2.get("type"):
                        G.add_edge(f"pattern_{i}", f"pattern_{j}", weight=1)
                    
                    # Connect patterns with text overlap if they have different types
                    elif (p1.get("text") and p2.get("text") and 
                          p1.get("type") != p2.get("type") and
                          (p1.get("text") in p2.get("text") or p2.get("text") in p1.get("text"))):
                        G.add_edge(f"pattern_{i}", f"pattern_{j}", weight=0.5)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Define node colors and sizes based on type
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            if G.nodes[node].get("node_type") == "pattern_type":
                node_colors.append("skyblue")
                node_sizes.append(1000)
            else:
                # Color by depth if available
                depth = G.nodes[node].get("depth", 1)
                color_intensity = min(1.0, depth / 5.0)  # Normalize depth to 0-1
                node_colors.append((1-color_intensity, 0.2, color_intensity))
                node_sizes.append(500)
        
        # Create layout
        layout = nx.spring_layout(G)
        
        # Draw network
        nx.draw_networkx(G, layout, node_color=node_colors, node_size=node_sizes,
                         font_size=8, edge_color="gray", alpha=0.8)
        
        # Add node labels
        node_labels = {}
        for node in G.nodes():
            if G.nodes[node].get("node_type") == "pattern_type":
                node_labels[node] = node
            else:
                # For pattern instances, show a snippet of text
                text = G.nodes[node].get("text", "")
                depth = G.nodes[node].get("depth", 1)
                node_labels[node] = f"D{depth}: {text[:15]}..." if text else f"D{depth}"
        
        nx.draw_networkx_labels(G, layout, labels=node_labels, font_size=8)
        
        plt.title("Network of Recursive Language Patterns")
        plt.axis("off")
        plt.tight_layout()
        
        # Save visualization
        output_file = os.path.join(self.output_dir, "pattern_network.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        self.logger.info(f"Pattern network visualization saved to {output_file}")
        return output_file
    
    def visualize_recursive_depth(self, visualization_type: str = "time_series") -> str:
        """
        Visualize the recursive depth of patterns over time or as a distribution.
        
        Args:
            visualization_type: Type of visualization - "time_series" or "distribution"
            
        Returns:
            Path to the generated visualization file
        """
        self.logger.info(f"Generating recursive depth visualization ({visualization_type})")
        
        # Ensure pattern history is available
        if not self.pattern_history and not self.load_patterns():
            self.logger.warning("No pattern history available for depth visualization")
            return ""
        
        # Extract timestamps and depths
        timestamps = []
        depths = []
        
        for entry in self.pattern_history:
            if "timestamp" in entry and "avg_depth" in entry:
                timestamps.append(datetime.fromtimestamp(entry["timestamp"]))
                depths.append(entry["avg_depth"])
        
        if not timestamps:
            self.logger.warning("No depth data found in pattern history")
            return ""
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        if visualization_type == "time_series":
            # Create time series plot
            plt.plot(timestamps, depths, marker='o', linestyle='-', color='blue', alpha=0.7)
            plt.title("Recursive Depth of Patterns Over Time")
            plt.xlabel("Time")
            plt.ylabel("Average Recursive Depth")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis as dates
            plt.gcf().autofmt_xdate()
            
            # Add moving average
            if len(depths) > 5:
                window_size = min(5, len(depths) // 2)
                moving_avg = np.convolve(depths, np.ones(window_size)/window_size, mode='valid')
                moving_avg_times = timestamps[window_size-1:]
                plt.plot(moving_avg_times, moving_avg, linestyle='-', color='red', 
                         alpha=0.8, label=f"{window_size}-point Moving Average")
                plt.legend()
            
            output_file = os.path.join(self.output_dir, "depth_time_series.png")
        
        else:  # distribution
            # Create histogram
            plt.hist(depths, bins=10, alpha=0.7, color='green')
            plt.title("Distribution of Recursive Pattern Depths")
            plt.xlabel("Recursive Depth")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add mean and median lines
            mean_depth = np.mean(depths)
            median_depth = np.median(depths)
            plt.axvline(mean_depth, color='red', linestyle='-', alpha=0.8, label=f"Mean: {mean_depth:.2f}")
            plt.axvline(median_depth, color='blue', linestyle='--', alpha=0.8, label=f"Median: {median_depth:.2f}")
            plt.legend()
            
            output_file = os.path.join(self.output_dir, "depth_distribution.png")
        
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        self.logger.info(f"Recursive depth visualization saved to {output_file}")
        return output_file
    
    def visualize_llm_impact(self, 
                            text_samples: List[str] = None, 
                            weights: List[float] = None) -> str:
        """
        Visualize the impact of different LLM weights on pattern detection.
        
        Args:
            text_samples: List of text samples to analyze (default: use built-in samples)
            weights: List of LLM weights to test (default: [0.0, 0.25, 0.5, 0.75, 1.0])
            
        Returns:
            Path to the generated visualization file
        """
        self.logger.info("Generating LLM impact visualization")
        
        if not self.analyzer:
            self.logger.warning("RecursivePatternAnalyzer instance required for LLM impact visualization")
            return ""
        
        # Default text samples
        if text_samples is None:
            text_samples = [
                "This sentence refers to itself in a recursive pattern.",
                "Language can be used to describe language, creating a meta-linguistic structure.",
                "The previous statement loops back to this statement, forming a linguistic loop.",
                "Words (which contain letters (which are symbols)) can be nested recursively."
            ]
        
        # Default weights
        if weights is None:
            weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Analyze text samples with different weights
        results = {}
        
        for sample in text_samples:
            sample_results = []
            for weight in weights:
                self.analyzer.adjust_llm_weight(weight)
                analysis = self.analyzer.analyze_text(sample)
                
                sample_results.append({
                    "weight": weight,
                    "pattern_count": len(analysis["patterns"]),
                    "confidence": analysis["confidence"],
                    "avg_depth": analysis["avg_depth"]
                })
            
            results[sample[:30] + "..." if len(sample) > 30 else sample] = sample_results
        
        # Create visualization
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot pattern counts
        ax = axs[0]
        for sample, sample_results in results.items():
            x = [r["weight"] for r in sample_results]
            y = [r["pattern_count"] for r in sample_results]
            ax.plot(x, y, marker='o', linestyle='-', label=sample)
        
        ax.set_title("Impact of LLM Weight on Pattern Count")
        ax.set_xlabel("LLM Weight")
        ax.set_ylabel("Pattern Count")
        ax.set_xticks(weights)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        # Plot confidence
        ax = axs[1]
        for sample, sample_results in results.items():
            x = [r["weight"] for r in sample_results]
            y = [r["confidence"] for r in sample_results]
            ax.plot(x, y, marker='o', linestyle='-', label=sample)
        
        ax.set_title("Impact of LLM Weight on Analysis Confidence")
        ax.set_xlabel("LLM Weight")
        ax.set_ylabel("Confidence")
        ax.set_xticks(weights)
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot average depth
        ax = axs[2]
        for sample, sample_results in results.items():
            x = [r["weight"] for r in sample_results]
            y = [r["avg_depth"] for r in sample_results]
            ax.plot(x, y, marker='o', linestyle='-', label=sample)
        
        ax.set_title("Impact of LLM Weight on Average Recursive Depth")
        ax.set_xlabel("LLM Weight")
        ax.set_ylabel("Average Depth")
        ax.set_xticks(weights)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)
        fig.suptitle("Impact of LLM Weight on Recursive Pattern Analysis", fontsize=16)
        
        # Save visualization
        output_file = os.path.join(self.output_dir, "llm_impact.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        self.logger.info(f"LLM impact visualization saved to {output_file}")
        return output_file
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report on recursive pattern analysis.
        
        Returns:
            Path to the generated report file
        """
        self.logger.info("Generating recursive pattern analysis report")
        
        # Ensure pattern data is available
        if not self.load_patterns() and not self.pattern_counts:
            if self.analyzer:
                self.pattern_counts = self.analyzer.get_pattern_statistics()["pattern_counts"]
                self.pattern_history = self.analyzer.pattern_history
                self.patterns = self.analyzer.recent_patterns
            else:
                self.logger.warning("No pattern data available for report generation")
                return ""
        
        # Generate report content
        report = []
        report.append("=" * 80)
        report.append("RECURSIVE PATTERN ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Add timestamp
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("-" * 40)
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        
        total_patterns = sum(self.pattern_counts.values()) if self.pattern_counts else 0
        report.append(f"Total Patterns Detected: {total_patterns}")
        
        if self.pattern_history:
            latest_timestamp = max(entry.get("timestamp", 0) for entry in self.pattern_history)
            latest_time = datetime.fromtimestamp(latest_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            report.append(f"Latest Analysis: {latest_time}")
            
            if any("avg_depth" in entry for entry in self.pattern_history):
                avg_depths = [entry["avg_depth"] for entry in self.pattern_history if "avg_depth" in entry]
                avg_depth = sum(avg_depths) / len(avg_depths) if avg_depths else 0
                report.append(f"Average Recursive Depth: {avg_depth:.2f}")
        
        report.append("")
        
        # Pattern distribution
        if self.pattern_counts:
            report.append("-" * 40)
            report.append("PATTERN TYPE DISTRIBUTION")
            report.append("-" * 40)
            
            # Find the maximum length of pattern type names for formatting
            max_len = max(len(str(t)) for t in self.pattern_counts.keys()) if self.pattern_counts else 0
            
            for pattern_type, count in self.pattern_counts.items():
                percentage = (count / total_patterns * 100) if total_patterns > 0 else 0
                bar = "█" * int(percentage / 5)  # Create a simple bar chart with blocks
                report.append(f"{pattern_type:<{max_len}} | {count:5} | {percentage:5.1f}% {bar}")
            
            report.append("")
        
        # Recent patterns
        if self.patterns:
            report.append("-" * 40)
            report.append(f"RECENT PATTERNS (showing {min(5, len(self.patterns))} of {len(self.patterns)})")
            report.append("-" * 40)
            
            for i, pattern in enumerate(self.patterns[:5]):  # Show at most 5 recent patterns
                pattern_type = pattern.get("type", "Unknown")
                text = pattern.get("text", "")
                confidence = pattern.get("confidence", 0)
                depth = pattern.get("depth", 0)
                
                report.append(f"Pattern {i+1}:")
                report.append(f"  Type: {pattern_type}")
                report.append(f"  Text: \"{text[:50]}...\"" if len(text) > 50 else f"  Text: \"{text}\"")
                report.append(f"  Confidence: {confidence:.2f}")
                report.append(f"  Recursive Depth: {depth}")
                report.append("")
        
        # Visualizations
        report.append("-" * 40)
        report.append("VISUALIZATIONS")
        report.append("-" * 40)
        
        # Generate visualizations for the report
        viz_files = []
        viz_files.append(("Pattern Distribution", self.visualize_pattern_distribution()))
        viz_files.append(("Pattern Network", self.visualize_pattern_network()))
        viz_files.append(("Recursive Depth (Time Series)", self.visualize_recursive_depth("time_series")))
        viz_files.append(("Recursive Depth (Distribution)", self.visualize_recursive_depth("distribution")))
        
        if self.analyzer:
            viz_files.append(("LLM Impact Analysis", self.visualize_llm_impact()))
        
        # Add visualization references to the report
        for viz_name, viz_file in viz_files:
            if viz_file:
                rel_path = os.path.relpath(viz_file, start=os.path.dirname(self.output_dir))
                report.append(f"{viz_name}: {rel_path}")
        
        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        # Format report as string
        formatted = "\n".join(report)
        
        # Save report to file
        output_file = os.path.join(self.output_dir, "recursive_pattern_report.txt")
        try:
            with open(output_file, 'w') as f:
                f.write(formatted)
            self.logger.info(f"Recursive pattern analysis report saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
            return ""
        
        return output_file

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create data and output directories
    data_dir = "data/recursive_patterns"
    output_dir = "output/visualizations"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer and visualizer
    analyzer = RecursivePatternAnalyzer(data_dir=data_dir)
    visualizer = RecursivePatternVisualizer(data_dir=data_dir, output_dir=output_dir, analyzer=analyzer)
    
    # Generate test data
    test_texts = [
        "This sentence refers to itself in a recursive pattern.",
        "Language can be used to describe language, creating a meta-linguistic structure.",
        "The previous statement loops back to this statement, forming a linguistic loop.",
        "Words (which contain letters (which are symbols)) can be nested recursively."
    ]
    
    for text in test_texts:
        analyzer.analyze_text(text)
    
    # Generate visualizations
    print("Generating pattern distribution visualization...")
    visualizer.visualize_pattern_distribution()
    
    print("Generating pattern network visualization...")
    visualizer.visualize_pattern_network()
    
    print("Generating recursive depth visualization (time series)...")
    visualizer.visualize_recursive_depth("time_series")
    
    print("Generating recursive depth visualization (distribution)...")
    visualizer.visualize_recursive_depth("distribution")
    
    print("Generating LLM impact visualization...")
    visualizer.visualize_llm_impact()
    
    # Generate report
    print("Generating comprehensive report...")
    report_file = visualizer.generate_report()
    
    if report_file:
        report = ""
        with open(report_file, 'r') as f:
            report = f.read()
        
        if report:
            print("\nReport summary:")
            print(report[:500] + "...\n") 