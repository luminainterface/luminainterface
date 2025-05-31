#!/usr/bin/env python3
"""
Generate Research Papers for Professor Grading
Creates multiple high-quality research papers across different academic fields
"""

import asyncio
import json
import time
import requests
from datetime import datetime
import os

class PaperGenerator:
    """Generate research papers for academic grading"""
    
    def __init__(self):
        self.backend_url = "http://localhost:5000"
        self.papers_generated = []
        
        # Sample paper topics for different academic fields
        self.paper_topics = [
            {
                'topic': 'AI Ethics in Healthcare Diagnostic Systems: Addressing Bias and Ensuring Equitable Patient Outcomes',
                'field': 'ai',
                'keywords': 'artificial intelligence, healthcare ethics, diagnostic bias, algorithmic fairness, patient equity, medical AI, bias detection, healthcare disparities',
                'target_journal': 'nejm',
                'context': 'This research investigates the ethical implications of AI-powered diagnostic systems in healthcare, with particular focus on identifying and mitigating algorithmic bias that could lead to disparate health outcomes across different patient populations.',
                'quality_score': 9.5,
                'humanization': 8,
                'fact_check': 9
            },
            {
                'topic': 'Precision Medicine and AI-Driven Personalized Treatment Protocols in Oncology',
                'field': 'medical',
                'keywords': 'precision medicine, personalized treatment, AI oncology, biomarker analysis, genomic profiling, cancer treatment, targeted therapy',
                'target_journal': 'lancet',
                'context': 'Comprehensive analysis of AI-driven precision medicine approaches in oncology, examining the integration of genomic data, biomarker analysis, and machine learning algorithms to develop personalized treatment protocols.',
                'quality_score': 9.7,
                'humanization': 7,
                'fact_check': 10
            },
            {
                'topic': 'Constitutional Implications of AI Decision-Making in Criminal Justice Systems',
                'field': 'law',
                'keywords': 'constitutional law, AI criminal justice, due process, algorithmic bias, judicial AI, criminal justice reform, constitutional rights',
                'target_journal': 'harvard-law',
                'context': 'Legal analysis of constitutional challenges posed by AI decision-making systems in criminal justice, examining due process implications, equal protection concerns, and the need for judicial oversight.',
                'quality_score': 9.4,
                'humanization': 9,
                'fact_check': 8
            },
            {
                'topic': 'Machine Learning Models for Climate Change Prediction and Mitigation Strategies',
                'field': 'engineering',
                'keywords': 'climate modeling, machine learning, environmental prediction, carbon reduction, sustainability, climate science, environmental engineering',
                'target_journal': 'nature',
                'context': 'Engineering approach to climate change mitigation using advanced machine learning models for prediction, optimization of renewable energy systems, and development of sustainable technologies.',
                'quality_score': 9.6,
                'humanization': 6,
                'fact_check': 9
            },
            {
                'topic': 'Quantum Computing Applications in Drug Discovery and Molecular Simulation',
                'field': 'physics',
                'keywords': 'quantum computing, drug discovery, molecular simulation, quantum algorithms, pharmaceutical research, computational chemistry',
                'target_journal': 'science',
                'context': 'Investigation of quantum computing applications in pharmaceutical research, focusing on molecular simulation capabilities, drug-target interaction modeling, and acceleration of drug discovery processes.',
                'quality_score': 9.8,
                'humanization': 5,
                'fact_check': 9
            },
            {
                'topic': 'Psychological Impact of Social Media Algorithms on Adolescent Mental Health',
                'field': 'psychology',
                'keywords': 'social media, adolescent psychology, mental health, algorithmic influence, digital wellness, social media addiction, youth development',
                'target_journal': 'pnas',
                'context': 'Psychological study examining the relationship between social media algorithmic content delivery and adolescent mental health outcomes, with focus on anxiety, depression, and social comparison behaviors.',
                'quality_score': 9.3,
                'humanization': 8,
                'fact_check': 8
            }
        ]
    
    def check_backend_status(self):
        """Check if the research backend is running"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def generate_paper(self, topic_data, use_enhanced_fact_check=False):
        """Generate a single research paper"""
        
        print(f"\n🔬 Generating Paper: {topic_data['topic'][:60]}...")
        print(f"   Field: {topic_data['field']} | Quality: {topic_data['quality_score']}/10 | Humanization: {topic_data['humanization']}/10")
        
        try:
            # Choose endpoint based on fact-checking preference
            endpoint = "/generate_paper_with_enhanced_fact_check" if use_enhanced_fact_check else "/generate_paper"
            
            # Send request to backend
            response = requests.post(
                f"{self.backend_url}{endpoint}",
                json=topic_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Create paper metadata
                paper_info = {
                    'topic': topic_data['topic'],
                    'field': topic_data['field'],
                    'quality_score': topic_data['quality_score'],
                    'humanization_level': topic_data['humanization'],
                    'fact_check_level': topic_data['fact_check'],
                    'enhanced_fact_check': use_enhanced_fact_check,
                    'generation_time': datetime.now().isoformat(),
                    'result': result
                }
                
                self.papers_generated.append(paper_info)
                
                print(f"   ✅ Generated successfully!")
                if use_enhanced_fact_check and 'enhanced_fact_check' in result:
                    fc = result['enhanced_fact_check']
                    print(f"   📊 Reliability Score: {(fc['overall_reliability_score'] * 100):.1f}%")
                    print(f"   🔍 Claims Checked: {fc['total_claims_checked']}")
                    if fc['suspicious_claims_found'] > 0:
                        print(f"   ⚠️ Suspicious Claims: {fc['suspicious_claims_found']}")
                
                return paper_info
            else:
                print(f"   ❌ Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return None
    
    def save_paper_to_file(self, paper_info, output_dir="papers_for_grading"):
        """Save individual paper to file"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create filename
        field = paper_info['field']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{field}_{timestamp}_quality_{paper_info['quality_score']}.html"
        filepath = os.path.join(output_dir, filename)
        
        # Get paper content
        result = paper_info['result']
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{paper_info['topic']}</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #fff;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
        }}
        .title {{
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .meta {{
            color: #666;
            font-style: italic;
            margin-bottom: 10px;
        }}
        .quality-info {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            font-size: 0.9rem;
        }}
        .content {{
            text-align: justify;
            font-size: 1.1rem;
        }}
        .fact-check-info {{
            background: #e8f5e8;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            font-size: 0.9rem;
        }}
        .grading-rubric {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
        }}
        h2 {{ color: #333; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
        h3 {{ color: #555; }}
        .page-break {{ page-break-before: always; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">{paper_info['topic']}</div>
        <div class="meta">
            Generated by AI Research Suite | Field: {paper_info['field'].title()} | 
            Target Journal: {paper_info.get('target_journal', 'N/A').upper()}
        </div>
        <div class="meta">
            Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
        </div>
    </div>

    <div class="quality-info">
        <h3>📊 Paper Quality Metrics</h3>
        <ul>
            <li><strong>Target Quality Score:</strong> {paper_info['quality_score']}/10</li>
            <li><strong>Humanization Level:</strong> {paper_info['humanization_level']}/10</li>
            <li><strong>Fact-Check Rigor:</strong> {paper_info['fact_check_level']}/10</li>
            <li><strong>Enhanced Web Fact-Check:</strong> {'Yes' if paper_info['enhanced_fact_check'] else 'No'}</li>
        </ul>
    </div>
"""

        # Add enhanced fact-check information if available
        if paper_info['enhanced_fact_check'] and 'enhanced_fact_check' in result:
            fc = result['enhanced_fact_check']
            html_content += f"""
    <div class="fact-check-info">
        <h3>🔍 Enhanced Fact-Check Results</h3>
        <ul>
            <li><strong>Reliability Score:</strong> {(fc['overall_reliability_score'] * 100):.1f}%</li>
            <li><strong>Claims Verified:</strong> {fc['total_claims_checked']}</li>
            <li><strong>Suspicious Claims:</strong> {fc['suspicious_claims_found']}</li>
            <li><strong>Processing Time:</strong> {fc['processing_time']:.2f}s</li>
        </ul>
        <p><strong>Recommendations:</strong></p>
        <ul>
            {''.join([f'<li>{rec}</li>' for rec in fc.get('recommendations', [])])}
        </ul>
    </div>
"""

        # Add grading rubric
        html_content += f"""
    <div class="grading-rubric">
        <h3>📝 Grading Rubric for Professor</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #f8f9fa;">
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Criteria</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Points</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Score</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Content Quality & Depth</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">25</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">____</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Academic Writing & Structure</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">20</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">____</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Methodology & Research Design</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">20</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">____</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Critical Analysis & Originality</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">15</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">____</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Citations & References</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">10</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">____</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Ethical Considerations</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">10</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">____</td>
            </tr>
            <tr style="background: #f8f9fa; font-weight: bold;">
                <td style="border: 1px solid #ddd; padding: 8px;">Total</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">100</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">____</td>
            </tr>
        </table>
        <p style="margin-top: 15px;"><strong>Comments:</strong></p>
        <div style="border: 1px solid #ddd; height: 100px; margin-top: 5px;"></div>
    </div>

    <div class="page-break"></div>

    <div class="content">
        {result.get('content', 'Content not available')}
    </div>

    <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9rem;">
        <p><strong>AI Generation Details:</strong></p>
        <ul>
            <li>Generated using Publication Excellence Generator v1.0</li>
            <li>Quality Target: {paper_info['quality_score']}/10</li>
            <li>Humanization Applied: Level {paper_info['humanization_level']}/10</li>
            <li>Enhanced Fact-Checking: {'Enabled' if paper_info['enhanced_fact_check'] else 'Standard'}</li>
            <li>Academic Field: {paper_info['field'].title()}</li>
        </ul>
    </div>
</body>
</html>
"""
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   📄 Saved to: {filepath}")
        return filepath
    
    def generate_summary_report(self, output_dir="papers_for_grading"):
        """Generate summary report for all papers"""
        
        summary_file = os.path.join(output_dir, "grading_summary_report.html")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Papers - Grading Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
        .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; border-bottom: 3px solid #007bff; padding-bottom: 15px; }}
        .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: #e8f4fd; border: 1px solid #007bff; border-radius: 8px; padding: 20px; text-align: center; }}
        .papers-list {{ margin-top: 30px; }}
        .paper-item {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 15px 0; }}
        .paper-title {{ font-weight: bold; color: #333; margin-bottom: 10px; }}
        .paper-meta {{ color: #666; font-size: 0.9rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f8f9fa; font-weight: bold; }}
        .quality-high {{ color: #28a745; font-weight: bold; }}
        .quality-medium {{ color: #ffc107; font-weight: bold; }}
        .quality-low {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 Research Papers - Grading Summary Report</h1>
        <p style="text-align: center; color: #666; font-style: italic;">
            Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")} | 
            Publication Excellence Generator v1.0
        </p>
        
        <div class="summary-stats">
            <div class="stat-card">
                <h3>📊 Total Papers</h3>
                <div style="font-size: 2rem; font-weight: bold; color: #007bff;">{len(self.papers_generated)}</div>
            </div>
            <div class="stat-card">
                <h3>🎯 Avg Quality Score</h3>
                <div style="font-size: 2rem; font-weight: bold; color: #28a745;">
                    {sum(p['quality_score'] for p in self.papers_generated) / len(self.papers_generated):.1f}/10
                </div>
            </div>
            <div class="stat-card">
                <h3>🔬 Academic Fields</h3>
                <div style="font-size: 2rem; font-weight: bold; color: #17a2b8;">
                    {len(set(p['field'] for p in self.papers_generated))}
                </div>
            </div>
            <div class="stat-card">
                <h3>🔍 Enhanced Fact-Checked</h3>
                <div style="font-size: 2rem; font-weight: bold; color: #6f42c1;">
                    {sum(1 for p in self.papers_generated if p['enhanced_fact_check'])}
                </div>
            </div>
        </div>
        
        <h2>📄 Papers Generated for Grading</h2>
        <table>
            <thead>
                <tr>
                    <th>Paper Title</th>
                    <th>Field</th>
                    <th>Quality Score</th>
                    <th>Humanization</th>
                    <th>Fact-Check</th>
                    <th>Enhanced FC</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for paper in self.papers_generated:
            quality_class = "quality-high" if paper['quality_score'] >= 9.5 else "quality-medium" if paper['quality_score'] >= 9.0 else "quality-low"
            
            html_content += f"""
                <tr>
                    <td>{paper['topic'][:80]}{'...' if len(paper['topic']) > 80 else ''}</td>
                    <td>{paper['field'].title()}</td>
                    <td class="{quality_class}">{paper['quality_score']}/10</td>
                    <td>{paper['humanization_level']}/10</td>
                    <td>{paper['fact_check_level']}/10</td>
                    <td>{'✅' if paper['enhanced_fact_check'] else '❌'}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
        
        <h2>📝 Grading Instructions for Professor</h2>
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 20px; margin: 20px 0;">
            <h3>🎯 Evaluation Criteria</h3>
            <ul>
                <li><strong>Content Quality (25 pts):</strong> Depth of analysis, relevance to field, innovation</li>
                <li><strong>Academic Writing (20 pts):</strong> Structure, clarity, professional tone</li>
                <li><strong>Methodology (20 pts):</strong> Research design, data analysis approach</li>
                <li><strong>Critical Analysis (15 pts):</strong> Original insights, critical thinking</li>
                <li><strong>Citations (10 pts):</strong> Proper referencing, source quality</li>
                <li><strong>Ethics (10 pts):</strong> Ethical considerations, responsible research</li>
            </ul>
            
            <h3>🤖 AI Detection Notes</h3>
            <ul>
                <li>Papers with higher humanization levels (7-10) should appear more natural</li>
                <li>Enhanced fact-checked papers have verified claims and reduced false specifics</li>
                <li>Look for patterns: overly precise statistics, fictional experts, perfect structure</li>
                <li>Quality scores indicate target publication level (9.5+ = top-tier journals)</li>
            </ul>
            
            <h3>📊 Expected Grade Distribution</h3>
            <ul>
                <li><strong>Quality 9.5-10:</strong> Expected A/A- grades (publication ready)</li>
                <li><strong>Quality 9.0-9.4:</strong> Expected B+/A- grades (strong academic work)</li>
                <li><strong>Quality 8.5-8.9:</strong> Expected B/B+ grades (good academic work)</li>
            </ul>
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9rem; text-align: center;">
            Generated by Publication Excellence Generator | Enhanced Research Backend v1.0<br>
            Total Generation Time: {datetime.now().isoformat()}<br>
            System Features: AI Failsafe Traps, Enhanced Fact-Checking, AI Humanization, Publication Excellence
        </div>
    </div>
</body>
</html>
"""
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n📋 Summary report saved to: {summary_file}")
        return summary_file

async def main():
    """Main function to generate papers for grading"""
    
    print("🏆 PUBLICATION EXCELLENCE GENERATOR - PAPERS FOR GRADING")
    print("=" * 70)
    print("Generating high-quality research papers for professor evaluation...")
    
    generator = PaperGenerator()
    
    # Check if backend is running
    print("\n🔍 Checking research backend status...")
    if not generator.check_backend_status():
        print("❌ Backend not running. Please start enhanced_research_backend.py first")
        print("   Command: python enhanced_research_backend.py")
        return
    
    print("✅ Research backend is running!")
    
    # Generate papers
    print(f"\n📄 Generating {len(generator.paper_topics)} research papers...")
    
    # Generate first batch with standard fact-checking
    print("\n📋 BATCH 1: Standard Fact-Checking")
    for i, topic in enumerate(generator.paper_topics[:3], 1):
        print(f"\n[{i}/3] ", end="")
        paper = await generator.generate_paper(topic, use_enhanced_fact_check=False)
        if paper:
            generator.save_paper_to_file(paper)
        await asyncio.sleep(1)  # Brief pause between generations
    
    # Generate second batch with enhanced fact-checking
    print("\n📋 BATCH 2: Enhanced Web-Search Fact-Checking")
    for i, topic in enumerate(generator.paper_topics[3:], 1):
        print(f"\n[{i}/3] ", end="")
        paper = await generator.generate_paper(topic, use_enhanced_fact_check=True)
        if paper:
            generator.save_paper_to_file(paper)
        await asyncio.sleep(1)  # Brief pause between generations
    
    # Generate summary report
    print("\n📊 Generating summary report...")
    generator.generate_summary_report()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 PAPER GENERATION COMPLETE!")
    print(f"   • Total Papers Generated: {len(generator.papers_generated)}")
    print(f"   • Academic Fields Covered: {len(set(p['field'] for p in generator.papers_generated))}")
    print(f"   • Enhanced Fact-Checked: {sum(1 for p in generator.papers_generated if p['enhanced_fact_check'])}")
    print(f"   • Average Quality Score: {sum(p['quality_score'] for p in generator.papers_generated) / len(generator.papers_generated):.1f}/10")
    print("\n📁 Files saved in: papers_for_grading/")
    print("📋 Open grading_summary_report.html for overview")
    print("\n🏆 Ready for professor grading!")

if __name__ == '__main__':
    asyncio.run(main()) 