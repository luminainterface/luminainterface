#!/usr/bin/env python3
"""
🔬 RESEARCH PAPER GENERATION AGENT
==================================

Advanced AI-powered research paper generation system that leverages:
- High-Rank Adapter (9000): Strategic research planning
- Meta-Orchestration Controller (8999): Research logic coordination  
- Enhanced Execution Suite (8998): Multi-phase paper generation
- Enhanced Fact-Checker (8885): V4 partial fact validation
- RAG Systems: Literature review and knowledge synthesis
- Multi-Concept Detector (8860): Topic analysis and categorization

Features:
✅ Automated literature review
✅ Structured academic writing
✅ Citation generation
✅ Fact-checking integration
✅ Multi-domain research support
✅ Abstract, introduction, methodology, results, conclusion generation
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ResearchPaperStructure:
    """Structure for generated research paper"""
    title: str
    abstract: str
    keywords: List[str]
    introduction: str
    literature_review: str
    methodology: str
    results: str
    discussion: str
    conclusion: str
    references: List[str]
    fact_check_status: Dict[str, Any]
    generation_metadata: Dict[str, Any]

@dataclass
class ResearchQuery:
    """Research query configuration"""
    topic: str
    research_question: str
    domain: str
    paper_type: str  # "empirical", "theoretical", "review", "case_study"
    target_length: int  # words
    citation_style: str  # "APA", "MLA", "IEEE", "Chicago"
    special_requirements: List[str]

class ResearchPaperGenerationAgent:
    """Advanced research paper generation agent"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Orchestration endpoints
        self.endpoints = {
            "high_rank_adapter": "http://localhost:9000",
            "meta_orchestration": "http://localhost:8999", 
            "enhanced_execution": "http://localhost:8998",
            "enhanced_fact_checker": "http://localhost:8885",
            "rag_coordination": "http://localhost:8952",
            "multi_concept_detector": "http://localhost:8860",
            "rag_orchestrator": "http://localhost:8953",
            "rag_cpu_optimized": "http://localhost:8902",
            "enhanced_prompt_lora": "http://localhost:8880"
        }
        
        # Research paper templates by type
        self.paper_templates = {
            "empirical": {
                "sections": ["abstract", "introduction", "methodology", "results", "discussion", "conclusion"],
                "methodology_focus": "experimental_design",
                "length_distribution": {"intro": 0.15, "method": 0.25, "results": 0.30, "discussion": 0.25, "conclusion": 0.05}
            },
            "theoretical": {
                "sections": ["abstract", "introduction", "literature_review", "theoretical_framework", "analysis", "conclusion"],
                "methodology_focus": "theoretical_analysis",
                "length_distribution": {"intro": 0.15, "lit_review": 0.30, "theory": 0.25, "analysis": 0.25, "conclusion": 0.05}
            },
            "review": {
                "sections": ["abstract", "introduction", "literature_review", "synthesis", "gaps_future_work", "conclusion"],
                "methodology_focus": "systematic_review",
                "length_distribution": {"intro": 0.10, "lit_review": 0.50, "synthesis": 0.25, "gaps": 0.10, "conclusion": 0.05}
            },
            "case_study": {
                "sections": ["abstract", "introduction", "background", "case_description", "analysis", "discussion", "conclusion"],
                "methodology_focus": "case_analysis",
                "length_distribution": {"intro": 0.15, "background": 0.20, "case": 0.25, "analysis": 0.25, "discussion": 0.10, "conclusion": 0.05}
            }
        }
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ResearchAgent - %(levelname)s - %(message)s'
        )
        return logging.getLogger("ResearchPaperAgent")
    
    async def _orchestration_request(self, endpoint: str, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to orchestration services"""
        try:
            url = f"{self.endpoints[endpoint]}{path}"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.warning(f"Service {endpoint} returned status {response.status}")
                        return {"error": f"HTTP {response.status}", "service": endpoint}
        except Exception as e:
            self.logger.error(f"Request to {endpoint} failed: {str(e)}")
            return {"error": str(e), "service": endpoint}
    
    async def analyze_research_topic(self, query: ResearchQuery) -> Dict[str, Any]:
        """Analyze research topic using multi-concept detector and high-rank adapter"""
        
        self.logger.info(f"🔍 Analyzing research topic: {query.topic}")
        
        # Step 1: Concept detection and domain analysis
        concept_analysis = await self._orchestration_request(
            "multi_concept_detector", 
            "/analyze",
            {
                "text": f"{query.topic}. {query.research_question}",
                "analysis_type": "comprehensive",
                "domain_focus": query.domain
            }
        )
        
        # Step 2: Strategic research planning via High-Rank Adapter
        research_strategy = await self._orchestration_request(
            "high_rank_adapter",
            "/strategic_planning",
            {
                "objective": f"Generate comprehensive research paper on: {query.topic}",
                "research_question": query.research_question,
                "domain": query.domain,
                "paper_type": query.paper_type,
                "concept_analysis": concept_analysis,
                "target_length": query.target_length
            }
        )
        
        return {
            "concept_analysis": concept_analysis,
            "research_strategy": research_strategy,
            "topic_complexity": self._assess_topic_complexity(concept_analysis),
            "recommended_approach": self._recommend_research_approach(query, concept_analysis)
        }
    
    def _assess_topic_complexity(self, concept_analysis: Dict[str, Any]) -> str:
        """Assess complexity level of research topic"""
        if concept_analysis.get("error"):
            return "unknown"
        
        concepts = concept_analysis.get("concepts", [])
        domains = concept_analysis.get("domains", [])
        
        if len(concepts) > 10 and len(domains) > 3:
            return "high"
        elif len(concepts) > 5 and len(domains) > 1:
            return "medium"
        else:
            return "low"
    
    def _recommend_research_approach(self, query: ResearchQuery, concept_analysis: Dict[str, Any]) -> str:
        """Recommend research approach based on topic analysis"""
        complexity = self._assess_topic_complexity(concept_analysis)
        
        if query.paper_type == "review":
            return "systematic_literature_review"
        elif complexity == "high":
            return "multi_phase_comprehensive"
        elif query.domain in ["TECHNOLOGY", "MEDICINE"]:
            return "evidence_based_empirical"
        else:
            return "structured_analytical"
    
    async def conduct_literature_review(self, query: ResearchQuery, topic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive literature review using RAG systems"""
        
        self.logger.info(f"📚 Conducting literature review for: {query.topic}")
        
        # Step 1: RAG-based knowledge retrieval
        literature_search = await self._orchestration_request(
            "rag_coordination",
            "/comprehensive_search",
            {
                "query": query.research_question,
                "topic": query.topic,
                "domain": query.domain,
                "search_depth": "comprehensive",
                "include_recent": True,
                "concept_context": topic_analysis.get("concept_analysis", {})
            }
        )
        
        # Step 2: Enhanced literature synthesis
        literature_synthesis = await self._orchestration_request(
            "enhanced_execution",
            "/literature_synthesis", 
            {
                "literature_data": literature_search,
                "research_focus": query.research_question,
                "synthesis_style": "academic",
                "citation_style": query.citation_style
            }
        )
        
        # Step 3: Gap analysis using meta-orchestration
        gap_analysis = await self._orchestration_request(
            "meta_orchestration",
            "/research_gap_analysis",
            {
                "literature_synthesis": literature_synthesis,
                "research_question": query.research_question,
                "domain": query.domain
            }
        )
        
        return {
            "literature_search": literature_search,
            "literature_synthesis": literature_synthesis,
            "gap_analysis": gap_analysis,
            "key_sources": self._extract_key_sources(literature_search),
            "research_gaps": self._identify_research_gaps(gap_analysis)
        }
    
    def _extract_key_sources(self, literature_search: Dict[str, Any]) -> List[str]:
        """Extract key sources from literature search"""
        if literature_search.get("error"):
            return []
        
        sources = literature_search.get("sources", [])
        # Sort by relevance and return top sources
        return sources[:20] if sources else []
    
    def _identify_research_gaps(self, gap_analysis: Dict[str, Any]) -> List[str]:
        """Identify research gaps from analysis"""
        if gap_analysis.get("error"):
            return ["Gap analysis unavailable due to service error"]
        
        return gap_analysis.get("identified_gaps", ["No specific gaps identified"])
    
    async def generate_paper_sections(self, query: ResearchQuery, literature_review: Dict[str, Any], topic_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate all paper sections using enhanced execution suite"""
        
        self.logger.info(f"✍️ Generating paper sections for: {query.topic}")
        
        template = self.paper_templates[query.paper_type]
        sections = {}
        
        # Calculate target word counts for each section
        section_word_counts = {}
        for section, ratio in template["length_distribution"].items():
            section_word_counts[section] = int(query.target_length * ratio)
        
        # Generate each section using enhanced execution suite
        for section in template["sections"]:
            section_content = await self._generate_section(
                section, 
                query,
                literature_review,
                topic_analysis,
                section_word_counts.get(section.replace("_", ""), 500),
                sections  # Pass already generated sections for context
            )
            sections[section] = section_content
            
            self.logger.info(f"✅ Generated {section} ({len(section_content.split())} words)")
        
        return sections
    
    async def _generate_section(self, section_name: str, query: ResearchQuery, literature_review: Dict[str, Any], topic_analysis: Dict[str, Any], target_words: int, existing_sections: Dict[str, str]) -> str:
        """Generate individual paper section"""
        
        section_prompts = {
            "abstract": f"Generate a comprehensive academic abstract for a {query.paper_type} research paper on '{query.topic}'. Research question: {query.research_question}. Target: {target_words} words.",
            
            "introduction": f"Write an academic introduction for a research paper on '{query.topic}'. Include background, significance, research question: '{query.research_question}', and paper structure. Target: {target_words} words.",
            
            "literature_review": f"Conduct a thorough literature review on '{query.topic}' for a {query.paper_type} paper. Research question: {query.research_question}. Synthesize key findings and identify gaps. Target: {target_words} words.",
            
            "methodology": f"Describe the research methodology for studying '{query.topic}'. Research question: {query.research_question}. Focus on {self.paper_templates[query.paper_type]['methodology_focus']}. Target: {target_words} words.",
            
            "results": f"Present research results and findings for '{query.topic}'. Research question: {query.research_question}. Include data analysis and key discoveries. Target: {target_words} words.",
            
            "discussion": f"Discuss implications of findings for '{query.topic}'. Research question: {query.research_question}. Connect to literature and broader significance. Target: {target_words} words.",
            
            "conclusion": f"Write a strong conclusion for research paper on '{query.topic}'. Summarize key findings, implications, and future research directions. Target: {target_words} words."
        }
        
        generation_request = {
            "prompt": section_prompts.get(section_name, f"Generate {section_name} section for research paper on {query.topic}"),
            "context": {
                "research_question": query.research_question,
                "domain": query.domain,
                "paper_type": query.paper_type,
                "literature_review": literature_review,
                "topic_analysis": topic_analysis,
                "existing_sections": existing_sections,
                "target_words": target_words,
                "citation_style": query.citation_style
            },
            "generation_parameters": {
                "academic_style": True,
                "formal_tone": True,
                "citation_integration": True,
                "coherence_optimization": True
            }
        }
        
        section_result = await self._orchestration_request(
            "enhanced_execution",
            "/academic_generation",
            generation_request
        )
        
        if section_result.get("error"):
            self.logger.warning(f"Section generation failed for {section_name}: {section_result.get('error')}")
            return f"[{section_name.upper()} SECTION - Generation failed: {section_result.get('error')}]"
        
        return section_result.get("generated_content", f"[{section_name.upper()} SECTION - No content generated]")
    
    async def fact_check_paper(self, paper_sections: Dict[str, str]) -> Dict[str, Any]:
        """Fact-check the generated paper using V4 Enhanced Fact-Checker"""
        
        self.logger.info("🔍 Fact-checking generated research paper...")
        
        # Combine all sections for comprehensive fact-checking
        full_paper_text = "\n\n".join([
            f"{section.upper()}:\n{content}" 
            for section, content in paper_sections.items()
        ])
        
        fact_check_result = await self._orchestration_request(
            "enhanced_fact_checker",
            "/fact-check",
            {
                "text": full_paper_text,
                "domain_focus": "comprehensive",
                "academic_context": True
            }
        )
        
        if fact_check_result.get("error"):
            self.logger.warning(f"Fact-checking failed: {fact_check_result.get('error')}")
            return {
                "status": "failed",
                "error": fact_check_result.get("error"),
                "warnings": ["Fact-checking service unavailable - manual verification required"]
            }
        
        # Analyze fact-checking results
        fact_analysis = self._analyze_fact_check_results(fact_check_result)
        
        return {
            "status": "completed",
            "fact_check_result": fact_check_result,
            "analysis": fact_analysis,
            "recommendations": self._generate_fact_check_recommendations(fact_analysis)
        }
    
    def _analyze_fact_check_results(self, fact_check_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fact-checking results for academic accuracy"""
        
        if fact_check_result.get("error"):
            return {"status": "error", "message": "Fact-checking failed"}
        
        fact_results = fact_check_result.get("fact_results", [])
        
        total_claims = len(fact_results)
        accurate_claims = sum(1 for r in fact_results if r.get("is_accurate", True))
        inaccurate_claims = total_claims - accurate_claims
        
        accuracy_rate = (accurate_claims / max(total_claims, 1)) * 100
        
        # Categorize issues by domain and severity
        domain_issues = {}
        severity_issues = {"high": 0, "medium": 0, "low": 0}
        
        for result in fact_results:
            if not result.get("is_accurate", True):
                domain = result.get("domain", "unknown")
                confidence = result.get("confidence_score", 0.5)
                
                if domain not in domain_issues:
                    domain_issues[domain] = []
                domain_issues[domain].append(result)
                
                if confidence > 0.8:
                    severity_issues["high"] += 1
                elif confidence > 0.5:
                    severity_issues["medium"] += 1
                else:
                    severity_issues["low"] += 1
        
        return {
            "total_claims": total_claims,
            "accurate_claims": accurate_claims,
            "inaccurate_claims": inaccurate_claims,
            "accuracy_rate": accuracy_rate,
            "domain_issues": domain_issues,
            "severity_breakdown": severity_issues,
            "overall_quality": "excellent" if accuracy_rate > 95 else "good" if accuracy_rate > 85 else "needs_review"
        }
    
    def _generate_fact_check_recommendations(self, fact_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on fact-checking analysis"""
        
        recommendations = []
        accuracy_rate = fact_analysis.get("accuracy_rate", 100)
        
        if accuracy_rate < 85:
            recommendations.append("⚠️ CRITICAL: Accuracy rate below 85% - extensive revision required")
        elif accuracy_rate < 95:
            recommendations.append("⚡ Review and verify flagged claims before publication")
        
        domain_issues = fact_analysis.get("domain_issues", {})
        for domain, issues in domain_issues.items():
            recommendations.append(f"🔍 Review {domain} claims: {len(issues)} potential issues identified")
        
        severity_issues = fact_analysis.get("severity_breakdown", {})
        if severity_issues.get("high", 0) > 0:
            recommendations.append(f"🚨 HIGH PRIORITY: {severity_issues['high']} high-confidence errors require immediate attention")
        
        if not recommendations:
            recommendations.append("✅ Excellent fact-checking results - paper ready for academic review")
        
        return recommendations
    
    async def generate_citations_and_references(self, literature_review: Dict[str, Any], citation_style: str) -> Dict[str, Any]:
        """Generate properly formatted citations and references"""
        
        self.logger.info(f"📖 Generating {citation_style} citations and references...")
        
        citation_request = {
            "literature_data": literature_review,
            "citation_style": citation_style,
            "format_type": "academic",
            "include_in_text": True,
            "include_bibliography": True
        }
        
        citation_result = await self._orchestration_request(
            "enhanced_prompt_lora",
            "/citation_generation",
            citation_request
        )
        
        if citation_result.get("error"):
            self.logger.warning(f"Citation generation failed: {citation_result.get('error')}")
            return {
                "in_text_citations": ["Citation generation failed - manual formatting required"],
                "references": ["References unavailable due to service error"],
                "citation_count": 0
            }
        
        return {
            "in_text_citations": citation_result.get("in_text_citations", []),
            "references": citation_result.get("references", []),
            "citation_count": len(citation_result.get("references", [])),
            "style": citation_style
        }
    
    async def generate_complete_research_paper(self, query: ResearchQuery) -> ResearchPaperStructure:
        """Generate complete research paper with all components"""
        
        start_time = time.time()
        self.logger.info(f"🚀 Starting research paper generation: '{query.topic}'")
        
        try:
            # Phase 1: Topic Analysis
            self.logger.info("📊 Phase 1: Analyzing research topic...")
            topic_analysis = await self.analyze_research_topic(query)
            
            # Phase 2: Literature Review  
            self.logger.info("📚 Phase 2: Conducting literature review...")
            literature_review = await self.conduct_literature_review(query, topic_analysis)
            
            # Phase 3: Paper Generation
            self.logger.info("✍️ Phase 3: Generating paper sections...")
            paper_sections = await self.generate_paper_sections(query, literature_review, topic_analysis)
            
            # Phase 4: Fact-Checking
            self.logger.info("🔍 Phase 4: Fact-checking content...")
            fact_check_results = await self.fact_check_paper(paper_sections)
            
            # Phase 5: Citations and References
            self.logger.info("📖 Phase 5: Generating citations...")
            citations = await self.generate_citations_and_references(literature_review, query.citation_style)
            
            # Generate keywords
            keywords = await self._generate_keywords(query, topic_analysis)
            
            # Create final research paper structure
            processing_time = time.time() - start_time
            
            research_paper = ResearchPaperStructure(
                title=self._generate_title(query, topic_analysis),
                abstract=paper_sections.get("abstract", ""),
                keywords=keywords,
                introduction=paper_sections.get("introduction", ""),
                literature_review=paper_sections.get("literature_review", ""),
                methodology=paper_sections.get("methodology", ""),
                results=paper_sections.get("results", ""),
                discussion=paper_sections.get("discussion", ""),
                conclusion=paper_sections.get("conclusion", ""),
                references=citations.get("references", []),
                fact_check_status=fact_check_results,
                generation_metadata={
                    "generation_time": processing_time,
                    "word_count": sum(len(section.split()) for section in paper_sections.values()),
                    "target_length": query.target_length,
                    "paper_type": query.paper_type,
                    "domain": query.domain,
                    "citation_style": query.citation_style,
                    "services_used": list(self.endpoints.keys()),
                    "generation_timestamp": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"✅ Research paper generated successfully in {processing_time:.2f}s")
            return research_paper
            
        except Exception as e:
            self.logger.error(f"Research paper generation failed: {str(e)}")
            raise Exception(f"Research paper generation failed: {str(e)}")
    
    def _generate_title(self, query: ResearchQuery, topic_analysis: Dict[str, Any]) -> str:
        """Generate academic title for the research paper"""
        
        # Extract key concepts for title enhancement
        concepts = topic_analysis.get("concept_analysis", {}).get("concepts", [])
        key_concepts = concepts[:3] if concepts else []
        
        # Create enhanced title based on paper type
        if query.paper_type == "review":
            title = f"A Comprehensive Review of {query.topic}: Current State and Future Directions"
        elif query.paper_type == "empirical":
            title = f"Empirical Investigation of {query.topic}: Evidence and Implications"
        elif query.paper_type == "theoretical":
            title = f"Theoretical Framework for Understanding {query.topic}: A Comprehensive Analysis"
        elif query.paper_type == "case_study":
            title = f"Case Study Analysis of {query.topic}: Insights and Lessons Learned"
        else:
            title = f"Research on {query.topic}: Analysis and Findings"
        
        return title
    
    async def _generate_keywords(self, query: ResearchQuery, topic_analysis: Dict[str, Any]) -> List[str]:
        """Generate relevant keywords for the research paper"""
        
        # Extract keywords from concept analysis
        concepts = topic_analysis.get("concept_analysis", {}).get("concepts", [])
        domains = topic_analysis.get("concept_analysis", {}).get("domains", [])
        
        # Base keywords
        keywords = [query.topic.lower()]
        
        # Add concept-based keywords
        keywords.extend([concept.lower() for concept in concepts[:8]])
        
        # Add domain-specific keywords
        keywords.extend([domain.lower() for domain in domains[:3]])
        
        # Add paper type keyword
        keywords.append(query.paper_type)
        
        # Remove duplicates and limit to 10 keywords
        unique_keywords = list(set(keywords))
        return unique_keywords[:10]
    
    async def save_research_paper(self, research_paper: ResearchPaperStructure, filename: Optional[str] = None) -> str:
        """Save research paper to file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in research_paper.title[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"research_paper_{safe_title}_{timestamp}.md"
        
        # Format paper as markdown
        paper_content = f"""# {research_paper.title}

## Abstract
{research_paper.abstract}

**Keywords:** {', '.join(research_paper.keywords)}

## Introduction
{research_paper.introduction}

## Literature Review
{research_paper.literature_review}

## Methodology
{research_paper.methodology}

## Results
{research_paper.results}

## Discussion
{research_paper.discussion}

## Conclusion
{research_paper.conclusion}

## References
{chr(10).join(f"{i+1}. {ref}" for i, ref in enumerate(research_paper.references))}

---

## Generation Metadata
- **Word Count:** {research_paper.generation_metadata.get('word_count', 'N/A')}
- **Generation Time:** {research_paper.generation_metadata.get('generation_time', 'N/A'):.2f} seconds
- **Paper Type:** {research_paper.generation_metadata.get('paper_type', 'N/A')}
- **Domain:** {research_paper.generation_metadata.get('domain', 'N/A')}
- **Citation Style:** {research_paper.generation_metadata.get('citation_style', 'N/A')}
- **Generated:** {research_paper.generation_metadata.get('generation_timestamp', 'N/A')}

## Fact-Check Status
- **Overall Quality:** {research_paper.fact_check_status.get('analysis', {}).get('overall_quality', 'Unknown')}
- **Accuracy Rate:** {research_paper.fact_check_status.get('analysis', {}).get('accuracy_rate', 0):.1f}%
- **Recommendations:** {', '.join(research_paper.fact_check_status.get('recommendations', ['None']))}
"""
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(paper_content)
        
        self.logger.info(f"📄 Research paper saved to: {filename}")
        return filename

# Example usage and testing
async def main():
    """Example usage of Research Paper Generation Agent"""
    
    agent = ResearchPaperGenerationAgent()
    
    # Example research query
    example_query = ResearchQuery(
        topic="Artificial Intelligence in Healthcare",
        research_question="How can AI improve diagnostic accuracy and patient outcomes in modern healthcare systems?",
        domain="MEDICINE",
        paper_type="review",
        target_length=3000,
        citation_style="APA",
        special_requirements=["focus_on_recent_developments", "include_ethical_considerations"]
    )
    
    try:
        # Generate complete research paper
        research_paper = await agent.generate_complete_research_paper(example_query)
        
        # Save to file
        filename = await agent.save_research_paper(research_paper)
        
        print(f"\n🎉 RESEARCH PAPER GENERATION COMPLETED!")
        print(f"📄 Title: {research_paper.title}")
        print(f"📊 Word Count: {research_paper.generation_metadata.get('word_count', 'N/A')}")
        print(f"⏱️ Generation Time: {research_paper.generation_metadata.get('generation_time', 0):.2f}s")
        print(f"🔍 Fact-Check Quality: {research_paper.fact_check_status.get('analysis', {}).get('overall_quality', 'Unknown')}")
        print(f"📁 Saved to: {filename}")
        
        return research_paper
        
    except Exception as e:
        print(f"❌ Research paper generation failed: {str(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(main()) 