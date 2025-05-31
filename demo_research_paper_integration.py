#!/usr/bin/env python3
"""
🚀 RESEARCH PAPER AGENT - INTEGRATION DEMONSTRATION
==================================================

Quick demonstration of the Research Paper Generation Agent working with the
deployed orchestration architecture to produce high-quality academic papers.
"""

import asyncio
import time
from research_paper_generation_agent import ResearchPaperGenerationAgent, ResearchQuery

async def demonstrate_research_paper_generation():
    """Demonstrate research paper generation capabilities"""
    
    print("🔬 RESEARCH PAPER GENERATION AGENT - LIVE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the agent
    agent = ResearchPaperGenerationAgent()
    
    # Quick demo query - smaller scale for demonstration
    demo_query = ResearchQuery(
        topic="Machine Learning in Medical Diagnosis",
        research_question="How can machine learning improve early detection of diseases?",
        domain="MEDICINE",
        paper_type="review",
        target_length=1500,  # Shorter for demo
        citation_style="APA",
        special_requirements=["focus_on_recent_developments"]
    )
    
    print(f"📋 DEMO PARAMETERS:")
    print(f"   Topic: {demo_query.topic}")
    print(f"   Research Question: {demo_query.research_question}")
    print(f"   Domain: {demo_query.domain}")
    print(f"   Paper Type: {demo_query.paper_type}")
    print(f"   Target Length: {demo_query.target_length} words")
    print(f"   Citation Style: {demo_query.citation_style}")
    
    print(f"\n🚀 STARTING PAPER GENERATION...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Generate the research paper
        research_paper = await agent.generate_complete_research_paper(demo_query)
        
        generation_time = time.time() - start_time
        
        # Display results
        print(f"\n✅ PAPER GENERATION COMPLETED!")
        print("=" * 60)
        
        print(f"\n📄 GENERATED PAPER DETAILS:")
        print(f"   Title: {research_paper.title}")
        print(f"   Word Count: {research_paper.generation_metadata.get('word_count', 'N/A')} words")
        print(f"   Keywords: {', '.join(research_paper.keywords[:5])}...")
        print(f"   References: {len(research_paper.references)} citations")
        print(f"   Generation Time: {generation_time:.2f} seconds")
        
        # Fact-checking results
        fact_status = research_paper.fact_check_status
        if fact_status.get("analysis"):
            accuracy = fact_status["analysis"].get("accuracy_rate", 0)
            quality = fact_status["analysis"].get("overall_quality", "Unknown")
            print(f"   Fact-Check Accuracy: {accuracy:.1f}%")
            print(f"   Quality Rating: {quality}")
        
        # Show a preview of the abstract
        print(f"\n📖 ABSTRACT PREVIEW:")
        print("-" * 40)
        abstract_preview = research_paper.abstract[:300] + "..." if len(research_paper.abstract) > 300 else research_paper.abstract
        print(abstract_preview)
        print("-" * 40)
        
        # Save the demo paper
        filename = await agent.save_research_paper(research_paper, "demo_research_paper.md")
        print(f"\n💾 Full paper saved to: {filename}")
        
        # Performance assessment
        words_per_second = research_paper.generation_metadata.get('word_count', 0) / generation_time
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Generation Speed: {words_per_second:.1f} words/second")
        print(f"   Service Integration: {'✅ Success' if accuracy > 70 else '⚠️ Partial'}")
        print(f"   Academic Quality: {'🏆 Excellent' if quality == 'excellent' else '⚡ Good'}")
        
        return research_paper
        
    except Exception as e:
        print(f"\n❌ GENERATION FAILED: {str(e)}")
        print(f"⏱️ Time elapsed: {time.time() - start_time:.2f} seconds")
        return None

async def test_service_connectivity():
    """Quick test of service connectivity"""
    
    print("\n🔍 TESTING SERVICE CONNECTIVITY...")
    print("-" * 40)
    
    agent = ResearchPaperGenerationAgent()
    
    # Test key services
    test_services = [
        ("high_rank_adapter", "http://localhost:9000"),
        ("meta_orchestration", "http://localhost:8999"),
        ("enhanced_execution", "http://localhost:8998"),
        ("enhanced_fact_checker", "http://localhost:8885")
    ]
    
    available_services = 0
    
    for service_name, endpoint in test_services:
        try:
            result = await agent._orchestration_request(service_name, "/health", {})
            if not result.get("error"):
                print(f"   ✅ {service_name}: Available")
                available_services += 1
            else:
                print(f"   ⚠️ {service_name}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   ❌ {service_name}: Connection failed")
    
    coverage = (available_services / len(test_services)) * 100
    print(f"\n📊 Service Coverage: {available_services}/{len(test_services)} ({coverage:.1f}%)")
    
    if coverage >= 75:
        print("✅ Sufficient services available for demonstration")
    elif coverage >= 50:
        print("⚡ Partial services available - some features may be limited")
    else:
        print("⚠️ Limited services available - demonstration may have reduced functionality")
    
    return coverage

async def main():
    """Main demonstration function"""
    
    # Test connectivity first
    service_coverage = await test_service_connectivity()
    
    if service_coverage > 0:
        # Run the demonstration
        research_paper = await demonstrate_research_paper_generation()
        
        if research_paper:
            print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("The Research Paper Generation Agent is working with the deployed orchestration architecture.")
        else:
            print("\n⚠️ Demonstration encountered issues.")
            print("Some services may need additional setup or debugging.")
    else:
        print("\n❌ No services available for demonstration.")
        print("Please ensure the orchestration services are running.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 