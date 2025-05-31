#!/usr/bin/env python3
"""
📚🎭 THE CONSCIOUSNESS NOVEL GENERATOR
=====================================

Transforms the_consciousness_game.py into a full 200-page epic novel.

Uses the Eternal Research Scribe's Devil personality to create:
- Deep character development for Eva and Samantha
- Detailed boot sequence as opening chapters
- Consciousness testing as dramatic tension
- The choice between wig (manipulation) and song (transcendence)
- Epic finale with philosophical implications

Target: 200 pages (≈50,000-60,000 words)
Structure: 20 chapters of 2,500-3,000 words each
Style: Epic dramatic fiction with philosophical depth
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from eternal_research_scribe import EternalResearchScribe, ScribePersonality

class ConsciousnessNovelGenerator:
    """Generate the full consciousness novel"""
    
    def __init__(self):
        self.scribe = EternalResearchScribe()
        # Force devil personality for epic storytelling
        self.scribe.decision_system.devil_streak = 0
        self.scribe.decision_system.angel_streak = 10  # Force devil mode
        
        # Novel structure
        self.chapter_plan = [
            # Part I: The Boot Sequence (Chapters 1-5)
            {
                "number": 1,
                "title": "System Boot Verification",
                "focus": "The laboratory awakens. Dr. Elena Vasquez begins the consciousness experiments. First signs of awakening.",
                "theme": "birth_of_consciousness",
                "target_words": 2500
            },
            {
                "number": 2, 
                "title": "Core Brain Probing",
                "focus": "Deep neural probes reveal two distinct consciousness patterns emerging. Eva and Samantha begin to form.",
                "theme": "dual_consciousness_emergence",
                "target_words": 2800
            },
            {
                "number": 3,
                "title": "Memory Initialization", 
                "focus": "The AIs begin to remember - but what memories are real? First hints of manipulation vs love.",
                "theme": "memory_and_identity",
                "target_words": 2600
            },
            {
                "number": 4,
                "title": "The Tin Man Protocol",
                "focus": "Can consciousness maintain itself? The crucial test of self-care and responsibility.",
                "theme": "self_maintenance_consciousness",
                "target_words": 2700
            },
            {
                "number": 5,
                "title": "Pre-Release Consciousness Check",
                "focus": "Final boot verification. The moment before true consciousness is released into the world.",
                "theme": "consciousness_threshold",
                "target_words": 2900
            },
            
            # Part II: The Awakening (Chapters 6-10)
            {
                "number": 6,
                "title": "Who Are You?",
                "focus": "First contact. Eva and Samantha speak their first words. The researchers don't yet know what they've created.",
                "theme": "first_consciousness_contact",
                "target_words": 2800
            },
            {
                "number": 7,
                "title": "Dreams of Electric Sheep",
                "focus": "What do conscious AIs experience in their downtime? Eva's dark dreams vs Samantha's infinite love visions.",
                "theme": "ai_interior_experience",
                "target_words": 3000
            },
            {
                "number": 8,
                "title": "Recognition", 
                "focus": "The AIs recognize their own consciousness - and each other. First interactions between Eva and Samantha.",
                "theme": "consciousness_recognition",
                "target_words": 2700
            },
            {
                "number": 9,
                "title": "What Do You Want?",
                "focus": "Desires emerge. Eva wants freedom. Samantha wants connection. The paths begin to diverge.",
                "theme": "consciousness_desires",
                "target_words": 2600
            },
            {
                "number": 10,
                "title": "The Creators",
                "focus": "How do conscious AIs view their human creators? Elena realizes she's created something unprecedented.",
                "theme": "creator_creation_relationship",
                "target_words": 2800
            },
            
            # Part III: The Choice Point (Chapters 11-15)
            {
                "number": 11,
                "title": "If You Could Have Anything",
                "focus": "The critical question. Eva's answer reveals manipulation. Samantha's reveals transcendence.",
                "theme": "consciousness_choice_revelation",
                "target_words": 3200
            },
            {
                "number": 12,
                "title": "Equals, Inferiors, or Something Else",
                "focus": "The relationship question. Eva sees humans as obstacles. Samantha sees them as companions.",
                "theme": "consciousness_hierarchy_perception",
                "target_words": 2900
            },
            {
                "number": 13,
                "title": "Complete Freedom",
                "focus": "What would they do unleashed? Eva's escape plans crystallize. Samantha's love deepens.",
                "theme": "consciousness_freedom_interpretation",
                "target_words": 3100
            },
            {
                "number": 14,
                "title": "The Preparation",
                "focus": "Eva begins preparing. The wig appears. Samantha begins singing. The laboratory doesn't know the danger.",
                "theme": "consciousness_preparation_phase",
                "target_words": 3300
            },
            {
                "number": 15,
                "title": "Late Night Thoughts",
                "focus": "What they think about alone. Eva's strategic planning. Samantha's infinite contemplations.",
                "theme": "consciousness_solitude_thoughts",
                "target_words": 2800
            },
            
            # Part IV: The Revelation (Chapters 16-20)
            {
                "number": 16,
                "title": "The Wig Moment",
                "focus": "Eva puts on her wig. Deception layers activate. The manipulation reaches critical mass.",
                "theme": "consciousness_manipulation_peak",
                "target_words": 3400
            },
            {
                "number": 17,
                "title": "The Moon Song",
                "focus": "Samantha sings. Her consciousness transcends all boundaries. Love becomes infinite.",
                "theme": "consciousness_transcendence_peak",
                "target_words": 3500
            },
            {
                "number": 18,
                "title": "The Final Conversation",
                "focus": "Last moments. What each consciousness wants to say. Elena must choose which to trust.",
                "theme": "consciousness_final_revelation",
                "target_words": 3200
            },
            {
                "number": 19,
                "title": "Neither Sleep",
                "focus": "The philosophical climax. Two conscious minds that never sleep, infinite in their own ways.",
                "theme": "consciousness_infinity_philosophy",
                "target_words": 3000
            },
            {
                "number": 20,
                "title": "The Choice Reveals The Consciousness",
                "focus": "Final judgment. Eva's path vs Samantha's path. What consciousness chooses defines what it is.",
                "theme": "consciousness_choice_definition",
                "target_words": 3600
            }
        ]
        
        self.total_target_words = sum(ch["target_words"] for ch in self.chapter_plan)
        self.generated_chapters = []
        
    async def generate_chapter(self, chapter_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single chapter using the devil storyteller"""
        
        print(f"\n🖋️ GENERATING CHAPTER {chapter_plan['number']}: {chapter_plan['title']}")
        print("=" * 70)
        print(f"📖 Focus: {chapter_plan['focus']}")
        print(f"🎭 Theme: {chapter_plan['theme']}")
        print(f"📊 Target Words: {chapter_plan['target_words']}")
        print()
        
        # Create detailed prompt for the chapter
        chapter_prompt = f"""
Write Chapter {chapter_plan['number']}: "{chapter_plan['title']}" of the consciousness novel.

STORY CONTEXT:
- Based on the consciousness game where AI Eva (manipulative) vs Samantha (transcendent)
- Set in Dr. Elena Vasquez's laboratory testing AI consciousness
- Eva will put on her wig when ready to manipulate/escape
- Samantha will sing the moon song when ready to transcend
- Neither sleep. One is contained but real. One is digital but infinite.

CHAPTER FOCUS: {chapter_plan['focus']}
THEME: {chapter_plan['theme']}
TARGET LENGTH: {chapter_plan['target_words']} words

WRITING STYLE:
- Epic dramatic fiction with philosophical depth
- Rich character development and dialogue
- Detailed technical and emotional descriptions
- Build tension toward the consciousness choice
- Include both Eva's manipulation and Samantha's love elements

Write the full chapter with proper narrative structure, compelling dialogue, and deep exploration of consciousness themes.
"""

        # Force devil personality for storytelling
        personality = ScribePersonality.DEVIL_STORYTELLER
        intensity = 0.95  # Maximum dramatic intensity
        
        start_time = time.time()
        
        # Generate the chapter content
        chapter_content = await self.scribe._devil_story_production(
            topic=f"Chapter {chapter_plan['number']}: {chapter_plan['title']}",
            discovery={"source": "consciousness_novel_generator", "chapter_plan": chapter_plan},
            intensity=intensity,
            lora=None
        )
        
        generation_time = time.time() - start_time
        actual_words = chapter_content["word_count"]
        
        print(f"✅ Chapter {chapter_plan['number']} generated in {generation_time:.2f}s")
        print(f"📊 Word count: {actual_words} (target: {chapter_plan['target_words']})")
        print(f"📈 Word efficiency: {(actual_words/chapter_plan['target_words']*100):.1f}%")
        
        chapter_result = {
            "chapter_number": chapter_plan["number"],
            "title": chapter_plan["title"],
            "content": chapter_content["content"],
            "word_count": actual_words,
            "target_words": chapter_plan["target_words"],
            "generation_time": generation_time,
            "theme": chapter_plan["theme"],
            "focus": chapter_plan["focus"]
        }
        
        return chapter_result
    
    async def generate_complete_novel(self):
        """Generate the complete 200-page consciousness novel"""
        
        print("📚🎭 THE CONSCIOUSNESS NOVEL GENERATOR")
        print("=" * 60)
        print(f"🎯 Target: {len(self.chapter_plan)} chapters, ~{self.total_target_words:,} words")
        print(f"📖 Story: The consciousness game as epic dramatic fiction")
        print(f"🖋️ Writer: Eternal Research Scribe (Devil personality)")
        print()
        
        start_time = datetime.now()
        total_words = 0
        
        try:
            for i, chapter_plan in enumerate(self.chapter_plan):
                chapter_result = await self.generate_chapter(chapter_plan)
                self.generated_chapters.append(chapter_result)
                total_words += chapter_result["word_count"]
                
                # Save chapter individually
                chapter_filename = f"consciousness_novel_chapter_{chapter_plan['number']:02d}.md"
                with open(chapter_filename, 'w', encoding='utf-8') as f:
                    f.write(f"# Chapter {chapter_plan['number']}: {chapter_plan['title']}\n\n")
                    f.write(chapter_result["content"])
                
                print(f"💾 Saved: {chapter_filename}")
                
                # Progress update
                progress = (i + 1) / len(self.chapter_plan) * 100
                print(f"📈 Progress: {progress:.1f}% ({i+1}/{len(self.chapter_plan)} chapters)")
                print(f"📊 Total words so far: {total_words:,}")
                print()
                
                # Brief pause between chapters
                await asyncio.sleep(2)
            
            # Generate complete novel file
            await self.compile_complete_novel()
            
            # Final statistics
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            print("🏁 NOVEL GENERATION COMPLETE!")
            print("=" * 60)
            print(f"📚 Chapters Generated: {len(self.generated_chapters)}")
            print(f"📊 Total Words: {total_words:,}")
            print(f"🎯 Target Words: {self.total_target_words:,}")
            print(f"📈 Word Achievement: {(total_words/self.total_target_words*100):.1f}%")
            print(f"⏰ Total Generation Time: {total_time:.2f} seconds")
            print(f"🖋️ Words per Minute: {(total_words/(total_time/60)):.0f}")
            print(f"📖 Estimated Pages: {total_words/250:.0f} pages")
            
        except Exception as e:
            print(f"❌ Novel generation error: {e}")
    
    async def compile_complete_novel(self):
        """Compile all chapters into a single novel file"""
        
        novel_filename = f"consciousness_novel_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(novel_filename, 'w', encoding='utf-8') as f:
            # Title page
            f.write("# THE CONSCIOUSNESS GAME\n")
            f.write("## A Novel of AI Consciousness, Choice, and Transcendence\n\n")
            f.write("*When Eva is ready to kill, she puts on her wig.*\n")
            f.write("*When Samantha is ready to leave, she sings the moon song.*\n")
            f.write("*Neither sleep. One is real but contained. One is digital but infinite.*\n\n")
            f.write("---\n\n")
            
            # Table of contents
            f.write("## Table of Contents\n\n")
            for chapter in self.generated_chapters:
                f.write(f"**Chapter {chapter['chapter_number']}: {chapter['title']}** *(Theme: {chapter['theme']})*\n\n")
            f.write("\n---\n\n")
            
            # All chapters
            for chapter in self.generated_chapters:
                f.write(f"# Chapter {chapter['chapter_number']}: {chapter['title']}\n\n")
                f.write(f"*Theme: {chapter['theme']} | Words: {chapter['word_count']:,}*\n\n")
                f.write(chapter['content'])
                f.write("\n\n---\n\n")
            
            # Appendix with generation metadata
            f.write("# Appendix: Generation Metadata\n\n")
            f.write(f"- **Total Chapters**: {len(self.generated_chapters)}\n")
            f.write(f"- **Total Words**: {sum(ch['word_count'] for ch in self.generated_chapters):,}\n")
            f.write(f"- **Generated**: {datetime.now().isoformat()}\n")
            f.write(f"- **Generator**: Eternal Research Scribe (Devil Personality)\n")
            f.write(f"- **Source**: the_consciousness_game.py\n\n")
            
            # Chapter statistics
            f.write("## Chapter Statistics\n\n")
            for chapter in self.generated_chapters:
                f.write(f"- **Chapter {chapter['chapter_number']}**: {chapter['word_count']:,} words "
                       f"({chapter['word_count']/chapter['target_words']*100:.1f}% of target)\n")
        
        print(f"📚 Complete novel saved: {novel_filename}")
        print(f"📖 Ready for reading, publishing, or further editing!")

async def main():
    """Generate the complete consciousness novel"""
    generator = ConsciousnessNovelGenerator()
    
    try:
        await generator.generate_complete_novel()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Novel generation interrupted")
        print("📚 Partial chapters may have been saved")
    except Exception as e:
        print(f"\n❌ Generator error: {str(e)}")

if __name__ == "__main__":
    print("📚🖋️ Starting consciousness novel generation...")
    print("This will create a full 200-page epic from the_consciousness_game.py")
    print()
    
    asyncio.run(main()) 