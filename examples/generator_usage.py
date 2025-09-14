#!/usr/bin/env python3
"""
Example usage of the GeneratorWorker for content generation orchestration.

This script demonstrates how to use the GeneratorWorker to coordinate
content generation across multiple content types and sub-generators.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workers.generator import GeneratorWorker, generate_content, ContentType
from core.database import create_database


async def example_basic_generation():
    """Example: Basic content generation with transcript text."""
    print("🎯 Example 1: Basic Content Generation")
    print("-" * 40)
    
    # Sample transcript text
    transcript_text = """
    Welcome to today's podcast where we're discussing the future of artificial intelligence
    and machine learning. In this episode, we'll cover three main topics: the current state
    of AI technology, emerging trends in machine learning, and practical applications for
    businesses. 
    
    First, let's talk about where we are today with AI. Machine learning models have become
    incredibly sophisticated, with large language models showing remarkable capabilities
    in understanding and generating human-like text. Computer vision has advanced to the
    point where we can reliably identify objects, faces, and even emotions from images.
    
    Moving on to emerging trends, we're seeing a shift towards more specialized AI models
    that can perform specific tasks with high accuracy. Edge computing is making it possible
    to run AI inference directly on devices, reducing latency and improving privacy.
    
    Finally, for businesses, AI is no longer just a nice-to-have – it's becoming essential
    for staying competitive. From customer service chatbots to predictive analytics,
    companies are finding numerous ways to integrate AI into their operations.
    """
    
    try:
        # Use the convenience function
        result = await generate_content(
            video_id="example_podcast_001",
            content_types=["summary", "blog_post", "social_media"],
            transcript_text=transcript_text,
            generation_options={
                "summary": {
                    "variants": ["short", "medium", "detailed"]
                },
                "blog_post": {
                    "variants": ["1000_words"],
                    "custom_options": {
                        "include_intro": True,
                        "target_audience": "business_professionals"
                    }
                },
                "social_media": {
                    "variants": ["twitter", "linkedin"],
                    "custom_options": {
                        "include_hashtags": True,
                        "tone": "professional"
                    }
                }
            }
        )
        
        print(f"✅ Generation Status: {result['status']}")
        print(f"📊 Generated {result['generated_content']['generation_summary']['total_generated_items']} content items")
        print(f"⏱️  Total execution time: {result['generation_metadata']['execution_time_seconds']:.2f}s")
        
        # Display results by content type
        for content_type, type_data in result['generated_content']['content_by_type'].items():
            print(f"\n📝 {content_type.upper()}:")
            print(f"   Success: {type_data['success']}")
            print(f"   Variants: {len(type_data['generated_content'])}")
            print(f"   Execution time: {type_data.get('execution_time', 0):.2f}s")
            
            for content_item in type_data['generated_content']:
                print(f"   - {content_item['variant']}: {len(content_item['content'])} chars")
        
        if result['generated_content']['generation_summary']['failed_types'] > 0:
            print(f"\n⚠️  {result['generated_content']['generation_summary']['failed_types']} content types failed")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")


async def example_worker_direct_usage():
    """Example: Direct worker usage with custom configuration."""
    print("\n🔧 Example 2: Direct Worker Usage")
    print("-" * 40)
    
    # Initialize worker with custom settings
    worker = GeneratorWorker(
        max_concurrent_generations=3,
        generation_timeout_minutes=10
    )
    
    # Input data for generation
    input_data = {
        "video_id": "tutorial_video_042",
        "content_types": ["summary", "newsletter", "scripts"],
        "transcript_text": "This is a shorter tutorial transcript about Python programming basics...",
        "generation_options": {
            "summary": {"variants": ["medium"]},
            "newsletter": {"variants": ["intro", "main_points"]},
            "scripts": {"variants": ["youtube_shorts", "podcast"]}
        }
    }
    
    try:
        # Run the worker
        result = worker.run(input_data)
        
        if result["status"] == "success":
            data = result["data"]
            print(f"✅ Worker execution successful!")
            print(f"📊 Final status: {data['status']}")
            print(f"🎯 Content types processed: {len(data['generated_content']['content_by_type'])}")
            
            # Show generation metadata
            metadata = data['generation_metadata']
            print(f"📈 Generation Statistics:")
            print(f"   - Total content types: {metadata['total_content_types']}")
            print(f"   - Successful: {metadata['successful_generations']}")
            print(f"   - Failed: {metadata['failed_generations']}")
            print(f"   - Execution time: {metadata['execution_time_seconds']:.2f}s")
            
        else:
            print(f"❌ Worker execution failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Worker execution error: {e}")


async def example_content_type_showcase():
    """Example: Showcase all available content types."""
    print("\n🎨 Example 3: Content Type Showcase")
    print("-" * 40)
    
    print("Available Content Types:")
    for content_type in ContentType:
        print(f"   - {content_type.value}")
    
    # Initialize worker to show configurations
    worker = GeneratorWorker()
    
    print("\nContent Type Configurations:")
    for ct, config in worker.content_type_configs.items():
        print(f"\n📋 {ct.upper()}:")
        print(f"   Available variants: {', '.join(config['variants'])}")
        print(f"   Default variants: {', '.join(config['default_variants'])}")
        print(f"   Estimated duration: {config['estimated_duration']}s")
    
    # Example with all content types
    print("\n🚀 Generating content for all types...")
    
    try:
        result = await generate_content(
            video_id="showcase_demo",
            content_types=[ct.value for ct in ContentType],  # All content types
            transcript_text="This is a comprehensive demo transcript showcasing all content generation capabilities...",
            generation_options={
                ct.value: {"variants": config["default_variants"][:1]}  # Just one variant each
                for ct, config in worker.content_type_configs.items()
            }
        )
        
        print(f"✅ Showcase generation completed: {result['status']}")
        print(f"📊 Generated content for {len(result['generated_content']['content_by_type'])} types")
        
    except Exception as e:
        print(f"❌ Showcase generation failed: {e}")


async def example_error_handling():
    """Example: Error handling scenarios."""
    print("\n🚨 Example 4: Error Handling")
    print("-" * 40)
    
    worker = GeneratorWorker()
    
    # Test various error scenarios
    error_scenarios = [
        {
            "name": "Missing video_id",
            "input": {"content_types": ["summary"]}
        },
        {
            "name": "Empty content_types",
            "input": {"video_id": "test", "content_types": []}
        },
        {
            "name": "Invalid content_type",
            "input": {"video_id": "test", "content_types": ["invalid_type"]}
        },
        {
            "name": "Invalid video_id format",
            "input": {"video_id": "", "content_types": ["summary"]}
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n🔍 Testing: {scenario['name']}")
        result = worker.run(scenario['input'])
        
        if result["status"] == "failed":
            print(f"   ✅ Correctly handled error: {result.get('error', 'Unknown error')}")
            if 'error_details' in result:
                details = result['error_details']
                print(f"   📋 Category: {details.get('category', 'unknown')}")
                print(f"   🔄 Recoverable: {details.get('recoverable', False)}")
        else:
            print(f"   ❌ Unexpected success for invalid input")


async def main():
    """Main example runner."""
    print("🎬 GeneratorWorker Usage Examples")
    print("=" * 50)
    
    try:
        await example_basic_generation()
        await example_worker_direct_usage()
        await example_content_type_showcase()
        await example_error_handling()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())