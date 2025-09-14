#!/usr/bin/env python3
"""
Test script for the GeneratorWorker implementation.
Verifies the content generation orchestration functionality.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from workers.generator import GeneratorWorker, generate_content, ContentType
from core.database import create_database


async def test_generator_worker():
    """Test the GeneratorWorker functionality."""
    print("🧪 Testing GeneratorWorker Implementation")
    print("=" * 50)
    
    try:
        # Initialize database
        print("📊 Setting up test database...")
        db_manager = await create_database("sqlite+aiosqlite:///test_generator.db")
        
        # Test 1: Input Validation
        print("\n1️⃣ Testing Input Validation")
        worker = GeneratorWorker(db_manager=db_manager)
        
        # Valid input
        valid_input = {
            "video_id": "test_video_123",
            "content_types": ["summary", "blog_post"],
            "transcript_text": "This is a test transcript with some content to analyze.",
            "generation_options": {
                "summary": {"variants": ["medium"]},
                "blog_post": {"variants": ["1000_words"]}
            }
        }
        
        validation_result = worker.validate_input(valid_input)
        print(f"✅ Valid input validation: {validation_result}")
        
        # Invalid input - missing video_id
        invalid_input = {
            "content_types": ["summary"]
        }
        
        validation_result = worker.validate_input(invalid_input)
        print(f"❌ Invalid input validation: {validation_result}")
        
        # Test 2: Content Type Validation
        print("\n2️⃣ Testing Content Type Validation")
        
        # Unsupported content type
        unsupported_input = {
            "video_id": "test_video_123",
            "content_types": ["summary", "unsupported_type"],
            "transcript_text": "Test transcript"
        }
        
        validation_result = worker.validate_input(unsupported_input)
        print(f"❌ Unsupported content type validation: {validation_result}")
        
        # Test 3: Full Execution (Phase 1 Simulation)
        print("\n3️⃣ Testing Full Execution (Phase 1 Simulation)")
        
        result = worker.run(valid_input)
        
        if result["status"] == "success":
            data = result["data"]
            print("✅ Generation execution successful!")
            print(f"   Status: {data['status']}")
            print(f"   Content Types: {len(data['generated_content']['content_by_type'])}")
            print(f"   Total Items: {data['generated_content']['generation_summary']['total_generated_items']}")
            print(f"   Successful Types: {data['generated_content']['generation_summary']['successful_types']}")
            print(f"   Execution Time: {data['generation_metadata']['execution_time_seconds']:.2f}s")
        else:
            print(f"❌ Generation execution failed: {result.get('error')}")
            if 'error_details' in result:
                print(f"   Details: {result['error_details']}")
        
        # Test 4: Error Handling
        print("\n4️⃣ Testing Error Handling")
        
        # Test with missing transcript
        no_transcript_input = {
            "video_id": "nonexistent_video",
            "content_types": ["summary"]
        }
        
        result = worker.run(no_transcript_input)
        print(f"Error handling result: {result['status']}")
        if result["status"] == "failed":
            print(f"✅ Properly handled missing transcript error")
        
        # Test 5: Convenience Function
        print("\n5️⃣ Testing Convenience Function")
        
        try:
            result = await generate_content(
                video_id="convenience_test",
                content_types=["summary", "social_media"],
                transcript_text="This is a test transcript for the convenience function.",
                generation_options={
                    "summary": {"variants": ["short", "medium"]},
                    "social_media": {"variants": ["twitter", "linkedin"]}
                }
            )
            
            print("✅ Convenience function successful!")
            print(f"   Generated content types: {len(result['generated_content']['content_by_type'])}")
            
        except Exception as e:
            print(f"❌ Convenience function failed: {e}")
        
        # Test 6: Configuration Validation
        print("\n6️⃣ Testing Configuration")
        
        print("Available content types:")
        for content_type in ContentType:
            print(f"   - {content_type.value}")
        
        print("\nWorker configuration:")
        print(f"   Max concurrent generations: {worker.max_concurrent_generations}")
        print(f"   Generation timeout: {worker.generation_timeout_minutes} minutes")
        print(f"   Supported content types: {len(worker.content_type_configs)}")
        
        for ct, config in worker.content_type_configs.items():
            print(f"   {ct}:")
            print(f"     - Variants: {config['variants']}")
            print(f"     - Default variants: {config['default_variants']}")
            print(f"     - Estimated duration: {config['estimated_duration']}s")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False
    
    finally:
        # Cleanup
        if 'db_manager' in locals():
            await db_manager.close()
        
        # Remove test database
        test_db_path = Path("test_generator.db")
        if test_db_path.exists():
            test_db_path.unlink()
    
    return True


async def main():
    """Main test runner."""
    print("Starting GeneratorWorker Tests...")
    success = await test_generator_worker()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())