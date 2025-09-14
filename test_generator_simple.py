#!/usr/bin/env python3
"""
Simple test script for the GeneratorWorker implementation.
Tests the core functionality without complex dependencies.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Direct imports to avoid settings issues
from workers.generator import GeneratorWorker, ContentType, SummaryVariant, SocialPlatform


async def test_generator_worker_basic():
    """Test basic GeneratorWorker functionality without database."""
    print("🧪 Testing GeneratorWorker Basic Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Worker Initialization
        print("1️⃣ Testing Worker Initialization")
        worker = GeneratorWorker()
        print(f"✅ Worker initialized: {worker.name}")
        print(f"   Max concurrent: {worker.max_concurrent_generations}")
        print(f"   Timeout: {worker.generation_timeout_minutes} minutes")
        
        # Test 2: Input Validation
        print("\n2️⃣ Testing Input Validation")
        
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
        
        # Invalid inputs
        invalid_inputs = [
            {},  # Missing required fields
            {"video_id": "test"},  # Missing content_types
            {"video_id": "test", "content_types": []},  # Empty content_types
            {"video_id": "test", "content_types": ["invalid_type"]},  # Unsupported type
            {"video_id": "", "content_types": ["summary"]},  # Invalid video_id
        ]
        
        for i, invalid_input in enumerate(invalid_inputs, 1):
            result = worker.validate_input(invalid_input)
            print(f"❌ Invalid input {i}: {result}")
        
        # Test 3: Content Type Configurations
        print("\n3️⃣ Testing Content Type Configurations")
        
        print("Available content types:")
        for content_type in ContentType:
            print(f"   - {content_type.value}")
        
        print("\nContent type configurations:")
        for ct, config in worker.content_type_configs.items():
            print(f"   {ct}:")
            print(f"     - Variants: {config['variants']}")
            print(f"     - Default variants: {config['default_variants']}")
            print(f"     - Estimated duration: {config['estimated_duration']}s")
        
        # Test 4: Generation Plan Creation
        print("\n4️⃣ Testing Generation Plan Creation")
        
        content_types = ["summary", "social_media"]
        generation_options = {
            "summary": {"variants": ["short", "medium"]},
            "social_media": {"variants": ["twitter", "linkedin"]}
        }
        
        plan = await worker._create_generation_plan(content_types, generation_options)
        print(f"✅ Generation plan created for {len(plan)} content types")
        
        for ct, ct_plan in plan.items():
            print(f"   {ct}: {len(ct_plan['variants'])} variants")
            print(f"     - Variants: {ct_plan['variants']}")
            print(f"     - Priority: {ct_plan['priority']}")
            print(f"     - Estimated duration: {ct_plan['estimated_duration']}s")
        
        # Test 5: Enum Values
        print("\n5️⃣ Testing Enum Values")
        
        print("Summary variants:")
        for variant in SummaryVariant:
            print(f"   - {variant.value}")
        
        print("Social platforms:")
        for platform in SocialPlatform:
            print(f"   - {platform.value}")
        
        # Test 6: Error Handling
        print("\n6️⃣ Testing Error Handling")
        
        test_error = ValueError("Test error message")
        error_result = worker.handle_error(test_error)
        
        print(f"✅ Error handling result:")
        print(f"   - Error type: {error_result['error_type']}")
        print(f"   - Category: {error_result['category']}")
        print(f"   - Recoverable: {error_result['recoverable']}")
        print(f"   - Suggested action: {error_result['suggested_action']}")
        
        # Test different error types
        test_errors = [
            ValueError("Invalid input"),
            asyncio.TimeoutError("Operation timed out"),
            Exception("Transcript not found"),
            RuntimeError("Unknown error")
        ]
        
        for error in test_errors:
            result = worker.handle_error(error)
            print(f"   {type(error).__name__}: {result['category']}")
        
        print("\n🎉 Basic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False


async def test_generation_plan_edge_cases():
    """Test edge cases in generation plan creation."""
    print("\n🔧 Testing Generation Plan Edge Cases")
    print("-" * 40)
    
    try:
        worker = GeneratorWorker()
        
        # Test with invalid variants
        content_types = ["summary", "social_media"]
        generation_options = {
            "summary": {"variants": ["invalid_variant", "medium"]},  # Mix of valid/invalid
            "social_media": {"variants": ["invalid_platform"]},     # All invalid
            "blog_post": {}  # No variants specified
        }
        
        plan = await worker._create_generation_plan(content_types, generation_options)
        print(f"✅ Handled invalid variants gracefully")
        
        for ct, ct_plan in plan.items():
            print(f"   {ct}: {ct_plan['variants']} (fallback to defaults if needed)")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False


async def main():
    """Main test runner."""
    print("Starting GeneratorWorker Basic Tests...")
    
    success1 = await test_generator_worker_basic()
    success2 = await test_generation_plan_edge_cases()
    
    if success1 and success2:
        print("\n✅ All basic tests passed!")
        print("\nNote: Full integration tests with database and queue")
        print("require proper environment setup and dependencies.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())