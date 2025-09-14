#!/usr/bin/env python3
"""
Test script for QualityWorker functionality.

Tests both transcript and content quality validation with various scenarios.
"""

import asyncio
import json
from pathlib import Path

# Add the project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from workers.quality import QualityWorker, TranscriptQualityMetrics, ContentQualityMetrics


def test_transcript_quality():
    """Test transcript quality validation with various scenarios."""
    print("=" * 60)
    print("TESTING TRANSCRIPT QUALITY VALIDATION")
    print("=" * 60)
    
    quality_worker = QualityWorker()
    
    # Test Case 1: High quality transcript
    print("\n1. High Quality Transcript Test")
    high_quality_input = {
        'target_id': 'test_video_001',
        'target_type': 'transcript',
        'content': """
        Welcome to this comprehensive tutorial on machine learning fundamentals. 
        In today's session, we will explore the key concepts that form the foundation 
        of artificial intelligence and data science. We'll start with supervised learning, 
        which involves training algorithms on labeled datasets to make predictions. 
        Then we'll move on to unsupervised learning, where we discover hidden patterns 
        in data without explicit labels. Finally, we'll discuss reinforcement learning, 
        where agents learn through trial and error by receiving rewards or penalties 
        for their actions. This comprehensive approach will give you a solid understanding 
        of the different paradigms in machine learning and help you choose the right 
        technique for your specific use case.
        """,
        'video_duration': 300,  # 5 minutes
        'video_metadata': {
            'title': 'Machine Learning Fundamentals',
            'language': 'en'
        }
    }
    
    result = quality_worker.run(high_quality_input)
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        data = result['data']
        print(f"Overall Score: {data['overall_score']:.3f}")
        print(f"Passed: {data['passed']}")
        print("Individual Scores:")
        for metric, score in data['individual_scores'].items():
            print(f"  {metric}: {score:.3f}")
        print("Recommendations:")
        for rec in data['recommendations']:
            print(f"  - {rec}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Test Case 2: Poor quality transcript (too short)
    print("\n2. Poor Quality Transcript Test (Too Short)")
    poor_quality_input = {
        'target_id': 'test_video_002',
        'target_type': 'transcript',
        'content': "Hi there. Machine learning is good. Thanks.",
        'video_duration': 300,  # 5 minutes - content is way too short
        'video_metadata': {
            'title': 'Machine Learning Brief',
            'language': 'en'
        }
    }
    
    result = quality_worker.run(poor_quality_input)
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        data = result['data']
        print(f"Overall Score: {data['overall_score']:.3f}")
        print(f"Passed: {data['passed']}")
        print("Individual Scores:")
        for metric, score in data['individual_scores'].items():
            print(f"  {metric}: {score:.3f}")
        print("Recommendations:")
        for rec in data['recommendations']:
            print(f"  - {rec}")
    
    # Test Case 3: Invalid input
    print("\n3. Invalid Input Test")
    invalid_input = {
        'target_id': 'test_video_003',
        'target_type': 'transcript',
        # Missing 'content' and 'video_duration'
    }
    
    result = quality_worker.run(invalid_input)
    print(f"Status: {result['status']}")
    if result['status'] == 'failed':
        print(f"Error: {result.get('error', 'Unknown error')}")


def test_content_quality():
    """Test content quality validation with various scenarios."""
    print("\n" + "=" * 60)
    print("TESTING CONTENT QUALITY VALIDATION")
    print("=" * 60)
    
    quality_worker = QualityWorker()
    
    # Test Case 1: High quality blog post
    print("\n1. High Quality Blog Post Test")
    blog_post_input = {
        'target_id': 'content_001',
        'target_type': 'content',
        'content': """# Machine Learning: A Comprehensive Guide

Machine learning has revolutionized how we approach complex problems in technology and business. This comprehensive guide explores the fundamental concepts that every data scientist should understand.

## What is Machine Learning?

Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping function from input variables to output variables. Common applications include classification and regression problems.

### Unsupervised Learning
Unsupervised learning works with unlabeled data to discover hidden patterns or structures. Clustering and dimensionality reduction are typical examples.

### Reinforcement Learning
Reinforcement learning involves an agent learning to make decisions through trial and error, receiving rewards or penalties for its actions.

## Conclusion

Understanding these fundamental concepts provides a solid foundation for diving deeper into machine learning applications and techniques. Each approach has its strengths and is suitable for different types of problems.
        """,
        'content_type': 'blog_post',
        'source_transcript': 'machine learning artificial intelligence data patterns algorithms supervised unsupervised reinforcement learning classification regression clustering'
    }
    
    result = quality_worker.run(blog_post_input)
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        data = result['data']
        print(f"Overall Score: {data['overall_score']:.3f}")
        print(f"Passed: {data['passed']}")
        print("Individual Scores:")
        for metric, score in data['individual_scores'].items():
            print(f"  {metric}: {score:.3f}")
        print("Recommendations:")
        for rec in data['recommendations']:
            print(f"  - {rec}")
    
    # Test Case 2: Poor quality social media post (too long)
    print("\n2. Poor Quality Social Media Post (Too Long)")
    social_media_input = {
        'target_id': 'content_002',
        'target_type': 'content',
        'content': """This is a very long social media post that goes way beyond the typical character limits of most social platforms. It contains far too much information for a social media format and lacks the conciseness that makes social content engaging. Social media posts should be short, punchy, and to the point, but this one rambles on and on without any clear structure or compelling call to action that would engage the audience effectively.""",
        'content_type': 'social_media',
        'source_transcript': 'social media marketing engagement audience'
    }
    
    result = quality_worker.run(social_media_input)
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        data = result['data']
        print(f"Overall Score: {data['overall_score']:.3f}")
        print(f"Passed: {data['passed']}")
        print("Individual Scores:")
        for metric, score in data['individual_scores'].items():
            print(f"  {metric}: {score:.3f}")
        print("Recommendations:")
        for rec in data['recommendations']:
            print(f"  - {rec}")
    
    # Test Case 3: Good quality summary
    print("\n3. Good Quality Summary Test")
    summary_input = {
        'target_id': 'content_003',
        'target_type': 'content',
        'content': """This video covers machine learning fundamentals, focusing on three main approaches. Supervised learning uses labeled data to train algorithms for classification and regression tasks. Unsupervised learning discovers patterns in unlabeled data through clustering and dimensionality reduction techniques. Reinforcement learning enables agents to learn optimal actions through trial and error with reward systems. These paradigms form the foundation of modern AI applications and provide different solutions for various data science challenges.""",
        'content_type': 'summary',
        'source_transcript': 'machine learning supervised unsupervised reinforcement learning algorithms classification regression clustering data science artificial intelligence'
    }
    
    result = quality_worker.run(summary_input)
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        data = result['data']
        print(f"Overall Score: {data['overall_score']:.3f}")
        print(f"Passed: {data['passed']}")
        print("Individual Scores:")
        for metric, score in data['individual_scores'].items():
            print(f"  {metric}: {score:.3f}")


def test_quality_metrics():
    """Test quality metrics and thresholds."""
    print("\n" + "=" * 60)
    print("TESTING QUALITY METRICS AND THRESHOLDS")
    print("=" * 60)
    
    print("\nTranscript Quality Thresholds:")
    for metric, threshold in TranscriptQualityMetrics.THRESHOLDS.items():
        print(f"  {metric}: {threshold}")
    
    print("\nContent Quality Length Requirements:")
    print("Minimum lengths:")
    for content_type, min_len in ContentQualityMetrics.MIN_LENGTHS.items():
        print(f"  {content_type}: {min_len} words")
    
    print("Maximum lengths:")
    for content_type, max_len in ContentQualityMetrics.MAX_LENGTHS.items():
        print(f"  {content_type}: {max_len} words")
    
    print(f"\nReadability Threshold: {ContentQualityMetrics.READABILITY_THRESHOLD}")


def main():
    """Run all quality worker tests."""
    print("QualityWorker Test Suite")
    print("=" * 60)
    
    try:
        test_quality_metrics()
        test_transcript_quality()
        test_content_quality()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()