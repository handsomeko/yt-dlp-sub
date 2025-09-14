#!/usr/bin/env python3
"""
Test script for dual transcript system - comparing YouTube auto-generated captions vs Whisper transcripts.

This script:
1. Downloads videos from a YouTube channel
2. Extracts YouTube auto-generated captions (saved as _auto.srt/txt)
3. Runs Whisper transcription (saved as _whisper.srt/txt)
4. Compares the two transcripts for quality metrics
5. Generates a comparison report
"""

import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime
import difflib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.channel_enumerator import ChannelEnumerator
from core.downloader import YouTubeDownloader
from core.subtitle_extractor_v2 import LanguageAgnosticSubtitleExtractor
from workers.transcriber import TranscribeWorker
from core.storage_paths_v2 import get_storage_paths_v2
from core.filename_sanitizer import sanitize_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DualTranscriptTester:
    """Test harness for comparing auto-generated vs Whisper transcripts."""
    
    def __init__(self):
        self.storage_paths = get_storage_paths_v2()
        self.downloader = YouTubeDownloader()
        self.subtitle_extractor = LanguageAgnosticSubtitleExtractor(translate_enabled=False)
        self.transcriber = TranscribeWorker()
        self.results = []
        
    def process_channel(self, channel_url: str, limit: int = 2) -> Dict[str, Any]:
        """Process videos from a channel with both transcript methods."""
        
        logger.info(f"Starting dual transcript test for channel: {channel_url}")
        logger.info(f"Processing limit: {limit} videos")
        
        # Enumerate videos from channel
        enumerator = ChannelEnumerator()
        channel_info = enumerator.get_channel_info(channel_url)
        
        if not channel_info:
            logger.error(f"Could not get channel info for: {channel_url}")
            return {"error": "Failed to get channel info"}
        
        videos = enumerator.get_all_videos(channel_url, limit=limit)
        
        if not videos:
            logger.error(f"No videos found for channel: {channel_url}")
            return {"error": "No videos found"}
        
        logger.info(f"Found {len(videos)} videos to process")
        
        # Process each video
        for i, video in enumerate(videos, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing video {i}/{len(videos)}: {video.get('title', 'Unknown')}")
            logger.info(f"Video ID: {video.get('video_id')}")
            logger.info(f"URL: {video.get('url')}")
            
            result = self.process_video(video)
            self.results.append(result)
            
            # Add a small delay between videos to avoid rate limiting
            if i < len(videos):
                time.sleep(2)
        
        # Generate comparison report
        report = self.generate_report()
        
        return {
            "channel_info": channel_info,
            "videos_processed": len(self.results),
            "report": report,
            "detailed_results": self.results
        }
    
    def process_video(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single video with both transcript methods."""
        
        video_id = video_info.get('video_id')
        video_url = video_info.get('url')
        video_title = video_info.get('title', 'Unknown')
        
        if not video_id or not video_url:
            return {"error": "Missing video ID or URL"}
        
        result = {
            "video_id": video_id,
            "video_title": video_title,
            "video_url": video_url,
            "auto_transcript": None,
            "whisper_transcript": None,
            "comparison": None,
            "errors": []
        }
        
        try:
            # Step 1: Download audio
            logger.info("Step 1: Downloading audio...")
            download_result = self.downloader.download_video(
                url=video_url,
                download_audio_only=True
            )
            
            if not download_result or not download_result.get('success'):
                result["errors"].append(f"Download failed: {download_result.get('error', 'Unknown error')}")
                return result
            
            storage_info = download_result.get('storage', {})
            channel_id = storage_info.get('channel_id')
            video_id = storage_info.get('video_id')
            video_title = storage_info.get('video_title')
            audio_path = download_result.get('audio_path')
            
            logger.info(f"Audio downloaded successfully: {audio_path}")
            
            # Step 2: Extract YouTube auto-generated captions
            logger.info("Step 2: Extracting YouTube auto-generated captions...")
            transcript_dir = self.storage_paths.get_transcript_dir(channel_id, video_id)
            
            subtitle_result = self.subtitle_extractor.extract_subtitles(
                video_url=video_url,
                output_dir=transcript_dir,
                video_id=video_id,
                video_title=video_title
            )
            
            if subtitle_result.success:
                # Find the auto transcript files
                auto_files = [f for f in subtitle_result.original_files if '_auto' in f]
                auto_txt_files = [f for f in auto_files if f.endswith('.txt')]
                
                if auto_txt_files:
                    with open(auto_txt_files[0], 'r', encoding='utf-8') as f:
                        result["auto_transcript"] = {
                            "text": f.read(),
                            "file": auto_txt_files[0],
                            "languages": subtitle_result.languages_found,
                            "methods": subtitle_result.methods_used
                        }
                    logger.info(f"Auto-generated captions extracted: {auto_txt_files[0]}")
                else:
                    result["errors"].append("No auto-generated text file found")
            else:
                result["errors"].append("Failed to extract auto-generated captions")
            
            # Step 3: Run Whisper transcription
            logger.info("Step 3: Running Whisper transcription...")
            
            # Prepare input for transcriber
            transcriber_input = {
                'video_id': video_id,
                'video_url': video_url,
                'channel_id': channel_id,
                'audio_path': audio_path,
                'video_title': video_title
            }
            
            # Execute Whisper transcription
            whisper_result = self.transcriber.execute(transcriber_input)
            
            if whisper_result.get('status') == 'success':
                whisper_txt_path = whisper_result.get('txt_path')
                
                if whisper_txt_path and Path(whisper_txt_path).exists():
                    with open(whisper_txt_path, 'r', encoding='utf-8') as f:
                        result["whisper_transcript"] = {
                            "text": f.read(),
                            "file": whisper_txt_path,
                            "method": whisper_result.get('extraction_method'),
                            "language": whisper_result.get('language'),
                            "word_count": whisper_result.get('word_count')
                        }
                    logger.info(f"Whisper transcript created: {whisper_txt_path}")
                else:
                    result["errors"].append(f"Whisper text file not found: {whisper_txt_path}")
            else:
                result["errors"].append(f"Whisper transcription failed: {whisper_result.get('error', 'Unknown error')}")
            
            # Step 4: Compare transcripts
            if result["auto_transcript"] and result["whisper_transcript"]:
                result["comparison"] = self.compare_transcripts(
                    result["auto_transcript"]["text"],
                    result["whisper_transcript"]["text"]
                )
                logger.info("Transcript comparison completed")
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            result["errors"].append(str(e))
        
        return result
    
    def compare_transcripts(self, auto_text: str, whisper_text: str) -> Dict[str, Any]:
        """Compare two transcripts for quality metrics."""
        
        # Basic metrics
        auto_words = auto_text.split()
        whisper_words = whisper_text.split()
        
        # Calculate similarity ratio
        similarity = difflib.SequenceMatcher(None, auto_text, whisper_text).ratio()
        
        # Count punctuation
        auto_punctuation = sum(1 for char in auto_text if char in '.,!?;:')
        whisper_punctuation = sum(1 for char in whisper_text if char in '.,!?;:')
        
        # Count capitalization (sentence starts)
        auto_caps = sum(1 for word in auto_words if word and word[0].isupper())
        whisper_caps = sum(1 for word in whisper_words if word and word[0].isupper())
        
        # Check for technical terms (simple heuristic)
        tech_terms = ['API', 'JavaScript', 'Python', 'function', 'variable', 'class', 
                      'method', 'array', 'object', 'database', 'server', 'client']
        
        auto_tech_count = sum(1 for term in tech_terms 
                              if term.lower() in auto_text.lower())
        whisper_tech_count = sum(1 for term in tech_terms 
                                if term.lower() in whisper_text.lower())
        
        return {
            "word_count": {
                "auto": len(auto_words),
                "whisper": len(whisper_words),
                "difference": len(whisper_words) - len(auto_words)
            },
            "character_count": {
                "auto": len(auto_text),
                "whisper": len(whisper_text),
                "difference": len(whisper_text) - len(auto_text)
            },
            "similarity_ratio": round(similarity, 3),
            "punctuation": {
                "auto": auto_punctuation,
                "whisper": whisper_punctuation,
                "difference": whisper_punctuation - auto_punctuation
            },
            "capitalization": {
                "auto": auto_caps,
                "whisper": whisper_caps,
                "difference": whisper_caps - auto_caps
            },
            "technical_terms": {
                "auto": auto_tech_count,
                "whisper": whisper_tech_count,
                "difference": whisper_tech_count - auto_tech_count
            }
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive comparison report."""
        
        successful = [r for r in self.results if r["comparison"]]
        failed = [r for r in self.results if not r["comparison"]]
        
        if not successful:
            return {
                "summary": "No successful comparisons",
                "total_videos": len(self.results),
                "failed_videos": len(failed)
            }
        
        # Aggregate metrics
        avg_similarity = sum(r["comparison"]["similarity_ratio"] for r in successful) / len(successful)
        
        total_auto_words = sum(r["comparison"]["word_count"]["auto"] for r in successful)
        total_whisper_words = sum(r["comparison"]["word_count"]["whisper"] for r in successful)
        
        total_auto_punct = sum(r["comparison"]["punctuation"]["auto"] for r in successful)
        total_whisper_punct = sum(r["comparison"]["punctuation"]["whisper"] for r in successful)
        
        report = {
            "summary": {
                "total_videos": len(self.results),
                "successful_comparisons": len(successful),
                "failed_comparisons": len(failed),
                "average_similarity": round(avg_similarity, 3)
            },
            "aggregate_metrics": {
                "total_words": {
                    "auto": total_auto_words,
                    "whisper": total_whisper_words,
                    "difference_percent": round((total_whisper_words - total_auto_words) / total_auto_words * 100, 2) if total_auto_words > 0 else 0
                },
                "total_punctuation": {
                    "auto": total_auto_punct,
                    "whisper": total_whisper_punct,
                    "difference_percent": round((total_whisper_punct - total_auto_punct) / total_auto_punct * 100, 2) if total_auto_punct > 0 else 0
                }
            },
            "individual_results": []
        }
        
        # Add individual video summaries
        for result in successful:
            video_summary = {
                "video_title": result["video_title"],
                "video_id": result["video_id"],
                "similarity": result["comparison"]["similarity_ratio"],
                "word_count_diff": result["comparison"]["word_count"]["difference"],
                "punctuation_diff": result["comparison"]["punctuation"]["difference"],
                "technical_terms_diff": result["comparison"]["technical_terms"]["difference"]
            }
            report["individual_results"].append(video_summary)
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str = "dual_transcript_report.json"):
        """Save the report to a JSON file."""
        
        report_with_metadata = {
            "test_timestamp": datetime.now().isoformat(),
            "test_type": "dual_transcript_comparison",
            **report
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_with_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved to: {output_path}")
        
        # Also create a human-readable summary
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("DUAL TRANSCRIPT COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if "summary" in report:
                f.write("SUMMARY\n")
                f.write("-" * 40 + "\n")
                for key, value in report["summary"].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            if "aggregate_metrics" in report:
                f.write("AGGREGATE METRICS\n")
                f.write("-" * 40 + "\n")
                metrics = report["aggregate_metrics"]
                f.write(f"Total Words - Auto: {metrics['total_words']['auto']}, "
                       f"Whisper: {metrics['total_words']['whisper']} "
                       f"({metrics['total_words']['difference_percent']:+.1f}%)\n")
                f.write(f"Total Punctuation - Auto: {metrics['total_punctuation']['auto']}, "
                       f"Whisper: {metrics['total_punctuation']['whisper']} "
                       f"({metrics['total_punctuation']['difference_percent']:+.1f}%)\n")
                f.write("\n")
            
            if "individual_results" in report:
                f.write("INDIVIDUAL VIDEO RESULTS\n")
                f.write("-" * 40 + "\n")
                for video in report["individual_results"]:
                    f.write(f"\nVideo: {video['video_title']}\n")
                    f.write(f"  Similarity: {video['similarity']:.3f}\n")
                    f.write(f"  Word Count Diff: {video['word_count_diff']:+d}\n")
                    f.write(f"  Punctuation Diff: {video['punctuation_diff']:+d}\n")
                    f.write(f"  Technical Terms Diff: {video['technical_terms_diff']:+d}\n")
        
        logger.info(f"Summary saved to: {summary_path}")


def main():
    """Main entry point for the test script."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dual transcript system')
    parser.add_argument('channel_url', nargs='?', default='https://www.youtube.com/@mikeynocode',
                       help='YouTube channel URL to test (default: @mikeynocode)')
    parser.add_argument('--limit', type=int, default=2,
                       help='Number of videos to process (default: 2)')
    parser.add_argument('--output', default='dual_transcript_report.json',
                       help='Output file for report (default: dual_transcript_report.json)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("DUAL TRANSCRIPT SYSTEM TEST")
    print("=" * 60)
    print(f"Channel: {args.channel_url}")
    print(f"Video Limit: {args.limit}")
    print(f"Output File: {args.output}")
    print("=" * 60 + "\n")
    
    # Run the test
    tester = DualTranscriptTester()
    result = tester.process_channel(args.channel_url, limit=args.limit)
    
    # Save the report
    if "report" in result:
        tester.save_report(result, args.output)
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("TEST COMPLETE - SUMMARY")
        print("=" * 60)
        
        if "summary" in result["report"]:
            summary = result["report"]["summary"]
            print(f"Videos Processed: {summary.get('total_videos', 0)}")
            print(f"Successful Comparisons: {summary.get('successful_comparisons', 0)}")
            print(f"Failed Comparisons: {summary.get('failed_comparisons', 0)}")
            print(f"Average Similarity: {summary.get('average_similarity', 0):.3f}")
        
        print(f"\nFull report saved to: {args.output}")
        print(f"Summary saved to: {args.output.replace('.json', '_summary.txt')}")
    else:
        print("\nERROR: Test failed")
        if "error" in result:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()