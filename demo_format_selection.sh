#!/bin/bash
# Demo: Format Selection Checkpoint Workflow

echo "======================================="
echo " Format Selection Checkpoint Demo"
echo "======================================="
echo ""
echo "This demo shows the two-checkpoint workflow:"
echo "1. Review Checkpoint - Approve/reject videos"
echo "2. Format Selection - Choose which formats to generate"
echo ""

# Demo 1: Process videos with format selection workflow
echo "-----------------------------------"
echo "Demo 1: Standard workflow with format selection"
echo "-----------------------------------"
echo ""

echo "Step 1: Process multiple videos"
echo "Command: yt-dl process url1 url2 url3"
python3 cli.py process \
  "https://www.youtube.com/watch?v=jNQXAC9IVRw" \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  "https://www.youtube.com/watch?v=kJQP7kiw5Fk"

echo ""
echo "Step 2: Review pending videos"
echo "Command: yt-dl review list"
python3 cli.py review list --limit 5

echo ""
echo "Step 3: Approve videos for content generation"
echo "Command: yt-dl review approve <video_ids>"
echo "(Videos now pending format selection)"

echo ""
echo "Step 4: List videos pending format selection"
echo "Command: yt-dl generate list --show-formats"
python3 cli.py generate list --show-formats

echo ""
echo "Step 5: Select specific formats for a video"
echo "Command: yt-dl generate select <video_id> --formats blog,social"
echo ""

echo "Step 6: Select all formats for multiple videos"
echo "Command: yt-dl generate select-all <video_ids>"
echo ""

echo "Step 7: Interactive format selection"
echo "Command: yt-dl generate interactive"
echo "This opens an interactive UI to select formats for each video"
echo ""

echo "Step 8: Start content generation"
echo "Command: yt-dl generate start"
echo "Creates jobs only for selected formats"
echo ""

echo "-----------------------------------"
echo "Demo 2: Skip format selection (use all formats)"
echo "-----------------------------------"
echo ""

echo "Approve with --skip-format-selection flag:"
echo "Command: yt-dl review approve <video_ids> --skip-format-selection"
echo "This immediately creates jobs for ALL formats"
echo ""

echo "-----------------------------------"
echo "Demo 3: Channel-specific default formats"
echo "-----------------------------------"
echo ""

echo "Set in .env file:"
echo "CHANNEL_DEFAULT_FORMATS=TED:blog,newsletter;MIT:summary,script"
echo ""
echo "Videos from TED will default to: blog, newsletter"
echo "Videos from MIT will default to: summary, script"
echo ""

echo "-----------------------------------"
echo "Demo 4: Bulk operations"
echo "-----------------------------------"
echo ""

echo "Select all formats for all pending videos:"
echo "Command: yt-dl generate select-all --all-pending"
echo ""

echo "Clear format selections:"
echo "Command: yt-dl generate clear <video_ids>"
echo ""

echo "Dry run to preview what would be generated:"
echo "Command: yt-dl generate start --dry-run"
echo ""

echo "======================================="
echo " Available Commands Summary"
echo "======================================="
echo ""
echo "Review Commands:"
echo "  yt-dl review list                    # List videos to review"
echo "  yt-dl review approve <ids>           # Approve videos"
echo "  yt-dl review reject <ids>            # Reject videos"
echo "  yt-dl review interactive             # Interactive review"
echo ""
echo "Generate Commands:"
echo "  yt-dl generate list                  # List pending format selection"
echo "  yt-dl generate select <ids> --formats blog,social"
echo "  yt-dl generate select-all <ids>      # Select all formats"
echo "  yt-dl generate interactive           # Interactive selection"
echo "  yt-dl generate clear <ids>           # Clear selections"
echo "  yt-dl generate start                 # Start generation"
echo ""
echo "Available Formats: blog, social, summary, newsletter, script"
echo ""
echo "======================================="
echo " Demo Complete!"
echo "======================================="