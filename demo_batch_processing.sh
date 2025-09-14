#!/bin/bash
# Demo: Batch Video Processing with Multiple URLs

echo "==================================="
echo " Batch Video Processing Demo"
echo "==================================="
echo ""

# Demo 1: Process multiple videos directly
echo "1. Processing multiple videos as arguments:"
echo "   yt-dl process url1 url2 url3"
python3 cli.py process \
  "https://www.youtube.com/watch?v=jNQXAC9IVRw" \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  "https://www.youtube.com/shorts/abc123"

echo ""
echo "-----------------------------------"
echo ""

# Demo 2: Process from file
echo "2. Processing URLs from a file:"
echo "   yt-dl process --from-file urls.txt"
cat > demo_urls.txt << EOF
# YouTube videos to process
https://www.youtube.com/watch?v=9bZkp7q19f0
https://www.youtube.com/watch?v=kJQP7kiw5Fk  
https://www.youtube.com/shorts/xyz789

# This is a channel (needs --all flag)
https://www.youtube.com/@TED
EOF

python3 cli.py process --from-file demo_urls.txt

echo ""
echo "-----------------------------------"
echo ""

# Demo 3: Combine multiple sources
echo "3. Combining command-line URLs with file:"
echo "   yt-dl process url1 --from-file urls.txt"
python3 cli.py process \
  "https://www.youtube.com/watch?v=newvideo" \
  --from-file demo_urls.txt

echo ""
echo "-----------------------------------"
echo ""

# Demo 4: Process channels with --all
echo "4. Process videos from channels:"
echo "   yt-dl process <urls> --all --limit 5"
python3 cli.py process \
  "https://www.youtube.com/@YouTube" \
  "https://www.youtube.com/@TED" \
  --all --limit 5

echo ""
echo "==================================="
echo " Demo Complete!"
echo "==================================="

# Clean up
rm -f demo_urls.txt