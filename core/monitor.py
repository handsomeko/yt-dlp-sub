"""Channel Monitor Module - RSS-based YouTube channel monitoring"""

import json
import feedparser
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import sqlite3

# CRITICAL FIX: Import rate limiting to prevent 429 errors
from .rate_limit_manager import get_rate_limit_manager

logger = logging.getLogger(__name__)


class ChannelMonitor:
    """Monitor YouTube channels for new videos using RSS feeds"""
    
    def __init__(self, db_path: str = None):
        """Initialize with database for tracking"""
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'data' / 'monitor.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Channels table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS channels (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked TIMESTAMP,
                enabled BOOLEAN DEFAULT 1
            )
        ''')
        
        # Videos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                channel_id TEXT,
                title TEXT,
                url TEXT,
                published TIMESTAMP,
                downloaded BOOLEAN DEFAULT 0,
                download_date TIMESTAMP,
                FOREIGN KEY (channel_id) REFERENCES channels (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_channel(self, channel_id: str, channel_name: str, channel_url: str = None) -> bool:
        """Add a channel to monitor"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if not channel_url:
                channel_url = f"https://www.youtube.com/channel/{channel_id}"
                
            cursor.execute('''
                INSERT OR REPLACE INTO channels (id, name, url)
                VALUES (?, ?, ?)
            ''', (channel_id, channel_name, channel_url))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding channel: {e}")
            return False
        finally:
            conn.close()
            
    def get_channels(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get list of monitored channels"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if enabled_only:
            cursor.execute('SELECT * FROM channels WHERE enabled = 1')
        else:
            cursor.execute('SELECT * FROM channels')
            
        columns = [description[0] for description in cursor.description]
        channels = []
        
        for row in cursor.fetchall():
            channels.append(dict(zip(columns, row)))
            
        conn.close()
        return channels
    
    def check_channel_for_new_videos(self, 
                                    channel_id: str, 
                                    days_back: int = 7) -> List[Dict[str, Any]]:
        """Check a channel for new videos via RSS"""
        feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        
        # CRITICAL FIX: Add rate limiting protection before feedparser.parse()
        rate_manager = get_rate_limit_manager()
        allowed, wait_time = rate_manager.should_allow_request('youtube.com')
        if not allowed:
            logger.warning(f"Rate limited - waiting {wait_time:.1f}s before RSS feed request for channel {channel_id}")
            time.sleep(wait_time)
        
        try:
            feed = feedparser.parse(feed_url)
            
            # Record the request result for rate limiting stats
            # feedparser doesn't give us HTTP status, so we use bozo flag as proxy
            success = not feed.get('bozo', False)
            rate_manager.record_request('youtube.com', success=success)
            if feed.bozo:
                return []
                
            cutoff_date = datetime.now() - timedelta(days=days_back)
            new_videos = []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for entry in feed.entries:
                video_id = entry.yt_videoid
                
                # Check if already in database
                cursor.execute('SELECT video_id FROM videos WHERE video_id = ?', (video_id,))
                if cursor.fetchone():
                    continue
                    
                # Parse publish date
                published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%S%z")
                published = published.replace(tzinfo=None)  # Remove timezone for comparison
                
                if published > cutoff_date:
                    video_data = {
                        'video_id': video_id,
                        'channel_id': channel_id,
                        'title': entry.title,
                        'url': entry.link,
                        'published': published.isoformat(),
                        'description': entry.get('summary', ''),
                        'thumbnail': entry.get('media_thumbnail', [{}])[0].get('url', '') if 'media_thumbnail' in entry else ''
                    }
                    
                    # Add to database
                    cursor.execute('''
                        INSERT INTO videos (video_id, channel_id, title, url, published)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (video_id, channel_id, entry.title, entry.link, published))
                    
                    new_videos.append(video_data)
                    
            # Update last checked time
            cursor.execute('''
                UPDATE channels SET last_checked = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (channel_id,))
            
            conn.commit()
            conn.close()
            
            return new_videos
            
        except Exception as e:
            # CRITICAL FIX: Record failed request for rate limiting stats
            rate_manager = get_rate_limit_manager()
            error_str = str(e).lower()
            is_429 = '429' in error_str or 'too many requests' in error_str
            rate_manager.record_request('youtube.com', success=False, is_429=is_429)
            
            logger.error(f"Error checking channel {channel_id}: {e}")
            return []
    
    def check_all_channels(self, days_back: int = 7) -> Dict[str, List[Dict[str, Any]]]:
        """Check all enabled channels for new videos"""
        channels = self.get_channels(enabled_only=True)
        all_new_videos = {}
        
        for channel in channels:
            new_videos = self.check_channel_for_new_videos(channel['id'], days_back)
            if new_videos:
                all_new_videos[channel['name']] = new_videos
                
        return all_new_videos
    
    def mark_video_downloaded(self, video_id: str) -> bool:
        """Mark a video as downloaded"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE videos 
                SET downloaded = 1, download_date = CURRENT_TIMESTAMP
                WHERE video_id = ?
            ''', (video_id,))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error marking video as downloaded: {e}")
            return False
        finally:
            conn.close()
    
    def get_pending_videos(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get videos that haven't been downloaded yet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT v.*, c.name as channel_name
            FROM videos v
            JOIN channels c ON v.channel_id = c.id
            WHERE v.downloaded = 0
            ORDER BY v.published DESC
        '''
        
        # Use parameterized query to prevent SQL injection
        if limit:
            # Validate limit is a positive integer
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("Limit must be a positive integer")
            query += ' LIMIT ?'
            cursor.execute(query, (limit,))
        else:
            cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        videos = []
        
        for row in cursor.fetchall():
            videos.append(dict(zip(columns, row)))
            
        conn.close()
        return videos
    
    def load_channels_from_json(self, json_file: str) -> int:
        """Load channels from a JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        channels = data.get('channels', [])
        added = 0
        
        for channel in channels:
            if self.add_channel(channel['id'], channel['name'], channel.get('url')):
                added += 1
                
        return added