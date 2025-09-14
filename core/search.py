"""
Full-text search functionality using SQLite FTS5 virtual tables.

Provides comprehensive search capabilities across video transcripts and generated content
with relevance scoring, filtering, pagination, and result highlighting.
"""

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import aiosqlite
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import DatabaseManager


class SearchType(Enum):
    """Search target types."""
    TRANSCRIPTS = "transcripts"
    CONTENT = "content"
    ALL = "all"


class SortOrder(Enum):
    """Search result sort orders."""
    RELEVANCE = "relevance"
    DATE = "date"
    TITLE = "title"


@dataclass
class SearchFilters:
    """Search filtering options."""
    channel_id: Optional[str] = None
    content_type: Optional[str] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    min_quality_score: Optional[float] = None
    topics: Optional[List[str]] = None


@dataclass 
class SearchResult:
    """Individual search result."""
    video_id: str
    title: str
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    content_type: Optional[str] = None
    snippet: str = ""
    rank: float = 0.0
    published_at: Optional[datetime] = None
    word_count: Optional[int] = None
    quality_score: Optional[float] = None
    url: Optional[str] = None


@dataclass
class SearchResults:
    """Paginated search results."""
    results: List[SearchResult]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    query: str
    search_time_ms: float


class SearchService:
    """
    Full-text search service using SQLite FTS5.
    
    Features:
    - Full-text search across transcripts and generated content
    - Relevance scoring using BM25 algorithm
    - Filtering by channel, date range, content type
    - Context snippets with match highlighting
    - Search suggestions and query completion
    - Pagination support
    - Multiple sort options
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    async def search_transcripts(
        self,
        query: str,
        channel_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: SortOrder = SortOrder.RELEVANCE
    ) -> SearchResults:
        """
        Search video transcripts with full-text search.
        
        Args:
            query: Search query string (supports FTS5 syntax)
            channel_id: Filter by specific channel
            since: Only include videos published after this date
            limit: Maximum results to return
            offset: Number of results to skip (for pagination)
            sort_by: Result ordering preference
            
        Returns:
            SearchResults with paginated transcript matches
        """
        start_time = datetime.now()
        
        # Prepare search query for FTS5
        fts_query = self._prepare_fts_query(query)
        
        raw_db_path = self.db_manager.database_url.replace("sqlite+aiosqlite:///", "")
        
        async with aiosqlite.connect(raw_db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Build main search query
            sql_parts = [
                """
                SELECT 
                    fts.video_id,
                    v.title,
                    v.channel_id,
                    c.channel_name,
                    v.published_at,
                    t.word_count,
                    t.quality_score,
                    snippet(transcripts_fts, 1, '<mark>', '</mark>', '...', 64) as snippet,
                    bm25(transcripts_fts) as rank,
                    'https://youtube.com/watch?v=' || fts.video_id as url
                FROM transcripts_fts fts
                JOIN videos v ON fts.video_id = v.video_id
                JOIN channels c ON v.channel_id = c.channel_id
                JOIN transcripts t ON v.video_id = t.video_id
                WHERE transcripts_fts MATCH ?
                """
            ]
            
            params = [fts_query]
            
            # Add filters
            if channel_id:
                sql_parts.append("AND v.channel_id = ?")
                params.append(channel_id)
                
            if since:
                sql_parts.append("AND v.published_at >= ?")
                params.append(since.isoformat())
            
            # Add sorting
            if sort_by == SortOrder.RELEVANCE:
                sql_parts.append("ORDER BY rank")
            elif sort_by == SortOrder.DATE:
                sql_parts.append("ORDER BY v.published_at DESC")
            elif sort_by == SortOrder.TITLE:
                sql_parts.append("ORDER BY v.title")
                
            # Add pagination
            sql_parts.append("LIMIT ? OFFSET ?")
            params.extend([limit, offset])
            
            main_query = " ".join(sql_parts)
            
            # Execute search
            cursor = await db.execute(main_query, params)
            rows = await cursor.fetchall()
            
            # Get total count for pagination
            count_query = """
                SELECT COUNT(*) as total
                FROM transcripts_fts fts
                JOIN videos v ON fts.video_id = v.video_id
                WHERE transcripts_fts MATCH ?
            """
            count_params = [fts_query]
            
            if channel_id:
                count_query += " AND v.channel_id = ?"
                count_params.append(channel_id)
                
            if since:
                count_query += " AND v.published_at >= ?"
                count_params.append(since.isoformat())
            
            cursor = await db.execute(count_query, count_params)
            total_count = (await cursor.fetchone())['total']
            
            # Convert results
            results = []
            for row in rows:
                result = SearchResult(
                    video_id=row['video_id'],
                    title=row['title'],
                    channel_id=row['channel_id'],
                    channel_name=row['channel_name'],
                    snippet=row['snippet'],
                    rank=row['rank'],
                    published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                    word_count=row['word_count'],
                    quality_score=row['quality_score'],
                    url=row['url']
                )
                results.append(result)
        
        # Calculate pagination info
        page = (offset // limit) + 1
        total_pages = (total_count + limit - 1) // limit
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResults(
            results=results,
            total_count=total_count,
            page=page,
            page_size=limit,
            total_pages=total_pages,
            query=query,
            search_time_ms=search_time
        )
    
    async def search_content(
        self,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: SortOrder = SortOrder.RELEVANCE
    ) -> SearchResults:
        """
        Search generated content with full-text search.
        
        Args:
            query: Search query string
            content_type: Filter by content type (blog, summary, twitter, etc.)
            limit: Maximum results to return
            offset: Number of results to skip
            sort_by: Result ordering preference
            
        Returns:
            SearchResults with paginated content matches
        """
        start_time = datetime.now()
        
        # Prepare search query for FTS5
        fts_query = self._prepare_fts_query(query)
        
        raw_db_path = self.db_manager.database_url.replace("sqlite+aiosqlite:///", "")
        
        async with aiosqlite.connect(raw_db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Build main search query
            sql_parts = [
                """
                SELECT 
                    fts.video_id,
                    v.title,
                    v.channel_id,
                    c.channel_name,
                    fts.content_type,
                    gc.quality_score,
                    v.published_at,
                    snippet(content_fts, 2, '<mark>', '</mark>', '...', 64) as snippet,
                    bm25(content_fts) as rank,
                    'https://youtube.com/watch?v=' || fts.video_id as url
                FROM content_fts fts
                JOIN generated_content gc ON fts.rowid = gc.id
                JOIN videos v ON gc.video_id = v.video_id
                JOIN channels c ON v.channel_id = c.channel_id
                WHERE content_fts MATCH ?
                """
            ]
            
            params = [fts_query]
            
            # Add content type filter
            if content_type:
                sql_parts.append("AND fts.content_type = ?")
                params.append(content_type)
            
            # Add sorting
            if sort_by == SortOrder.RELEVANCE:
                sql_parts.append("ORDER BY rank")
            elif sort_by == SortOrder.DATE:
                sql_parts.append("ORDER BY v.published_at DESC")
            elif sort_by == SortOrder.TITLE:
                sql_parts.append("ORDER BY v.title")
                
            # Add pagination
            sql_parts.append("LIMIT ? OFFSET ?")
            params.extend([limit, offset])
            
            main_query = " ".join(sql_parts)
            
            # Execute search
            cursor = await db.execute(main_query, params)
            rows = await cursor.fetchall()
            
            # Get total count
            count_query = """
                SELECT COUNT(*) as total
                FROM content_fts fts
                JOIN generated_content gc ON fts.rowid = gc.id
                WHERE content_fts MATCH ?
            """
            count_params = [fts_query]
            
            if content_type:
                count_query += " AND fts.content_type = ?"
                count_params.append(content_type)
            
            cursor = await db.execute(count_query, count_params)
            total_count = (await cursor.fetchone())['total']
            
            # Convert results
            results = []
            for row in rows:
                result = SearchResult(
                    video_id=row['video_id'],
                    title=row['title'],
                    channel_id=row['channel_id'],
                    channel_name=row['channel_name'],
                    content_type=row['content_type'],
                    snippet=row['snippet'],
                    rank=row['rank'],
                    published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                    quality_score=row['quality_score'],
                    url=row['url']
                )
                results.append(result)
        
        # Calculate pagination info
        page = (offset // limit) + 1
        total_pages = (total_count + limit - 1) // limit
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResults(
            results=results,
            total_count=total_count,
            page=page,
            page_size=limit,
            total_pages=total_pages,
            query=query,
            search_time_ms=search_time
        )
    
    async def advanced_search(
        self,
        query: str,
        search_type: SearchType = SearchType.ALL,
        filters: Optional[SearchFilters] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: SortOrder = SortOrder.RELEVANCE
    ) -> SearchResults:
        """
        Advanced search with comprehensive filtering options.
        
        Args:
            query: Search query string
            search_type: What to search (transcripts, content, or both)
            filters: Additional filtering options
            limit: Maximum results to return
            offset: Number of results to skip
            sort_by: Result ordering preference
            
        Returns:
            Combined search results from requested sources
        """
        if filters is None:
            filters = SearchFilters()
            
        results = []
        
        if search_type in [SearchType.TRANSCRIPTS, SearchType.ALL]:
            transcript_results = await self.search_transcripts(
                query=query,
                channel_id=filters.channel_id,
                since=filters.since,
                limit=limit if search_type == SearchType.TRANSCRIPTS else limit // 2,
                offset=offset,
                sort_by=sort_by
            )
            results.extend(transcript_results.results)
        
        if search_type in [SearchType.CONTENT, SearchType.ALL]:
            content_results = await self.search_content(
                query=query,
                content_type=filters.content_type,
                limit=limit if search_type == SearchType.CONTENT else limit // 2,
                offset=offset if search_type == SearchType.CONTENT else 0,
                sort_by=sort_by
            )
            results.extend(content_results.results)
        
        # Re-sort combined results if needed
        if search_type == SearchType.ALL:
            if sort_by == SortOrder.RELEVANCE:
                results.sort(key=lambda r: r.rank)
            elif sort_by == SortOrder.DATE:
                results.sort(key=lambda r: r.published_at or datetime.min, reverse=True)
            elif sort_by == SortOrder.TITLE:
                results.sort(key=lambda r: r.title)
            
            # Apply limit to combined results
            results = results[:limit]
        
        # Calculate approximate totals
        total_count = len(results)
        if search_type == SearchType.ALL:
            # This is approximate - in production you'd want separate count queries
            total_count = (transcript_results.total_count if 'transcript_results' in locals() else 0) + \
                         (content_results.total_count if 'content_results' in locals() else 0)
        
        page = (offset // limit) + 1
        total_pages = (total_count + limit - 1) // limit
        
        return SearchResults(
            results=results,
            total_count=total_count,
            page=page,
            page_size=limit,
            total_pages=total_pages,
            query=query,
            search_time_ms=0.0  # Would need to track across all sub-searches
        )
    
    async def get_search_suggestions(
        self,
        partial_query: str,
        limit: int = 10
    ) -> List[str]:
        """
        Generate search suggestions based on partial query.
        
        Uses word frequency analysis from transcripts and content to suggest
        common terms and phrases that start with the partial query.
        
        Args:
            partial_query: Incomplete search term
            limit: Maximum suggestions to return
            
        Returns:
            List of suggested search terms
        """
        if len(partial_query) < 2:
            return []
            
        raw_db_path = self.db_manager.database_url.replace("sqlite+aiosqlite:///", "")
        
        async with aiosqlite.connect(raw_db_path) as db:
            # Get suggestions from transcript content
            # This is a simplified approach - in production you might maintain
            # a separate suggestions index
            suggestions = []
            
            # Search for terms that start with the partial query
            query = f"{partial_query}*"
            
            # Get common terms from transcripts
            cursor = await db.execute("""
                SELECT v.title, COUNT(*) as frequency
                FROM transcripts_fts fts
                JOIN videos v ON fts.video_id = v.video_id
                WHERE transcripts_fts MATCH ?
                GROUP BY v.title
                ORDER BY frequency DESC, v.title
                LIMIT ?
            """, [query, limit // 2])
            
            rows = await cursor.fetchall()
            for row in rows:
                title_words = row[0].lower().split()
                for word in title_words:
                    if word.startswith(partial_query.lower()) and len(word) > len(partial_query):
                        if word not in suggestions:
                            suggestions.append(word)
            
            # Get suggestions from content keywords
            cursor = await db.execute("""
                SELECT fts.content_type, COUNT(*) as frequency  
                FROM content_fts fts
                WHERE content_fts MATCH ?
                GROUP BY fts.content_type
                ORDER BY frequency DESC
                LIMIT ?
            """, [query, limit // 2])
            
            rows = await cursor.fetchall()
            for row in rows:
                content_type = row[0]
                if content_type.startswith(partial_query.lower()) and content_type not in suggestions:
                    suggestions.append(content_type)
            
            return suggestions[:limit]
    
    def highlight_matches(
        self,
        text: str,
        query: str,
        highlight_tag: str = "mark"
    ) -> str:
        """
        Add HTML highlighting around query matches in text.
        
        Args:
            text: Text to highlight matches in
            query: Search query to highlight
            highlight_tag: HTML tag to use for highlighting
            
        Returns:
            Text with highlighted matches
        """
        if not query or not text:
            return text
            
        # Split query into individual terms
        terms = query.lower().split()
        
        # Create regex pattern for case-insensitive matching
        for term in terms:
            # Escape special regex characters
            escaped_term = re.escape(term)
            pattern = re.compile(f'({escaped_term})', re.IGNORECASE)
            text = pattern.sub(f'<{highlight_tag}>\\1</{highlight_tag}>', text)
        
        return text
    
    def _prepare_fts_query(self, query: str) -> str:
        """
        Prepare user query for FTS5 search.
        
        Handles special characters and applies query optimization for better results.
        
        Args:
            query: Raw user query
            
        Returns:
            FTS5-compatible query string
        """
        # Remove or escape special FTS5 characters
        # FTS5 special chars: " * - ( ) AND OR NOT NEAR
        
        # Simple approach: quote the entire query for exact phrase matching
        # or split into terms for OR matching
        query = query.strip()
        
        if not query:
            return '""'
        
        # If query contains quotes, assume user wants phrase search
        if '"' in query:
            return query
        
        # For multi-word queries, create OR search with individual terms
        # and phrase search for better recall
        terms = query.split()
        
        if len(terms) == 1:
            # Single term - use prefix matching
            return f'"{terms[0]}"*'
        else:
            # Multi-term - combine phrase and individual terms
            phrase_query = f'"{query}"'
            term_queries = [f'"{term}"' for term in terms]
            return f'({phrase_query} OR {" OR ".join(term_queries)})'
    
    async def search_by_topics(
        self,
        topics: List[str],
        channel_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> SearchResults:
        """
        Search videos by extracted topics from AI summaries.
        
        Args:
            topics: List of topics to search for
            channel_id: Filter by specific channel
            limit: Maximum results to return
            offset: Number of results to skip
            
        Returns:
            SearchResults with videos matching the topics
        """
        start_time = datetime.now()
        
        if not topics:
            return SearchResults(
                results=[], total_count=0, page=1, page_size=limit,
                total_pages=0, query="", search_time_ms=0
            )
        
        raw_db_path = self.db_manager.database_url.replace("sqlite+aiosqlite:///", "")
        
        async with aiosqlite.connect(raw_db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Build topic search conditions
            topic_conditions = []
            params = []
            
            for topic in topics:
                topic_conditions.append("v.extracted_topics LIKE ?")
                params.append(f"%{topic}%")
            
            topic_clause = " OR ".join(topic_conditions)
            
            # Build main query
            sql = f"""
                SELECT 
                    v.video_id,
                    v.title,
                    v.channel_id,
                    c.channel_name,
                    v.published_at,
                    v.ai_summary,
                    v.extracted_topics,
                    t.word_count,
                    t.quality_score,
                    'https://youtube.com/watch?v=' || v.video_id as url,
                    0 as rank
                FROM videos v
                JOIN channels c ON v.channel_id = c.channel_id
                LEFT JOIN transcripts t ON v.video_id = t.video_id
                WHERE v.ai_summary IS NOT NULL
                AND v.extracted_topics IS NOT NULL
                AND ({topic_clause})
            """
            
            # Add channel filter
            if channel_id:
                sql += " AND v.channel_id = ?"
                params.append(channel_id)
            
            sql += " ORDER BY v.published_at DESC"
            
            # Get total count
            count_sql = sql.replace(
                "SELECT v.video_id, v.title, v.channel_id, c.channel_name, v.published_at, v.ai_summary, v.extracted_topics, t.word_count, t.quality_score, 'https://youtube.com/watch?v=' || v.video_id as url, 0 as rank FROM",
                "SELECT COUNT(*) as total FROM"
            )
            
            cursor = await db.execute(count_sql, params)
            row = await cursor.fetchone()
            total_count = row['total'] if row else 0
            
            # Get paginated results
            sql += f" LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
            
            results = []
            for row in rows:
                # Create snippet from summary
                snippet = row['ai_summary'][:200] + "..." if row['ai_summary'] else ""
                
                result = SearchResult(
                    video_id=row['video_id'],
                    title=row['title'],
                    channel_id=row['channel_id'],
                    channel_name=row['channel_name'],
                    snippet=snippet,
                    rank=row['rank'],
                    published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                    word_count=row['word_count'],
                    quality_score=row['quality_score'],
                    url=row['url']
                )
                results.append(result)
        
        # Calculate pagination info
        page = (offset // limit) + 1
        total_pages = (total_count + limit - 1) // limit
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResults(
            results=results,
            total_count=total_count,
            page=page,
            page_size=limit,
            total_pages=total_pages,
            query=f"topics: {', '.join(topics)}",
            search_time_ms=search_time
        )
    
    async def get_popular_searches(
        self,
        limit: int = 10,
        days: int = 30
    ) -> List[Tuple[str, int]]:
        """
        Get popular search terms from search history.
        
        Note: This would require implementing search history tracking.
        For now, returns placeholder data.
        
        Args:
            limit: Maximum terms to return
            days: Look back this many days
            
        Returns:
            List of (search_term, frequency) tuples
        """
        # Placeholder implementation - in production you'd track search queries
        return [
            ("artificial intelligence", 45),
            ("machine learning", 38),
            ("web development", 32),
            ("python programming", 28),
            ("data science", 25),
            ("react tutorial", 22),
            ("cryptocurrency", 18),
            ("cloud computing", 15),
            ("cybersecurity", 12),
            ("blockchain", 10)
        ][:limit]
    
    async def get_related_videos(
        self,
        video_id: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Find videos related to the given video based on transcript similarity.
        
        Uses TF-IDF style similarity based on common terms in transcripts.
        
        Args:
            video_id: Source video to find related videos for
            limit: Maximum related videos to return
            
        Returns:
            List of related video search results
        """
        raw_db_path = self.db_manager.database_url.replace("sqlite+aiosqlite:///", "")
        
        async with aiosqlite.connect(raw_db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # First, get the source video's transcript content
            cursor = await db.execute("""
                SELECT content_text, title
                FROM transcripts t
                JOIN videos v ON t.video_id = v.video_id
                WHERE t.video_id = ?
            """, [video_id])
            
            source_row = await cursor.fetchone()
            if not source_row:
                return []
            
            source_content = source_row['content_text']
            source_title = source_row['title']
            
            # Extract key terms (simplified approach)
            # In production, you'd use proper TF-IDF or embeddings
            words = re.findall(r'\b[a-zA-Z]{4,}\b', source_content.lower())
            common_words = set(['this', 'that', 'with', 'have', 'they', 'from', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'some', 'these', 'many', 'then', 'them', 'than', 'like', 'well', 'were'])
            
            # Get most frequent meaningful terms
            word_freq = {}
            for word in words:
                if word not in common_words and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top terms
            top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if not top_terms:
                return []
            
            # Build search query from top terms
            search_terms = [term[0] for term in top_terms[:5]]
            search_query = " OR ".join([f'"{term}"' for term in search_terms])
            
            # Search for related videos
            cursor = await db.execute("""
                SELECT 
                    fts.video_id,
                    v.title,
                    v.channel_id,
                    c.channel_name,
                    v.published_at,
                    t.word_count,
                    t.quality_score,
                    bm25(transcripts_fts) as rank,
                    'https://youtube.com/watch?v=' || fts.video_id as url
                FROM transcripts_fts fts
                JOIN videos v ON fts.video_id = v.video_id
                JOIN channels c ON v.channel_id = c.channel_id
                JOIN transcripts t ON v.video_id = t.video_id
                WHERE transcripts_fts MATCH ?
                AND fts.video_id != ?
                ORDER BY rank
                LIMIT ?
            """, [search_query, video_id, limit])
            
            rows = await cursor.fetchall()
            
            results = []
            for row in rows:
                result = SearchResult(
                    video_id=row['video_id'],
                    title=row['title'],
                    channel_id=row['channel_id'],
                    channel_name=row['channel_name'],
                    rank=row['rank'],
                    published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                    word_count=row['word_count'],
                    quality_score=row['quality_score'],
                    url=row['url']
                )
                results.append(result)
            
            return results


# ========================================
# Utility Functions
# ========================================

async def create_search_service(database_url: str = "sqlite+aiosqlite:///data/yt-dl-sub.db") -> SearchService:
    """Create and initialize a search service."""
    db_manager = DatabaseManager(database_url)
    await db_manager.initialize()
    return SearchService(db_manager)


# ========================================
# CLI Testing Functions
# ========================================

async def test_search_functionality():
    """Test the search functionality with sample data."""
    print("Testing search functionality...")
    
    try:
        # Create search service
        search_service = await create_search_service()
        
        # Test search suggestions
        print("\n=== Testing Search Suggestions ===")
        suggestions = await search_service.get_search_suggestions("machine")
        print(f"Suggestions for 'machine': {suggestions}")
        
        suggestions = await search_service.get_search_suggestions("ai")
        print(f"Suggestions for 'ai': {suggestions}")
        
        # Test popular searches
        print("\n=== Testing Popular Searches ===")
        popular = await search_service.get_popular_searches(5)
        for term, freq in popular:
            print(f"  {term}: {freq} searches")
        
        # Test highlight functionality
        print("\n=== Testing Text Highlighting ===")
        sample_text = "This is a sample text about machine learning and artificial intelligence."
        highlighted = search_service.highlight_matches(sample_text, "machine learning")
        print(f"Original: {sample_text}")
        print(f"Highlighted: {highlighted}")
        
        print("\n✅ Search functionality tests passed!")
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        raise


if __name__ == "__main__":
    # Run tests when executed directly
    asyncio.run(test_search_functionality())