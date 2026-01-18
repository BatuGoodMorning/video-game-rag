"""Chunking strategies for game data.

Implements a hybrid approach:
1. Summary Chunk (~200 tokens): Game name + key facts for quick retrieval
2. Detail Chunks (~512 tokens, 100 token overlap): Full descriptions, gameplay, plot
3. Similarity Chunks: Describe relationships between similar games for multi-hop reasoning
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import defaultdict


class ChunkType(Enum):
    """Type of chunk for different retrieval purposes."""
    SUMMARY = "summary"
    DETAIL = "detail"
    SIMILARITY = "similarity"


@dataclass
class Chunk:
    """A text chunk with metadata for embedding."""
    
    text: str
    chunk_type: ChunkType
    game_name: str
    platform: str
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Metadata for filtering
    genres: list[str] = field(default_factory=list)
    release_date: str = ""
    sales_millions: Optional[float] = None
    developer: str = ""
    
    # Source tracking
    source_section: str = ""  # e.g., "description", "plot", "gameplay", "similarity"
    
    # For similarity chunks: list of related games
    related_games: list[str] = field(default_factory=list)
    
    def to_metadata(self) -> dict:
        """Convert to metadata dict for vector store."""
        # Pinecone doesn't support float in metadata, convert to string
        sales_str = None
        if self.sales_millions is not None:
            sales_str = str(round(self.sales_millions, 2))
        
        return {
            "chunk_type": self.chunk_type.value,
            "game_name": self.game_name,
            "platform": self.platform,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "genres": self.genres,
            "release_date": self.release_date,
            "sales_millions": sales_str,
            "developer": self.developer,
            "source_section": self.source_section,
            "related_games": self.related_games,
        }
    
    @property
    def id(self) -> str:
        """Generate unique ID for this chunk (ASCII only for Pinecone compatibility)."""
        import unicodedata
        import re
        
        # Remove non-ASCII characters, keep only ASCII
        safe_name = unicodedata.normalize('NFKD', self.game_name)
        safe_name = safe_name.encode('ascii', 'ignore').decode('ascii')
        # Replace any remaining non-alphanumeric with underscore
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', safe_name)
        # Remove multiple underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        # Limit length
        safe_name = safe_name[:50]
        
        return f"{safe_name}_{self.platform}_{self.chunk_type.value}_{self.chunk_index}"


class GameChunker:
    """Chunks game data using hybrid strategy including similarity chunks."""
    
    def __init__(
        self,
        summary_max_tokens: int = 200,
        detail_chunk_size: int = 512,
        chunk_overlap: int = 100,
        chars_per_token: float = 4.0,
    ):
        """Initialize chunker with size parameters."""
        self.summary_max_chars = int(summary_max_tokens * chars_per_token)
        self.detail_chunk_chars = int(detail_chunk_size * chars_per_token)
        self.overlap_chars = int(chunk_overlap * chars_per_token)
    
    def create_summary_chunk(self, game: dict) -> Chunk:
        """Create a summary chunk with key game facts."""
        parts = [game["name"]]
        
        # Add genres
        if game.get("genres"):
            parts.append(f"Genres: {', '.join(game['genres'])}")
        
        # Add platform
        parts.append(f"Platform: {game['platform']}")
        
        # Add release date
        if game.get("release_date"):
            parts.append(f"Released: {game['release_date']}")
        
        # Add sales
        if game.get("sales_millions"):
            sales = game['sales_millions']
            try:
                if isinstance(sales, (int, float)):
                    sales_float = float(sales)
                elif isinstance(sales, str):
                    sales_float = float(sales)
                else:
                    sales_float = float(sales)
                parts.append(f"Sales: {sales_float:.1f} million copies")
            except (ValueError, TypeError, AttributeError):
                parts.append(f"Sales: {sales} million copies")
        
        # Add developer/publisher
        if game.get("developer"):
            parts.append(f"Developer: {game['developer']}")
        if game.get("publisher"):
            parts.append(f"Publisher: {game['publisher']}")
        
        # Add truncated description
        if game.get("description"):
            desc = game["description"]
            remaining_chars = self.summary_max_chars - len(" | ".join(parts))
            if remaining_chars > 50:
                truncated_desc = desc[:remaining_chars - 3] + "..." if len(desc) > remaining_chars else desc
                parts.append(truncated_desc)
        
        summary_text = " | ".join(parts)
        
        return Chunk(
            text=summary_text[:self.summary_max_chars],
            chunk_type=ChunkType.SUMMARY,
            game_name=game["name"],
            platform=game["platform"],
            chunk_index=0,
            total_chunks=1,
            genres=game.get("genres", []),
            release_date=game.get("release_date", ""),
            sales_millions=game.get("sales_millions"),
            developer=game.get("developer", ""),
            source_section="summary",
        )
    
    def create_detail_chunks(self, game: dict) -> list[Chunk]:
        """Create detail chunks from game content with overlap."""
        chunks = []
        
        # Combine all detailed content
        sections = []
        
        if game.get("description"):
            sections.append(("description", game["description"]))
        
        if game.get("plot"):
            sections.append(("plot", f"Plot: {game['plot']}"))
        
        if game.get("gameplay"):
            sections.append(("gameplay", f"Gameplay: {game['gameplay']}"))
        
        if game.get("reception"):
            sections.append(("reception", f"Reception: {game['reception']}"))
        
        # Process each section
        chunk_index = 0
        for section_name, content in sections:
            section_chunks = self._split_with_overlap(
                content, 
                self.detail_chunk_chars, 
                self.overlap_chars
            )
            
            for text in section_chunks:
                # Prepend game name for context
                prefixed_text = f"{game['name']}: {text}"
                
                chunk = Chunk(
                    text=prefixed_text,
                    chunk_type=ChunkType.DETAIL,
                    game_name=game["name"],
                    platform=game["platform"],
                    chunk_index=chunk_index,
                    total_chunks=0,
                    genres=game.get("genres", []),
                    release_date=game.get("release_date", ""),
                    sales_millions=game.get("sales_millions"),
                    developer=game.get("developer", ""),
                    source_section=section_name,
                )
                chunks.append(chunk)
                chunk_index += 1
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def create_similarity_chunks(self, games: list[dict]) -> list[Chunk]:
        """Create similarity chunks describing relationships between games.
        
        This enables better multi-hop reasoning by explicitly stating
        which games are similar and why.
        """
        chunks = []
        
        # Group games by genre
        genre_to_games = defaultdict(list)
        for game in games:
            for genre in game.get("genres", []):
                genre_to_games[genre].append(game)
        
        # Group games by developer
        developer_to_games = defaultdict(list)
        for game in games:
            if game.get("developer"):
                developer_to_games[game["developer"]].append(game)
        
        # Group games by platform
        platform_to_games = defaultdict(list)
        for game in games:
            platform_to_games[game["platform"]].append(game)
        
        # For each game, create a similarity chunk
        for game in games:
            similar_games = self._find_similar_games(
                game, games, genre_to_games, developer_to_games
            )
            
            if similar_games:
                similarity_text = self._format_similarity_chunk(game, similar_games)
                related_game_names = [g["name"] for g in similar_games]
                
                chunk = Chunk(
                    text=similarity_text,
                    chunk_type=ChunkType.SIMILARITY,
                    game_name=game["name"],
                    platform=game["platform"],
                    chunk_index=0,
                    total_chunks=1,
                    genres=game.get("genres", []),
                    release_date=game.get("release_date", ""),
                    sales_millions=game.get("sales_millions"),
                    developer=game.get("developer", ""),
                    source_section="similarity",
                    related_games=related_game_names,
                )
                chunks.append(chunk)
        
        return chunks
    
    def _find_similar_games(
        self,
        game: dict,
        all_games: list[dict],
        genre_to_games: dict,
        developer_to_games: dict,
        max_similar: int = 5,
    ) -> list[dict]:
        """Find games similar to the given game based on various criteria."""
        game_name = game["name"]
        game_genres = set(game.get("genres", []))
        game_developer = game.get("developer", "")
        
        # Score each other game
        scores = []
        for other_game in all_games:
            if other_game["name"] == game_name:
                continue
            
            score = 0
            reasons = []
            
            # Genre overlap
            other_genres = set(other_game.get("genres", []))
            genre_overlap = game_genres & other_genres
            if genre_overlap:
                score += len(genre_overlap) * 2
                reasons.append(f"same genre: {', '.join(genre_overlap)}")
            
            # Same developer
            if game_developer and other_game.get("developer") == game_developer:
                score += 3
                reasons.append(f"same developer: {game_developer}")
            
            # Same platform
            if game.get("platform") == other_game.get("platform"):
                score += 1
                reasons.append(f"same platform")
            
            if score > 0:
                scores.append((other_game, score, reasons))
        
        # Sort by score and return top matches
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scores[:max_similar]]
    
    def _format_similarity_chunk(
        self,
        game: dict,
        similar_games: list[dict],
    ) -> str:
        """Format a similarity chunk text."""
        game_name = game["name"]
        game_genres = game.get("genres", [])
        
        parts = [f"{game_name} is similar to the following games:"]
        
        for similar in similar_games:
            similar_name = similar["name"]
            similar_genres = similar.get("genres", [])
            
            # Find shared genres
            shared_genres = set(game_genres) & set(similar_genres)
            
            reason_parts = []
            if shared_genres:
                reason_parts.append(f"shared genres: {', '.join(shared_genres)}")
            if game.get("developer") == similar.get("developer"):
                reason_parts.append(f"same developer: {game.get('developer')}")
            if game.get("platform") == similar.get("platform"):
                reason_parts.append(f"same platform: {game.get('platform')}")
            
            reason_str = "; ".join(reason_parts) if reason_parts else "similar style"
            parts.append(f"- {similar_name} ({reason_str})")
        
        # Add recommendation text
        parts.append(f"\nIf you enjoy {game_name}, you might also like: {', '.join([g['name'] for g in similar_games])}.")
        
        return "\n".join(parts)
    
    def _split_with_overlap(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> list[str]:
        """Split text into chunks with overlap."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                search_start = max(end - 100, start)
                last_period = text.rfind(". ", search_start, end)
                if last_period > search_start:
                    end = last_period + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap
            
            # Avoid infinite loop
            if start <= chunks[-1] if chunks else 0:
                start = end
        
        return chunks
    
    def chunk_game(self, game: dict) -> list[Chunk]:
        """Create summary and detail chunks for a game."""
        chunks = []
        
        # Always create summary chunk
        summary = self.create_summary_chunk(game)
        chunks.append(summary)
        
        # Create detail chunks if there's substantial content
        detail_chunks = self.create_detail_chunks(game)
        chunks.extend(detail_chunks)
        
        return chunks
    
    def chunk_all_games(self, games: list[dict]) -> list[Chunk]:
        """Create summary and detail chunks for all games."""
        all_chunks = []
        
        for game in games:
            game_chunks = self.chunk_game(game)
            all_chunks.extend(game_chunks)
        
        return all_chunks
    
    def get_stats(self, chunks: list[Chunk]) -> dict:
        """Get statistics about chunks."""
        summary_chunks = [c for c in chunks if c.chunk_type == ChunkType.SUMMARY]
        detail_chunks = [c for c in chunks if c.chunk_type == ChunkType.DETAIL]
        similarity_chunks = [c for c in chunks if c.chunk_type == ChunkType.SIMILARITY]
        
        return {
            "total_chunks": len(chunks),
            "summary_chunks": len(summary_chunks),
            "detail_chunks": len(detail_chunks),
            "similarity_chunks": len(similarity_chunks),
            "avg_summary_length": sum(len(c.text) for c in summary_chunks) / len(summary_chunks) if summary_chunks else 0,
            "avg_detail_length": sum(len(c.text) for c in detail_chunks) / len(detail_chunks) if detail_chunks else 0,
            "avg_similarity_length": sum(len(c.text) for c in similarity_chunks) / len(similarity_chunks) if similarity_chunks else 0,
            "unique_games": len(set(c.game_name for c in chunks)),
        }
