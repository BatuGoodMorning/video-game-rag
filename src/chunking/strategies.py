"""Chunking strategies for game data.

Implements a hybrid approach:
1. Summary Chunk (~200 tokens): Game name + key facts for quick retrieval
2. Detail Chunks (~512 tokens, 100 token overlap): Full descriptions, gameplay, plot
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ChunkType(Enum):
    """Type of chunk for different retrieval purposes."""
    SUMMARY = "summary"
    DETAIL = "detail"


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
    source_section: str = ""  # e.g., "description", "plot", "gameplay"
    
    def to_metadata(self) -> dict:
        """Convert to metadata dict for vector store."""
        return {
            "chunk_type": self.chunk_type.value,
            "game_name": self.game_name,
            "platform": self.platform,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "genres": self.genres,
            "release_date": self.release_date,
            "sales_millions": self.sales_millions,
            "developer": self.developer,
            "source_section": self.source_section,
        }
    
    @property
    def id(self) -> str:
        """Generate unique ID for this chunk."""
        safe_name = self.game_name.replace(" ", "_").replace("/", "_")[:50]
        return f"{safe_name}_{self.platform}_{self.chunk_type.value}_{self.chunk_index}"


class GameChunker:
    """Chunks game data using hybrid strategy."""
    
    def __init__(
        self,
        summary_max_tokens: int = 200,
        detail_chunk_size: int = 512,
        chunk_overlap: int = 100,
        chars_per_token: float = 4.0,  # Approximate
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
        
        # Add sales (instead of Metacritic)
        if game.get("sales_millions"):
            parts.append(f"Sales: {game['sales_millions']:.1f} million copies")
        
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
                    total_chunks=0,  # Will be set later
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
                # Look for sentence end near chunk boundary
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
        """Create all chunks for a game."""
        chunks = []
        
        # Always create summary chunk
        summary = self.create_summary_chunk(game)
        chunks.append(summary)
        
        # Create detail chunks if there's substantial content
        detail_chunks = self.create_detail_chunks(game)
        chunks.extend(detail_chunks)
        
        return chunks
    
    def chunk_all_games(self, games: list[dict]) -> list[Chunk]:
        """Create chunks for all games."""
        all_chunks = []
        
        for game in games:
            game_chunks = self.chunk_game(game)
            all_chunks.extend(game_chunks)
        
        return all_chunks
    
    def get_stats(self, chunks: list[Chunk]) -> dict:
        """Get statistics about chunks."""
        summary_chunks = [c for c in chunks if c.chunk_type == ChunkType.SUMMARY]
        detail_chunks = [c for c in chunks if c.chunk_type == ChunkType.DETAIL]
        
        return {
            "total_chunks": len(chunks),
            "summary_chunks": len(summary_chunks),
            "detail_chunks": len(detail_chunks),
            "avg_summary_length": sum(len(c.text) for c in summary_chunks) / len(summary_chunks) if summary_chunks else 0,
            "avg_detail_length": sum(len(c.text) for c in detail_chunks) / len(detail_chunks) if detail_chunks else 0,
            "unique_games": len(set(c.game_name for c in chunks)),
        }
