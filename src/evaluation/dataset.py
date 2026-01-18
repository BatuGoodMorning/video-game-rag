"""Synthetic dataset generation for RAG evaluation.

Generates test queries from game data using LLM to create
realistic question-answer pairs for evaluation.
"""

import json
import random
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.config import config


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""
    
    query: str
    query_type: str  # "simple", "comparison", "recommendation"
    relevant_games: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "relevant_games": self.relevant_games,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "expected_keywords": self.expected_keywords,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EvalQuery":
        return cls(
            query=data["query"],
            query_type=data["query_type"],
            relevant_games=data.get("relevant_games", []),
            relevant_chunk_ids=data.get("relevant_chunk_ids", []),
            expected_keywords=data.get("expected_keywords", []),
            metadata=data.get("metadata", {}),
        )


class SyntheticDatasetGenerator:
    """Generates synthetic evaluation datasets from game data."""
    
    # Templates for different query types
    SIMPLE_TEMPLATES = [
        "What is {game} about?",
        "Tell me about {game}",
        "What genre is {game}?",
        "Who developed {game}?",
        "When was {game} released?",
        "What platforms is {game} available on?",
        "How well did {game} sell?",
        "What is the gameplay like in {game}?",
    ]
    
    COMPARISON_TEMPLATES = [
        "What games are similar to {game}?",
        "Recommend games like {game}",
        "If I liked {game}, what else should I play?",
        "What are alternatives to {game}?",
        "Games similar to {game} but on {platform}?",
    ]
    
    COMPLEX_TEMPLATES = [
        "Compare {game1} and {game2}",
        "Which is better: {game1} or {game2}?",
        "What do {game1} and {game2} have in common?",
        "Differences between {game1} and {game2}?",
    ]
    
    def __init__(
        self,
        games: list[dict],
        google_api_key: Optional[str] = None,
        use_llm: bool = True,
    ):
        """Initialize the generator.
        
        Args:
            games: List of game dictionaries
            google_api_key: API key for LLM-based generation
            use_llm: Whether to use LLM for more diverse queries
        """
        self.games = games
        self.use_llm = use_llm
        
        if use_llm:
            api_key = google_api_key or config.GOOGLE_API_KEY
            if api_key:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.8,
                )
            else:
                self.use_llm = False
                self.llm = None
        else:
            self.llm = None
        
        # Index games by various attributes
        self._index_games()
    
    def _index_games(self):
        """Create indexes for quick lookup."""
        self.games_by_genre = {}
        self.games_by_platform = {}
        self.games_by_developer = {}
        
        for game in self.games:
            # By genre
            for genre in game.get("genres", []):
                if genre not in self.games_by_genre:
                    self.games_by_genre[genre] = []
                self.games_by_genre[genre].append(game)
            
            # By platform
            platform = game.get("platform", "Unknown")
            if platform not in self.games_by_platform:
                self.games_by_platform[platform] = []
            self.games_by_platform[platform].append(game)
            
            # By developer
            developer = game.get("developer", "")
            if developer:
                if developer not in self.games_by_developer:
                    self.games_by_developer[developer] = []
                self.games_by_developer[developer].append(game)
    
    def generate_simple_queries(
        self,
        num_queries: int = 50,
    ) -> list[EvalQuery]:
        """Generate simple factual queries about specific games."""
        queries = []
        sampled_games = random.sample(
            self.games, 
            min(num_queries, len(self.games))
        )
        
        for game in sampled_games:
            template = random.choice(self.SIMPLE_TEMPLATES)
            query_text = template.format(game=game["name"])
            
            # Determine expected keywords based on template
            expected_keywords = [game["name"]]
            if "genre" in template.lower():
                expected_keywords.extend(game.get("genres", []))
            if "developer" in template.lower():
                if game.get("developer"):
                    expected_keywords.append(game["developer"])
            
            query = EvalQuery(
                query=query_text,
                query_type="simple",
                relevant_games=[game["name"]],
                expected_keywords=expected_keywords,
                metadata={"template": template, "source_game": game["name"]},
            )
            queries.append(query)
        
        return queries
    
    def generate_comparison_queries(
        self,
        num_queries: int = 30,
    ) -> list[EvalQuery]:
        """Generate comparison/recommendation queries."""
        queries = []
        sampled_games = random.sample(
            self.games,
            min(num_queries, len(self.games))
        )
        
        for game in sampled_games:
            template = random.choice(self.COMPARISON_TEMPLATES)
            
            # Handle platform-specific template
            if "{platform}" in template:
                platforms = list(self.games_by_platform.keys())
                platform = random.choice(platforms)
                query_text = template.format(game=game["name"], platform=platform)
            else:
                query_text = template.format(game=game["name"])
            
            # Find games that should be relevant (same genre or developer)
            similar_games = []
            for genre in game.get("genres", []):
                if genre in self.games_by_genre:
                    for similar in self.games_by_genre[genre]:
                        if similar["name"] != game["name"]:
                            similar_games.append(similar["name"])
            
            # Deduplicate and limit
            similar_games = list(set(similar_games))[:5]
            
            query = EvalQuery(
                query=query_text,
                query_type="comparison",
                relevant_games=[game["name"]] + similar_games,
                expected_keywords=[game["name"]] + similar_games[:2],
                metadata={"template": template, "source_game": game["name"]},
            )
            queries.append(query)
        
        return queries
    
    def generate_complex_queries(
        self,
        num_queries: int = 20,
    ) -> list[EvalQuery]:
        """Generate complex multi-game comparison queries."""
        queries = []
        
        for _ in range(min(num_queries, len(self.games) // 2)):
            # Pick two games from the same genre for meaningful comparison
            genres = list(self.games_by_genre.keys())
            genre = random.choice(genres)
            genre_games = self.games_by_genre.get(genre, [])
            
            if len(genre_games) >= 2:
                game1, game2 = random.sample(genre_games, 2)
                template = random.choice(self.COMPLEX_TEMPLATES)
                query_text = template.format(
                    game1=game1["name"],
                    game2=game2["name"]
                )
                
                query = EvalQuery(
                    query=query_text,
                    query_type="complex",
                    relevant_games=[game1["name"], game2["name"]],
                    expected_keywords=[game1["name"], game2["name"]],
                    metadata={
                        "template": template,
                        "shared_genre": genre,
                    },
                )
                queries.append(query)
        
        return queries
    
    def generate_llm_queries(
        self,
        num_queries: int = 20,
    ) -> list[EvalQuery]:
        """Generate more diverse queries using LLM."""
        if not self.use_llm or not self.llm:
            return []
        
        queries = []
        sampled_games = random.sample(
            self.games,
            min(num_queries, len(self.games))
        )
        
        for game in sampled_games:
            prompt = f"""Generate a natural question a user might ask about the video game "{game['name']}".
The game is a {', '.join(game.get('genres', ['game']))} on {game.get('platform', 'PC')}.

Respond with ONLY the question, nothing else. Make it conversational and natural.
Examples of good questions:
- "What makes this game special?"
- "Is this game worth playing?"
- "What's the story about?"
"""
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                query_text = response.content.strip().strip('"')
                
                query = EvalQuery(
                    query=query_text,
                    query_type="simple",
                    relevant_games=[game["name"]],
                    expected_keywords=[game["name"]],
                    metadata={"source": "llm", "source_game": game["name"]},
                )
                queries.append(query)
            except Exception as e:
                continue  # Skip on error
        
        return queries
    
    def generate_full_dataset(
        self,
        simple: int = 50,
        comparison: int = 30,
        complex: int = 20,
        llm_generated: int = 20,
    ) -> list[EvalQuery]:
        """Generate a complete evaluation dataset.
        
        Args:
            simple: Number of simple queries
            comparison: Number of comparison queries
            complex: Number of complex queries
            llm_generated: Number of LLM-generated queries
            
        Returns:
            Combined list of all queries
        """
        all_queries = []
        
        all_queries.extend(self.generate_simple_queries(simple))
        all_queries.extend(self.generate_comparison_queries(comparison))
        all_queries.extend(self.generate_complex_queries(complex))
        
        if self.use_llm:
            all_queries.extend(self.generate_llm_queries(llm_generated))
        
        # Shuffle
        random.shuffle(all_queries)
        
        return all_queries
    
    def save_dataset(
        self,
        queries: list[EvalQuery],
        path: Path,
    ):
        """Save dataset to JSON file."""
        data = [q.to_dict() for q in queries]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_dataset(
        self,
        path: Path,
    ) -> list[EvalQuery]:
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvalQuery.from_dict(d) for d in data]

