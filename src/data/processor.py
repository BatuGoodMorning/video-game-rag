"""Data processing module for cleaning and preparing game data."""

import json
import re
from pathlib import Path
from typing import Optional

from .wikipedia_client import GameInfo


class GameDataProcessor:
    """Processes and cleans game data for embedding."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize processor with data directory."""
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Clean text by removing wiki markup and normalizing whitespace."""
        if not text:
            return ""

        # Remove wiki-style references like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def process_game(self, game: GameInfo) -> dict:
        """Process a single game's data."""
        return {
            "name": game.name.strip(),
            "platform": game.platform,
            "description": self.clean_text(game.description),
            "release_date": game.release_date,
            "genres": game.genres,
            "developer": game.developer.strip() if game.developer else "",
            "publisher": game.publisher.strip() if game.publisher else "",
            "sales_millions": game.sales_millions,
            "plot": self.clean_text(game.plot),
            "gameplay": self.clean_text(game.gameplay),
            "reception": self.clean_text(game.reception),
            "wikipedia_url": game.wikipedia_url,
        }

    def process_all_games(
        self, games_by_platform: dict[str, list[GameInfo]]
    ) -> list[dict]:
        """Process all games from all platforms."""
        processed = []

        for platform, games in games_by_platform.items():
            for game in games:
                processed_game = self.process_game(game)
                # Keep games with either description or sales data
                if processed_game["description"] or processed_game["sales_millions"]:
                    processed.append(processed_game)

        return processed

    def save_raw_data(
        self, games_by_platform: dict[str, list[GameInfo]], filename: str = "games_raw.json"
    ):
        """Save raw game data to JSON."""
        output_path = self.data_dir / "raw" / filename

        # Convert to serializable format
        data = {}
        for platform, games in games_by_platform.items():
            data[platform] = [game.to_dict() for game in games]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved raw data to {output_path}")
        return output_path

    def save_processed_data(
        self, processed_games: list[dict], filename: str = "games_processed.json"
    ):
        """Save processed game data to JSON."""
        output_path = self.data_dir / "processed" / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_games, f, indent=2, ensure_ascii=False)

        print(f"Saved processed data to {output_path}")
        return output_path

    def load_processed_data(
        self, filename: str = "games_processed.json"
    ) -> list[dict]:
        """Load processed game data from JSON."""
        input_path = self.data_dir / "processed" / filename

        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_stats(self, processed_games: list[dict]) -> dict:
        """Get statistics about the processed data."""
        stats = {
            "total_games": len(processed_games),
            "platforms": {},
            "games_with_sales": 0,
            "games_with_plot": 0,
            "games_with_gameplay": 0,
            "total_sales_millions": 0,
            "genres": {},
        }

        for game in processed_games:
            # Platform counts
            platform = game["platform"]
            stats["platforms"][platform] = stats["platforms"].get(platform, 0) + 1

            # Content counts
            if game.get("sales_millions"):
                stats["games_with_sales"] += 1
                stats["total_sales_millions"] += game["sales_millions"]
            if game.get("plot"):
                stats["games_with_plot"] += 1
            if game.get("gameplay"):
                stats["games_with_gameplay"] += 1

            # Genre counts
            for genre in game.get("genres", []):
                stats["genres"][genre] = stats["genres"].get(genre, 0) + 1

        return stats
