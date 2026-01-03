#!/usr/bin/env python3
"""Script to fetch game data from Wikipedia."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data import WikipediaGameFetcher, GameDataProcessor


def main():
    """Fetch and process game data."""
    print("=" * 60)
    print("Video Game RAG - Data Fetcher")
    print("=" * 60)

    # Initialize fetcher and processor
    processor = GameDataProcessor(config.DATA_DIR)

    with WikipediaGameFetcher() as fetcher:
        # Fetch games for all platforms
        games_by_platform = fetcher.fetch_all_platforms(
            games_per_platform=config.GAMES_PER_PLATFORM,
            delay=0.5,  # Rate limiting
        )

        # Save raw data
        processor.save_raw_data(games_by_platform)

        # Process and save
        processed_games = processor.process_all_games(games_by_platform)
        processor.save_processed_data(processed_games)

        # Print stats
        stats = processor.get_stats(processed_games)
        print("\n" + "=" * 60)
        print("Data Collection Stats")
        print("=" * 60)
        print(f"Total games: {stats['total_games']}")
        print(f"By platform: {stats['platforms']}")
        print(f"With Metacritic score: {stats['games_with_metacritic']}")
        print(f"With plot summary: {stats['games_with_plot']}")
        print(f"With gameplay info: {stats['games_with_gameplay']}")
        print(f"Top genres: {dict(sorted(stats['genres'].items(), key=lambda x: -x[1])[:10])}")


if __name__ == "__main__":
    main()

