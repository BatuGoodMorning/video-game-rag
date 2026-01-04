#!/usr/bin/env python3
"""Script to fetch game data from Wikipedia best-selling lists."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data import WikipediaGameFetcher, GameDataProcessor


def main():
    """Fetch and process game data from best-selling lists."""
    print("=" * 60)
    print("Video Game RAG - Data Fetcher (Best-Selling Lists)")
    print("=" * 60)
    print("\nData sources:")
    print("  - List of best-selling PC games")
    print("  - List of best-selling PlayStation 4 video games")
    print("  - List of best-selling PlayStation 5 video games")
    print("  - List of best-selling Nintendo Switch video games")
    print()

    # Initialize fetcher and processor
    processor = GameDataProcessor(config.DATA_DIR)

    with WikipediaGameFetcher() as fetcher:
        # Fetch games for all platforms
        games_by_platform = fetcher.fetch_all_platforms(delay=0.3)

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
        print(f"With sales data: {stats['games_with_sales']}")
        print(f"Total sales: {stats['total_sales_millions']:.1f} million copies")
        print(f"With plot summary: {stats['games_with_plot']}")
        print(f"With gameplay info: {stats['games_with_gameplay']}")
        print(f"Top genres: {dict(sorted(stats['genres'].items(), key=lambda x: -x[1])[:10])}")
        
        # Print sample games
        print("\n" + "=" * 60)
        print("Sample Games (first 5)")
        print("=" * 60)
        for game in processed_games[:5]:
            sales = f"{game['sales_millions']:.1f}M" if game.get('sales_millions') else "N/A"
            print(f"  - {game['name']} ({game['platform']}) - Sales: {sales}")


if __name__ == "__main__":
    main()
