"""Wikipedia API client for fetching video game information."""

import re
import time
from dataclasses import dataclass, field
from typing import Optional
import wikipediaapi
import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm


@dataclass
class GameInfo:
    """Structured game information."""

    name: str
    platform: str
    description: str = ""
    release_date: str = ""
    genres: list[str] = field(default_factory=list)
    developer: str = ""
    publisher: str = ""
    metacritic_score: Optional[int] = None
    plot: str = ""
    gameplay: str = ""
    reception: str = ""
    wikipedia_url: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "platform": self.platform,
            "description": self.description,
            "release_date": self.release_date,
            "genres": self.genres,
            "developer": self.developer,
            "publisher": self.publisher,
            "metacritic_score": self.metacritic_score,
            "plot": self.plot,
            "gameplay": self.gameplay,
            "reception": self.reception,
            "wikipedia_url": self.wikipedia_url,
        }


class WikipediaGameFetcher:
    """Fetches video game information from Wikipedia."""

    # Wikipedia list pages for each platform
    PLATFORM_LISTS = {
        "PC": [
            "List of best-selling PC games",
            "List of Game of the Year awards",
        ],
        "PS5": [
            "List of PlayStation 5 games",
        ],
        "Switch": [
            "List of Nintendo Switch games (A–F)",
            "List of Nintendo Switch games (G–P)",
            "List of Nintendo Switch games (Q–Z)",
        ],
    }

    def __init__(self, user_agent: str = "VideoGameRAG/1.0"):
        """Initialize Wikipedia API client."""
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )
        self.http_client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=30.0,
        )

    def _extract_games_from_list_page(
        self, list_page_title: str, limit: int = 100
    ) -> list[str]:
        """Extract game names from a Wikipedia list page."""
        games = []

        # Use MediaWiki API to get links from the page
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": list_page_title,
            "prop": "links",
            "pllimit": "max",
            "format": "json",
        }

        try:
            response = self.http_client.get(api_url, params=params)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    continue
                links = page_data.get("links", [])
                for link in links:
                    title = link.get("title", "")
                    # Filter out non-game pages
                    if self._is_likely_game_page(title):
                        games.append(title)
                        if len(games) >= limit:
                            break
        except Exception as e:
            print(f"Error fetching list page {list_page_title}: {e}")

        return games[:limit]

    def _is_likely_game_page(self, title: str) -> bool:
        """Check if a Wikipedia title is likely a game page."""
        # Skip common non-game patterns
        skip_patterns = [
            r"^List of",
            r"^Category:",
            r"^Template:",
            r"^Wikipedia:",
            r"^Help:",
            r"^File:",
            r"^Portal:",
            r"\(company\)$",
            r"\(developer\)$",
            r"\(publisher\)$",
            r"\(console\)$",
            r"\(platform\)$",
            r"^PlayStation",
            r"^Nintendo",
            r"^Xbox",
            r"^Sega",
            r"^Atari",
        ]

        for pattern in skip_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return False

        return True

    def _extract_metacritic_score(self, text: str) -> Optional[int]:
        """Extract Metacritic score from text."""
        # Common patterns for Metacritic scores
        patterns = [
            r"Metacritic[:\s]+(\d{1,3})/100",
            r"Metacritic[:\s]+(\d{1,3})%",
            r"(\d{1,3})/100[^\d]*Metacritic",
            r"Metacritic.*?(\d{1,3})/100",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 100:
                    return score
        return None

    def _extract_release_date(self, text: str) -> str:
        """Extract release date from text."""
        # Common date patterns
        patterns = [
            r"released[:\s]+([A-Za-z]+ \d{1,2}, \d{4})",
            r"release date[:\s]+([A-Za-z]+ \d{1,2}, \d{4})",
            r"(\d{4})",  # Just year as fallback
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_genres(self, text: str) -> list[str]:
        """Extract game genres from text."""
        genre_keywords = [
            "action",
            "adventure",
            "role-playing",
            "RPG",
            "shooter",
            "first-person",
            "third-person",
            "platformer",
            "puzzle",
            "strategy",
            "simulation",
            "sports",
            "racing",
            "fighting",
            "horror",
            "survival",
            "open world",
            "sandbox",
            "stealth",
            "MMORPG",
            "roguelike",
            "metroidvania",
            "visual novel",
            "JRPG",
        ]

        found_genres = []
        text_lower = text.lower()

        for genre in genre_keywords:
            if genre.lower() in text_lower:
                found_genres.append(genre)

        return found_genres[:5]  # Limit to 5 genres

    def fetch_game_details(self, game_name: str, platform: str) -> Optional[GameInfo]:
        """Fetch detailed information about a specific game."""
        try:
            page = self.wiki.page(game_name)

            if not page.exists():
                return None

            # Get full text
            full_text = page.text

            # Extract sections
            sections = {}
            for section in page.sections:
                sections[section.title.lower()] = section.text

            # Build GameInfo
            game = GameInfo(
                name=game_name,
                platform=platform,
                description=page.summary[:1000] if page.summary else "",
                wikipedia_url=page.fullurl,
            )

            # Extract structured data
            game.release_date = self._extract_release_date(full_text)
            game.metacritic_score = self._extract_metacritic_score(full_text)
            game.genres = self._extract_genres(full_text)

            # Get specific sections
            for key in ["plot", "story", "synopsis"]:
                if key in sections:
                    game.plot = sections[key][:2000]
                    break

            for key in ["gameplay", "game mechanics", "mechanics"]:
                if key in sections:
                    game.gameplay = sections[key][:2000]
                    break

            for key in ["reception", "critical reception", "reviews"]:
                if key in sections:
                    game.reception = sections[key][:2000]
                    break

            # Try to extract developer/publisher from infobox
            game.developer = self._extract_infobox_field(game_name, "developer")
            game.publisher = self._extract_infobox_field(game_name, "publisher")

            return game

        except Exception as e:
            print(f"Error fetching {game_name}: {e}")
            return None

    def _extract_infobox_field(self, page_title: str, field: str) -> str:
        """Extract a field from Wikipedia infobox using API."""
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "revisions",
            "rvprop": "content",
            "rvsection": "0",
            "format": "json",
        }

        try:
            response = self.http_client.get(api_url, params=params)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    continue
                revisions = page_data.get("revisions", [])
                if revisions:
                    content = revisions[0].get("*", "")
                    # Simple regex to find infobox field
                    pattern = rf"\|\s*{field}\s*=\s*([^\n\|]+)"
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        # Clean up wiki markup
                        value = match.group(1).strip()
                        value = re.sub(r"\[\[([^\]|]+)\|?[^\]]*\]\]", r"\1", value)
                        value = re.sub(r"\{\{[^}]+\}\}", "", value)
                        return value.strip()
        except Exception:
            pass

        return ""

    def fetch_games_for_platform(
        self, platform: str, limit: int = 100, delay: float = 0.5
    ) -> list[GameInfo]:
        """Fetch games for a specific platform."""
        if platform not in self.PLATFORM_LISTS:
            raise ValueError(f"Unknown platform: {platform}")

        print(f"\nFetching games for {platform}...")

        # Get game names from list pages
        game_names = []
        for list_page in self.PLATFORM_LISTS[platform]:
            names = self._extract_games_from_list_page(list_page, limit=limit * 2)
            game_names.extend(names)

        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in game_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        game_names = unique_names[:limit]
        print(f"Found {len(game_names)} potential games")

        # Fetch details for each game
        games = []
        for name in tqdm(game_names, desc=f"Fetching {platform} games"):
            game = self.fetch_game_details(name, platform)
            if game and game.description:  # Only keep games with descriptions
                games.append(game)
            time.sleep(delay)  # Rate limiting

            if len(games) >= limit:
                break

        print(f"Successfully fetched {len(games)} games for {platform}")
        return games

    def fetch_all_platforms(
        self, games_per_platform: int = 100, delay: float = 0.5
    ) -> dict[str, list[GameInfo]]:
        """Fetch games for all platforms."""
        all_games = {}

        for platform in self.PLATFORM_LISTS.keys():
            games = self.fetch_games_for_platform(
                platform, limit=games_per_platform, delay=delay
            )
            all_games[platform] = games

        total = sum(len(games) for games in all_games.values())
        print(f"\nTotal games fetched: {total}")

        return all_games

    def close(self):
        """Close HTTP client."""
        self.http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

