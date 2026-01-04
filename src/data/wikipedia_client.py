"""Wikipedia API client for fetching video game information from best-selling lists.

Uses table parsing from best-selling game lists for high-quality data:
- https://en.wikipedia.org/wiki/List_of_best-selling_Nintendo_Switch_video_games
- https://en.wikipedia.org/wiki/List_of_best-selling_PC_games
- https://en.wikipedia.org/wiki/List_of_best-selling_PlayStation_4_video_games
- https://en.wikipedia.org/wiki/List_of_best-selling_PlayStation_5_video_games
"""

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
    sales_millions: Optional[float] = None  # Sales in millions
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
            "sales_millions": self.sales_millions,
            "plot": self.plot,
            "gameplay": self.gameplay,
            "reception": self.reception,
            "wikipedia_url": self.wikipedia_url,
        }


class WikipediaGameFetcher:
    """Fetches video game information from Wikipedia best-selling lists."""

    # Best-selling game list pages for each platform
    PLATFORM_LISTS = {
        "PC": ["List of best-selling PC games"],
        "PlayStation": [
            "List of best-selling PlayStation 4 video games",
            "List of best-selling PlayStation 5 video games",
        ],
        "Switch": ["List of best-selling Nintendo Switch video games"],
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

    def _fetch_page_html(self, page_title: str) -> Optional[str]:
        """Fetch HTML content of a Wikipedia page."""
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "parse",
            "page": page_title,
            "format": "json",
            "prop": "text",
        }

        try:
            response = self.http_client.get(api_url, params=params)
            data = response.json()
            
            if "parse" in data and "text" in data["parse"]:
                return data["parse"]["text"]["*"]
        except Exception as e:
            print(f"Error fetching HTML for {page_title}: {e}")
        
        return None

    def _parse_sales_number(self, text: str) -> Optional[float]:
        """Parse sales number from text (e.g., '45.33 million' -> 45.33)."""
        if not text:
            return None
        
        text = text.strip().lower()
        
        # Remove references like [1], [2]
        text = re.sub(r'\[.*?\]', '', text)
        
        # Pattern for numbers with million/billion
        patterns = [
            r'([\d,.]+)\s*million',
            r'([\d,.]+)\s*m\b',
            r'([\d,.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    num_str = match.group(1).replace(',', '')
                    return float(num_str)
                except ValueError:
                    continue
        
        return None

    def _clean_text(self, text: str) -> str:
        """Clean text by removing wiki markup and references."""
        if not text:
            return ""
        
        # Remove reference markers
        text = re.sub(r'\[.*?\]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()

    def _extract_game_link(self, cell) -> tuple[str, str]:
        """Extract game name and Wikipedia URL from a table cell."""
        link = cell.find('a', href=True)
        if link:
            href = link.get('href', '')
            title = link.get('title', '') or link.get_text(strip=True)
            
            if href.startswith('/wiki/') and ':' not in href:
                url = f"https://en.wikipedia.org{href}"
                name = self._clean_text(title)
                return name, url
        
        # Fallback to cell text
        return self._clean_text(cell.get_text()), ""

    def _parse_best_selling_table(
        self, html: str, platform: str
    ) -> list[dict]:
        """Parse best-selling games table from HTML."""
        games = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all wikitables
        tables = soup.find_all('table', class_='wikitable')
        
        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue
            
            # Get header row to understand column structure
            header_row = rows[0]
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
            
            # Find relevant column indices
            name_idx = None
            sales_idx = None
            developer_idx = None
            publisher_idx = None
            date_idx = None
            
            for i, header in enumerate(headers):
                if 'title' in header or 'game' in header or 'name' in header:
                    name_idx = i
                elif 'copies' in header or 'sales' in header or 'sold' in header:
                    sales_idx = i
                elif 'developer' in header:
                    developer_idx = i
                elif 'publisher' in header:
                    publisher_idx = i
                elif 'release' in header or 'date' in header:
                    date_idx = i
            
            # If no name column found, assume first column
            if name_idx is None:
                name_idx = 0
            
            # Parse data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) <= name_idx:
                    continue
                
                # Extract game name and URL
                name, url = self._extract_game_link(cells[name_idx])
                
                if not name or len(name) < 2:
                    continue
                
                # Skip if it looks like a header or category
                if name.lower() in ['total', 'notes', 'references', 'see also']:
                    continue
                
                game_data = {
                    "name": name,
                    "platform": platform,
                    "wikipedia_url": url,
                    "sales_millions": None,
                    "developer": "",
                    "publisher": "",
                    "release_date": "",
                }
                
                # Extract sales
                if sales_idx is not None and sales_idx < len(cells):
                    sales_text = cells[sales_idx].get_text()
                    game_data["sales_millions"] = self._parse_sales_number(sales_text)
                
                # Extract developer
                if developer_idx is not None and developer_idx < len(cells):
                    game_data["developer"] = self._clean_text(cells[developer_idx].get_text())
                
                # Extract publisher
                if publisher_idx is not None and publisher_idx < len(cells):
                    game_data["publisher"] = self._clean_text(cells[publisher_idx].get_text())
                
                # Extract release date
                if date_idx is not None and date_idx < len(cells):
                    game_data["release_date"] = self._clean_text(cells[date_idx].get_text())
                
                games.append(game_data)
        
        return games

    def _extract_genres(self, text: str) -> list[str]:
        """Extract game genres from text."""
        genre_keywords = [
            "action", "adventure", "role-playing", "RPG", "shooter",
            "first-person", "third-person", "platformer", "puzzle",
            "strategy", "simulation", "sports", "racing", "fighting",
            "horror", "survival", "open world", "sandbox", "stealth",
            "MMORPG", "roguelike", "metroidvania", "visual novel", "JRPG",
            "battle royale", "party", "rhythm", "card game",
        ]

        found_genres = []
        text_lower = text.lower()

        for genre in genre_keywords:
            if genre.lower() in text_lower:
                found_genres.append(genre)

        return found_genres[:5]

    def _extract_release_date(self, text: str) -> str:
        """Extract release date from text."""
        patterns = [
            r'released[:\s]+([A-Za-z]+ \d{1,2}, \d{4})',
            r'release date[:\s]+([A-Za-z]+ \d{1,2}, \d{4})',
            r'(\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def fetch_game_details(self, game_data: dict) -> Optional[GameInfo]:
        """Fetch detailed information about a specific game from its Wikipedia page."""
        try:
            name = game_data["name"]
            page = self.wiki.page(name)

            if not page.exists():
                # Try with "(video game)" suffix
                page = self.wiki.page(f"{name} (video game)")
                if not page.exists():
                    # Return basic info from table
                    return GameInfo(
                        name=name,
                        platform=game_data["platform"],
                        sales_millions=game_data.get("sales_millions"),
                        developer=game_data.get("developer", ""),
                        publisher=game_data.get("publisher", ""),
                        release_date=game_data.get("release_date", ""),
                        wikipedia_url=game_data.get("wikipedia_url", ""),
                    )

            # Get full text
            full_text = page.text

            # Extract sections
            sections = {}
            for section in page.sections:
                sections[section.title.lower()] = section.text

            # Build GameInfo
            game = GameInfo(
                name=name,
                platform=game_data["platform"],
                description=page.summary[:1500] if page.summary else "",
                wikipedia_url=page.fullurl or game_data.get("wikipedia_url", ""),
                sales_millions=game_data.get("sales_millions"),
                developer=game_data.get("developer", ""),
                publisher=game_data.get("publisher", ""),
            )

            # Extract or use provided release date
            if game_data.get("release_date"):
                game.release_date = game_data["release_date"]
            else:
                game.release_date = self._extract_release_date(full_text)
            
            # Extract genres
            game.genres = self._extract_genres(full_text)

            # Get specific sections
            for key in ["plot", "story", "synopsis", "premise"]:
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

            # Try to extract developer/publisher from infobox if not already set
            if not game.developer:
                game.developer = self._extract_infobox_field(name, "developer")
            if not game.publisher:
                game.publisher = self._extract_infobox_field(name, "publisher")

            return game

        except Exception as e:
            print(f"Error fetching details for {game_data['name']}: {e}")
            # Return basic info
            return GameInfo(
                name=game_data["name"],
                platform=game_data["platform"],
                sales_millions=game_data.get("sales_millions"),
                developer=game_data.get("developer", ""),
                publisher=game_data.get("publisher", ""),
                wikipedia_url=game_data.get("wikipedia_url", ""),
            )

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
                    pattern = rf"\|\s*{field}\s*=\s*([^\n\|]+)"
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        value = re.sub(r"\[\[([^\]|]+)\|?[^\]]*\]\]", r"\1", value)
                        value = re.sub(r"\{\{[^}]+\}\}", "", value)
                        return value.strip()[:100]
        except Exception:
            pass

        return ""

    def fetch_games_for_platform(
        self, platform: str, delay: float = 0.5
    ) -> list[GameInfo]:
        """Fetch games for a specific platform from best-selling lists."""
        if platform not in self.PLATFORM_LISTS:
            raise ValueError(f"Unknown platform: {platform}")

        print(f"\nFetching best-selling games for {platform}...")

        all_table_games = []
        
        # Parse all list pages for this platform
        for list_page in self.PLATFORM_LISTS[platform]:
            print(f"  Parsing: {list_page}")
            html = self._fetch_page_html(list_page)
            
            if html:
                table_games = self._parse_best_selling_table(html, platform)
                all_table_games.extend(table_games)
                print(f"  Found {len(table_games)} games in table")
        
        # Remove duplicates by name
        seen = set()
        unique_games = []
        for game in all_table_games:
            if game["name"] not in seen:
                seen.add(game["name"])
                unique_games.append(game)
        
        print(f"Total unique games found: {len(unique_games)}")

        # Fetch details for each game
        games = []
        for game_data in tqdm(unique_games, desc=f"Fetching {platform} details"):
            game = self.fetch_game_details(game_data)
            if game and (game.description or game.sales_millions):
                games.append(game)
            time.sleep(delay)

        print(f"Successfully fetched {len(games)} games for {platform}")
        return games

    def fetch_all_platforms(
        self, delay: float = 0.5
    ) -> dict[str, list[GameInfo]]:
        """Fetch games for all platforms."""
        all_games = {}

        for platform in self.PLATFORM_LISTS.keys():
            games = self.fetch_games_for_platform(platform, delay=delay)
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
