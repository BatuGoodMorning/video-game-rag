"""Data fetching and processing modules."""

from .wikipedia_client import WikipediaGameFetcher
from .processor import GameDataProcessor

__all__ = ["WikipediaGameFetcher", "GameDataProcessor"]

