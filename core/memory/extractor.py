"""
MemoryExtractor — Novel Video Factory v4
LLM-based extraction of story entities for the knowledge store.
"""
import json
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class MemoryExtractor:
    def __init__(self, llm_adapter, config: dict = None):
        self.llm = llm_adapter
        self.config = config or {}

    def _gen(self, prompt: str, system: str, temperature: float = 0.1) -> str:
        return self.llm.generate(prompt, system_prompt=system, temperature=temperature)

    def _gen_json(self, prompt: str, system: str, temperature: float = 0.1) -> str:
        return self.llm.generate_json(prompt, system_prompt=system, temperature=temperature)

    def extract_all(self, text: str, existing_characters: list = None,
                    existing_relationships: list = None) -> Dict:
        """Extract all entity types from a text chunk in one structured call."""
        existing_names = [c.get("canonical_name", "") for c in (existing_characters or [])]
        known = ", ".join(existing_names) if existing_names else "none yet"

        system = (
            "You are a visual character extractor for a Korean manhwa AI pipeline. "
            "Extract ALL named entities AND narrative events from the following text. "
            f"Known characters already in the database: [{known}]. "
            "For new characters, extract static visual DNA as booru-style tags. "
            "For ALL characters (new and existing), extract their CURRENT dynamic state (outfit, injuries, emotion). "
            "Output ONLY valid JSON with this structure:\n"
            '{"characters": [{"canonical_name": "Name", "visual_dna": {'
            '"subject": "1boy or 1girl", "age": "e.g. 20 years old, teenager, ancient", '
            '"hair": "black short hair", "eyes": "sharp brown eyes", "build": "athletic", '
            '"clothing": "white martial arts robe", "accessories": ""}, '
            '"current_state": {"outfit": "torn white robe", "injuries": "bleeding cheek", "emotion": "angry"}}], '
            '"locations": [{"canonical_name": "Name", "description": "brief", '
            '"visual_tags": "stone courtyard, ancient pillars, morning mist"}], '
            '"events": [{"summary": "Brief description of the action", '
            '"importance": 8, "involved_characters": ["Name1", "Name2"], '
            '"location": "Name"}], '
            '"world_concepts": [{"concept_type": "power_system", "name": "Qi Cultivation", '
            '"description": "cultivators absorb spiritual energy"}], '
            '"relationships": [{"char1": "Name1", "char2": "Name2", "type": "rivals", '
            '"description": "brief context"}]} '
            "Note: 'importance' is a 1-10 scale where 10 is a major battle or plot twist. "
            "Return empty arrays if nothing found. NO extra text outside the JSON."
        )
        result = self._gen_json(text[:3000], system)
        try:
            return json.loads(result)
        except Exception as e:
            logger.warning(f"extract_all JSON parse failed: {e}")
        return {"characters": [], "locations": [], "world_concepts": [], "relationships": []}

    def extract_world_style(self, text: str) -> str:
        """Extract a short visual style string for consistent art direction."""
        system = (
            "You are a visual art director for manhwa. "
            "Describe the visual aesthetic of this world in 10-15 comma-separated booru tags. "
            "Focus on: era/time period, architecture style, climate, color palette, atmosphere. "
            "Examples: 'ancient China, stone pagodas, misty mountains, imperial colors' "
            "or 'modern Seoul, neon lights, urban fantasy, rainy streets'. "
            "Output ONLY the comma-separated tags, nothing else."
        )
        result = self._gen(text[:1500], system)
        tags = result.strip().strip("`").split("\n")[0].strip()
        return tags[:300]
