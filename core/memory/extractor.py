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

    def extract_characters(self, text: str, existing_characters: list = None) -> Dict:
        existing_names = [c.get("canonical_name", "") for c in (existing_characters or []) if c.get("canonical_name")]
        relevant_names = [n for n in existing_names if n.lower() in text.lower()]
        known = ", ".join(relevant_names) if relevant_names else "none yet"

        system = (
            "You are a visual character extractor for a Korean manhwa AI pipeline. "
            "Extract ALL named characters from the following text. "
            f"Relevant known characters already in the database: [{known}]. "
            "For new characters, extract static visual DNA as booru-style tags. "
            "For ALL characters (new and existing), extract their CURRENT dynamic state (outfit, injuries, emotion). "
            "Output ONLY valid JSON with this structure:\n"
            '{"characters": [{"canonical_name": "Name", "visual_dna": {'
            '"subject": "1boy or 1girl", "age": "e.g. 20 years old, teenager, ancient", '
            '"hair": "black short hair", "eyes": "sharp brown eyes", "build": "athletic", '
            '"clothing": "white martial arts robe", "accessories": ""}, '
            '"current_state": {"outfit": "torn white robe", "injuries": "bleeding cheek", "emotion": "angry"}}]} '
            "Return empty array if nothing found. NO extra text outside the JSON."
        )
        prompt = text[:3000]
        logger.info(f"  [Compression] Characters: {len(existing_names)} total -> {len(relevant_names)} relevant. Prompt length: {len(prompt)} chars.")
        
        max_t = self.config.get("models", {}).get("llm", {}).get("character_max_tokens", 1200)
        result = self.llm.generate_json(prompt, system_prompt=system, temperature=0.1, max_tokens=max_t)
        try:
            return json.loads(result)
        except Exception as e:
            logger.warning(f"extract_characters JSON parse failed: {e}")
            return {"_parse_error": True, "_raw_text": result}

    def extract_locations(self, text: str) -> Dict:
        system = (
            "You are a visual location extractor for a Korean manhwa AI pipeline. "
            "Extract ALL locations from the following text. "
            "Output ONLY valid JSON with this structure:\n"
            '{"locations": [{"canonical_name": "Name", "description": "brief", '
            '"visual_tags": "stone courtyard, ancient pillars, morning mist"}]} '
            "Return empty array if nothing found. NO extra text outside the JSON."
        )
        max_t = self.config.get("models", {}).get("llm", {}).get("location_max_tokens", 800)
        result = self.llm.generate_json(text[:3000], system_prompt=system, temperature=0.1, max_tokens=max_t)
        try:
            return json.loads(result)
        except Exception as e:
            logger.warning(f"extract_locations JSON parse failed: {e}")
            return {"_parse_error": True, "_raw_text": result}

    def extract_events(self, text: str, existing_characters: list = None) -> Dict:
        existing_names = [c.get("canonical_name", "") for c in (existing_characters or []) if c.get("canonical_name")]
        relevant_names = [n for n in existing_names if n.lower() in text.lower()]
        known = ", ".join(relevant_names) if relevant_names else "none yet"

        system = (
            "You are a narrative event extractor for a Korean manhwa AI pipeline. "
            "Extract ALL narrative events, actions, and relationships from the text. "
            f"Relevant known characters: [{known}]. "
            "Output ONLY valid JSON with this structure:\n"
            '{"events": [{"summary": "Brief description of the action", '
            '"importance": 8, "involved_characters": ["Name1", "Name2"], '
            '"location": "Name"}], '
            '"relationships": [{"char1": "Name1", "char2": "Name2", "type": "rivals", '
            '"description": "brief context"}]} '
            "Note: 'importance' is a 1-10 scale where 10 is a major battle or plot twist. "
            "Return empty arrays if nothing found. NO extra text outside the JSON."
        )
        prompt = text[:3000]
        logger.info(f"  [Compression] Events: {len(existing_names)} total -> {len(relevant_names)} relevant. Prompt length: {len(prompt)} chars.")

        max_t = self.config.get("models", {}).get("llm", {}).get("event_max_tokens", 2000)
        result = self.llm.generate_json(prompt, system_prompt=system, temperature=0.1, max_tokens=max_t)
        try:
            return json.loads(result)
        except Exception as e:
            logger.warning(f"extract_events JSON parse failed: {e}")
            return {"_parse_error": True, "_raw_text": result}

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
