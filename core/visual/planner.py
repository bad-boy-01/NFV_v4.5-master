"""
Scene Planner — Novel Video Factory v4
Converts text chunks into structured narrative scenes and cinematic shots.
"""
import json
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class ScenePlanner:
    """
    Breaks text chunks into narrative scenes (the 'showrunner' role).
    Each scene groups related sentences with location/character/mood info.
    """
    def __init__(self, llm_adapter, config: dict = None):
        self.llm = llm_adapter
        self.config = config or {}

    def plan_scenes(self, text_chunk: str, chapter: int = 1, events: List[Dict] = None) -> List[Dict]:
        """Convert a text chunk into a list of visual scenes."""
        events_context = ""
        if events:
            # Provide the LLM with the ground-truth events so it doesn't skip them
            events_text = "\n".join([f"- {e.get('summary', '')}" for e in events])
            events_context = (
                f"\nCRITICAL: Ensure your scenes cover the following key events:\n"
                f"{events_text}\n"
            )

        system = (
            "You are a Korean manhwa storyboard showrunner. "
            "Your goal is to turn the provided text chunk into a detailed, scene-by-scene storyboard.\n\n"
            "MANDATORY RULES:\n"
            "1. NO SUMMARIZATION: You must include EVERY sentence from the text in the 'narration_text' fields.\n"
            "2. HIGH SCENE DENSITY: Generate at least 15 scenes for this chunk. Every sentence or action must be a scene.\n"
            "3. NO SKIPPING: If a sentence is not in a scene, you have failed.\n"
            "4. A new scene starts for every action, dialogue, or change in camera angle.\n"
            "5. The combined 'narration_text' of all scenes MUST equal the source text 100%.\n\n"
            "For each scene output a JSON object with:\n"
            '  "scene_id": "SCxxx" (sequential),\n'
            '  "location": "current location",\n'
            '  "characters": ["Name1", "Name2"],\n'
            '  "emotion": "neutral|happy|angry|sad|fearful|fighting|focused|shocked",\n'
            '  "action": "detailed description of character movements and scene environment",\n'
            '  "camera_angle": "close-up|medium shot|wide shot|aerial|low angle",\n'
            '  "lighting": "cinematic lighting description",\n'
            '  "visual_prompt_tags": "comma-separated booru tags (e.g., 1boy, black hair, looking at viewer)",\n'
            '  "narration_text": "THE EXACT SENTENCE FROM THE TEXT",\n'
            '  "complexity": 5\n\n'
            "FORMAT RULES:\n"
            "- Output ONLY a valid JSON array of objects [{},{}].\n"
            "- NO extra text before or after the JSON."
            f"{events_context}"
        )

        response = self.llm.generate_json(text_chunk, system_prompt=system, temperature=0.2)
        
        try:
            scenes = json.loads(response)
            if isinstance(scenes, dict) and "scenes" in scenes:
                scenes = scenes["scenes"]
            if not isinstance(scenes, list):
                if isinstance(scenes, dict):
                    scenes = [scenes]
                else:
                    scenes = []
        except Exception as e:
            logger.warning(f"Scene planner JSON parse failed: {e}")
            scenes = []

        if not scenes:
            logger.warning("Scene planner returned empty — creating fallback scene")
            scenes = [self._fallback_scene(text_chunk, chapter)]

        # Ensure all required fields are present
        for sc in scenes:
            sc.setdefault("location", "Unknown Location")
            sc.setdefault("characters", [])
            sc.setdefault("emotion", "neutral")
            sc.setdefault("action", "continuation")
            sc.setdefault("camera_angle", "medium shot")
            sc.setdefault("lighting", "cinematic lighting")
            sc.setdefault("visual_prompt_tags", "")
            sc.setdefault("narration_text", "")
            sc.setdefault("complexity", 5)

        return scenes

    def _fallback_scene(self, text: str, chapter: int) -> Dict:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        narration = ". ".join(sentences[:3]) + "." if sentences else text[:200]
        return {
            "location": "Unknown",
            "characters": [],
            "emotion": "neutral",
            "action": "Story continues",
            "camera_angle": "medium shot",
            "lighting": "natural daylight",
            "visual_prompt_tags": "interior scene, detailed background",
            "narration_text": narration,
            "complexity": 5,
        }
