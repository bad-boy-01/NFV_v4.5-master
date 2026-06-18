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
            "Break the following story text into visual SCENES.\n\n"
            "CRITICAL RULES:\n"
            "- DO NOT SUMMARIZE. Include EVERY single detail from the text.\n"
            "- Break this chunk into AT LEAST 6-8 SCENES. Each scene = one image panel.\n"
            "- The 'narration_text' for all scenes COMBINED must contain 100% of the provided text.\n"
            "- A new scene starts when location, time, or main focus changes.\n\n"
            "For each scene output a JSON object with:\n"
            '  "scene_id": "SC001" (sequential),\n'
            '  "location": "place name",\n'
            '  "characters": ["Name1", "Name2"],\n'
            '  "emotion": "neutral|happy|angry|sad|fearful|fighting|focused|shocked",\n'
            '  "action": "short action description",\n'
            '  "camera_angle": "close-up|medium shot|wide shot|aerial|low angle",\n'
            '  "lighting": "cinematic lighting description",\n'
            '  "visual_prompt_tags": "comma-separated booru tags — NO character names, use 1boy/1girl",\n'
            '  "narration_text": "EXACT sentences from the text for TTS (DO NOT SKIP ANY SENTENCES)",\n'
            '  "complexity": 1-10\n\n'
            "FORMAT RULES:\n"
            "- DO NOT include character names in visual_prompt_tags (use 1boy/1girl instead)\n"
            "- Output ONLY a JSON array. NO extra text outside the array."
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
