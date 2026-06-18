"""
Memory Database — Novel Video Factory v4
SQLAlchemy-backed knowledge store for story entities.

Tables:
  Character     — visual DNA + canonical name
  Location      — visual tags + background image path
  Relationship  — character interactions
  WorldConcept  — lore, items, power systems
"""
import json
import logging
import os
from contextlib import contextmanager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import Column, Integer, JSON, String, Text, create_engine
    from sqlalchemy.orm import DeclarativeBase, sessionmaker

    class Base(DeclarativeBase):
        pass

    class Character(Base):
        __tablename__ = "characters"
        id = Column(String, primary_key=True)
        canonical_name = Column(String, nullable=False, unique=True)
        visual_dna = Column(JSON, default=dict)

    class Location(Base):
        __tablename__ = "locations"
        id = Column(Integer, primary_key=True, autoincrement=True)
        canonical_name = Column(String, nullable=False, unique=True)
        description = Column(Text, default="")
        visual_tags = Column(Text, default="")
        background_path = Column(String, default="")

    class Relationship(Base):
        __tablename__ = "relationships"
        id = Column(Integer, primary_key=True, autoincrement=True)
        character_a = Column(String)
        character_b = Column(String)
        relationship_type = Column(String, default="other")
        description = Column(Text, default="")

    class WorldConcept(Base):
        __tablename__ = "world_concepts"
        id = Column(Integer, primary_key=True, autoincrement=True)
        concept_type = Column(String, default="misc")
        name = Column(String)
        description = Column(Text, default="")

    SQLALCHEMY_AVAILABLE = True

except ImportError:
    logger.warning("SQLAlchemy not installed — using in-memory dict fallback")
    SQLALCHEMY_AVAILABLE = False


class MemoryEngine:
    """
    Central knowledge store for the novel's characters, locations, and lore.
    Falls back to an in-memory dict if SQLAlchemy is not installed.
    """
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        mem_dir = os.path.join(project_dir, "memory")
        os.makedirs(mem_dir, exist_ok=True)

        self._in_memory = not SQLALCHEMY_AVAILABLE
        if self._in_memory:
            self._chars: Dict[str, dict] = {}
            self._locs: Dict[str, dict] = {}
            self._rels: List[dict] = []
            self._concepts: List[dict] = []
            return

        db_path = os.path.join(mem_dir, "novel_memory.db")
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def Session(self):
        if self._in_memory:
            yield None
            return
        s = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ── Characters ───────────────────────────────────────────────────────────
    def add_character(self, char_id: str, name: str, visual_dna: dict):
        if self._in_memory:
            if name in self._chars:
                self._chars[name]["visual_dna"].update(visual_dna)
            else:
                self._chars[name] = {"id": char_id, "canonical_name": name,
                                     "visual_dna": visual_dna}
            return

        with self.Session() as s:
            existing = s.query(Character).filter_by(canonical_name=name).first()
            if existing:
                merged = {**existing.visual_dna, **visual_dna}
                existing.visual_dna = merged
            else:
                s.add(Character(id=char_id, canonical_name=name, visual_dna=visual_dna))

    def get_character_by_name(self, name: str) -> Optional[Dict]:
        if self._in_memory:
            return self._chars.get(name)

        with self.Session() as s:
            c = s.query(Character).filter_by(canonical_name=name).first()
            if c:
                return {"id": c.id, "canonical_name": c.canonical_name,
                        "visual_dna": c.visual_dna}
        return None

    def get_all_characters(self) -> List[Dict]:
        if self._in_memory:
            return list(self._chars.values())

        with self.Session() as s:
            return [{"id": c.id, "canonical_name": c.canonical_name,
                     "visual_dna": c.visual_dna}
                    for c in s.query(Character).all()]

    # ── Locations ─────────────────────────────────────────────────────────────
    def add_location(self, name: str, description: str = "", visual_tags: str = ""):
        if self._in_memory:
            if name not in self._locs:
                self._locs[name] = {"canonical_name": name, "description": description,
                                    "visual_tags": visual_tags, "background_path": ""}
            return

        with self.Session() as s:
            if not s.query(Location).filter_by(canonical_name=name).first():
                s.add(Location(canonical_name=name, description=description,
                               visual_tags=visual_tags))

    def get_location_by_name(self, name: str) -> Optional[Dict]:
        if self._in_memory:
            return self._locs.get(name)

        with self.Session() as s:
            loc = s.query(Location).filter_by(canonical_name=name).first()
            if loc:
                return {"canonical_name": loc.canonical_name,
                        "description": loc.description,
                        "visual_tags": loc.visual_tags,
                        "background_path": loc.background_path}
        return None

    def get_all_locations(self) -> List[Dict]:
        if self._in_memory:
            return list(self._locs.values())

        with self.Session() as s:
            return [{"canonical_name": l.canonical_name, "description": l.description,
                     "visual_tags": l.visual_tags, "background_path": l.background_path}
                    for l in s.query(Location).all()]

    def update_location_background(self, name: str, path: str):
        if self._in_memory:
            if name in self._locs:
                self._locs[name]["background_path"] = path
            return

        with self.Session() as s:
            loc = s.query(Location).filter_by(canonical_name=name).first()
            if loc:
                loc.background_path = path

    # ── Relationships ─────────────────────────────────────────────────────────
    def add_relationship(self, char_a: str, char_b: str, rel_type: str = "other",
                         description: str = ""):
        char_1, char_2 = sorted([char_a, char_b])
        if self._in_memory:
            self._rels.append({"character_a": char_1, "character_b": char_2,
                                "type": rel_type, "description": description})
            return

        with self.Session() as s:
            existing = s.query(Relationship).filter(
                Relationship.character_a == char_1,
                Relationship.character_b == char_2,
            ).first()
            if not existing:
                s.add(Relationship(character_a=char_1, character_b=char_2,
                                   relationship_type=rel_type, description=description))

    def get_all_relationships(self) -> List[Dict]:
        if self._in_memory:
            return self._rels

        with self.Session() as s:
            return [{"character_a": r.character_a, "character_b": r.character_b,
                     "type": r.relationship_type, "description": r.description}
                    for r in s.query(Relationship).all()]

    def get_relationship_staging(self, character_names: List[str]) -> str:
        """Returns a short comma-separated staging tag for scene composition."""
        rels = self.get_all_relationships()
        tags = []
        for rel in rels:
            if rel["character_a"] in character_names and rel["character_b"] in character_names:
                rt = rel.get("type", "other")
                if rt == "rivals":
                    tags.append("confrontational stance, tension")
                elif rt in ("allies", "friends"):
                    tags.append("standing together, comradery")
                elif rt == "romance":
                    tags.append("close proximity, warm atmosphere")
        return ", ".join(tags[:2]) if tags else ""

    # ── World Concepts ────────────────────────────────────────────────────────
    def add_world_concept(self, concept_type: str, name: str, description: str = ""):
        if self._in_memory:
            self._concepts.append({"type": concept_type, "name": name,
                                   "description": description})
            return

        with self.Session() as s:
            if not s.query(WorldConcept).filter_by(name=name).first():
                s.add(WorldConcept(concept_type=concept_type, name=name,
                                   description=description))

    # ── Export ────────────────────────────────────────────────────────────────
    def export_to_json(self) -> Dict:
        return {
            "characters": self.get_all_characters(),
            "locations": self.get_all_locations(),
            "relationships": self.get_all_relationships(),
        }
