# SPDX-License-Identifier: Apache-2.0

"""Voice / text command parser -- converts natural language to structured MissionIntent.

Input:  "Survey north ridge, avoid high-interference corridor, report in 10 min"
Output: MissionIntent(type=RECON, area=NorthRidge, constraints=[AvoidAA], deadline=600s)

The parser uses a keyword-matching approach as a baseline.  Production
deployment may swap in a domain-adapted language model (see strix.llm)
for richer understanding, but the keyword fallback remains as a
degraded-mode capability when LLM inference is unavailable.

Supported languages: English (primary), Bulgarian (partial -- operator
may issue requests in native language).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from strix.brain import Constraint, MissionArea, MissionIntent, MissionType, Vec3

logger = logging.getLogger("strix.nlp.intent_parser")

# ---------------------------------------------------------------------------
# Named areas (lookup table -- in production loaded from scenario config)
# ---------------------------------------------------------------------------

_NAMED_AREAS: dict[str, MissionArea] = {
    "north ridge": MissionArea(center=Vec3(0, 500, 0), radius_m=300),
    "south valley": MissionArea(center=Vec3(0, -400, 0), radius_m=400),
    "east flank": MissionArea(center=Vec3(500, 0, 0), radius_m=250),
    "west approach": MissionArea(center=Vec3(-500, 0, 0), radius_m=350),
    "hilltop": MissionArea(center=Vec3(200, 300, 80), radius_m=150),
    "bridge": MissionArea(center=Vec3(-100, 200, 0), radius_m=100),
    "checkpoint alpha": MissionArea(center=Vec3(300, 100, 0), radius_m=200),
    "checkpoint bravo": MissionArea(center=Vec3(-200, 400, 0), radius_m=200),
    "landing zone": MissionArea(center=Vec3(0, 0, 0), radius_m=100),
    "fob": MissionArea(center=Vec3(0, -100, 0), radius_m=500),
}

# ---------------------------------------------------------------------------
# Keyword -> MissionType mapping
# ---------------------------------------------------------------------------

_MISSION_KEYWORDS: dict[str, MissionType] = {
    "recon": MissionType.RECON,
    "reconnaissance": MissionType.RECON,
    "scout": MissionType.RECON,
    "survey": MissionType.RECON,
    "observe": MissionType.RECON,
    "watch": MissionType.RECON,
    "isr": MissionType.RECON,
    "strike": MissionType.STRIKE,
    "interdict": MissionType.INTERDICT,
    "attack": MissionType.STRIKE,
    "engage": MissionType.STRIKE,
    "destroy": MissionType.STRIKE,
    "neutralize": MissionType.STRIKE,
    "suppress": MissionType.STRIKE,
    "escort": MissionType.ESCORT,
    "protect": MissionType.ESCORT,
    "guard": MissionType.ESCORT,
    "accompany": MissionType.ESCORT,
    "defend": MissionType.DEFEND,
    "hold": MissionType.DEFEND,
    "secure": MissionType.DEFEND,
    "patrol": MissionType.PATROL,
    "monitor": MissionType.PATROL,
    "loiter": MissionType.PATROL,
    "overwatch": MissionType.PATROL,
    "relay": MissionType.RELAY,
    "comms": MissionType.RELAY,
    "bridge comms": MissionType.RELAY,
}

# ---------------------------------------------------------------------------
# Constraint keywords
# ---------------------------------------------------------------------------

_CONSTRAINT_KEYWORDS: dict[str, str] = {
    "avoid aa": "Avoid restricted air corridor",
    "avoid sam": "Avoid high-interference zone",
    "avoid civilian": "Avoid civilian areas",
    "low altitude": "Maintain low altitude profile",
    "high altitude": "Maintain high altitude",
    "stealth": "Minimize sensor exposure",
    "silent": "Radio silence -- passive sensors only",
    "fast": "Maximize speed, accept higher risk",
    "safe": "Minimize risk, accept slower execution",
    "observation only": "Observation-focused action profile requested",
    "broad autonomy": "Expanded autonomy envelope requested by caller",
}

# ---------------------------------------------------------------------------
# Time extraction patterns
# ---------------------------------------------------------------------------

_TIME_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"(\d+)\s*min(?:ute)?s?", re.IGNORECASE), 60.0),
    (re.compile(r"(\d+)\s*sec(?:ond)?s?", re.IGNORECASE), 1.0),
    (re.compile(r"(\d+)\s*h(?:our)?s?", re.IGNORECASE), 3600.0),
    (re.compile(r"asap|immediately|urgent", re.IGNORECASE), 0.0),  # special: ASAP
]

# ---------------------------------------------------------------------------
# Drone count extraction
# ---------------------------------------------------------------------------

_COUNT_PATTERN = re.compile(r"(?:with|use|send|deploy)\s+(\d+)\s+(?:drone|uav|unit)s?", re.IGNORECASE)
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "ten": 10, "twelve": 12}
_COUNT_WORD_PATTERN = re.compile(
    r"(?:with|use|send|deploy)\s+(" + "|".join(_COUNT_WORDS.keys()) + r")\s+(?:drone|uav|unit)s?", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Priority extraction
# ---------------------------------------------------------------------------

_PRIORITY_KEYWORDS: dict[str, float] = {
    "critical": 1.0,
    "urgent": 0.9,
    "high priority": 0.8,
    "priority": 0.7,
    "normal": 0.5,
    "low priority": 0.3,
    "when possible": 0.2,
}


# ---------------------------------------------------------------------------
# IntentParser
# ---------------------------------------------------------------------------


class IntentParser:
    """Convert natural language commands to structured mission intents.

    Examples::

        parser = IntentParser()

        intent = parser.parse("Survey north ridge, avoid high-interference corridor, report in 10 min")
        # -> MissionIntent(type=RECON, area=NorthRidge, constraints=[AvoidAA], deadline=600s)

        intent = parser.parse("Send 6 drones to secure checkpoint alpha, high priority")
        # -> MissionIntent(type=DEFEND, area=CheckpointAlpha, drone_count=6,
        #                  constraints=[...])

    The parser is intentionally simple -- keyword matching with regex.
    For production, plug in a richer language-model provider via ``strix.llm``
    and use this parser as the fallback for degraded-mode operation.
    """

    def __init__(self, named_areas: dict[str, MissionArea] | None = None) -> None:
        self._named_areas = named_areas or _NAMED_AREAS

    def parse(self, text: str) -> MissionIntent:
        """Parse a natural language command into a MissionIntent.

        Parameters
        ----------
        text : str
            Free-form text command from the operator.

        Returns
        -------
        MissionIntent
            Structured intent with type, area, constraints, and deadline.
        """
        if len(text) > 2000:
            text = text[:2000]
        text_lower = text.lower().strip()
        logger.info("Parsing: '%s'", text)

        mission_type = self._extract_mission_type(text_lower)
        area = self.extract_area(text_lower)
        constraints = self.extract_constraints(text_lower)
        deadline = self.extract_timeline(text_lower)
        drone_count = self._extract_drone_count(text_lower)
        priority = self._extract_priority(text_lower)

        intent = MissionIntent(
            mission_type=mission_type,
            area=area,
            constraints=constraints,
            deadline_s=deadline,
            priority=priority,
            drone_count=drone_count,
            description=text,
        )

        logger.info(
            "Parsed: type=%s area=%s constraints=%d deadline=%.0fs drones=%s",
            mission_type.name,
            area is not None,
            len(constraints),
            deadline,
            drone_count,
        )
        return intent

    def extract_constraints(self, text: str) -> list[Constraint]:
        """Extract operational constraints from the command text.

        Matches known constraint keywords and builds Constraint objects.
        """
        text_lower = text.lower()
        found: list[Constraint] = []

        for keyword, description in _CONSTRAINT_KEYWORDS.items():
            if keyword in text_lower:
                is_avoid = keyword.startswith("avoid")
                area = None
                if is_avoid:
                    # Try to find an area for the avoidance zone
                    area = self._find_constraint_area(keyword, text_lower)

                found.append(
                    Constraint(
                        name=keyword.replace(" ", "_"),
                        description=description,
                        area=area,
                        avoid=is_avoid,
                    )
                )

        return found

    def extract_area(self, text: str) -> Optional[MissionArea]:
        """Extract the mission area from the command text.

        Tries named area lookup first, then falls back to coordinate
        extraction (e.g., "at 300,400,50").
        """
        text_lower = text.lower()

        # Named area lookup
        for name, area in self._named_areas.items():
            if name in text_lower:
                logger.debug("Matched named area: %s", name)
                return area

        # Coordinate extraction: "at X,Y,Z" or "coordinates X Y Z"
        coord_match = re.search(r"(?:at|coordinates?|coords?|position)\s+([-\d.]+)[,\s]+([-\d.]+)(?:[,\s]+([-\d.]+))?", text_lower)
        if coord_match:
            try:
                x = float(coord_match.group(1))
                y = float(coord_match.group(2))
                z = float(coord_match.group(3)) if coord_match.group(3) else 0.0
                return MissionArea(center=Vec3(x, y, z), radius_m=200.0)
            except ValueError:
                return None

        return None

    def extract_timeline(self, text: str) -> float:
        """Extract the mission deadline from the command text.

        Returns the deadline in seconds.  Defaults to 600s if not specified.
        """
        text_lower = text.lower()

        for pattern, multiplier in _TIME_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                if multiplier == 0.0:
                    # ASAP/urgent -> 60 second deadline
                    return 60.0
                value = float(match.group(1))
                return value * multiplier

        # Check for "report in" / "complete by" / "within" constructs
        report_match = re.search(r"(?:report|complete|finish|within|in)\s+(\d+)\s*(?:min|m)", text_lower)
        if report_match:
            return float(report_match.group(1)) * 60.0

        return 600.0  # default 10 minutes

    # -- Private extraction helpers ------------------------------------------

    @staticmethod
    def _extract_mission_type(text: str) -> MissionType:
        """Find the mission type keyword in the text."""
        # Check multi-word keywords first
        for keyword, mtype in sorted(_MISSION_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if keyword in text:
                return mtype
        return MissionType.RECON  # default: if unclear, recon is safest

    def _extract_drone_count(self, text: str) -> Optional[int]:
        """Extract the requested number of drones."""
        # Numeric count
        match = _COUNT_PATTERN.search(text)
        if match:
            return int(match.group(1))

        # Word count
        match = _COUNT_WORD_PATTERN.search(text)
        if match:
            return _COUNT_WORDS.get(match.group(1).lower())

        return None

    @staticmethod
    def _extract_priority(text: str) -> float:
        """Extract mission priority from keywords."""
        for keyword, priority in sorted(_PRIORITY_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if keyword in text:
                return priority
        return 0.5  # default: normal priority

    def _find_constraint_area(self, keyword: str, text: str) -> Optional[MissionArea]:
        """Try to find a named area associated with an avoidance constraint."""
        # Look for area names near the constraint keyword
        pos = text.find(keyword)
        if pos < 0:
            return None

        # Search in a window around the keyword
        window = text[max(0, pos - 20) : pos + len(keyword) + 40]
        for name, area in self._named_areas.items():
            if name in window:
                return area
        return None
