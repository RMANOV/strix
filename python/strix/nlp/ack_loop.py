"""Commander Acknowledgment Loop -- confirm understanding before execution.

Military operations demand *positive acknowledgment* before execution.
STRIX never acts on an ambiguous order.  This module generates a structured
confirmation message from a parsed intent and plan, then evaluates the
commander's response.

Flow::

    Commander: "Recon north ridge with 4 drones, avoid SAM corridor"
    STRIX:     "Understood: recon of north ridge with 4 drones.
                Avoiding SAM corridor.  Estimated time: 8 minutes.
                Current regime: PATROL.  Confirm?"
    Commander: "Confirmed" / "Потвърждавам" / "Abort" / "Change to 6 drones"
"""

from __future__ import annotations

import logging
import re
from enum import Enum, auto

from strix.brain import MissionIntent, MissionPlan, MissionType

logger = logging.getLogger("strix.nlp.ack_loop")


class AckResponse(Enum):
    """Possible commander responses to a confirmation prompt."""

    CONFIRMED = auto()
    REJECTED = auto()
    MODIFY = auto()
    UNCLEAR = auto()


# Keywords that signal confirmation (EN + BG)
_CONFIRM_KEYWORDS = {
    "confirm",
    "confirmed",
    "go",
    "go ahead",
    "execute",
    "proceed",
    "affirm",
    "affirmative",
    "roger",
    "wilco",
    "yes",
    "da",
    "approved",
    # Bulgarian
    "потвърждавам",
    "потвърди",
    "действай",
    "напред",
    "да",
    "изпълни",
}

# Keywords that signal rejection
_REJECT_KEYWORDS = {
    "abort",
    "cancel",
    "negative",
    "no",
    "stop",
    "hold",
    "scratch",
    "belay",
    "не",
    "отмени",
    "стоп",
    "отказ",
}

# Keywords that signal modification
_MODIFY_KEYWORDS = {
    "change",
    "modify",
    "update",
    "instead",
    "actually",
    "make it",
    "switch",
    "промени",
    "смени",
}


class AckLoop:
    """Confirm understanding with the commander before executing.

    Usage::

        ack = AckLoop()
        prompt = ack.generate_confirmation(intent, plan)
        # Display or speak `prompt` to the commander

        response = "Confirmed"
        result = ack.accept_response(response)
        if result == AckResponse.CONFIRMED:
            execute(plan)
    """

    # -- Mission type display names ------------------------------------------
    _MISSION_NAMES = {
        MissionType.RECON: "reconnaissance",
        MissionType.STRIKE: "strike",
        MissionType.ESCORT: "escort",
        MissionType.DEFEND: "defense",
        MissionType.PATROL: "patrol",
        MissionType.RELAY: "communications relay",
    }

    def generate_confirmation(self, intent: MissionIntent, plan: MissionPlan) -> str:
        """Generate a human-readable confirmation message.

        The message summarizes:
        - Mission type and target area
        - Number of drones allocated
        - Active constraints
        - Estimated duration
        - Current battlespace regime
        - Request for explicit confirmation

        Parameters
        ----------
        intent : MissionIntent
            The parsed commander intent.
        plan : MissionPlan
            The generated mission plan.

        Returns
        -------
        str
            A formatted confirmation string for the commander.
        """
        mission_name = self._MISSION_NAMES.get(intent.mission_type, "mission")
        n_drones = len(plan.assignments)

        lines = [f"Understood: {mission_name}"]

        # Area
        if intent.area:
            lines[0] += f" at designated area (radius {intent.area.radius_m:.0f}m)"

        # Drone count
        lines.append(f"Allocating {n_drones} drone{'s' if n_drones != 1 else ''}.")

        # Constraints
        if intent.constraints:
            constraint_strs = [c.description or c.name for c in intent.constraints]
            lines.append("Constraints: " + "; ".join(constraint_strs) + ".")

        # Timing
        if plan.estimated_duration_s > 0:
            minutes = plan.estimated_duration_s / 60.0
            if minutes >= 1.0:
                lines.append(f"Estimated time: {minutes:.0f} minute{'s' if minutes != 1 else ''}.")
            else:
                lines.append(f"Estimated time: {plan.estimated_duration_s:.0f} seconds.")

        # Regime
        lines.append(f"Current regime: {plan.regime.name}.")

        # Confidence
        if plan.confidence < 0.6:
            lines.append(f"WARNING: plan confidence is low ({plan.confidence:.0%}).")

        # Confirmation request
        lines.append("Confirm?")

        confirmation = "\n".join(lines)
        logger.info("Generated confirmation:\n%s", confirmation)
        return confirmation

    def accept_response(self, response: str) -> AckResponse:
        """Evaluate the commander's response to a confirmation prompt.

        Parameters
        ----------
        response : str
            Free-form text response from the commander.

        Returns
        -------
        AckResponse
            CONFIRMED, REJECTED, MODIFY, or UNCLEAR.
        """
        text = response.lower().strip()
        logger.info("Evaluating response: '%s'", text)

        # Check rejection first (higher priority than confirm)
        for keyword in _REJECT_KEYWORDS:
            if keyword in text:
                logger.info("Response classified: REJECTED (keyword='%s')", keyword)
                return AckResponse.REJECTED

        # Check modification
        for keyword in _MODIFY_KEYWORDS:
            if keyword in text:
                logger.info("Response classified: MODIFY (keyword='%s')", keyword)
                return AckResponse.MODIFY

        # Check confirmation
        for keyword in _CONFIRM_KEYWORDS:
            if keyword in text:
                logger.info("Response classified: CONFIRMED (keyword='%s')", keyword)
                return AckResponse.CONFIRMED

        logger.warning("Response unclear: '%s'", text)
        return AckResponse.UNCLEAR

    def generate_rejection_ack(self) -> str:
        """Generate acknowledgment for a rejected plan."""
        return "Understood. Mission cancelled. Awaiting new orders."

    def generate_modification_prompt(self) -> str:
        """Prompt the commander for modification details."""
        return "Ready for modifications. What would you like to change?"

    def generate_unclear_prompt(self) -> str:
        """Re-prompt when the response is unclear."""
        return "Could not understand response. Please confirm, modify, or abort."
