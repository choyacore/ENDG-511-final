"""
recommender.py – Rule-based styling recommendation engine.

Maps (hair_color, hair_length) → up to 3 styling tips.
All rules live in config.py so they can be extended without touching logic.

ENDG 511 – Team 14
"""

from config import get_recommendation


def get_tips(color: str, length: str) -> list[str]:
    """
    Return styling tips for the detected hair attributes.

    Parameters
    ----------
    color  : e.g. 'brown', 'blonde', 'black', 'dark' (fallback)
    length : 'short' | 'medium' | 'long' | 'unknown'

    Returns
    -------
    List of up to 3 tip strings.
    """
    if length == "unknown":
        return ["Centre your face in frame for accurate length analysis."]

    return get_recommendation(color, length)[:3]
