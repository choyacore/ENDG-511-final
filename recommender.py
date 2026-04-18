"""
recommender.py – Rule-based styling recommendation engine.

Maps (hair_color, hair_length) → up to 3 styling tips.
All rules live in config.py so they can be extended without touching logic.

ENDG 511 – Team 1
"""

from config import get_recommendation


def get_tips(color: str, length: str) -> list[str]:
    """
    styling tips for the detected hair attributes.

    Parameters

    color  : e.g. 'brown', 'blonde', 'black' etc. (from color_classifier.py)
    length : 'short' , 'medium' , 'long'  or 'unknown' (from length_classifier.py)
    recommendation rules are defined in config.py
    Returns

    List  3 tips .
    """
    if length == "unknown":
        return ["Centre your face in frame for accurate length analysis."]

    return get_recommendation(color, length)[:3]



