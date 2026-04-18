"""
recommender.py  Rule-based styling recommendation engine.

Maps (hair_color, hair_length) → up to 3 styling tips.
All rules live in config.py so they can be extended without touching logic.

ENDG 511 Team 1
"""

from config import get_recommendation

#This function gets the color and lenght from color_classifier and length_classifier 
#Uses the config file for recommendation 
def get_tips(color: str, length: str) -> list[str]:

    if length == "unknown":
        return ["Centre your face in frame for accurate length analysis."]

    return get_recommendation(color, length)[:3]



