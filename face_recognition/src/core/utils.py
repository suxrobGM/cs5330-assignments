def format_deepface_results(results: list[dict]) -> str:
    """
    Formats the results of DeepFace analysis for multiple faces into a multiline string.
    Args:
        results (list): The result from DeepFace analysis containing an array of dicts.
    Returns:
        str: A formatted multiline string displaying the analysis for all faces.
    """
    formatted_result = "DeepFace Analysis Results:\n-----------------------------\n"
    for idx, result in enumerate(results):
        emotion: str = result.get("dominant_emotion", "N/A")
        age: str = result.get("age", "N/A")
        gender: str = result.get("dominant_gender", "N/A")
        race: str = result.get("dominant_race", "N/A")
        
        formatted_result += f"Face {idx + 1}:\n"
        formatted_result += f"  Emotion: {emotion.capitalize()}\n"
        formatted_result += f"  Age: {age}\n"
        formatted_result += f"  Gender: {gender}\n"
        formatted_result += f"  Race: {race.capitalize()}\n"
        formatted_result += "-----------------------------\n"
    
    return formatted_result
