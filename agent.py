

def intent_router(user_query):
    
    prediction_keywords = ["predict", "forecast", "projection", "outcome"]
    stats_keywords = ["stats", "statistics", "data", "numbers"]
    if any(keyword in user_query.lower() for keyword in prediction_keywords):
        return "predict"
    elif any(keyword in user_query.lower() for keyword in stats_keywords):
        return "stats"
    else:
        return "general"
