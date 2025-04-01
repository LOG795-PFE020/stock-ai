from langchain_core.tools import tool

@tool
def update_investment_profile(
    risk_tolerance: str = None, 
    investment_goals: str = None, 
    preferred_sectors: list = None
):
    """Update the user's investment profile."""
    global user_profile
    
    if risk_tolerance:
        user_profile['risk_tolerance'] = risk_tolerance
    if investment_goals:
        user_profile['investment_goals'] = investment_goals
    if preferred_sectors:
        user_profile['preferred_sectors'] = preferred_sectors
    
    save_profile(user_id, user_profile)
    return f"Profile updated successfully: {user_profile}"

# Integrate with OpenAI function calling
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125").bind_tools([update_investment_profile])
