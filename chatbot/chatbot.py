import sqlite3
import json
import sys
import os
from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    sys.exit(1)
logger.info("OPENAI_API_KEY is configured")

# LangSmith tracing configuration (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
else:
    logger.warning("LANGSMITH_API_KEY not set; tracing disabled.")

# Initialize OpenAI model
try:
    model = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    sys.exit(1)

# Constants for profile setup
RISK_TOLERANCE_OPTIONS = {'1': 'low', '2': 'medium', '3': 'high'}
INVESTMENT_GOALS_OPTIONS = {
    '1': 'Conservative Income', '2': 'Balanced Growth', '3': 'Aggressive Growth',
    '4': 'Retirement Planning', '5': 'Short-term Trading'
}
AVAILABLE_SECTORS = {
    '1': 'Technology', '2': 'Healthcare', '3': 'Financial Services',
    '4': 'Consumer Goods', '5': 'Energy', '6': 'Real Estate',
    '7': 'Industrial', '8': 'Communications', '9': 'Materials', '10': 'Utilities'
}

# Database initialization
def init_database():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS profiles
                 (user_id TEXT PRIMARY KEY, risk_tolerance TEXT, investment_goals TEXT, preferred_sectors TEXT)''')
    conn.commit()
    conn.close()

init_database()

# Profile management functions
def load_profile(user_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT risk_tolerance, investment_goals, preferred_sectors FROM profiles WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"risk_tolerance": row[0], "investment_goals": row[1], "preferred_sectors": json.loads(row[2])}
    return None

def save_profile(user_id, profile):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO profiles (user_id, risk_tolerance, investment_goals, preferred_sectors) VALUES (?, ?, ?, ?)",
              (user_id, profile['risk_tolerance'], profile['investment_goals'], json.dumps(profile['preferred_sectors'])))
    conn.commit()
    conn.close()

# Initialize user profile
user_id = "user2"
user_profile = load_profile(user_id) or {}
if not user_profile:
    def setup_user_profile():
        print("\n=== Investment Profile Setup ===\n")
        print("What is your risk tolerance level?")
        for k, v in RISK_TOLERANCE_OPTIONS.items():
            print(f"{k}. {v.capitalize()}")
        user_profile['risk_tolerance'] = RISK_TOLERANCE_OPTIONS.get(input("Enter the number (1-3): "), "medium")

        print("\nWhat are your primary investment goals?")
        for k, v in INVESTMENT_GOALS_OPTIONS.items():
            print(f"{k}. {v}")
        user_profile['investment_goals'] = INVESTMENT_GOALS_OPTIONS.get(input("Enter the number (1-5): "), "Balanced Growth")

        print("\nWhich sectors interest you? (Choose up to 3)")
        for k, v in AVAILABLE_SECTORS.items():
            print(f"{k}. {v}")
        selected_sectors = []
        while len(selected_sectors) < 3:
            choice = input(f"Enter sector number (1-10) or 'done' ({len(selected_sectors)}/3): ")
            if choice.lower() == 'done' and selected_sectors:
                break
            if choice in AVAILABLE_SECTORS and AVAILABLE_SECTORS[choice] not in selected_sectors:
                selected_sectors.append(AVAILABLE_SECTORS[choice])
        user_profile['preferred_sectors'] = selected_sectors or ["Technology"]

        print("\n=== Your Investment Profile ===")
        print(f"Risk Tolerance: {user_profile['risk_tolerance'].capitalize()}")
        print(f"Investment Goals: {user_profile['investment_goals']}")
        print("Preferred Sectors:", ", ".join(user_profile['preferred_sectors']))
        save_profile(user_id, user_profile)

    setup_user_profile()

# Define state schema
class State(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], add_messages]
    user_profile: dict

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a financial advisor specializing in stocks. "
                          "The user has the following profile:\n"
                          "- Risk Tolerance: {risk_tolerance}\n"
                          "- Investment Goals: {investment_goals}\n"
                          "- Preferred Sectors: {preferred_sectors}\n"
                          "Use this profile to provide personalized advice. "
                          "If the user asks to update their profile, use the update_profile tool."),
    MessagesPlaceholder(variable_name="messages"),
])

# Message trimmer
trimmer = trim_messages(max_tokens=10000, strategy="last", token_counter=model, include_system=True, allow_partial=False, start_on="human")

# Tool for profile updates
@tool
def update_profile(risk_tolerance: str = None, investment_goals: str = None, preferred_sectors: list = None):
    """Update the user's investment profile."""
    global user_profile
    if risk_tolerance and risk_tolerance.lower() in RISK_TOLERANCE_OPTIONS.values():
        user_profile['risk_tolerance'] = risk_tolerance.lower()
    if investment_goals and investment_goals in INVESTMENT_GOALS_OPTIONS.values():
        user_profile['investment_goals'] = investment_goals
    if preferred_sectors and all(s in AVAILABLE_SECTORS.values() for s in preferred_sectors):
        user_profile['preferred_sectors'] = preferred_sectors
    save_profile(user_id, user_profile)
    return f"Profile updated: Risk Tolerance: {user_profile['risk_tolerance']}, Goals: {user_profile['investment_goals']}, Sectors: {', '.join(user_profile['preferred_sectors'])}"

# Bind tools to model
model_with_tools = model.bind_tools([update_profile])

# Define model calling function
def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    formatted_messages = prompt_template.format_messages(
        risk_tolerance=state["user_profile"]["risk_tolerance"],
        investment_goals=state["user_profile"]["investment_goals"],
        preferred_sectors=", ".join(state["user_profile"]["preferred_sectors"]),
        messages=trimmed_messages
    )
    response = model_with_tools.invoke(formatted_messages)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call['name'] == "update_profile":
            result = update_profile(**tool_call['args'])
            return {"messages": [AIMessage(content=result)]}
    return {"messages": [response]}

# Build LangGraph workflow
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Config for persistence
config = {"configurable": {"thread_id": user_id}}

# Streaming response function
def generate_response(user_message):
    try:
        input_messages = [HumanMessage(content=user_message)]
        stream = app.stream(
            {"messages": input_messages, "user_profile": user_profile},
            config,
            stream_mode="updates"  # Use "updates" for custom state
        )
        for chunk in stream:
            logger.debug(f"Stream chunk: {chunk}")
            #print(f"Raw chunk: {chunk}")  # Print raw output for debugging
            if "model" in chunk:
                update = chunk["model"]
                if "messages" in update:
                    message = update["messages"][0]
                    if isinstance(message, AIMessage):
                        if message.content:
                            print(message.content, end="|", flush=True)
                        elif message.tool_calls:
                            print(f"Tool call: {message.tool_calls}", end="|", flush=True)
        print()  # Newline after response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        print("Sorry, an error occurred. Please try again.", flush=True)

# Main execution loop
if __name__ == "__main__":
    print("Welcome to your Financial Advisor Chatbot! Type 'exit' to quit.")
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            generate_response(user_input)
    except KeyboardInterrupt:
        print("\nSaving profile and exiting...")
        print("Goodbye!")