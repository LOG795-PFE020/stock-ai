import sqlite3
import openai
import json
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
import re
import logging
import random
from datetime import timedelta
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Check if the OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    sys.exit(1)
else:
    logger.info("OPENAI_API_KEY is configured")

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json; charset=utf-8"

# Initialize OpenAI client
try:
    client = ChatOpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    sys.exit(1)

# Constants for profile setup
RISK_TOLERANCE_OPTIONS = {
    '1': 'low',
    '2': 'medium',
    '3': 'high'
}

INVESTMENT_GOALS_OPTIONS = {
    '1': 'Conservative Income (Focus on stable, dividend-paying investments)',
    '2': 'Balanced Growth (Mix of growth and income)',
    '3': 'Aggressive Growth (Focus on capital appreciation)',
    '4': 'Retirement Planning',
    '5': 'Short-term Trading'
}

AVAILABLE_SECTORS = {
    '1': 'Technology',
    '2': 'Healthcare',
    '3': 'Financial Services',
    '4': 'Consumer Goods',
    '5': 'Energy',
    '6': 'Real Estate',
    '7': 'Industrial',
    '8': 'Communications',
    '9': 'Materials',
    '10': 'Utilities'
}

# Database initialization
def init_database():
    """Initialize SQLite database with profiles table."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS profiles
                 (user_id TEXT PRIMARY KEY, risk_tolerance TEXT, investment_goals TEXT, preferred_sectors TEXT)''')
    conn.commit()
    conn.close()

init_database()

# Profile management functions
def load_profile(user_id):
    """Load user profile from the database."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT risk_tolerance, investment_goals, preferred_sectors FROM profiles WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "risk_tolerance": row[0],
            "investment_goals": row[1],
            "preferred_sectors": json.loads(row[2])  # Stored as JSON string
        }
    return None

def save_profile(user_id, profile):
    """Save or update user profile in the database."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO profiles (user_id, risk_tolerance, investment_goals, preferred_sectors) VALUES (?, ?, ?, ?)",
              (user_id, profile['risk_tolerance'], profile['investment_goals'], json.dumps(profile['preferred_sectors'])))
    conn.commit()
    conn.close()

# Initialize user profile and memory
user_id = "user1"  # Hardcoded for testing; replace with dynamic user identification in production
user_profile = {}

# Set up SQL chat message history
chat_message_history = SQLChatMessageHistory(
    session_id=user_id,
    connection="sqlite:///chat_history.db"
)

# Create tables for chat history if they don't exist
def init_chat_history_db():
    """Initialize SQLite database with messages table."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS message_store
                 (session_id TEXT, message TEXT, additional_kwargs TEXT, type TEXT, created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

init_chat_history_db()

# Set up memory with ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(api_key=openai_api_key),
    chat_memory=chat_message_history,
    max_token_limit=1000,
    return_messages=True
)

# Load or set up user profile
profile = load_profile(user_id)
if profile:
    user_profile.update(profile)
else:
    def setup_user_profile():
        """Interactive function to set up user's investment profile."""
        print("\n=== Investment Profile Setup ===\n")
        
        # Risk Tolerance
        print("What is your risk tolerance level?")
        for key, value in RISK_TOLERANCE_OPTIONS.items():
            print(f"{key}. {value.capitalize()}")
        while True:
            risk_choice = input("\nEnter the number of your choice (1-3): ")
            if risk_choice in RISK_TOLERANCE_OPTIONS:
                user_profile['risk_tolerance'] = RISK_TOLERANCE_OPTIONS[risk_choice]
                break
            print("Invalid choice. Please try again.")

        # Investment Goals
        print("\nWhat are your primary investment goals?")
        for key, value in INVESTMENT_GOALS_OPTIONS.items():
            print(f"{key}. {value}")
        while True:
            goal_choice = input("\nEnter the number of your choice (1-5): ")
            if goal_choice in INVESTMENT_GOALS_OPTIONS:
                user_profile['investment_goals'] = INVESTMENT_GOALS_OPTIONS[goal_choice]
                break
            print("Invalid choice. Please try again.")

        # Preferred Sectors
        print("\nWhich sectors interest you? (Choose up to 3)")
        for key, value in AVAILABLE_SECTORS.items():
            print(f"{key}. {value}")
        selected_sectors = []
        while len(selected_sectors) < 3:
            sector_choice = input(f"\nEnter sector number (1-10) or 'done' to finish {len(selected_sectors)}/3: ")
            if sector_choice.lower() == 'done':
                if len(selected_sectors) > 0:
                    break
                print("Please select at least one sector.")
                continue
            if sector_choice in AVAILABLE_SECTORS and AVAILABLE_SECTORS[sector_choice] not in selected_sectors:
                selected_sectors.append(AVAILABLE_SECTORS[sector_choice])
            elif sector_choice in AVAILABLE_SECTORS:
                print("You've already selected this sector.")
            else:
                print("Invalid choice. Please try again.")
        user_profile['preferred_sectors'] = selected_sectors

        # Display summary
        print("\n=== Your Investment Profile ===")
        print(f"Risk Tolerance: {user_profile['risk_tolerance'].capitalize()}")
        print(f"Investment Goals: {user_profile['investment_goals']}")
        print("Preferred Sectors:", ", ".join(user_profile['preferred_sectors']))
        print("\nYou can now start chatting with your financial advisor!")
        
        # Save the new profile
        save_profile(user_id, user_profile)

    setup_user_profile()

# Function definition for updating profile
update_profile_function = {
    "name": "update_profile",
    "description": "Update the user's financial profile based on their input.",
    "parameters": {
        "type": "object",
        "properties": {
            "risk_tolerance": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "User's risk tolerance level"
            },
            "investment_goals": {
                "type": "string",
                "description": "User's investment goals (e.g., growth, income)"
            },
            "preferred_sectors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Sectors the user is interested in (e.g., tech, healthcare)"
            }
        },
        "required": []
    }
}

def generate_response(user_message):
    try:
        # Save user message to memory first
        memory.save_context({"input": user_message}, {"output": ""})
        
        # Load conversation history from memory
        memory_messages = memory.load_memory_variables({})["history"]
        
        # Extract summary and recent messages
        summary_content = ""
        conversation_history = []
        for msg in memory_messages:
            if msg.type == "system":
                summary_content = msg.content
            elif msg.type == "human":
                conversation_history.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                conversation_history.append({"role": "assistant", "content": msg.content})
        
        # Create system message with profile and summary
        system_content = (
            "You are a financial advisor specializing in stocks. "
            f"The user has the following established profile:\n"
            f"- Risk Tolerance: {user_profile['risk_tolerance']}\n"
            f"- Investment Goals: {user_profile['investment_goals']}\n"
            f"- Preferred Sectors: {', '.join(user_profile['preferred_sectors'])}\n\n"
            "Use this profile information to provide personalized advice. "
            "Always reference specific details from past conversations, including exact investment amounts and stocks mentioned. "
            "Be precise when recalling past information.\n"
            f"Previous conversation summary: {summary_content}"
        )
        
        # First API call with function calling
        response = client.invoke(
            [
                {"role": "system", "content": system_content},
                *conversation_history,
                {"role": "user", "content": user_message}
            ],
            tools=[{
                "type": "function",
                "function": update_profile_function
            }]
        )
        
        # Handle function call if present
        if response.additional_kwargs.get('tool_calls'):
            tool_call = response.additional_kwargs['tool_calls'][0]
            if isinstance(tool_call, dict):
                function_name = tool_call.get('function', {}).get('name')
                function_args = tool_call.get('function', {}).get('arguments', '{}')
            else:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

            if function_name == "update_profile":
                # Extract function arguments
                function_args = json.loads(function_args)
                
                # Update user profile
                for key, value in function_args.items():
                    if key == "preferred_sectors" and value:
                        user_profile[key].extend([sector for sector in value if sector not in user_profile[key]])
                    elif value:
                        user_profile[key] = value
                
                # Save updated profile to database
                save_profile(user_id, user_profile)
                
                # Second API call for final response
                second_response = client.invoke(
                    [
                        {"role": "system", "content": system_content},
                        *conversation_history,
                        {"role": "assistant", "content": "Profile updated successfully. " + response.content}
                    ]
                )
                final_message = second_response.content
                
                # Save assistant response to memory
                memory.save_context({"input": user_message}, {"output": final_message})
                return final_message
        
        # No function call; direct response
        else:
            final_message = response.content
            memory.save_context({"input": user_message}, {"output": final_message})
            return final_message
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error. Please try again."

# Main execution
if __name__ == "__main__":
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            response = generate_response(user_input)
            print(f"Advisor: {response}")
    except KeyboardInterrupt:
        print("\nSaving conversation history before exit...")
        # Ensure the last context is saved
        memory.save_context({"input": "exit"}, {"output": "Conversation ended."})
        print("Goodbye!")