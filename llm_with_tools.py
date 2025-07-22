import requests
import json
import os
from dotenv import load_dotenv
from typing import TypedDict, Sequence, Annotated

# LangChain imports
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Set up Azure OpenAI environment
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment="GPTO4_training",
    api_version="2024-12-01-preview",
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"]
)

# Define tools
@tool
def weather_tool(location: str) -> str:
    """STRICT WEATHER TOOL - Only for current weather conditions. Requires exact location format 'City,CountryCode'."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid=426c9a09ea63f1c63a84e43b7c3fbe1d&units=metric"
        data = requests.get(url).json()
        if data.get('cod') != 200:
            raise ValueError(data.get('message', 'Unknown error'))
        return f"Weather in {location}: {data['weather'][0]['description'].capitalize()}, {data['main']['temp']}°C, Humidity: {data['main']['humidity']}%"
    except Exception as e:
        return f"WEATHER_ERROR: {str(e)}"

@tool
def dictionary_tool(word: str) -> str:
    """Return the dictionary definition of a word."""
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        definition = data[0]["meanings"][0]["definitions"][0]["definition"]
        return f"Definition of {word}: {definition}"
    return "No definition found."

@tool
def news_tool(topic: str = "general") -> str:
    """STRICT NEWS TOOL - Only for recent headlines. Valid topics: business, technology, science, health."""
    try:
        valid_topics = ['business', 'technology', 'science', 'health', 'general']
        topic = topic.lower() if topic.lower() in valid_topics else 'general'
        url = f"https://newsapi.org/v2/top-headlines?category={topic}&apiKey=6789ceab523a4a8690d965e4c1c3502b"
        articles = requests.get(url).json().get('articles', [])[:3]
        if not articles:
            raise ValueError("No articles found")
        return f"Latest {topic} news:\n" + "\n".join([f"• {a['title']}" for a in articles])
    except Exception as e:
        return f"NEWS_ERROR: {str(e)}"

# Prepare tools and LLM
tools = [weather_tool, dictionary_tool, news_tool]
llm_with_tools = llm.bind_tools(tools)

# Define State
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

# Define Nodes
def chatbot_node(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_runner_node(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, "tool_calls"):
        return {"messages": []}
    
    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        try:
            tool = next(t for t in tools if t.name == tool_name)
            result = tool.invoke(tool_args)
            tool_results.append(
                ToolMessage(
                    content=result,  # No need to json.dumps since we're returning string
                    name=tool_name,
                    tool_call_id=tool_id
                )
            )
        except Exception as e:
            tool_results.append(
                ToolMessage(
                    content=f"Error: {str(e)}",
                    name=tool_name,
                    tool_call_id=tool_id
                )
            )
    
    return {"messages": tool_results}

# Define Conditional Edges
def stop_or_tools(state: State):
    messages = state["messages"]
    if not messages:
        return END
    
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        return "chat"
    
    if isinstance(last_message, ToolMessage):
        return "chat"
    
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return END
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    return END

# Build Graph
memory = MemorySaver()
graph = StateGraph(State)
graph.add_node("chat", chatbot_node)
graph.add_node("tools", tool_runner_node)
graph.set_entry_point("chat")
graph.add_edge("tools", "chat")
graph.add_conditional_edges("chat", stop_or_tools)
compiled_graph = graph.compile(checkpointer=memory)



# from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import HumanMessage, AIMessage

def process_query(user_query: str, message_history: list, thread_id: str = "demo-thread") -> list:
    try:
        # Step 1: Convert existing chat history to LangChain message format
        lc_messages = [HumanMessage(content=msg["content"]) if msg["role"] == "user"
                       else AIMessage(content=msg["content"])
                       for msg in message_history if msg["role"] in ("user", "assistant")]

        # Step 2: Add the current user query as a new HumanMessage
        lc_messages.append(HumanMessage(content=user_query))

        # Step 3: Call LangGraph with full history
        result = compiled_graph.invoke(
            {"messages": lc_messages},
            config={"configurable": {"thread_id": thread_id}}
        )

        # Step 4: Find only the final assistant response (skip ToolMessages)
        final_response = None
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
        elif isinstance(result, list):
            messages = result
        else:
            messages = []

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                final_response = msg.content
                break

        # Step 5: Return only the assistant message to be appended in UI
        return [{"role": "assistant", "content": final_response}] if final_response else []

    except Exception as e:
        return [{"role": "assistant", "content": f" Error: {str(e)}"}]



# CLI Interface
# if __name__ == "__main__":
#     print(" LangGraph Assistant: Type 'exit' to quit")
#     thread_id = "user-123"
    
#     while True:
#         user_input = input("\nYou: ").strip()
#         if user_input.lower() in ("exit", "quit"):
#             break
            
#         try:
#             result = compiled_graph.invoke(
#                 {"messages": [HumanMessage(content=user_input)]},
#                 config={
#                     "configurable": {"thread_id": thread_id},
#                     "recursion_limit": 20
#                 }
#             )
#             print("Assistant:", result["messages"][-1].content)
#         except Exception as e:
#             print(f"Error: {str(e)}")