import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os
import requests
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, START
import functools
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Ensure the keys are loaded correctly
assert OPENAI_API_KEY is not None, "OPENAI_API_KEY is not set"
assert PINECONE_API_KEY is not None, "PINECONE_API_KEY is not set"
assert PINECONE_INDEX_NAME is not None, "PINECONE_INDEX_NAME is not set"
assert OPENWEATHER_API_KEY is not None, "OPENWEATHER_API_KEY is not set"

# Define the LLM
llm = ChatOpenAI(model="gpt-4o")

# Getting vector index
embeddings = OpenAIEmbeddings()
docsearch = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)

# Define the Pinecone search tool
@tool
def pinecone_search(query: Annotated[str, "The query to search in Pinecone."]):
    """
    Search the Pinecone vector database with the provided query.
    
    Args:
    query (str): The query to search in Pinecone.
    
    Returns:
    str: The top result from the Pinecone search.
    """
    response = docsearch.similarity_search(query)
    if response:
        return response[0].page_content
    else:
        return "No matches found."

# Define OpenWeather tool
@tool
def get_weather(city: Annotated[str, "The city to get weather information for."]):
    """
    Get the weather information for the specified city using the OpenWeather API.
    
    Args:
    city (str): The city to get weather information for.
    
    Returns:
    dict: The weather information for the specified city.
    """
    api_key = OPENWEATHER_API_KEY
    if not api_key:
        return {"error": "OpenWeather API key is not set."}

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city}"
    response = requests.get(complete_url)
    return response.json()

# Define a tool for suggesting clothing using LLM response
@tool
def suggest_clothing_llm(weather_data: Annotated[dict, "Weather data for clothing suggestions."]):
    """
    Suggest clothing based on weather data using LLM response.
    
    Args:
    weather_data (dict): The weather data to base the suggestions on.
    
    Returns:
    str: The clothing suggestion.
    """
    temp = weather_data.get('main', {}).get('temp', 0) - 273.15  # Convert from Kelvin to Celsius
    weather_description = weather_data.get('weather', [{}])[0].get('description', 'weather')
    
    prompt = f"The current temperature is {temp:.1f} degrees Celsius and the weather is {weather_description}. Based on this, suggest appropriate clothing."
    llm_response = llm(prompt)
    
    return llm_response

# Define a general response tool using LLM
@tool
def general_response(query: Annotated[str, "General queries"]):
    """
    Provide general responses using LLM.
    
    Args:
    query (str): The input query.
    
    Returns:
    str: The LLM response.
    """
    return llm(query)

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

# General agent using GPT-4o with general response tool
general_agent = create_agent(
    llm,
    [general_response],
    system_message="You should respond to general queries."
)
general_node = functools.partial(agent_node, agent=general_agent, name="GeneralAgent")

# RAG agent using Pinecone
rag_agent = create_agent(
    llm,
    [pinecone_search],
    system_message="You should provide knowledge-based responses using the Pinecone knowledge base."
)
rag_node = functools.partial(agent_node, agent=rag_agent, name="RAGAgent")

# Weather agent using OpenWeather API and clothing suggestion tool
weather_agent = create_agent(
    llm,
    [get_weather, suggest_clothing_llm],
    system_message="You should provide weather information and clothing suggestions based on the weather."
)
weather_node = functools.partial(agent_node, agent=weather_agent, name="WeatherAgent")

from typing import Annotated, Sequence, TypedDict
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

from langgraph.prebuilt import ToolNode

# Define tools
tools = [pinecone_search, get_weather, suggest_clothing_llm, general_response]
tool_node = ToolNode(tools)

# Define the router logic
from typing import Literal

def router(state) -> Literal["call_tool", "__end__", "RAGAgent", "WeatherAgent", "GeneralAgent"]:
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content.lower()
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    if "weather" in content:
        if "wear" in content or "clothes" in content or "dress" in content:
            return "WeatherAgent"
        else:
            # Ask if the user wants clothing suggestions
            state["messages"].append(
                AIMessage(content="Do you want to know what to wear in this weather?", name="WeatherAgent")
            )
            return "WeatherAgent"
    elif "business" in content:  # Check if the query is business-related
        return "RAGAgent"
    else:
        return "GeneralAgent"

# Define the graph
workflow = StateGraph(AgentState)

workflow.add_node("GeneralAgent", general_node)
workflow.add_node("RAGAgent", rag_node)
workflow.add_node("WeatherAgent", weather_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "GeneralAgent",
    router,
    {"WeatherAgent": "WeatherAgent", "RAGAgent": "RAGAgent", "call_tool": "call_tool", "__end__": END}
)

workflow.add_conditional_edges(
    "RAGAgent",
    router,
    {"WeatherAgent": "WeatherAgent", "GeneralAgent": "GeneralAgent", "call_tool": "call_tool", "__end__": END}
)
workflow.add_conditional_edges(
    "WeatherAgent",
    router,
    {"RAGAgent": "RAGAgent", "GeneralAgent": "GeneralAgent", "call_tool": "call_tool", "__end__": END}
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "GeneralAgent": "GeneralAgent",
        "RAGAgent": "RAGAgent",
        "WeatherAgent": "WeatherAgent",
    },
)
workflow.add_edge(START, "GeneralAgent")
graph = workflow.compile()

# Main loop to handle user input and generate responses
while True:
    print("while 1")
    content = input("you: ")
    if content.lower() == "exit" or content.lower() == "quit":
        break
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=content #"What is the weather in New York today?"
                )
            ],
        },
        {"recursion_limit": 150},
    )
    for s in events:
        print(s)
        print("----")
