# Overview

This repository contains three separate projects, each designed to demonstrate different functionalities using LangChain, OpenAI, and other tools for developing AI agents. The projects are organized into three folders: `Agent_Executor`, `AgentWithPinecone`, and `multiAgentWorkFlow`. Below are detailed descriptions of each project and how they work.

## Table of Contents

- [Agent Executor](#agent-executor)
- [Agent With Pinecone](#agent-with-pinecone)
- [Multi-Agent Workflow](#multi-agent-workflow)

## Agent Executor

This project demonstrates the creation and execution of a LangChain agent using OpenAI's GPT-4o and Tavily search tools. The agent is designed to handle user inputs and provide responses based on previous interactions and predefined actions.

### Key Components

- **Environment Setup:** API keys for OpenAI and Tavily are loaded using the `dotenv` library.
- **LangChain Agent Creation:** Utilizes `langchain` and `langchain_community` libraries to create an agent that integrates with OpenAI and Tavily.
- **Agent State Management:** Manages input, chat history, intermediate steps, and outcomes using a custom `AgentState` class.
- **Graph Definition:** Defines the workflow of the agent using nodes and edges, enabling conditional and normal transitions based on agent decisions.
- **Agent Execution Loop:** Continuously prompts the user for input and processes responses using the defined agent workflow.

### How It Works

1. **Initialization:** Load environment variables and set up API keys.
2. **Agent Creation:** Define the LangChain agent with tools and prompts.
3. **State Definition:** Create a state object to manage the agent's input, history, and outcomes.
4. **Node and Edge Definition:** Set up nodes for agent actions and tool execution, along with conditional and normal edges to handle transitions.
5. **Execution Loop:** Continuously read user input, invoke the agent, and print responses until the user exits.

## Agent With Pinecone

This project focuses on integrating Pinecone for vector storage and search capabilities alongside OpenAI for embedding generation and weather data retrieval.

### Key Components

- **Environment Setup:** API keys for OpenAI, Pinecone, and OpenWeather are loaded using the `dotenv` library.
- **Pinecone Initialization:** Initializes Pinecone with the specified index and configuration.
- **PDF Processing:** Reads and processes PDF documents to create text embeddings using OpenAI's embeddings.
- **Tool Definition:** Defines tools for Pinecone search, weather information retrieval, and a dummy tool for extracting city names.
- **Graph Definition:** Sets up a workflow to route tasks between different agents based on the input content.

### How It Works

1. **Initialization:** Load environment variables and set up API keys.
2. **Pinecone Setup:** Initialize Pinecone and create an index if it doesn't exist.
3. **PDF Import:** Read PDF files, generate embeddings, and store them in Pinecone.
4. **Tool Definition:** Create tools for searching Pinecone, retrieving weather data, and extracting city names.
5. **Node and Edge Definition:** Define nodes for each agent and tool, with conditional edges to route tasks based on the agent's decisions.
6. **Execution Loop:** Continuously prompt the user for input, invoke the appropriate agent, and print responses until the user exits.

## Multi-Agent Workflow

This project demonstrates a multi-agent collaboration approach, where specialized agents handle different tasks or domains, and route tasks to the correct "expert" agent.

### Key Components

- **Environment Setup:** API keys for OpenAI and Tavily are loaded using the `dotenv` library.
- **Agent Creation:** Helper functions to create agents for specific tasks, such as research and chart generation.
- **Tool Definition:** Define tools for Tavily search and executing Python code in a REPL.
- **Graph Definition:** Set up a workflow with nodes for each agent and tool, and conditional edges to manage transitions based on the agent's outputs.

### How It Works

1. **Initialization:** Load environment variables and set up API keys.
2. **Agent and Tool Creation:** Define agents for research and chart generation, and tools for Tavily search and Python REPL.
3. **State Definition:** Create a state object to manage messages and track the most recent sender.
4. **Node and Edge Definition:** Define nodes for each agent and tool, with conditional edges to route tasks based on the agent's outputs.
5. **Execution Loop:** Continuously read user input, invoke the appropriate agent, and print responses until the user exits.

## Getting Started

To run these projects, follow the steps below:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-repo-url.git


2. **Install Dependencies:**

Navigate to each project folder and install the required dependencies:

'''bash
cd Agent_Executor
pip install -r requirements.txt

cd ../AgentWithPinecone
pip install -r requirements.txt

cd ../multiAgentWorkFlow
pip install -r requirements.txt

3. **Set Up Environment Variables:**

Create a .env file in each project folder with the necessary API keys:

OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
OPENWEATHER_API_KEY=your_openweather_api_key

