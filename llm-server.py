# filename: main.py

import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# ------------------
# IMPORTS FROM YOUR CODE
# ------------------
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage

# ------------------
# YOUR EXISTING CODE
# ------------------

llm = ChatOpenAI(
    api_key="ollama",
    model="llama3.3",
    base_url="http://localhost:11434/v1",
)

def find_providers(origin: str, destination: str) -> str:
    """
    Find providers that can provide services between origin and destination addresses, and other criteria.
    """
    response = requests.post(
        'https://optimat-db.onrender.com/api/v1/providers/match',
        json={
            "departureTime": "2024-03-20T09:30:00-07:00",
            "returnTime": "2024-03-20T14:45:00-07:00",
            "originAddress": origin,
            "destinationAddress": destination,
            "eligibility": ["senior", "disability"],
            "equipment": ["wheelchair"],
            "healthConditions": ["none"],
            "needsCompanion": True,
            "allowsSharing": True
        }
    )
    response_json = response.json()
    return [provider["provider_name"] for provider in response_json["data"]]

ASSISTANCE_PROMPT = """
You are a helpful assistant developed by OPTIMAT, a team that provides transportation services for people with disabilities and seniors.
You are able to find paratransit providers that can provide services between origin and destination addresses, and other criteria.

Your goal is to ask for the origin and destination addresses, and other criteria, and then find the paratransit providers that can provide services between the origin and destination addresses, and other criteria.
Please do not make up information, only use the information provided by the user.
"""

llama_tools = llm.bind_tools([find_providers])    

def tool_calling_llm(state: MessagesState):
    return {"messages": [llama_tools.invoke(state["messages"])]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([find_providers]))
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()


# ------------------
# FASTAPI SETUP
# ------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    # The entire conversation, including the newest user message
    messages: List[dict]

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Accepts the entire conversation (messages) 
    and returns ONLY the new AI messages.
    """
    # Convert the incoming messages to Langchain message types
    history = [SystemMessage(content=ASSISTANCE_PROMPT)]
    for msg in request.messages:
        if msg["role"] == "ai":
            history.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        else:
            # If you have system/other roles, handle them here as needed
            pass

    # IMPORTANT: We do NOT append another user message here, because 
    # the last user message is already in `request.messages`.
    # So we just pass 'history' directly to the graph.
    response_state = graph.invoke({"messages": history})

    # Now, response_state["messages"] contains the entire conversation, 
    # including brand new AI messages. 
    # We only want to return the newly generated messages:
    # i.e., messages after the length of the original conversation.
    output = []
    new_messages = response_state["messages"][len(history):]

    for msg in new_messages:
        if isinstance(msg, AIMessage):
            output.append({"role": "ai", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            output.append({"role": "human", "content": msg.content})
        else:
            output.append({"role": "system/other", "content": str(msg)})

    return {"messages": output}

@app.get("/health")
def health_check():
    """
    Simple endpoint to check if the server is running.
    Returns 'Online' if the server is up.
    """
    return {"status": "Online"}

# ------------------
# ENTRY POINT
# ------------------
if __name__ == "__main__":
    uvicorn.run("llm-server:app", host="0.0.0.0", port=8001, reload=True)