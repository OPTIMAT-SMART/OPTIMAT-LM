from typing import Annotated
from transformers import pipeline
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# Initialize Phi model using pipeline
pipe = pipeline("text-generation", model="microsoft/phi-4", device="cuda")


def chatbot(state: State):
    # Combine messages into a single string
    conversation = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            conversation += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            conversation += f"Assistant: {message.content}\n"
    
    # Generate response using pipeline
    response = pipe(conversation + "Assistant:", max_length=512)[0]['generated_text']
    
    # Extract just the assistant's response
    response = response.split("Assistant:")[-1].strip()
    
    return {"messages": [AIMessage(content=response)]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()