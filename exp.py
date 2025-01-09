import requests
from langchain.tools import BaseTool
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.schema import AgentAction, AgentFinish
import re

# Create custom tool for navigation API calls
class NavigationTool(BaseTool):
    name: str = "navigation"
    description: str = "Validates if origin and destination addresses are valid locations"
    
    def _run(self, origin: str, destination: str):
        response = requests.post(
            'http://127.0.0.1:5000/api/navigate',
            json={"origin": origin, "destination": destination},
            headers={"Content-Type": "application/json"}
        )
        return response.json()

    def _arun(self, origin: str, destination: str):
        raise NotImplementedError("Async not supported")

# Create prompt template for the agent
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    input_variables: list[str]

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

template = """You are a helpful transportation assistant. When a user asks about travel between two locations:
1. ALWAYS extract the origin and destination
2. Use the navigation tool with the EXACT locations mentioned
3. Provide a helpful response based on the tool's output

Example:
Question: "I want to go from Seattle to Portland"
Thought: I need to validate these locations
Action: navigation
Action Input: {"origin": "Seattle", "destination": "Portland"}

Question: {input}
Thought: """

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        # Check if this is a final answer
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Extract action and action input
        action_match = re.search(r'Action: (.*?)[\n]', llm_output, re.DOTALL)
        action_input_match = re.search(r'Action Input: (.*)', llm_output, re.DOTALL)
        
        if not action_match or not action_input_match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            
        action = action_match.group(1).strip()
        action_input = action_input_match.group(1).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

def create_agent(pipe):
    # Initialize tools and agent
    nav_tool = NavigationTool()
    tools = [nav_tool]

    # Create LangChain wrapper for Phi-4
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create the agent
    prompt = CustomPromptTemplate(
        template=template,
        input_variables=["input"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    output_parser = CustomOutputParser()

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        tools=tools,
        stop=["Observation:", "Final Answer:"]
    )
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def handle_chat_input(agent_executor, user_input):
    try:
        response = agent_executor.invoke({"input": user_input})
        return f"Assistant: {response['output']}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage:
    from transformers import pipeline
    # Initialize your pipeline here
    pipe = pipeline("text-generation", model="microsoft/phi-4", device="cuda")
    
    agent_executor = create_agent(pipe)
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        response = handle_chat_input(agent_executor, user_input)
        print(response)