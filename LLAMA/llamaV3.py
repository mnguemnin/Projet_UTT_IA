from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool
from langchain_groq import ChatGroq

# Define a mock search tool for demonstration purposes
@tool
def search_tool(query: str) -> str:
    return f"Searching the web for: {query}"

# Create a new class that extends ChatGroq and implements bind_tools
class ChatGroqWithTools(ChatGroq):
    def __init__(self, model, temperature, api_key):
        super().__init__(model=model, temperature=temperature, api_key=api_key)
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools
        return self  # Return the object to keep chaining possible

# Initialize the LLM with our new class
llm = ChatGroqWithTools(
    model="llama3-70b-8192",
    temperature=0,
    api_key="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"  # Replace with your actual Groq API key
)

# Bind the tools (in this case, we only have the search tool)
llm.bind_tools([search_tool])

# Create tools list
tools = [search_tool]

# Initialize the agent using the new LLM class
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # You can change agent type if needed
    verbose=True
)

# Execute the agent with a test query
response = agent.run("What are the best travel destinations in France?")
print(response)
