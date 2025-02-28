from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool, Tool
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the Groq client
model = init_chat_model("llama3-8b-8192", model_provider="groq", api_key="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")
memory = InMemoryChatMessageHistory(session_id="test-session")
# Initialize the search tool
duckduckgo_search = DuckDuckGoSearchRun()
# Create a structured prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, if you need to have up to date information, you can call the tool DuckDuckGo Search"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# TOOLS


# Define a simple tool for the agent

tools = [

         Tool(
             name="DuckDuckGo Search",
             func=duckduckgo_search.run,
             description="Useful only to search the internet about a query and return up-to-date relevant results."
         )
         ]

# Create an agent with tools and prompt
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "test-session"}}

# Example conversation
"""print(
    agent_with_chat_history.invoke(
        {"input": "Hi, I'm Polly! What's the output of magic_function of 3?"}, config
    )["output"]
)
print("---")
print(agent_with_chat_history.invoke({"input": "Remember my name?"}, config)["output"])
print("---")
print(
    agent_with_chat_history.invoke({"input": "What was that output again?"}, config)[
        "output"
    ]
)"""

# Real-time user interaction
while True:
    user_question = input("You: ")
    if user_question.lower() == 'exit':
        print("Goodbye! Have a great day!")
        break

    # Agent handles the conversation and tool invocation
    response = agent_with_chat_history.invoke({"input": user_question}, config)
    assistant_answer = response["output"]
    print(f"Assistant: {assistant_answer}\n")
