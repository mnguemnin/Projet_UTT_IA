import os
import sys
from pathlib import Path
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.callbacks.manager import CallbackManager

from template.template import CustomPromptTemplate, read_template
from parse.parser import CustomOutputParser
from langchain.memory import ConversationBufferMemory  # Import memory class
import streamlit as st

# DuckDuckGo search setup
duckduckgo_search = DuckDuckGoSearchRun()

def setup_agent(chatbot_name, memory, callbacks):
    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=duckduckgo_search.run,
            description="\U0001F30D Useful tool to search the Internet for a query and return relevant and up-to-date results."
        ),
    ]

    prompt = CustomPromptTemplate(
        template=read_template(str(Path(__file__).resolve().parent.parent / "template" / "base2.txt")).replace(
            "{chatbot_name}", chatbot_name),
        tools=tools,
        input_variables=["input", "intermediate_steps", "chat_history"]
    )

    output_parser = CustomOutputParser()

    llm = init_chat_model("llama3-8b-8192", model_provider="groq",
                          api_key="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")

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        callbacks=callbacks
    )
    return agent_executor

st.title("🌍✈️ Les Vacanciers - Travel Chatbot 🏝️🧳")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("💬 Type your travel-related question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    callbacks = CallbackManager([])
    agent_executor = setup_agent("Les Vacanciers", st.session_state.memory, callbacks)
    response = chat_with_agent(user_input, "Les Vacanciers", st.session_state.memory, callbacks)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(f"🌴 {response} 🏖️")
