import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun

# Configuration de la page
st.set_page_config(page_title="Les vacanciers", layout="wide")

# Initialisation du modèle LLM
model = init_chat_model("llama3-8b-8192", model_provider="groq",
                        api_key="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")


# Gestion de l'historique des messages
def get_memory():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory(session_id="streamlit-session")
    return st.session_state.chat_history


# Initialisation de l'outil de recherche DuckDuckGo
duckduckgo_search = DuckDuckGoSearchRun()

# Définition du template de prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. If you need up-to-date information, use the DuckDuckGo Search tool."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Définition des outils
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=duckduckgo_search.run,
        description="Useful only to search the internet about a query and return up-to-date relevant results."
    )
]

# Création de l'agent
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: get_memory(),
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "streamlit-session"}}

# Interface utilisateur Streamlit
st.title("🤖 IA de Hans ")

# Zone d'affichage des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie utilisateur
user_input = st.chat_input("Pose-moi une question...")
if user_input:
    # Ajouter le message utilisateur à l'affichage
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Obtenir la réponse du modèle
    response = agent_with_chat_history.invoke({"input": user_input}, config)
    assistant_response = response["output"]

    # Afficher la réponse de l'IA
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
