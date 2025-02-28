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
import requests
import re
from pathlib import Path



# DuckDuckGo search setup
duckduckgo_search = DuckDuckGoSearchRun()


class GoogleImageSearch:


    def __init__(self, api_key, cx):
        self.api_key = api_key
        self.cx = cx
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search_images(self, query):
        params = {
            "q": f"{query} tourisme",  # Ajouter "tourisme" pour améliorer la pertinence
            "cx": self.cx,
            "key": self.api_key,
            "searchType": "image",  # Rechercher uniquement des images
            "num": 1  # Limiter à 1 image
        }
        response = requests.get(self.base_url, params=params)

        if response.status_code == 200:
            results = response.json()
            if "items" in results and len(results["items"]) > 0:
                return results["items"][0]["link"]  # Retourne la première URL d'image
            else:
                return None  # Aucun résultat trouvé
        else:
            print(f"Erreur lors de la recherche d'images : {response.status_code}")
            return None  # En cas d'erreur, retourne None


def remplacer_photos(texte, searcher):
    """
    Remplace les placeholders [photo de ...] par des liens d'images trouvés dynamiquement.

    :param texte: Texte contenant les placeholders [photo de ...]
    :param searcher: Instance de GoogleImageSearch pour récupérer les images
    :return: Texte avec les liens d'images insérés
    """

    def remplacement(match):
        lieu = match.group(1).strip()  # Extrait le texte entre [photo de ...]
        image_url = searcher.search_images(lieu)  # Recherche d'image pour le lieu
        print(image_url)
        print(match)
        print(lieu)
        return f" ![Photo de {lieu}]({image_url}) "



    # Remplacement des placeholders dynamiquement
    texte_modifie = re.sub(r"(?i)\[(.*?)\]", remplacement, texte)
    return texte_modifie

searcher = google_image_search = GoogleImageSearch(api_key="AIzaSyDvWP8Pxvo5xMOPcfAG3LDjDjbeNE4oxBg", cx="c00cc1bbcd37248c7")


def chat_with_agent(user_input: str, chatbot_name: str, memory, callbacks):
    # Set up agent
    agent_executor = setup_agent(chatbot_name, memory, callbacks)

    # Run agent with user input
    try:
        # Capture the full response
        response = agent_executor.run(user_input, callbacks=callbacks)

        # Si la réponse est sous forme de chaîne, la diviser par "Réponse finale :"
        if isinstance(response, str):
            # Divisez la réponse à partir de "Réponse finale :"
            response_parts = response.split("Réponse finale :")

            # Si "Réponse finale :" est trouvé, prendre la partie après
            if len(response_parts) > 1:
                final_response = response_parts[-1].strip()  # La dernière partie après "Réponse finale :"
                return final_response

            # Si "Réponse finale :" n'est pas trouvé, retourner la réponse entière
            return response.strip()

        # Si la réponse est un dictionnaire, accédez à l'élément 'output'
        if isinstance(response, dict):
            return response.get("output", "Aucune réponse trouvée.")

        # Par défaut, retourner la réponse brute
        return response.strip()

    except Exception as e:
        return f"Une erreur est survenue : {str(e)}"

def setup_agent(chatbot_name, memory, callbacks):
    google_image_search = GoogleImageSearch(api_key="AIzaSyDvWP8Pxvo5xMOPcfAG3LDjDjbeNE4oxBg", cx="c00cc1bbcd37248c7")

    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=duckduckgo_search.run,
            description="\U0001F30D Useful tool to search the Internet for a query and return relevant and up-to-date results."
        ),

    ]
    template_path = Path(__file__).parent / "template" / "base2.txt"

    prompt = CustomPromptTemplate(
        template=read_template(template_path).replace(
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
    response = remplacer_photos(response, searcher)
    response='🌴'+response+'🏖️'
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(f"🏖️ {response} 🌴", unsafe_allow_html=True)



