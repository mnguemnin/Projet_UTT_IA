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


# Initialisation des outils de recherche

class UnsplashImageSearch:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.unsplash.com/search/photos"

    def search_images(self, query):
        headers = {
            "Authorization": f"Client-ID {self.api_key}"
        }
        params = {
            "query": query,
            "per_page": 5  # Limiter à 5 images
        }
        response = requests.get(self.base_url, headers=headers, params=params)
        print(query )
        if response.status_code == 200:
            results = response.json()
            image_urls = [result['urls']['regular'] for result in results['results']]
            return image_urls
        else:
            return []

def setup_agent(chatbot_name, memory, callbacks):
    #unsplash_image_search = UnsplashImageSearch(api_key="HeiIxV4E5Oz7xMLYuxTSRMaHls04Ix_8GqOv1XC")
    google_image_search = GoogleImageSearch(api_key="AIzaSyDvWP8Pxvo5xMOPcfAG3LDjDjbeNE4oxBg", cx="c00cc1bbcd37248c7")
    # Define tools
    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=duckduckgo_search.run,
            description="Outil utile pour rechercher sur Internet une requête et renvoyer les résultats pertinents et à jour."
        ),


    ]

    # Set up the prompt template using base.txt and the tools list
    prompt = CustomPromptTemplate(
        template=read_template(str(Path(__file__).resolve().parent.parent / "template" / "base2.txt")).replace(
            "{chatbot_name}", chatbot_name),
        tools=tools,
        input_variables=["input", "intermediate_steps", "chat_history"]
    )

    # Output parser
    output_parser = CustomOutputParser()

    # Initialize the language model
    llm = init_chat_model("llama3-8b-8192", model_provider="groq",
                          api_key="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")

    # Create LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Extract tool names
    tool_names = [tool.name for tool in tools]

    # Create agent
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    # Create AgentExecutor, using ConversationBufferMemory as memory
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,  # Now using the correct memory type
        callbacks=callbacks
    )
    return agent_executor


"""def chat_with_agent(user_input: str, chatbot_name: str, memory, callbacks):
    # Set up agent
    agent_executor = setup_agent(chatbot_name, memory, callbacks)

    # Run agent with user input
    try:
        response = agent_executor.run(user_input, callbacks=callbacks)
        if isinstance(response, dict):
            return response.get("output")
        else:
            return response
    except Exception as e:
        return f"Error occurred: {str(e)}"
        """

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
        print(match)
        return f"![Photo de {lieu}]({image_url})"

    # Remplacement des placeholders dynamiquement
    texte_modifie = re.sub(r"\[photo (.*?)\]", remplacement, texte)
    return texte_modifie


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
searcher = google_image_search = GoogleImageSearch(api_key="AIzaSyDvWP8Pxvo5xMOPcfAG3LDjDjbeNE4oxBg", cx="c00cc1bbcd37248c7")

if __name__ == "__main__":
    google_image_search = GoogleImageSearch(api_key="AIzaSyDvWP8Pxvo5xMOPcfAG3LDjDjbeNE4oxBg", cx="c00cc1bbcd37248c7")
    print(google_image_search.search_images('images Douala'))
    # Initialize chatbot and memory
    chatbot_name = "TolkAI"
    #user_input = "Who are you?"

    # Use ConversationBufferMemory for memory (this will store conversation history)
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Define callbacks (using CallbackManager or your own callback functions)
    callbacks = CallbackManager([])  # Empty list of callbacks, but can be extended

    # Get response from agent
    #response = chat_with_agent(user_input, chatbot_name, memory, callbacks)

    # Print the response
    #print(response)
    while True:
        # Take user input
        user_input = input("You: ")

        # If the user wants to exit, break the loop
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Get the response from the agent
        response = chat_with_agent(user_input, chatbot_name, memory, callbacks)
        response = remplacer_photos(response, searcher)
        # Print the agent's response
        print(f"{chatbot_name}: {response}")
