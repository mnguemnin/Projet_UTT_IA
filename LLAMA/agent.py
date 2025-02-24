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

# DuckDuckGo search setup
duckduckgo_search = DuckDuckGoSearchRun()

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
        if response.status_code == 200:
            results = response.json()
            image_urls = [result['urls']['regular'] for result in results['results']]
            return image_urls
        else:
            return []

def setup_agent(chatbot_name, memory, callbacks):
    unsplash_image_search = UnsplashImageSearch(api_key="HeiIxV4E5Oz7xMLYuxTSRMaHls04Ix_8GqOv1XC")
    # Define tools
    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=duckduckgo_search.run,
            description="Outil utile pour rechercher sur Internet une requête et renvoyer les résultats pertinents et à jour."
        ),
        Tool(
            name="Image Search",
            func=unsplash_image_search.search_images,
            description="Outil dédié pour rechercher des images pertinentes en fonction de la requête."
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


if __name__ == "__main__":
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

        # Print the agent's response
        print(f"{chatbot_name}: {response}")
