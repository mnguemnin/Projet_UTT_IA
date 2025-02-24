import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
#from template.template import CustomPromptTemplate, read_template

# Configuration de la page
st.set_page_config(page_title="Les vacanciers", layout="wide")

# Initialisation du modèle LLM
model = init_chat_model("llama3-8b-8192", model_provider="groq",
                        api_key="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")
duckduckgo_search = DuckDuckGoSearchRun()


# Gestion de l'historique des messages
def get_memory():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory(session_id="streamlit-session")
    return st.session_state.chat_history



# Outil de recherche d'images via DuckDuckGo
class ImageSearchTool:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()

    def search_images(self, query):
        # Rechercher des images avec DuckDuckGo
        results = self.search.run(query)
        # Extraire les URLs des images depuis les résultats
        image_urls = []
        for result in results.get("results", []):
            if "image" in result:
                image_urls.append(result["image"])
        return image_urls


# Créer une instance de l'outil de recherche d'images
image_search_tool = ImageSearchTool()

"""prompt = CustomPromptTemplate(
        template=read_template(str(Path(__file__).resolve().parent.parent / "template" / "base.txt")).replace(
            "{chatbot_name}", chatbot_name),
        tools=tools,
        input_variables=["input", "intermediate_steps", "chat_history"]
    )
"""
    # Instantiate a CustomOutputParser object for parsing output
#output_parser = CustomOutputParser()
# Définir les outils dans le chatbot
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=duckduckgo_search.run,
        description="Outil utile pour rechercher sur Internet une requête et renvoyer les résultats pertinents et à jour."
    ),
    Tool(
        name="DuckDuckGo Image Search",
        func=image_search_tool.search_images,
        description="Outil dédié pour rechercher des images pertinentes en fonction de la requête."
    ),
]

# Définition du template de prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Vous êtes un assistant utile, vous aidez les gens à planifier leur voyage dans un pays ou une ville. Si vous avez besoin d'informations à jour, utilisez l'outil de recherche DuckDuckGo. Voici un exemple de la manière dont vous structurez vos réponses.\n"
                   "Question : Je vais à Paris\n"
                   "Réponse :\n"
                   "C'est génial ! Paris a tellement à offrir. Voici quelques lieux incontournables selon vos centres d'intérêt :\n\n"
                   "**Lieux iconiques**\n"
                   "Tour Eiffel – À ne pas manquer, surtout la nuit lorsqu'elle scintille !\n"
                   "<img src='{{Tour_Eiffel_Image}}' />\n"
                   "Musée du Louvre – Abritant la Mona Lisa et d'innombrables chefs-d'œuvre.\n"
                   "<img src='{{Louvre_Image}}' />\n"
                   "Cathédrale Notre-Dame – Toujours en restauration, mais la zone est magnifique.\n"
                   "<img src='{{NotreDame_Image}}' />\n"
                   "Arc de Triomphe & Champs-Élysées – Parfait pour une promenade pittoresque et du shopping.\n"
                   "<img src='{{ArcTriomphe_Image}}' />\n"
                   "Basilique du Sacré-Cœur & Montmartre – Magnifique basilique avec une vue incroyable sur la ville.\n"
                   "<img src='{{SacreCoeur_Image}}' />\n"
                   "**Voulez-vous des recommandations pour des intérêts spécifiques, comme des joyaux cachés ou la vie nocturne ?** 😊"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
#llm_chain = LLMChain(llm=model, prompt=prompt)

# Création de l'agent avec l'outil de recherche d'images
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, callbacks=callbacks)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: get_memory(),
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "streamlit-session"}}

# Interface utilisateur Streamlit
st.title("🤖 IA de Hans")

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

    try:
        # Obtenir la réponse de l'agent
        response = agent_with_chat_history.invoke({"input": user_input}, config)
        assistant_response = response["output"]

        # Recherche d'images si nécessaire
        if "Paris" in user_input or "Tour Eiffel" in user_input or "Musée du Louvre" in user_input:
            # Chercher des images des lieux mentionnés
            image_results = image_search_tool.search_images(user_input)
            if image_results:
                # Ajouter des images à la réponse
                for image_url in image_results[:3]:  # Limiter à 3 images
                    assistant_response += f"\n\n![Image]({image_url})"

        # Afficher la réponse de l'IA avec les titres en gras et les images
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    except Exception as e:
        # En cas d'erreur, afficher un message générique
        st.session_state.messages.append({"role": "assistant", "content": "Désolé, une erreur s'est produite lors de la recherche d'informations."})
        with st.chat_message("assistant"):
            st.markdown("Désolé, une erreur s'est produite lors de la recherche d'informations.")
