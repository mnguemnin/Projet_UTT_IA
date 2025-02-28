import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
import requests
import spacy

# Charger le modèle de langue française pour l'extraction des noms de lieux
nlp = spacy.load("fr_core_news_sm")

# Configuration de la page Streamlit
st.set_page_config(page_title="Les vacanciers", layout="wide")

# Initialisation du modèle LLM
model = init_chat_model("llama3-8b-8192", model_provider="groq",
                        api_key="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")

# Outil de recherche DuckDuckGo
duckduckgo_search = DuckDuckGoSearchRun()

# Gestion de l'historique des messages
def get_memory():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory(session_id="streamlit-session")
    return st.session_state.chat_history

# Classe pour la recherche d'images via Unsplash
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

# Classe pour la recherche de vidéos via YouTube Data API
class YouTubeVideoSearch:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3/search"

    def search_videos(self, query):
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 3,  # Limiter à 3 vidéos
            "key": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            results = response.json()
            video_urls = [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in results['items']]
            return video_urls
        else:
            return []

# Initialisation des outils de recherche d'images et de vidéos
unsplash_image_search = UnsplashImageSearch(api_key="HeiIxV4E5Oz7xMLYuxTSRMaHls04Ix_8GqOv1XCycUQ")
youtube_video_search = YouTubeVideoSearch(api_key="AIzaSyBHi1MbeTX7MaExnAh4xrsnKILpLZiKtYo")

# Définir les outils dans le chatbot
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=duckduckgo_search.run,
        description="Outil utile pour rechercher sur Internet une requête et renvoyer les résultats pertinents et à jour."
    ),
    Tool(
        name="Unsplash Image Search",
        func=unsplash_image_search.search_images,
        description="Outil dédié pour rechercher des images pertinentes en fonction de la requête."
    ),
    Tool(
        name="YouTube Video Search",
        func=youtube_video_search.search_videos,
        description="Outil dédié pour rechercher des vidéos pertinentes en fonction de la requête."
    ),
]

# Définition du template de prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Vous êtes un assistant utile, vous aidez les gens à planifier leur voyage dans n'importe quelle ville ou site touristique dans le monde. Si vous avez besoin d'informations à jour, utilisez l'outil de recherche DuckDuckGo. Voici un exemple de la manière dont vous structurez vos réponses.\n"
                   "Question : Je vais à [Nom de la ville ou du lieu]\n"
                   "Réponse :\n"
                   "C'est génial ! [Nom de la ville ou du lieu] a tellement à offrir. Voici quelques lieux incontournables selon vos centres d'intérêt :\n\n"
                   "**Lieux iconiques**\n"
                   "[Lieu 1] – [Description du lieu].\n"
                   "<img src='{{Image_Lieu_1}}' />\n"
                   "[Lieu 2] – [Description du lieu].\n"
                   "<img src='{{Image_Lieu_2}}' />\n"
                   "[Lieu 3] – [Description du lieu].\n"
                   "<img src='{{Image_Lieu_3}}' />\n"
                   "**Voulez-vous des recommandations pour des intérêts spécifiques, comme des joyaux cachés ou la vie nocturne ?** 😊"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Création de l'agent avec les outils
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: get_memory(),
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "streamlit-session"}}

# Fonction pour extraire le nom du lieu
def extract_place_name(text):
    doc = nlp(text)
    places = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
    return places[0] if places else text  # Retourne le premier lieu trouvé ou la requête entière

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
        # Extraire le nom du lieu de la requête
        place_name = extract_place_name(user_input)

        # Obtenir la réponse de l'agent
        response = agent_with_chat_history.invoke({"input": user_input}, config)
        assistant_response = response["output"]

        # Recherche d'images pour le lieu mentionné
        image_results = unsplash_image_search.search_images(place_name)
        if image_results:
            # Ajouter des images à la réponse
            for image_url in image_results:
                assistant_response += f"\n\n![Image]({image_url})"

        # Recherche de vidéos pour le lieu mentionné
        if "vidéo" in user_input.lower() or "video" in user_input.lower():
            video_results = youtube_video_search.search_videos(place_name)
            if video_results:
                # Ajouter des vidéos à la réponse
                for video_url in video_results:
                    assistant_response += f"\n\n[Regarder la vidéo]({video_url})"

        # Afficher la réponse de l'IA avec les titres en gras et les images/vidéos
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    except Exception as e:
        # En cas d'erreur, afficher un message générique
        st.session_state.messages.append({"role": "assistant", "content": "Désolé, une erreur s'est produite lors de la recherche d'informations."})
        with st.chat_message("assistant"):
            st.markdown("Désolé, une erreur s'est produite lors de la recherche d'informations.")
