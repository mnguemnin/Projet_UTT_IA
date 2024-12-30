import streamlit as st
#from streamlit_chat import message

# Page Configuration
st.set_page_config(page_title="Travel Agency Chatbot", page_icon="🌍")

# App Title
st.title("Travel Agency Chatbot – Your Travel Companion 🌏")

# Initialize session state for chatbot conversation if not already done
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Chatbot response simulation function (replace this with actual chatbot logic)
def chatbot_response(user_input):
    # Sample logic for demonstration
    if "flight" in user_input.lower():
        return "Sure, I can help you with flight bookings. Where are you planning to travel?"
    elif "hotel" in user_input.lower():
        return "We have great hotel deals! Could you share your destination?"
    elif "package" in user_input.lower():
        return "Our travel packages are perfect for vacations. Would you like to know more about them?"
    else:
        return "I'm here to help with your travel plans. Could you tell me what you're looking for?"

# Chat Interface
st.sidebar.header("Chat with Travel Bot")
user_input = st.sidebar.text_input("Type your question:", placeholder="E.g., I want to book a flight to Paris.")

if st.sidebar.button("Send") and user_input:
    # Add user message to conversation history
    st.session_state['conversation'].append({"user": user_input})

    # Generate chatbot response and add to conversation history
    bot_response = chatbot_response(user_input)
    st.session_state['conversation'].append({"bot": bot_response})

# Display chat history
for entry in st.session_state['conversation']:
    if "user" in entry:
        #message(entry['user'], is_user=True, key=f"user_{st.session_state['conversation'].index(entry)}")
        ""
    elif "bot" in entry:
        ""
        #message(entry['bot'], key=f"bot_{st.session_state['conversation'].index(entry)}")

# Additional Features (optional)
# A section for suggested questions
st.sidebar.subheader("Suggested Questions")
st.sidebar.markdown(
    """- How can I book a flight?
    - Do you offer travel packages?
    - Can you recommend a hotel?
    - What are the top destinations for 2024?"""
)
