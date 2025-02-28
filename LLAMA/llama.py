from groq import Groq

# Initialize the Groq client
client = Groq(
    api_key="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",
)



def main():
    print("Welcome to the Q&A Chatbot!")
    print("Ask me anything, and I'll do my best to provide helpful answers.")
    print("Type 'exit' to end the chat.\n")

    while True:
        # Get the user's question
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            print("Goodbye! Have a great day!")
            break

        # Interact with the Groq API
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": user_question}
            ],
            temperature=1,
            max_tokens=500,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Display the assistant's answer
        assistant_answer = response.choices[0].message.content
        print(f"Assistant: {assistant_answer}\n")

if __name__ == "__main__":
    main()
