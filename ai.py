import nltk
import streamlit as st
import speech_recognition as sr
from transformers import pipeline

# Load and preprocess the data using the chatbot algorithm
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    return data

def preprocess_data(data):
    # Tokenize and preprocess the data
    tokens = nltk.word_tokenize(data)
    return tokens

# Define a function to transcribe speech into text using the speech recognition algorithm
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Please say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"🗣️ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("❌ Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.write("❌ Could not request results from the speech recognition service.")
    return ""

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering")

# Modify the chatbot function to take both text and speech input from the user
def chatbot_response(user_input, context):
    # Use the transformer model to get the answer
    result = qa_pipeline(question=user_input, context=context)
    return result['answer']

# Create a Streamlit app
def main():
    st.set_page_config(page_title="Speech-Enabled Chatbot", page_icon="🤖", layout="wide")

    st.sidebar.subheader("🗣️ Speech-Enabled Chatbot")
    st.sidebar.markdown("### About")
    st.sidebar.info("This is a speech-enabled chatbot application built using Streamlit, NLTK, and SpeechRecognition.")

    st.sidebar.markdown("### Contact Info")
    st.sidebar.info("Email: danielayomideh@gmail.com\n\nPhone: +234 706 715 9089")

    st.sidebar.markdown("### Links")
    st.sidebar.markdown("[GitHub](https://github.com)\n[LinkedIn](https://linkedin.com)")

    st.sidebar.markdown("### Help")
    st.sidebar.info("For any issues, please refer to the documentation or contact support.")

    st.sidebar.markdown("### Suggestion Box")
    suggestion = st.sidebar.text_area("Enter your suggestion:")
    if st.sidebar.button("Submit Suggestion"):
        with open("suggestions.txt", "a") as file:
            file.write(suggestion + "\n")
        st.sidebar.success("Suggestion submitted!")

    st.title("🗣️ Speech-Enabled Chatbot")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    input_type = st.radio("Choose input type:", ("Text", "Speech"))

    if input_type == "Text":
        user_input = st.text_input("Enter your message:")
        if st.button("Send"):
            if user_input:
                st.session_state.conversation.append(f"🗣️ You: {user_input}")
                response = chatbot_response(user_input, data)
                st.session_state.conversation.append(f"🤖 Chatbot: {response}")
    elif input_type == "Speech":
        if st.button("Record"):
            user_input = transcribe_speech()
            if user_input:
                st.session_state.conversation.append(f"🗣️ You: {user_input}")
                response = chatbot_response(user_input, data)
                st.session_state.conversation.append(f"🤖 Chatbot: {response}")

    for message in st.session_state.conversation:
        st.write(message)

if __name__ == "__main__":
    file_path = "The Kamasutra.txt"
    data = load_data(file_path)
    tokens = preprocess_data(data)
    main()