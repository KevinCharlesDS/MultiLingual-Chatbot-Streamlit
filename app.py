import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.schema.output_parser import StrOutputParser
from googletrans import Translator
import os

# 🔒 Secure API Key Storage
sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")   

# 🚨 Check if the API key is available
if not sec_key or "hf_" not in sec_key:
    st.error("❌ API Key is missing or incorrect! Set HUGGINGFACEHUB_API_TOKEN.")
    st.stop()  

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"max_length": 100, "temperature": 0.7},  # 🔹 Reduced response length
    huggingfacehub_api_token=sec_key
)

# 🎯 Function to generate region-specific prompts
def generate_prompt(question, variant):
    instructions = {
        "UK English": "Answer in UK English using proper British spelling and phrasing. Keep responses short and factually correct.",
        "USA English": "Answer in American English with US spelling. Keep responses clear, direct, and professional.",
        "Indian English": "Answer in casual Indian English (Hinglish) with phrases like 'arre yaar', 'bhai', 'chal na'. Keep it friendly but accurate.",
        "Tamil": "Answer in Tamil language. Keep it short, clear, and accurate."
    }

    instruction = instructions.get(variant, "Answer in English.")  # Default fallback
    return f"{instruction}\nQuestion: {question}\nAnswer: "


# 🤖 Function to process chatbot response
def chatbot_response(question, variant):
    prompt_text = generate_prompt(question, variant)
    template = PromptTemplate(template=prompt_text, input_variables=["question"])

    # Use LangChain sequence processing
    chain = template | llm | StrOutputParser()

    try:
        full_response = chain.invoke({"question": question}).strip()

        # 🔹 Clean up response (Remove unnecessary prompt instructions)
        if "Answer:" in full_response:
            clean_response = full_response.split("Answer:")[-1].strip()
        else:
            clean_response = full_response

        # 🌍 Translate to Tamil if needed
        if variant == "Tamil":
            translator = Translator()
            try:
                clean_response = translator.translate(clean_response, src='en', dest='ta').text
            except Exception:
                clean_response = "⚠️ Translation failed. Please try again."

        return clean_response

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# 🎨 Streamlit UI
st.title("🗣️ Multi-Language Chatbot")
st.markdown("**Choose a language style and ask your question!**")

# Dropdown for language selection
variant = st.selectbox("📌 **Select Language Style:**", ["UK English", "USA English", "Indian English", "Tamil"])

# User input for the question
question = st.text_input("💬 **Ask your question:**")

# Generate chatbot response when button is clicked
if st.button("Get Response"):
    if question.strip():
        response = chatbot_response(question, variant)
        st.markdown("## 🤖 **Chatbot Response:**")
        st.success(response)  # Displays in a green box
    else:
        st.warning("⚠️ Please enter a question.")
