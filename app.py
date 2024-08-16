from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Sidebar for selecting the application mode
st.sidebar.title("Select Application Mode")
app_mode = st.sidebar.selectbox("Choose one", ["AI Assistant", "PDF Summarizer", "Language Translator", "Ask Questions from PDF", "Code Generator", "Ask Question for Web Link"])

# Streamlit framework
st.title('Multi Function APP')

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

# Ollama LLaMA2 LLM
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

def extract_text_from_pdf(uploaded_file):
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pdf_text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pdf_text += page.get_text()
    return pdf_text

def extract_text_from_web(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

if app_mode == "AI Assistant":
    input_text = st.text_input("Enter your question")
    if input_text:
        st.write(chain.invoke({"question": input_text}))

elif app_mode == "PDF Summarizer":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write(f"Extracted text length: {len(pdf_text)} characters")

        if len(pdf_text) > 5000:  # Adjust the length as needed
            pdf_text = pdf_text[:5000]  # Truncate text if too long

        summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Please summarize the following text."),
                ("user", f"Text: {pdf_text}")
            ]
        )
        summary_chain = summary_prompt | llm | output_parser
        summary = summary_chain.invoke({"question": "Summarize the text"})
        st.write(summary)

elif app_mode == "Language Translator":
    input_text = st.text_input("Enter text to translate")
    target_language = st.text_input("Enter target language (e.g., 'es' for Spanish)")
    if input_text and target_language:
        translation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Please translate the following text."),
                ("user", f"Text: {input_text} to {target_language}")
            ]
        )
        translation_chain = translation_prompt | llm | output_parser
        translation = translation_chain.invoke({"question": "Translate the text"})
        st.write(translation)

elif app_mode == "Ask Questions from PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write(f"Extracted text length: {len(pdf_text)} characters")

        question = st.text_input("Enter your question about the PDF")
        if question:
            question_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant. Please answer the following question based on the provided text."),
                    ("user", f"Text: {pdf_text}"),
                    ("user", f"Question: {question}")
                ]
            )
            question_chain = question_prompt | llm | output_parser
            answer = question_chain.invoke({"question": question})
            st.write(answer)

elif app_mode == "Code Generator":
    code_prompt = st.text_input("Enter a prompt for code generation")
    if code_prompt:
        code_generation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Please generate code based on the following prompt."),
                ("user", f"Prompt: {code_prompt}")
            ]
        )
        code_generation_chain = code_generation_prompt | llm | output_parser
        generated_code = code_generation_chain.invoke({"question": code_prompt})
        st.code(generated_code, language='python')

elif app_mode == "Ask Question for Web Link":
    web_url = st.text_input("Enter a web link")
    if web_url:
        web_text = extract_text_from_web(web_url)
        st.write(f"Extracted text length: {len(web_text)} characters")

        question = st.text_input("Enter your question about the web content")
        if question:
            question_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant. Please answer the following question based on the provided text."),
                    ("user", f"Text: {web_text}"),
                    ("user", f"Question: {question}")
                ]
            )
            question_chain = question_prompt | llm | output_parser
            answer = question_chain.invoke({"question": question})
            st.write(answer)