import argparse
import pickle
import requests
import xmltodict
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise Exception("OpenAI API key not set")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO
def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)


def create_embeddings(sitemap_url):
    r = requests.get(sitemap_url)
    xml = r.text
    raw = xmltodict.parse(xml)

    pages = []
    for info in raw['urlset']['url']:
        url = info['loc']
        pages.append({'text': extract_text_from(url), 'source': url})

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        log_message = f"Processed {page['source']} into {len(splits)} chunks"
        socketio.emit('log', {'message': log_message})

    store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key=openai_api_key), metadatas=metadatas)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)


def get_answer(question, chat_history):
    with open("faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)

    llm = OpenAI(temperature=0.4)
    _template = """Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    template = """You are an AI assistant for answering questions about specific site that 
    was loaded in prior steps. You are given the following extracted parts of 
    a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say "Hmm, I'm not sure.".
    Don't try to make up an answer. If the question is not about
    the site or the content therein, politely inform them that you are tuned
    to only answer questions about the site that you were instructed to scrape.
    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown:"""
    QA = PromptTemplate(template=template, input_variables=["question", "context"])

    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )

    result = qa_chain({"question": question, "chat_history": chat_history})
    return result['answer']
