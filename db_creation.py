import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import pathlib
import subprocess
import tempfile
import pickle


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")


def get_source():
    for file_name in os.listdir("db\\"):
        if file_name.endswith('.txt'):
            name, ext = os.path.splitext(file_name)
            with open(os.path.join("db\\", file_name), 'r', encoding="utf8") as file:
                yield Document(page_content=file.read(), metadata={"source": name})


def embed_chunks(sources): 
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=720, chunk_overlap=0)
    for source in sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    return Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory="vector_db\\")

search_index = embed_chunks(get_source())