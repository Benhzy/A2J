import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.docstore.document import Document
import requests
from flask import Flask, redirect, url_for, render_template, request 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
import pathlib
import subprocess
import tempfile
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from translate import translate

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('samplefrontendmahesh.html')



def generate_prompt():
    prompt_template = """
    Chat History:
    ---------
    {chat_history}
    ---------
    Context:
    ---------
    {context}
    ---------
    Legal Question: 
    ---------
    {question}

    
    Instructions:
    Use the context provided to answer the legal question, taking into consideration of the chat history if any. If you are not certain about the answer, please indicate that you do not have the necessary information and recommend that the user seek legal advice from lawyers. To obtain legal aid, provide the user with a link to 'https://www.probono.sg/'.

    Please structure your response in the following format:
    User: [question here]
    AI: [reply here] (If the Choosen Language is not English, please translate your reply to the Choosen Language.)


"""
    return prompt_template




load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
query_list = []

def ask_bot(query, language = "English"):
    if language != "English":
        query = translate(query, 'English')
    global query_list
    search_index = Chroma(persist_directory='vector_db', embedding_function=OpenAIEmbeddings())
    query_list.append(query)
    if len(query_list) > 3:
        query_list.pop(0)
    query_sum = ' '.join(query_list)
    embeddings = OpenAIEmbeddings()
    docs = search_index.similarity_search(query_sum, k=3)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    PROMPT = PromptTemplate(
        template = generate_prompt(),
        input_variables=["context", "question", "chat_history"],
    )
    chain = load_qa_chain(
    OpenAI(temperature=0),
    chain_type="stuff",
    prompt=PROMPT,
    memory=memory,
    )
    query = query
    chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    result = chain.memory.buffer.split("AI:")[-1]
    if language != "English":
        result = translate(result, language)
    return result



@app.route('/get')
def get_bot_response():

    userText=request.args.get("msg")
    return (str(ask_bot(userText)))

if __name__=='__main__':
    app.run()
