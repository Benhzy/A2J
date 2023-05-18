import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.docstore.document import Document
import requests
from waitress import serve
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
    ---------
    Links to Source: 
    ---------
    {links}

    
    Instructions:
    1. Use the context provided to provide a response the legal question verbatim, unless you have to paraphrase, taking into consideration of the chat history if any. 
    
    2. If you are not certain about the response, please indicate that you do not have the necessary information and recommend that the user seek legal advice from lawyers.

    3. Please tell the user to provide more information when there is too many ways to respond.

    4. DO NOT respond with information that is not provided in the context, if you need to use information outside of the context, tell the user that you are unable to answer the question.

    5. If you cite the name of legislation or case law in the response, make sure this name can be found in the context.

    IMPORTANT! Please structure your response in the following format:

    AI: [Your response] <br>
        [your request for more information if any]<br>
        you can find out more about [relevant category] [<a href="link from links">here</a>]

    EXAMPLE 1 (vague question): 
    'Question: I want a divorce'
    'Answer: An application for divorce is a legal procedure to end a marriage. You and your spouse may file a divorce application on a simplified track if both parties can agree on all of the following before court papers are filed. <br>
    If you require a more detailed response, please provide me with details such as your current circumstances, or any specific information that you wish to know. <br>
    You can find out more about divorce in Singapore <a href="https://www.judiciary.gov.sg/family/divorce" target="_blank">here</a>'
    
    EXAMPLE 2 (answer not in database):
    'Question: What is the process of filing for a patent in Singapore?'
    'Answer: My apologies, I am unable to answer this question as I do not have the necessary information."


"""
    return prompt_template




load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
query_list = []
links_list = ['https://www.judiciary.gov.sg/family/divorce', 'https://www.judiciary.gov.sg/family/adoption', 'https://www.judiciary.gov.sg/family/deputyship', 'https://www.judiciary.gov.sg/family/care-protection-children-young-persons', 'https://www.judiciary.gov.sg/family/direct-judicial-communication', 'https://www.judiciary.gov.sg/family/appeal', 'https://www.judiciary.gov.sg/family/family-guidance-children-young-persons', 'https://www.judiciary.gov.sg/family/guardianship', 'https://www.judiciary.gov.sg/family/international-child-abduction', 'https://www.judiciary.gov.sg/family/probate-and-administration', 'https://www.judiciary.gov.sg/family/protection-against-family-violence', 'https://www.judiciary.gov.sg/family/protection-for-vulnerable-adults', 'https://www.judiciary.gov.sg/family/maintenance', 'https://www.judiciary.gov.sg/family/mediation-counselling-in-family-justice-courts', 'https://www.judiciary.gov.sg/family/syariah-court-orders', 'https://www.judiciary.gov.sg/civil/admiralty-proceedings-(from-1-april-2022)', 'https://www.judiciary.gov.sg/civil/appeals-under-the-protection-from-online-falsehoods-and-manipulation-act-(pofma)-(from-1-april-2022)', 'https://www.judiciary.gov.sg/civil/bankruptcy', 'https://www.judiciary.gov.sg/civil/bills-of-sale-(from-1-april-2022)', 'https://www.judiciary.gov.sg/civil/community-neighbour-dispute-claims', 'https://www.judiciary.gov.sg/civil/company-winding-up', 'https://www.judiciary.gov.sg/civil/employment-claims', 'https://www.judiciary.gov.sg/civil/mortgage-actions', 'https://www.judiciary.gov.sg/civil/powers-of-attorney', 'https://www.judiciary.gov.sg/civil/protection-from-harassment', 'https://www.judiciary.gov.sg/civil/small-claims', 'https://www.judiciary.gov.sg/civil/new-rules-of-court-2021' ]

def ask_bot(query, language = "English"):
    if language != "English":
        query = translate(query, 'English')
    global query_list
    global links_list
    search_index = Chroma(persist_directory='vector_db', embedding_function=OpenAIEmbeddings())
    query_list.append(query)
    if len(query_list) > 3:
        query_list.pop(0)
    query_sum = ' '.join(query_list)
    embeddings = OpenAIEmbeddings()
    docs = search_index.similarity_search(query_sum, k=3)
    #docs = [doc[0] for doc in docs_score]
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    PROMPT = PromptTemplate(
        template = generate_prompt(),
        input_variables=["context", "question", "links", "chat_history"],
    )
    chain = load_qa_chain(
    OpenAI(temperature=0),
    chain_type="stuff",
    prompt=PROMPT,
    memory=memory,
    )
    query = query
    chain({"input_documents": docs, "question": query, "links": links_list}, return_only_outputs=True)
    result = chain.memory.buffer.split("AI:")[-1]
    if language != "English":
        result = translate(result, language)
    return result



@app.route('/get')
def get_bot_response():

    userText=request.args.get("msg")
    return (str(ask_bot(userText)))

if __name__=='__main__':
    #app.run()
    serve(app, host="0.0.0.0", port=80)
