import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY="AIzaSyDllBzoIy4jAMam0J0LNU02QBKc4MgmrTI"
GROQ_API_KEY="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"


os.environ["GOOGLE_API_KEY"] = "AIzaSyDllBzoIy4jAMam0J0LNU02QBKc4MgmrTI"

st.sidebar.image("logo.jpg",width=180)
st.sidebar.title("About us")
user_input = st.sidebar.text_input('Welcome    to    Alferix Inc!    We    are    an    IT    consulting    company    located    in    Houston,    Texas.    We    started    our    journey    as    DreamFleet Inc.    In    2017    and    quickly    established    ourselves    as    a    leader    in    our    industry. Since    then,    we    have    achieved    more    than    100 % revenue    growth    every    year,    a    testament    to    the    hard work    and    dedication    of    our    team.')
st.title("Alferix  Chatbot")

llm=ChatGroq(groq_api_key=GROQ_API_KEY,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
if 'embedding_done' not in st.session_state:
    st.session_state.embedding_done = False
    
def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./Data") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings



prompt1=st.text_input("Welcome    to   the   Alferix    Chat     Bot    Q&A. Please    ask    your    HR    Policy    related    questions    below.")

left_column,right_column=st.columns(2)
with left_column:
    if st.button("Search"):
        vector_embedding() 
                                                                                                                                                        
                                                                                                                                                                                    

  

        



import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

