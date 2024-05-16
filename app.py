import os

import openai
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
import streamlit as st

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint


from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

if 'result' not in st.session_state:
    st.session_state['result'] = []

if 'query' not in st.session_state:
    st.session_state['query'] = []


def get_text_from_pdf(pdf_file):
    pdf_doc = PyPDF2.PdfReader(pdf_file)
    pdf_text = ''
    
    for i,page in enumerate(pdf_doc.pages):
        page_content = pdf_doc.pages[i].extract_text()
        pdf_text += page_content
    
    return pdf_text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap= 100)
    text_chunks = text_splitter.split_text(text)
    
    return text_chunks

def get_retriever(chunked_text):
    openai_embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunked_text, embedding=openai_embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    return retriever



# this chain also has a memory capability
def get_chain(retriever):
    # llm = ChatOpenAI()
    llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature= 0.6
)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory = st.session_state.memory,
            
    )

    return chain


def show_history():
    for i in range(len(st.session_state['query'])):
        with st.chat_message('user'):
            st.write(st.session_state['query'][i])
        with st.chat_message('ai'):
            st.write(st.session_state['result'][i]['answer'])


def main():
    with st.sidebar:
        pdf_file = st.file_uploader(label='Upload the pdf file',type='pdf',help="Upload your file here")

    if pdf_file:
        
        st.title("PDF chatbot")
        # getting the text from the pdf_file
        text = get_text_from_pdf(pdf_file)

        # getting chunks from the text
        chunked_text = get_text_chunks(text)
        
        # getting retriever for similarity search
        retriever = get_retriever(chunked_text)

        #chain
        chain = get_chain(retriever=retriever)

        # container
        history_container = st.container()
        query_container = st.container()

        with st.spinner('please wait...'):
            query = query_container.text_input(label="Enter the query")

            if query:
                    result = chain.invoke({'question':query})
                    st.session_state.query.append(query)
                    st.session_state.result.append(result)
                    with history_container:
                        show_history()
                
    else:
        st.title("Enter the pdf file")
        st.session_state.query = []
        st.session_state.result = []



if __name__=="__main__":
    main()