import streamlit as st
import fitz  # PyMuPDF
import openai
import faiss
import numpy as np
import os
import pickle
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="API Request Generator ISO/JSON", layout="wide")


st.write(
    "OPENAI_API_KEY has been set",
    os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],
)


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        chunks = text.split("<API-REQUEST-INFO>")
        chunks = [("Request" + chunk).strip() for chunk in chunks if chunk.strip()]
        return chunks

def create_or_load_index(pdf_path, index_directory):
    if not os.path.exists(index_directory):
        os.makedirs(index_directory)

    index_file = os.path.join(index_directory, "faiss_index.faiss")
    documents_file = os.path.join(index_directory, "documents.pkl")

    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    if os.path.exists(index_file) and os.path.exists(documents_file):
        print("Loading existing index and documents...")
        vector_store = FAISS.load_local(index_directory, embedding_model, "faiss_index", allow_dangerous_deserialization=True)
        with open(documents_file, "rb") as f:
            documents = pickle.load(f)
    else:
        print("Creating new index and documents...")
        pdf_text = extract_text_from_pdf(pdf_path)
        text_splitter = CustomTextSplitter()
        chunks = text_splitter.split_text(pdf_text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        vector_store = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
        vector_store.save_local(index_directory, "faiss_index")

        with open(documents_file, "wb") as f:
            pickle.dump(documents, f)

    return vector_store, documents

def create_rag_system(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    prompt_template = """You are an AI assistant tasked with answering questions about financial requests. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa

def ask_question(qa, question):
    result = qa({"query": question})
    return result["result"], result["source_documents"]

pdf_path = "Requests.pdf"
index_directory = "faiss_index"

vector_store, documents = create_or_load_index(pdf_path, index_directory)

rag_system = create_rag_system(vector_store)


st.title("API Request Generator ISO/JSON in format 💻🛠️")
st.markdown("This tool helps you generate structured financial requests. Just type your request, and It'll help you create the appropriate JSON or ISO 8583 format.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What financial request would you like to generate?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        answer, sources = ask_question(rag_system, prompt)
        
        message_placeholder.markdown(answer)
        
        try:
            json_start = answer.index('{')
            json_end = answer.rindex('}') + 1
            json_str = answer[json_start:json_end]
            json_obj = json.loads(json_str)
            st.json(json_obj)
        except:
            pass
        
        with st.expander("View Sources"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i + 1}:**")
                st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                st.markdown("---")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.markdown("### How to use:")
    st.markdown("1. Type your financial request in the chat input.")
    st.markdown("2. Press Enter to generate the structured format.")
    st.markdown("3. View the generated JSON or ISO 8583 format.")
    st.markdown("4. Explore the sources for more information.")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []


def main():
    pdf_path = "Requests.pdf"
    index_directory = "faiss_index"

    vector_store, documents = create_or_load_index(pdf_path, index_directory)

    global rag_system
    rag_system = create_rag_system(vector_store)

if __name__ == "__main__":
    main()
