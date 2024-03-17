# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a simple standalone implementation showing the RAG (Retrieval-Augmented Generation) pipeline
using NVIDIA AI Foundational models. It uses a Streamlit UI and a minimalistic RAG pipeline implementation.

The application allows users to:
1. Upload documents to a knowledge base
2. Create or load a vector store for the documents
3. Chat with an AI assistant (Envie) that retrieves relevant information from the knowledge base
   and generates responses based on the user's query and the retrieved context.

The application consists of the following components:
1. Document Loader: Handles uploading and storing documents in a directory.
2. Embedding Model and LLM: Initializes the embedding model and language model for vector storage and text generation.
3. Vector Database Store: Creates or loads a vector store for efficient document retrieval.
4. LLM Response Generation and Chat: Handles user queries, retrieves relevant documents, and generates AI responses.
"""

import os
import logging
from typing import List, Optional
import pickle
import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS, Chroma, Milvus
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_documents(docs_dir: str) -> List[str]:
    """
    Load raw documents from the specified directory.

    Args:
        docs_dir (str): Path to the directory containing the documents.

    Returns:
        List[str]: A list of raw document contents.
    """
    loader = DirectoryLoader(docs_dir)
    raw_documents = loader.load()
    return raw_documents

def create_or_load_vector_store(
    docs_dir: str,
    vector_store_path: str,
    use_existing_vector_store: bool,
    document_embedder: NVIDIAEmbeddings,
    vector_store_type: str,
) -> Optional[FAISS]:
    """
    Create or load a vector store based on the user's preference.

    Args:
        docs_dir (str): Path to the directory containing the documents.
        vector_store_path (str): Path to the vector store file.
        use_existing_vector_store (bool): Whether to use an existing vector store if available.
        document_embedder (NVIDIAEmbeddings): The embedding model for documents.
        vector_store_type (str): The type of vector store to use (FAISS, Chroma, or Milvus).

    Returns:
        Optional[FAISS]: The vector store object, or None if no documents are available.
    """
    raw_documents = load_documents(docs_dir)

    if not raw_documents:
        logging.warning("No documents available to process!")
        return None

    vector_store_exists = os.path.exists(vector_store_path)
    if use_existing_vector_store and vector_store_exists:
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)
        logging.info("Existing vector store loaded successfully.")
    else:
        with st.spinner("Splitting documents into chunks..."):
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            documents = text_splitter.split_documents(raw_documents)

        with st.spinner("Adding document chunks to vector database..."):
            if vector_store_type == "FAISS":
                vectorstore = FAISS.from_documents(documents, document_embedder)
            elif vector_store_type == "Chroma":
                vectorstore = Chroma.from_documents(documents, document_embedder)
            elif vector_store_type == "Milvus":
                vectorstore = Milvus.from_documents(documents, document_embedder)
            else:
                raise ValueError(f"Invalid vector store type: {vector_store_type}")

        with st.spinner("Saving vector store"):
            with open(vector_store_path, "wb") as f:
                pickle.dump(vectorstore, f)
        logging.info("Vector store created and saved.")

    return vectorstore

def initialize_models(api_key: str) -> tuple:
    """
    Initialize the embedding model and language model.

    Args:
        api_key (str): The NVIDIA AI Playground API key.

    Returns:
        tuple: A tuple containing the LLM (ChatNVIDIA), document embedder (NVIDIAEmbeddings),
               and query embedder (NVIDIAEmbeddings).
    """
    os.environ["NVIDIA_API_KEY"] = api_key
    llm = ChatNVIDIA(model="mixtral_8x7b")
    document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")
    query_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="query")
    return llm, document_embedder, query_embedder

def generate_response(
    prompt_template: ChatPromptTemplate,
    llm: ChatNVIDIA,
    vectorstore: FAISS,
    user_input: str,
) -> str:
    """
    Generate a response to the user's query by retrieving relevant documents and generating text.

    Args:
        prompt_template (ChatPromptTemplate): The prompt template for the LLM.
        llm (ChatNVIDIA): The language model for text generation.
        vectorstore (FAISS): The vector store for document retrieval.
        user_input (str): The user's query.

    Returns:
        str: The generated response.
    """
    chain = prompt_template | llm | StrOutputParser()
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])
    augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}\n"
    response = chain.run({"input": augmented_user_input})
    return response

import shutil

def clear_documents_and_vector_store():
    """
    Clear the uploaded documents directory and remove the vector store file.
    """
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    VECTOR_STORE_PATH = "vectorstore.pkl"

    # Clear the uploaded documents directory
    if os.path.exists(DOCS_DIR):
        shutil.rmtree(DOCS_DIR)
        st.success("Uploaded documents directory cleared.")
    else:
        st.warning("No documents found in the directory.")

    # Remove the vector store file
    if os.path.exists(VECTOR_STORE_PATH):
        os.remove(VECTOR_STORE_PATH)
        st.success("Vector store file removed.")
    else:
        st.warning("Vector store file not found.")

def main():
    # Set Streamlit page configuration
    st.set_page_config(layout="wide")

    # Sidebar for document upload
    with st.sidebar:
        DOCS_DIR = os.path.abspath("./uploaded_docs")
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)
        st.subheader("Add to the Knowledge Base")
        with st.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Upload a file to the Knowledge Base:", accept_multiple_files=True
            )
            submitted = st.form_submit_button("Upload!")

        if uploaded_files and submitted:
            for uploaded_file in uploaded_files:
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.read())
        st.subheader("Clear Documents and Vector Store")
        if st.button("Clear"):
            clear_documents_and_vector_store() 
    # Initialize models
    NVIDIA_API_KEY = ""
    llm, document_embedder, query_embedder = initialize_models(NVIDIA_API_KEY)

    # Vector store setup
    VECTOR_STORE_PATH = "vectorstore.pkl"
    with st.sidebar:
        use_existing_vector_store = st.radio(
            "Use existing vector store if available", ["Yes", "No"], horizontal=True
        )
        vector_store_type = st.selectbox(
            "Select vector store type",
            ["FAISS", "Chroma", "Milvus"],
            index=0,
            help="Choose the type of vector store to use for document storage and retrieval.",
        )

    vectorstore = create_or_load_vector_store(
        DOCS_DIR,
        VECTOR_STORE_PATH,
        use_existing_vector_store == "Yes",
        document_embedder,
        vector_store_type,
    )

    if vectorstore is None:
        st.warning("No documents available to process!")
        return

    # Chat UI
    st.subheader("Chat with your AI Assistant, Envie!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user.",
            ),
            ("user", "{input}"),
        ]
    )

    user_input = st.chat_input("Can you tell me what NVIDIA is known for?")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = generate_response(prompt_template, llm, vectorstore, user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()