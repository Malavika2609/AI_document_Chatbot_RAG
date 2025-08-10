Multi-PDF RAG Chatbot using LLAMA3:

Description:
This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer user queries by retrieving relevant information from multiple PDF documents. Leveraging vector similarity search and powerful language models, the chatbot provides accurate, context-aware answers based on the content of uploaded PDFs.

Project Goal:
To build an interactive chatbot that can:

    Ingest and process multiple PDF documents

    Split text intelligently for better retrieval

    Index document chunks using vector embeddings and FAISS

    Retrieve relevant information based on user queries

    Generate concise, context-rich answers with a language model (ChatGroq)

    Enable easy querying over a large set of documents in a scalable way

Steps:

    Install dependencies from requirements.txt

    Add your PDF files in the project folder or specify paths

    Run the app

    Enter your query in the chatbot interface

    The system performs:

        Text extraction and splitting from PDFs

        Embedding and indexing with FAISS

        Retrieval of relevant chunks based on query

        Answer generation using ChatGroq model

    View the generated answer based on retrieved context
    
Tech stack:
  Langchain
  HuggingFace
  Streamlit
  Groq
  LLama3
  RAG
  FAISS (Vector DB)
  
Result:

    A responsive chatbot that can handle complex queries over multiple PDFs

    Accurate retrieval with relevant contextual answers

    Easy integration with vector stores and LLMs for scalable document QA

    Modular code base for extending to other document types or models
