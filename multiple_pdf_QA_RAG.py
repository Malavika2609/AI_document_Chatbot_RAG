import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

key=os.getenv("GROQ_API_KEY")







def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    ## Embedding Using Huggingface

    embeddings= HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # lightweight and fast
    )
  
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    model=ChatGroq(groq_api_key=key,
             model_name="llama-3.3-70b-versatile")

    prompt_template="""
You are a legal text analyzer. Check whether the following obligation is clearly stated or fulfilled in the context below by following the below steps.
    Note:Consider If the Vendor will replace or modify the product and remove the infringement is there in the below paragraph, the obligation is fullfilled and give pass and rationale.
    Assume that if a company is supplying data, technology, or services, it is likely acting as a vendor — even if it’s not explicitly labeled as such.
    Vendor name would have mentioned in the below paragraph and may not have mentioned explicitly as vendor.

    Step 1: look if the infringement can be removed by vendor as mentioned in the obligation.
    Step 2: If step 1 is true, go for step 3 or else respand as "Fail".
    Step 3: Determine if the paragraph supports continued use of the product, either explicitly or implicitly. Check if any conditions must be met (e.g. no downgrade, no interference) and whether they are satisfied.
    If the step 3 is true, consider obligation fulfilled and respond as "Pass" else respond as "Fail".

<context>
{context}
<context>
Obligation:{question}

"""


    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    
    embeddings=HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # lightweight and fast
    )

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using LLAMA3💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
