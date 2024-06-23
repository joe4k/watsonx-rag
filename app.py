from langchain_community.document_loaders import UnstructuredWordDocumentLoader

from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# document processing modules
from unstructured.partition.pdf import partition_pdf

# 
from pdf_loaders import *
from text_splitters import *
from embedding_models import *
from vectors import *
from index_documents import *
from query_rertrievers import *
from rerank import *
from llm_models import *
from prompts import *

# general os modules
import os
from os import listdir
from os.path import isfile, join
import pathlib
from dotenv import load_dotenv

# Read creadentials from local .env file in the same directory as this script
def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    os.environ["WATSONX_APIKEY"] = globals()["api_key"]
    globals()["watsonx_project_id"] = os.getenv("watsonx_project_id", None)
    os.environ["WATSONX_PROJECT_ID"] = globals()["watsonx_project_id"] 
    globals()["url"] = os.getenv("url", None)
    os.environ["WATSONX_URL"] = globals()["url"]
    globals()["locale"] = os.getenv("locale", None)

    globals()["gcounter"] = 0

# PDF loader optinos
pdf_tools = ["pypdf","pymupdf","mathpixpdf","unstructured","pypdfium2","pdfminer","pdfplumber"]
# Chunking options
splitter_tools = ["recursive","semantic"]
# Embedding models
embedding_models = ["sentence-transformers/all-MiniLM-L6-v2","ibm/slate-125m-english-rtrvr","ibm/slate-30m-english-rtrvr","sentence-transformers/all-minilm-l12-v2","baai/bge-large-en-v1"]
# vector stores
vector_stores = ["chroma", "faiss", "milvus"]
# retriever types
retrieverTypes = ["vectorestore","multiquery","ensemble"]

max_tokens = 1000
min_tokens = 20
decoding_method="greedy"
stop_sequences=[]
temperature=0
 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    nresults = 5

    # Retrieve values required for invocation of LLMs from the .env file
    # Initialize variables from .env
    setupVars.get_credentials()

    # Load env variables
    get_credentials()
    
    pdf_tool = pdf_tools[0]
    splitter_tool =splitter_tools[0]
    embedding_model = embedding_models[0]
    vector_store = vector_stores[0]
    
    vectorDB = index_docs(pdf_tool,splitter_tool,vector_store,embedding_model)
    retriever = getRetriever(vectorDB, "vectorstore", 5)
    #compression_retriever = reranker(retriever,"ms-marco-MiniLM-L-12-v2")

    llm = get_model(max_tokens,min_tokens,decoding_method,stop_sequences,temperature)

    prompt = getPrompt()

    #qaChain = (
    #    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    #    | prompt
    #    | llm
    #)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # sample question
    question = "who is the CEO of IBM?"
    
    for chunk in qa_chain.stream(question):
        print(chunk, end="", flush=True)

    #results = vectorDB.similarity_search(question)
    #print(results[0].page_content)

    


if __name__ == "__main__":
    main()