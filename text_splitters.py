from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/

def text_to_chunks(data,chunk_method="recursive"):
    match chunk_method:
        case "recursive":
            # Recursively splits text. This splitting is trying to keep related pieces of text next to each other. 
            # This is the recommended way to start splitting text.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            #text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
            chunks = text_splitter.split_documents(data)
        case "semantic":
            # First splits on sentences. Then combines ones next to each other if they are semantically similar enough. 
            # Taken from Greg Kamradt
            text_splitter = SemanticChunker(HuggingFaceEmbeddings())
            # Semantic splitter requires as input a list of strings, not a list of langchain Documents
            documents_text=[]
            for d in data:
                documents_text.append(d.page_content)
            chunks = text_splitter.create_documents(documents_text)
        case _:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            #text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
            chunks = text_splitter.split_documents(data)

    return chunks
