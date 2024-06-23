from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus


# https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/

import os
from os.path import join
from dotenv import load_dotenv

        

def chunks_to_vectors(chunks,embedding,vector_store="chroma"):
    match vector_store:
        case "chroma":
            # Recursively splits text. This splitting is trying to keep related pieces of text next to each other. 
            # This is the recommended way to start splitting text.
            vectorDB= Chroma.from_documents(documents=chunks, embedding=embedding,)
        case "faiss":
            vectorDB = FAISS.from_documents(documents=chunks, embedding=embedding,)
        case "milvus":
            load_dotenv(join(os.getcwd(),".env.milvus"))
            URI = f'https://{os.getenv("milvus_host")}:{os.getenv("milvus_port")}'
            vectorDB = Milvus.from_documents(documents=chunks, embedding=embedding,
                                             connection_args={"uri": URI, "user": os.getenv("milvus_user"), "password": os.getenv("milvus_password"), "secure":True},
            )
        case _:
            vectorDB = Chroma.from_documents(documents=chunks, embedding=embedding,)

    return vectorDB