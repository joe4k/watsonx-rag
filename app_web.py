import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, util

from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

api_key=""
url=""
watsonx_project_id=""
model_id=""
max_tokens=1000
min_tokens=20
decoding = DecodingMethods.GREEDY
stop_sequences = ['.', '\n']
temperature = 0.7

# Return WatsonxLLM model object with the specific parameters
def get_model(model_type,max_tokens,min_tokens,decoding,stop_sequences,temperature):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences,
        GenParams.TEMPERATURE: temperature
    }
    print("url: ", url)
    model = WatsonxLLM(
        model_id=model_type,
        params=generate_params,
        url=url,
        apikey=api_key,
        project_id=watsonx_project_id
    )      
    return model

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
    globals()["watsonx_project_id"] = os.getenv("watsonx_project_id", None)
    globals()["url"] = os.getenv("url", None)
    globals()["model_id"] = os.getenv("model_id",None)
    globals()["locale"] = os.getenv("locale", None)

    globals()["gcounter"] = 0


get_credentials()

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

#vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(),)
model_name = "Salesforce/SFR-Embedding-2_R"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
#embedding = SentenceTransformer("Salesforce/SFR-Embedding-2_R")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding,)


# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

llm = get_model(model_id,max_tokens,min_tokens,decoding,stop_sequences,temperature)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "What is Task Decomposition?"
print("query: ", query)
response = rag_chain.invoke("What is Task Decomposition?")
print("response: ", response)