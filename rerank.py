
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI


def reranker(retriever,model_name=None):

    #compressor = FlashrankRerank(model=llm)
    #compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compressor = FlashrankRerank(model=model_name)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

