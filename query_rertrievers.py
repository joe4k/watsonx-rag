

# https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/


def getRetriever(vectorDB, retrieverType="vectorstore",k=5):
    match retrieverType:
        case "vectorstore":
            retriever = vectorDB.as_retriever(search_kwargs={"k": k})
        case "multiquery":
            print("Not supported at this time")
            retriever = vectorDB.as_retriever(search_kwargs={"k": k})
        case "ensemble":
            print("Not supported at this time")
            retriever = vectorDB.as_retriever(search_kwargs={"k": k})    
        case _:
            retriever = vectorDB.as_retriever(search_kwargs={"k": k})
    return retriever


