from langchain_core.prompts import PromptTemplate

prompt_template = """
Use the following information information in the Context to answer the user's Question.
If you don't know the answer, say I don't know, do not try to make up an answer.
Format the responses properly to be easily read.

Context: {context}
Question: {question}
"""

# RAG prompt
##from langchain import hub

# Loads the latest version
###prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")


def getPrompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])

    return prompt