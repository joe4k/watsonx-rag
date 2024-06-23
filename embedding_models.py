from langchain_huggingface import HuggingFaceEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings


import os


# https://python.langchain.com/v0.2/docs/integrations/text_embedding/

# https://python.langchain.com/v0.2/docs/integrations/text_embedding/ibm_watsonx/

def get_embedding(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    match model_name:
        case "sentence-transformers/all-MiniLM-L6-v2":
            # One of the popular embeddings on Huggingface
            # Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        case "ibm/slate-125m-english-rtrvr":
            embed_params = {
                EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
                EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
            }

            embedding = WatsonxEmbeddings(
                model_id=model_name,
                url=os.getenv("WATSONX_URL","https://us-south.ml.cloud.ibm.com"),
                project_id=os.getenv("PROJECT_ID"),
                params=embed_params,
            )
        case _:
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

    return embedding
