from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM

import setupVars

# Return WatsonxLLM model object with the specific parameters
def get_model(max_tokens,min_tokens,decoding,stop_sequences,temperature):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences,
        GenParams.TEMPERATURE: temperature
    }

    model = WatsonxLLM(
        model_id=setupVars.configVars["model_id"],
        params=generate_params,
        url=setupVars.configVars["url"],
        apikey=setupVars.configVars["api_key"],
        project_id=setupVars.configVars["watsonx_project_id"]
    )      
    return model