# For reading credentials from the .env file
import os
from dotenv import load_dotenv

configVars={}

def get_credentials():
    global configVars
    
    load_dotenv()

    print("type of configVars: ", type(configVars))
    # Update the global variables that will be used for authentication in another function
    configVars["api_key"] = os.getenv("api_key", None)
    configVars["watsonx_project_id"] = os.getenv("watsonx_project_id", None)
    configVars["url"] = os.getenv("url", None)
    configVars["space_id"] = os.getenv("space_id", None)
    configVars["model_id"] = os.getenv("model_id", None)
    configVars["classification_deployment_id"] = os.getenv("classification_deployment_id", None)
    configVars["question_deployment_id"] = os.getenv("question_deployment_id", None)
    configVars["programming_deployment_id"] = os.getenv("programming_deployment_id", None)