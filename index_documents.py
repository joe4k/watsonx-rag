# 
from pdf_loaders import *
from text_splitters import *
from embedding_models import *
from vectors import *

# general os modules
import os
from os import listdir
from os.path import isfile, join
import pathlib
from dotenv import load_dotenv


def index_docs(pdf_tool,splitter_tool,vector_store,embedding_model):
    document_directory_path=os.getenv("watsonx-challenge-directory")
    document_directory = pathlib.Path(document_directory_path)
    if not document_directory.exists():
        print("Error!! directory: ", document_directory, " does not exist")
    
    
    doc_files = [f for f in listdir(document_directory) if isfile(join(document_directory, f))]
    print(doc_files)
    
    # Get embedding for the specified model
    embedding = get_embedding(embedding_model)
    
    for f in doc_files:
        filepath = join(document_directory, f)
        # Returns a List[Element] present in the pages of the parsed pdf document
        ##elements = partition_pdf(filepath)
        ##elements_hi_res = partition_pdf(filepath,strategy="hi_res")
        ##loader = UnstructuredPDFLoader(filepath)
        data = pdf_to_text(filepath,pdf_tool)
        chunks = text_to_chunks(data,chunk_method=splitter_tool)
        vectorDB = chunks_to_vectors(chunks,embedding,vector_store)
        
    
        for c in chunks:
            print("###################")
            print("\n")
            print(c)
            print("\n")
        #print(chunks)
    
    #for e in elements:
    #    print(e)
    #    break
    return vectorDB