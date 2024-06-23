from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import MathpixPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFPlumberLoader

# Langchain supports multiple utilities for loading PDF files
# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/


# Return data (text) extracted from pdf file
# pdf_file should be complete path to the pdf file
def pdf_to_text(pdf_file,pdf_tool="pypdf"):

    match pdf_tool:
        case "pypdf":
            # Load PDF using pypdf into array of documents, where each document contains the page content 
            # and metadata with page number.
            loader = PyPDFLoader(pdf_file)
        case "pymupdf":
            # PyMuPDFLoader is the fastest of the PDF parsing options, and contains detailed metadata 
            # about the PDF and its pages, as well as returns one document per page.
            loader = PyMuPDFLoader(pdf_file)
        case "mathpixpdf":
            # Inspired by Daniel Gross's https://gist.github.com/danielgross/3ab4104e14faccc12b49200843adab21
            loader = MathpixPDFLoader(pdf_file)
        case "unstructured":
            # The unstructured[all-docs] package currently supports loading of text files, powerpoints, html, pdfs, images, and more
            loader = UnstructuredPDFLoader(pdf_file)
            #loader = UnstructuredPDFLoader(pdf_file,strategy="hi_res")
        case "pypdfium2":
            # 
            loader = PyPDFium2Loader(pdf_file)
        case "pdfminer":
            #
            loader = PDFMinerLoader(pdf_file)
        case "pdfplumber":
            loader = PDFPlumberLoader(pdf_file)
        case _:
            loader = PyPDFLoader(pdf_file)

    data = loader.load()
    return data


