import os
import arxiv
import requests
import tempfile

from langchain_community.document_loaders import PyPDFLoader,TextLoader,WebBaseLoader,ArxivLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def inject_title(docs,title):
    for doc in docs:
        doc.metadata['title'] = title
    return docs

_splitter = RecursiveCharacterTextSplitter(chunk_size = 700,chunk_overlap = 150)
_markdown_splitter = RecursiveCharacterTextSplitter.from_language(chunk_size = 700,chunk_overlap = 150,language='markdown')

def pdf_loader(file_path,file_name):
    pdf_docs = PyPDFLoader(file_path).load()
    docs = _splitter.split_documents(inject_title(pdf_docs,Path(file_name).stem))
    return docs


def markdown_loader(file_path,file_name):
    md_docs = TextLoader(file_path).load()
    docs = _markdown_splitter.split_documents(inject_title(md_docs,Path(file_name).stem))
    return docs

def text_loader(file_path,file_name):
    text_docs = TextLoader(file_path).load()
    docs = _splitter.split_documents(inject_title(text_docs,Path(file_name).stem))
    return docs

def web_loader(url):
    web_docs = WebBaseLoader(url).load()
    title = (web_docs[0].metadata.get('title') or url) if web_docs else url
    docs = _splitter.split_documents(inject_title(web_docs,title))
    return docs


def paper_loader(paper_query):
    search = arxiv.Search(query=paper_query, max_results=1)
    paper = next(arxiv.Client().results(search))

    pdf_url = paper.entry_id.replace("/abs/", "/pdf/") + ".pdf"

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(requests.get(pdf_url).content)
    tmp.close()

    try:
        p_docs = PyMuPDFLoader(tmp.name).load()
        title = paper.title or paper_query
        docs = _splitter.split_documents(inject_title(p_docs,title))
    finally:
        os.remove(tmp.name) 
    return docs


def document_loader(source,file_name):
    
    if source.startswith(("http://", "https://")):
        return web_loader(source)
    
    ext = Path(source).suffix.lower()
    
    if(ext == '.pdf'):
        return pdf_loader(source,file_name)
    if(ext == '.txt'):
        return text_loader(source,file_name)
    if(ext == '.md'):
        return markdown_loader(source,file_name)
