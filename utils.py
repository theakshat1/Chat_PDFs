import PyPDF2
import fitz
import tempfile
import asyncio
from typing import List, Dict, Tuple, Set, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import os
import base64
from dotenv import load_dotenv
import hashlib
import json
from pathlib import Path
from pydantic import BaseModel
import os
import shutil
import chromadb

load_dotenv()


CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_file_hash(file_bytes: bytes) -> str:

    return hashlib.md5(file_bytes).hexdigest()

def get_cache_path(file_hash: str) -> Path:
    """Get the cache file path for a given hash"""
    return CACHE_DIR / f"{file_hash}.json"

async def extract_text_from_pdf(pdf_file) -> tuple[str, str, bytes, dict]:
    """
    Extract text content from a PDF file
    Returns tuple of (text, filename, pdf_bytes, metadata)
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    page_texts = {}  # Store text by page number

    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        text += page_text
        page_texts[i] = page_text

    # Get PDF bytes and metadata
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()

    metadata = {
        "num_pages": len(pdf_reader.pages),
        "page_texts": page_texts,
        "file_size": len(pdf_bytes)
    }

    return text, pdf_file.name, pdf_bytes, metadata

async def process_single_pdf(pdf_file) -> Tuple[List[Document], Dict]:
    """Process a single PDF file with caching"""
    pdf_file.seek(0)
    file_hash = get_file_hash(pdf_file.read())
    cache_path = get_cache_path(file_hash)

    # Check cache
    if cache_path.exists():
        with cache_path.open() as f:
            cached_data = json.load(f)
            documents = [Document(**doc) for doc in cached_data['documents']]
            return documents, cached_data['metadata']

    text, filename, pdf_bytes, metadata = await extract_text_from_pdf(pdf_file)

    # Split text into smaller chunks with more overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200,  
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  
    )
    chunks = text_splitter.split_text(text)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": filename,
                "chunk_id": i,
                "text": chunk,
                "file_hash": file_hash,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    # Cache the results
    cache_data = {
        'documents': [doc.dict() for doc in documents],
        'metadata': metadata
    }
    with cache_path.open('w') as f:
        json.dump(cache_data, f)

    return documents, metadata

async def process_pdfs(pdf_files: List[Any]) -> Tuple[List[Document], Dict[str, bytes]]:
    """
    Process multiple PDF files concurrently and return documents with metadata
    """
    documents = []
    pdf_bytes_dict = {}

    # Process PDFs concurrently
    tasks = [process_single_pdf(pdf_file) for pdf_file in pdf_files]
    results = await asyncio.gather(*tasks)

    for pdf_file, (docs, metadata) in zip(pdf_files, results):
        documents.extend(docs)  # Combines documents from all PDFs
        pdf_bytes_dict[pdf_file.name] = pdf_file.read()

    return documents, pdf_bytes_dict

def highlight_pdf(pdf_bytes: bytes, text_to_highlight: str, page_number: int = None) -> tuple[bytes, int, int]:
    """Highlights the specified text in the PDF and returns the highlighted PDF along with page info."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        current_page = page_number if page_number is not None else 0
        current_page = min(max(0, current_page), total_pages - 1)

        text_chunks = []
        for sentence in text_to_highlight.replace('. ', '.|').replace('? ', '?|').replace('! ', '!|').split('|'):
            if sentence:
                phrases = sentence.split(',')
                text_chunks.extend([p.strip() for p in phrases if len(p.strip()) > 5])

        # Search and highlight text
        highlights_found = False
        pages_to_search = [current_page] if page_number is not None else range(total_pages)
        
        for pg_num in pages_to_search:
            page = doc.load_page(pg_num)
            page.clean_contents()  
            
            for chunk in text_chunks:
                text_instances = page.search_for(chunk.strip())
                if text_instances:
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=(1, 0.8, 0))
                        highlight.set_opacity(0.4)
                        highlight.update()
                        highlights_found = True
                        current_page = pg_num
            
            if highlights_found and page_number is None:
                break

        return doc.tobytes(), total_pages, current_page + 1
    except Exception as e:
        print(f"Error highlighting PDF: {str(e)}")
        return pdf_bytes, 1, 1
    finally:
        if 'doc' in locals():
            doc.close()

def setup_qa_chain(documents: List[Document], current_pdfs: List[str] = None) -> Tuple[Any, Any]:
    """Setup RAG chain with documents"""
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    chroma_settings = chromadb.Settings(
        is_persistent=True,
        persist_directory=".chroma_db",
        anonymized_telemetry=False
    )
    
    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            client_settings=chroma_settings,
            collection_name="pdf_collection"
        )
        
        # Add new documents to existing collection
        if documents:
            vectorstore.add_documents(documents)
            
    except Exception as e:
        # If collection doesn't exist, create new one
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            client_settings=chroma_settings,
            collection_name="pdf_collection"
        )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    # Setup retriever with filter for current PDFs
    search_kwargs = {"k": 4}
    if current_pdfs:
        search_kwargs["filter"] = {"source": {"$in": current_pdfs}}
        
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    template = """Answer the question based ONLY on the following context. Do not use any other information:
    {context}
    
    Question: {question}

    Instructions:
    - Use ONLY the information provided in the context above
    - Do not make assumptions or use external knowledge

    Your response should follow this structure:
    Answer: [Your main response here]
    Sources: [List the document name]
    Conflicts: [If any conflicting information exists, explain here]

    Remember to:
    - Draw connections between documents
    - Only use information from the provided context
    Avoid using markdown or HTML formatting in your response
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

def get_response(rag_chain, retriever, question: str, chat_history: List[Tuple[str, str]], current_pdfs: List[str] = None) -> Tuple[str, List[Document], str]:
    """
    Get response from the RAG chain with cross-document context
    Returns: (answer, sources, context_used)
    """
    # Get relevant documents and their content
    docs = retriever.invoke(question)
    
    # Filter docs to only include current PDFs if specified
    if current_pdfs:
        docs = [doc for doc in docs if doc.metadata.get("source") in current_pdfs]
    
    context_used = format_docs(docs)  # Get the actual context used

    # Get answer
    answer = rag_chain.invoke(question)

    sources = []
    seen_sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen_sources and (not current_pdfs or source in current_pdfs):
            sources.append(doc)
            seen_sources.add(source)

    return answer, sources, context_used

def get_pdf_display_base64(pdf_bytes: bytes) -> str:
    """
    Convert PDF bytes to base64 for display in HTML
    """
    try:
        if not pdf_bytes:
            raise ValueError("Empty PDF bytes")
        return base64.b64encode(pdf_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error converting PDF to base64: {str(e)}")
        return ""

def format_docs(docs):
    """Format documents into a single string with source information."""
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content
        formatted_docs.append(f"[Source: {source}]\n{content}")
    return "\n\n".join(formatted_docs)

class Document(BaseModel):
    page_content: str
    metadata: Dict[str, Any]