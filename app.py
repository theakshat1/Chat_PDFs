import streamlit as st
from utils import process_pdfs, setup_qa_chain, get_response, highlight_pdf, get_pdf_display_base64
import os
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple, Dict
import os
import shutil

load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

# Set page configuration
st.set_page_config(
page_title="Chat with PDFs",
layout="wide",
initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""

<style> .uploadedFile { border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin: 5px 0; } .stProgress > div > div > div > div { background-color: #00ff00; } .pdf-viewer { border: none; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); } </style>
""", unsafe_allow_html=True)

# Title and description
st.title("💬 Chat with Multiple PDFs")
st.markdown("""
Upload multiple PDF documents and ask questions across their content.
The application will use Google's Gemini AI to provide accurate answers based on all uploaded documents.

Features:

Multiple PDF upload support
Cross-document querying
Text highlighting
Source attribution
Document caching """)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "pdf_bytes_dict" not in st.session_state:
    st.session_state.pdf_bytes_dict = {}
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# Create main layout columns
left_col, right_col = st.columns([3, 2])

with left_col:
# File upload section with drag and drop
    st.markdown("### Upload PDFs")
    uploaded_files = st.file_uploader(
    "Drag and drop your PDFs here",
    type="pdf",
    accept_multiple_files=True,
    help="You can upload multiple PDF files (max 200MB each)"
    )


# Process PDFs when new files are uploaded
if uploaded_files and all(hasattr(f, 'read') for f in uploaded_files):
    # Process files only if they are valid file objects
    # Check if we have new files to process
    current_files = {f.name: f for f in uploaded_files}
    previous_files = {f.name: f for f in st.session_state.uploaded_files}
    
    if current_files != previous_files:
        with st.spinner("Processing PDFs..."):
            # Create progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Process PDFs asynchronously
            async def process_files():
                try:
                    # Clear existing cache directory
                    import shutil
                    if os.path.exists(".chroma_db"):
                        shutil.rmtree(".chroma_db")
                    
                    # Process all PDFs
                    documents, pdf_bytes_dict = await process_pdfs(uploaded_files)
                    
                    # Setup RAG chain with all documents
                    rag_chain, retriever = setup_qa_chain(documents)
                    
                    # Update session state
                    st.session_state.rag_chain = rag_chain
                    st.session_state.retriever = retriever
                    st.session_state.uploaded_files = uploaded_files
                    st.session_state.pdf_bytes_dict = pdf_bytes_dict
                    st.session_state.processing_complete = True
                    
                    return len(uploaded_files)
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
                    return 0
            
            # Run async processing
            with ThreadPoolExecutor() as executor:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                num_processed = loop.run_until_complete(process_files())
                loop.close()
            
            if num_processed > 0:
                st.success(f"Successfully processed {num_processed} PDF(s)!")

    # Display uploaded files with metadata
    if st.session_state.processing_complete:
        st.markdown("### Uploaded Documents")
        for file in uploaded_files:
            with st.expander(f"📄 {file.name}"):
                st.markdown(f"**Size:** {file.size / 1024:.1f} KB")
                if st.button(f"Remove {file.name}"):
                    # Remove file from session state
                    st.session_state.uploaded_files = [
                        f for f in st.session_state.uploaded_files if f.name != file.name
                    ]
                    # Clear the RAG chain and retriever
                    st.session_state.rag_chain = None
                    st.session_state.retriever = None
                    # Clear the PDF bytes dictionary
                    st.session_state.pdf_bytes_dict = {}
                    # Clear processing complete flag
                    st.session_state.processing_complete = False
                    st.rerun()
    
    # PDF Viewer Triggered by "View in PDF" Button
    if st.session_state.processing_complete:
        st.markdown("### Ask Questions")
        question = st.text_input(
            "Type your question here",
            placeholder="Ask a question about the uploaded documents..."
        )
        
        if question and st.session_state.rag_chain:
            with st.spinner("Searching through documents..."):
                # Get response with context
                answer, sources, context_used = get_response(
                    st.session_state.rag_chain,
                    st.session_state.retriever,
                    question,
                    st.session_state.chat_history
                )
                
                # Update chat history
                st.session_state.chat_history.append((question, answer))
                
                # Display answer in a card-like container
                st.markdown("### Answer:")
                
                # Split the answer into sections and format them
                sections = answer.split('\n')
                answer_content = ""
                sources_content = ""
                conflicts_content = ""
                
                for line in sections:
                    line = line.strip()
                    if line.startswith('Answer:'):
                        answer_content = line.replace('Answer:', '').strip()
                    elif line.startswith('Sources:'):
                        sources_content = line.replace('Sources:', '').strip()
                    elif line.startswith('Conflicts:'):
                        conflicts_content = line.replace('Conflicts:', '').strip()
                    elif line:  # Add non-empty lines to the appropriate section
                        if answer_content and not sources_content:
                            answer_content += f" {line}"
                        elif sources_content and not conflicts_content:
                            sources_content += f" {line}"
                        elif conflicts_content:
                            conflicts_content += f" {line}"

                # Display the main answer
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; padding:20px; border-radius:5px; margin-bottom:20px; background-color:#262730; color:white;">
                        <p style="font-size:16px; margin:0;">{answer_content}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display sources if present
                if sources_content:
                    st.markdown(
                        f"""
                        <div style="background-color:#1E1E1E; padding:15px; border-radius:5px; margin-bottom:20px; color:white;">
                            <strong>Sources Used:</strong><br>
                            <p style="margin:5px 0 0 0;">{sources_content}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Display conflicts if present
                if conflicts_content and conflicts_content.lower() != "none":
                    st.markdown(
                        f"""
                        <div style="background-color:#332D1E; padding:15px; border-radius:5px; margin-bottom:20px; color:white;">
                            <strong>Conflicting Information:</strong><br>
                            <p style="margin:5px 0 0 0;">{conflicts_content}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Display sources grouped by document
                with st.expander("📚 View Sources", expanded=True):
                    # Group sources by document
                    sources_by_doc = {}
                    for source in sources:
                        doc_name = source.metadata.get("source", "Unknown")
                        if doc_name not in sources_by_doc:
                            sources_by_doc[doc_name] = []
                        sources_by_doc[doc_name].append(source)
                    
                    # Display sources grouped by document
                    for doc_name, doc_sources in sources_by_doc.items():
                        st.markdown(f"### From {doc_name}:")
                        for i, source in enumerate(doc_sources, 1):
                            st.markdown(f"""
                            <div style="background-color:#262730; padding:10px; border-radius:5px; margin:10px 0; color:white;">
                                {source.page_content}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Update to pass context for highlighting
                            if st.button(f"🔍 View in PDF {doc_name}", key=f"view_pdf_{doc_name}_{i}"):
                                st.session_state.current_pdf = {
                                    'name': doc_name,
                                    'text': context_used,  # Use the actual context
                                    'page_number': None
                                }
                                st.rerun()
                        st.markdown("---")
            
            # Display chat history
            if st.session_state.chat_history:
                with st.expander("💬 Chat History", expanded=False):
                    for q, a in reversed(st.session_state.chat_history):
                        # Clean up the answer text by removing any HTML tags
                        clean_answer = a.replace('</div>', '').replace('<div>', '')
                        
                        st.markdown("**Question:**")
                        st.markdown(f"""
                        <div style="background-color:#262730; padding:15px; border-radius:5px; margin:10px 0; color:white;">
                            {q}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("**Answer:**")
                        st.markdown(f"""
                        <div style="background-color:#262730; padding:15px; border-radius:5px; margin:10px 0; color:white;">
                            {clean_answer}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<hr style='border-top: 2px solid #4a4a4a;'>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### PDF Viewer")
    if st.session_state.current_pdf and st.session_state.current_pdf.get('name') in st.session_state.pdf_bytes_dict:
        pdf_name = st.session_state.current_pdf['name']
        text_to_highlight = st.session_state.current_pdf['text']

        try:
            # Get PDF bytes
            pdf_bytes = st.session_state.pdf_bytes_dict[pdf_name]
            
            # Add page navigation
            highlighted_pdf_bytes, total_pages, current_page = highlight_pdf(pdf_bytes, text_to_highlight)
            
            # Page navigation
            page_number = st.number_input(
                "Page", 
                min_value=1, 
                max_value=total_pages,
                value=current_page,
                step=1,
                key="page_number"
            )
            
            # Highlight text on the selected page
            highlighted_pdf_bytes, _, _ = highlight_pdf(pdf_bytes, text_to_highlight, page_number - 1)
            
            # Convert to base64 for display
            b64_pdf = get_pdf_display_base64(highlighted_pdf_bytes)
            
            if b64_pdf:
                # Display current page info
                st.markdown(f"**Viewing page {page_number} of {total_pages}**")
                
                # Display PDF with highlights
                pdf_display = f'''
                    <iframe
                        src="data:application/pdf;base64,{b64_pdf}#page={page_number}"
                        width="100%"
                        height="800px"
                        class="pdf-viewer"
                        type="application/pdf"
                    >
                    </iframe>
                '''
                st.markdown(pdf_display, unsafe_allow_html=True)
                
                # Navigation buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("⬅️ Previous", disabled=page_number <= 1):
                        st.session_state.page_number = max(1, page_number - 1)
                        st.rerun()
                with col2:
                    if st.button("➡️ Next", disabled=page_number >= total_pages):
                        st.session_state.page_number = min(total_pages, page_number + 1)
                        st.rerun()
                with col3:
                    # Add download button
                    st.download_button(
                        label="⬇️ Download",
                        data=highlighted_pdf_bytes,
                        file_name=f"highlighted_{pdf_name}",
                        mime="application/pdf"
                    )
            else:
                st.error("Error displaying PDF. Please try again.")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    else:
        st.info("Select a source to view the PDF with highlights")
        
        # Display some instructions
        st.markdown("""
        #### How to view PDFs:
        1. Click on "🔍 View in PDF" next to any source
        2. Use page navigation to move through the document
        3. The relevant text will be highlighted in yellow
        4. Use the download button to save the highlighted PDF
        """)