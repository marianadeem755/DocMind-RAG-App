import streamlit as st
import tempfile
import os
import re
import time
from typing import List, Dict, Any, Tuple
import pickle
import hashlib
from dotenv import load_dotenv
import logging
import uuid
import base64
from io import BytesIO
import io
import fitz  # PyMuPDF for PDF image extraction
from PIL import Image 
# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Document Processing
import PyPDF2
import docx
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

# Vector Store
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Embeddings and LLM
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain

# For flowchart generation
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Configure page
st.set_page_config(
    page_title="DocMind RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "document_sources" not in st.session_state:
    st.session_state.document_sources = {}
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3-70b-8192"
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 250
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 50
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.1
if "max_token_limit" not in st.session_state:
    st.session_state.max_token_limit = 8192
if "cache_dir" not in st.session_state:
    st.session_state.cache_dir = os.path.join(tempfile.gettempdir(), "docmind_cache")
    os.makedirs(st.session_state.cache_dir, exist_ok=True)
if "learning_style" not in st.session_state:
    st.session_state.learning_style = "Standard"
# Get API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.sidebar.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    logger.error("GROQ_API_KEY not found in environment variables")

# Create a sidebar for configuration options
st.sidebar.title("DocMind RAG System")

# Model Configuration
with st.sidebar.expander("🤖 Model Configuration", expanded=False):
    model_options = {
        "llama3-70b-8192": "Llama-3-70b-8192",
        "llama3-8b-8192": "Llama-3-8b-8192"
    }
    
    st.session_state.selected_model = st.selectbox(
        "Select Model", 
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.selected_model)
    )
    st.session_state.chunk_size = st.slider(
        "Chunk Size", 
        min_value=100, 
        max_value=2000, 
        value=st.session_state.chunk_size, 
        step=100,
        help="Size of text chunks for processing"
    )
    
    st.session_state.chunk_overlap = st.slider(
        "Chunk Overlap", 
        min_value=0, 
        max_value=500, 
        value=st.session_state.chunk_overlap, 
        step=50,
        help="Overlap between consecutive chunks"
    )
    
    st.session_state.temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.temperature, 
        step=0.1,
        help="Controls randomness in responses (0=deterministic, 1=creative)"
    )

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info("📚 DocumentsMind RAG System")
st.sidebar.markdown("Powered by Groq & LangChain")
# Add this function to extract images from PDFs
def extract_images_with_context(file_path):
    """Extract images from PDF files along with surrounding text context"""
    images = []
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Get page text for context
            page_text = page.get_text()
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_format = base_image.get("ext", "png").lower()
                
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.width > 100 and image.height > 100:
                        # Get image rectangle on page
                        img_rect = page.get_image_bbox(img_info)
                        
                        # Extract text near the image (context)
                        # Get text before and after the image within a certain range
                        context_text = extract_text_around_rect(page, img_rect, radius=200)
                        
                        buffered = io.BytesIO()
                        image.save(buffered, format=image_format.upper())
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        images.append({
                            "data": img_str,
                            "format": image_format,
                            "page": page_num + 1,
                            "width": image.width,
                            "height": image.height,
                            "context": context_text,  # Store context with image
                            "page_text": page_text    # Store full page text for broader context
                        })
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
        print(f"Extracted {len(images)} images with context from {file_path}")
        return images
    except Exception as e:
        print(f"Error extracting images: {e}")
        return []

def extract_text_around_rect(page, rect, radius=200):
    """Extract text around a rectangle on a page"""
    # Expand the rectangle by the radius
    expanded_rect = (
        max(0, rect[0] - radius),
        max(0, rect[1] - radius),
        min(page.rect.width, rect[2] + radius),
        min(page.rect.height, rect[3] + radius)
    )
    
    # Get text in the expanded rectangle
    return page.get_text("text", clip=expanded_rect)

# Add this function to find semantically relevant images
def find_relevant_images(query, document_images, embeddings):
    """Find images that are semantically relevant to the query"""
    if not document_images:
        return []
    
    relevant_images = []
    
    # Create query embedding
    query_embedding = embeddings.embed_query(query)
    
    # For each document with images
    for doc_name, images in document_images.items():
        for img in images:
            if "context" in img:
                # Create embedding for image context
                try:
                    context_embedding = embeddings.embed_query(img["context"])
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        [query_embedding],
                        [context_embedding]
                    )[0][0]
                    
                    # If similarity exceeds threshold, consider relevant
                    if similarity > 0.3:  # Adjust threshold as needed
                        relevant_images.append({
                            **img,
                            "source": doc_name,
                            "similarity": similarity
                        })
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
    
    # Sort by relevance
    relevant_images.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    
    # Return top results (limit to avoid too many images)
    return relevant_images[:3]

# Add this function to store images in session state
def process_document_with_images(file, progress_bar=None):
    """Process a document and extract text and images"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Get document loader
        loader = get_loader_for_file(tmp_path)
        
        # Load document
        docs = loader.load()
        
        # Update progress if provided
        if progress_bar:
            progress_bar.progress(0.3)
        
        # Extract images if PDF
        images = []
        if tmp_path.lower().endswith('.pdf'):
            images = extract_images_with_context(tmp_path)
            
        # Update progress if provided
        if progress_bar:
            progress_bar.progress(0.6)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(docs)
        
        # Store source information for each chunk
        for chunk in chunks:
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = file.name
        
        # Update progress if provided
        if progress_bar:
            progress_bar.progress(1.0)
            
        return chunks, images
    
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None, []
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

# Helper Functions
def get_document_hash(file_bytes: bytes) -> str:
    """Generate a hash for a document to use as a unique identifier"""
    return hashlib.md5(file_bytes).hexdigest()


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF files"""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX files"""
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT files"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def get_loader_for_file(file_path: str):
    """Return the appropriate document loader based on file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension == '.docx':
        return Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def process_document_with_improved_images(file, progress_bar=None):
    """Process a document and extract text and images with context"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Get document loader
        loader = get_loader_for_file(tmp_path)
        
        # Load document
        docs = loader.load()
        
        # Update progress if provided
        if progress_bar:
            progress_bar.progress(0.3)
        
        # Extract images with context if PDF
        images = []
        if tmp_path.lower().endswith('.pdf'):
            images = extract_images_with_context(tmp_path)
            
        # Update progress if provided
        if progress_bar:
            progress_bar.progress(0.6)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(docs)
        
        # Store source information for each chunk
        for chunk in chunks:
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = file.name
        
        # Update progress if provided
        if progress_bar:
            progress_bar.progress(1.0)
            
        return chunks, images
    
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None, []
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
def save_vector_store(vector_store, name):
    """Save the vector store to disk for persistence"""
    cache_path = os.path.join(st.session_state.cache_dir, f"{name}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(vector_store, f)


def load_vector_store(name):
    """Load a vector store from disk if it exists"""
    cache_path = os.path.join(st.session_state.cache_dir, f"{name}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def initialize_llm():
    """Initialize the language model with current settings"""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        return None
    
    try:
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_token_limit,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    except Exception as e:
        st.error(f"Error initializing language model: {str(e)}")
        logger.error(f"Error initializing language model: {str(e)}")
        return None


def initialize_embeddings():
    """Initialize embedding model"""
    try:
        # Using HuggingFace embeddings as Groq doesn't provide embedding API
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=os.path.join(st.session_state.cache_dir, "hf_models")
        )
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        logger.error(f"Error initializing embeddings: {str(e)}")
        return None


# Improved get_learning_style_prompt function to ensure visual elements for visual learners
def get_learning_style_prompt(learning_style):
    """Return prompt instructions based on learning style"""
    styles = {
        "Visual learner": """
        FORMAT YOUR ANSWER FOR A VISUAL LEARNER:
        - Always include a Markdown header: "## Visual Learner Format"
        - Use descriptive language that creates mental images
        - Structure information spatially (use tables, diagrams when possible)
        - Suggest visual metaphors for complex concepts
        - Include a Mermaid.js diagram to visualize the key concepts
        - Always include at least one table to organize the information
        - Create a flowchart for process-related information
        
        Example visual elements you MUST include:
        
        ```mermaid
        graph TD
            A[Concept] --> B[Related Concept]
            B --> C[Example]
            A --> D[Another Related Concept]
        ```
        
        | Concept | Description | Example |
        |---------|-------------|---------|
        | Key Idea 1 | Description 1 | Example 1 |
        | Key Idea 2 | Description 2 | Example 2 |
        
        DO NOT omit the table and diagram - they are essential for visual learning.
        """,
        
        "Auditory learner": """
        FORMAT YOUR ANSWER FOR AN AUDITORY LEARNER:
        - Always include a Markdown header: "## Auditory Learner Format"
        - Use conversational language and rhetorical questions
        - Include examples of how you'd explain this verbally
        - Suggest mnemonics or rhymes to remember key points
        - Use a dialogue format where appropriate (Q&A)
        - Repeat key points in different ways
        """,
        
        "Reading/writing learner": """
        Format your answer for a reading/writing learner:
        1. Use precise and detailed textual explanations
        2. Include well-structured paragraphs with topic sentences
        3. Provide detailed lists and definitions
        4. Use quotes and references when appropriate
        5. Include clear headings, subheadings, and a logical progression of ideas
        6. Emphasize key points through repetition in written form
        """,
        
        "Kinesthetic learner": """
        Format your answer for a kinesthetic learner:
        1. Focus on practical applications and real-world examples
        2. Include step-by-step processes and procedures
        3. Relate concepts to physical actions or sensations
        4. Use case studies and scenarios that involve doing or experiencing
        5. Include interactive elements or suggestions for hands-on activities
        6. Break information into actionable chunks with clear sequences
        """
    }
    
    return styles.get(learning_style, "")
# Enhanced function to perform QA with visual elements for visual learners
def perform_qa_with_images(question, vector_store, k=4, learning_style="Standard"):
    """Perform question answering using RAG with learning style adaptation and semantic image search"""
    llm = initialize_llm()
    if not llm:
        return None
    
    # Initialize embeddings for semantic search
    embeddings = initialize_embeddings()
    if not embeddings:
        return None
    
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    learning_style_instructions = get_learning_style_prompt(learning_style)
    
    # Add specific instructions for visual elements when visual learner is selected
    visual_instructions = ""
    if learning_style == "Visual learner":
        visual_instructions = """
        You MUST include these visual elements in your response:
        1. A table summarizing key points
        2. Use of spatial organization for information
        
        Do not mention that you're including these elements because of instructions - just include them naturally.
        """
    
    # Enhanced RAG prompt with better context integration
    qa_template = f"""
    You are a helpful assistant that answers questions based on provided context.
    
    Context:
    {{context}}
    
    Question: {{question}}
    
    Instructions:
    1. Answer the question based ONLY on the provided context.
    2. If you don't know the answer based on the context, say "I don't have enough information to answer this question."
    3. Provide specific references to the source documents when possible.
    4. Be concise but thorough in your response.
    5. Format your answer in a clear, readable way with appropriate headings, bullet points, and structure.
    6. If the question is about a general topic not specifically covered in the documents, synthesize information from the context to provide a helpful response.
    7. If the question requires reasoning beyond what's in the documents, use the provided context as a foundation and clearly indicate when you're making inferences.
    
    {learning_style_instructions}
    {visual_instructions}
    
    Answer:
    """
    
    QA_PROMPT = PromptTemplate(
        template=qa_template, 
        input_variables=["context", "question"]
    )
    
    chain_type_kwargs = {"prompt": QA_PROMPT}
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    
    result = qa({"query": question})
    
    # Extract source information
    sources = []
    source_documents = []
    try:
        for doc in result.get("source_documents", []):
            source_documents.append(doc)
            if "source" in doc.metadata:
                if doc.metadata["source"] not in sources:
                    sources.append(doc.metadata["source"])
    except:
        pass
    
    # Find semantically relevant images
    relevant_images = []
    if "document_images" in st.session_state:
        # Find images related to the query semantically
        query_relevant_images = find_relevant_images(
            question, 
            st.session_state.document_images, 
            embeddings
        )
        relevant_images.extend(query_relevant_images)
        
        # Also include images from source documents
        for source in sources:
            if source in st.session_state.document_images:
                # Filter to most relevant images from this source
                source_images = st.session_state.document_images[source]
                if source_images:
                    # Get up to 2 images from each source document
                    relevant_images.extend(source_images[:2])
    
    # Always initialize flowchart_result to None
    flowchart_result = None
                
    # Generate a flowchart for visual learners
    if learning_style == "Visual learner":
        try:
            flowchart_result = generate_hierarchical_flowchart(vector_store, question)
        except Exception as e:
            logger.error(f"Error generating flowchart: {str(e)}")
            flowchart_result = None
    
    # Remove duplicates by checking data strings
    unique_images = []
    unique_data_strings = set()
    for img in relevant_images:
        if img["data"][:50] not in unique_data_strings:  # Use start of data as identifier
            unique_data_strings.add(img["data"][:50])
            unique_images.append(img)
    
    return {
        "answer": result.get("result", "No answer generated"),
        "sources": sources,
        "relevant_images": unique_images,
        "flowchart": flowchart_result
    }


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF files"""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text
# Enhanced generate_summary function for visual learners
def generate_summary(vector_store, learning_style: str = "Standard"):
    """Generate a summary of the documents with learning style adaptation"""
    llm = initialize_llm()
    if not llm:
        return None
    
    # Get documents from vector store
    docs = vector_store.similarity_search("", k=10)
    
    # Limit number of documents for summary to avoid token limit issues
    if len(docs) > 10:
        docs = docs[:10]
    
    learning_style_instructions = get_learning_style_prompt(learning_style)
    
    # Add specific instructions for visual elements when visual learner is selected
    visual_instructions = ""
    if learning_style == "Visual learner":
        visual_instructions = """
        You MUST include these visual elements in your response:
        1. A table summarizing key points from the documents
        2. A mermaid flowchart showing the relationship between main concepts
        3. Use of spatial organization for information
        
        Do not mention that you're including these elements because of instructions - just include them naturally.
        """
    
    # Use map_reduce for more efficient summarization of large documents
    map_prompt_template = """
    Write a concise summary of the following text:
    "{text}"
    CONCISE SUMMARY:
    """
    MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    
    combine_prompt_template = f"""
    You are a helpful assistant that combines multiple document summaries into a comprehensive overview.
    
    Below are summaries of different sections of a document or multiple documents:
    {{text}}
    
    Create a final comprehensive summary that captures the key information from all these summaries.
    
    {learning_style_instructions}
    {visual_instructions}
    
    The summary should be well-organized, coherent, and highlight the most important points.
    Use appropriate headings, bullet points, and structure to make the summary easy to read and understand.
    
    COMPREHENSIVE SUMMARY:
    """
    COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
    
    summary_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        verbose=True
    )
    
    summary = summary_chain.run(docs)
    
    # Extract source information
    sources = []
    for doc in docs:
        if "source" in doc.metadata:
            if doc.metadata["source"] not in sources:
                sources.append(doc.metadata["source"])
    
    # Generate flowchart for visual learners
    flowchart_result = None
    if learning_style == "Visual learner":
        # Add the function call to generate a flowchart
        try:
            flowchart_result = generate_hierarchical_flowchart(vector_store, "Create a concept map of the main ideas in these documents")
        except Exception as e:
            logger.error(f"Error generating flowchart: {str(e)}")
            flowchart_result = None
    
    # Extract key topics for visual learners to create a table visualization
    table_data = None
    if learning_style == "Visual learner":
        try:
            keywords = extract_keywords(vector_store, num_keywords=8)
            table_data = {
                "keywords": keywords,
                "explanation": "Key topics extracted from the documents"
            }
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            table_data = None
    
    return {
        "summary": summary,
        "sources": sources,
        "flowchart": flowchart_result,
        "table_data": table_data
    }

# Add this function to display the analysis results properly
def display_analysis_results(result, analysis_type, learning_style):
    """Display analysis results with appropriate visual elements based on learning style"""
    st.markdown(f"### {analysis_type} Results")
    st.markdown(result["result"])
    
    # For visual learners, always show the flowchart
    if learning_style == "Visual learner" and "flowchart" in result and result["flowchart"]:
        st.markdown("### Visual Representation")
        st.image(f"data:image/png;base64,{result['flowchart']['visual']}", 
                 caption=result["flowchart"].get("title", f"Visual representation of {analysis_type}"), 
                 use_column_width=True)

def generate_flowchart(vector_store):
    """Generate a flowchart based on document content or extract existing diagrams"""
    llm = initialize_llm()
    if not llm:
        return None
    
    # First check if any documents contain diagrams/images
    docs = vector_store.similarity_search("diagram OR image OR figure OR chart", k=5)
    
    # Check if any documents mention diagrams
    has_diagrams = any(
        "diagram" in doc.page_content.lower() or 
        "image" in doc.page_content.lower() or
        "figure" in doc.page_content.lower() or
        "chart" in doc.page_content.lower()
        for doc in docs
    )
    
    if has_diagrams:
        # Extract information about existing diagrams
        diagram_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the following text and identify any diagrams, images, or figures that are referenced.
            For each visual element found, provide:
            1. Its title or description
            2. The page/section where it appears
            3. A brief explanation of what it shows
            
            If no diagrams are found, say "No diagrams found in the documents."
            
            Text:
            {text}
            
            Diagram Analysis:
            """
        )
        
        chain = LLMChain(llm=llm, prompt=diagram_prompt)
        result = chain.run(text="\n\n".join([doc.page_content for doc in docs]))
        
        return {
            "type": "description",
            "content": result,
            "visual": None
        }
    else:
        # No diagrams found, create a new flowchart
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        
        flowchart_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the following text and identify key concepts, processes, or entities that could be organized into a flowchart.
            
            Text:
            {text}
            
            Please extract:
            1. A list of 5-10 key nodes (main concepts)
            2. A descriptive title for the flowchart
            
            Format your response as JSON:
            {{
                "title": "The flowchart title",
                "nodes": ["Node1", "Node2", "Node3", ...],
                "connections": [
                    ["Node1", "Node2", "description of connection"],
                    ["Node2", "Node3", "description of connection"],
                    ...
                ]
            }}
            
            Only provide the JSON with no other text or explanation.
            """
        )
        
        chain = LLMChain(llm=llm, prompt=flowchart_prompt)
        result = chain.run(text=combined_text)
        
        try:
            import json
            flowchart_data = json.loads(result)
            img_str = create_flowchart_image(flowchart_data)
            
            return {
                "type": "flowchart",
                "content": flowchart_data,
                "visual": img_str
            }
        except Exception as e:
            st.error(f"Error creating flowchart: {str(e)}")
            return None

def create_flowchart_image(flowchart_data):
    """Create a flowchart image from structured data"""
    try:
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in flowchart_data["nodes"]:
            G.add_node(node)
        
        # Add edges
        for connection in flowchart_data["connections"]:
            from_node, to_node, label = connection
            G.add_edge(from_node, to_node, label=label)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.title(flowchart_data["title"], fontsize=16)
        
        # Set up the layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.8)
        
        # Draw edges
        curved_edges = [edge for edge in G.edges()]
        nx.draw_networkx_edges(
            G, pos, edgelist=curved_edges, width=2, alpha=0.5, 
            connectionstyle="arc3,rad=0.1", arrowsize=20
        )
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=10, 
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7), 
            rotate=False, label_pos=0.5
        )
        
        plt.axis("off")
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode()
        
        return img_str
    except Exception as e:
        st.error(f"Error generating flowchart: {str(e)}")
        return None


import re
def extract_keywords(vector_store, num_keywords=10):
    """Extract key topics/keywords from the documents"""
    llm = initialize_llm()
    if not llm:
        return None

    # Get a sample of documents from vector store
    docs = vector_store.similarity_search("", k=5)

    # Combine document contents
    combined_text = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt for keyword extraction with clear formatting instructions
    keyword_prompt = PromptTemplate(
        input_variables=["text", "num_keywords"],
        template="""
        Extract exactly {num_keywords} important keywords or key phrases from the following text.
        Return ONLY the keywords, one per line, with no numbering, explanations, or additional text.

        TEXT: {text}
        KEYWORDS:
        """
    )
    
    chain = LLMChain(llm=llm, prompt=keyword_prompt)
    result = chain.run(text=combined_text, num_keywords=num_keywords)

    # Clean and format the keywords
    cleaned_result = result.strip()

    # Remove common prefixes and explanatory text
    unwanted_patterns = [
        "Here are the", "most important", "keywords or key phrases", "extracted from the text:",
        "separated by commas:", "The", "keywords are:", "Here's the list of", "1.", "2.", "3.",
        "•", "*", "-", "KEYWORDS:", "Key topics:"
    ]
    
    for pattern in unwanted_patterns:
        cleaned_result = cleaned_result.replace(pattern, "")

    # Split by common delimiters (newlines, commas) and clean each item
    if '\n' in cleaned_result:
        keywords = [k.strip() for k in cleaned_result.split('\n') if k.strip()]
    else:
        keywords = [k.strip() for k in cleaned_result.split(',') if k.strip()]

    # Further clean any numbering or bullets that might remain
    keywords = [re.sub(r'^\d+\.\s*', '', k).strip() for k in keywords]

    # Ensure each keyword is 1-3 words maximum
    filtered_keywords = []
    for k in keywords:
        words = k.split()
        if len(words) <= 10:
            filtered_keywords.append(k)
        else:
            filtered_keywords.append(' '.join(words[:3]))

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = [k for k in filtered_keywords if not (k in seen or seen.add(k))]

    # Exclude the first keyword and first topic
    if len(unique_keywords) > 1:
        return unique_keywords[1:]  # Return all except the first keyword
    return unique_keywords  # Return the list if it has one or no keywords

    
    # Ensure we return exactly num_keywords if possible
    return unique_keywords[:num_keywords]
def analyze_sentiment(vector_store):
    """Analyze the overall sentiment of the documents"""
    llm = initialize_llm()
    if not llm:
        return None
    
    # Get a sample of documents from vector store
    docs = vector_store.similarity_search("", k=5)
    
    # Combine document contents
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt for sentiment analysis
    sentiment_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze the sentiment of the following text. Determine if it is:
        1. Strongly Positive
        2. Positive
        3. Neutral
        4. Negative
        5. Strongly Negative
        
        Also provide a brief explanation of your assessment.
        
        TEXT:
        {text}
        
        SENTIMENT ANALYSIS:
        """
    )
    
    chain = LLMChain(llm=llm, prompt=sentiment_prompt)
    result = chain.run(text=combined_text)
    
    return result

# Add this new function to generate proper hierarchical flowcharts
def generate_hierarchical_flowchart(vector_store, question):
    """Generate a hierarchical flowchart based on document content and question"""
    llm = initialize_llm()
    if not llm:
        return None
    
    # Get related documents
    docs = vector_store.similarity_search(question, k=5)
    
    # Combine document contents
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    
    flowchart_prompt = PromptTemplate(
        input_variables=["text", "question"],
        template="""
        Analyze the following text and the user's question. Create a hierarchical flowchart or concept map showing the key concepts and their relationships.

        User question: {question}
        
        Text:
        {text}
        
        Create a hierarchical flowchart with 5-10 nodes that effectively visualizes the main concepts related to the question.
        
        Format your response as JSON:
        {{
            "title": "A clear title for the flowchart",
            "nodes": ["Main Concept", "Sub-concept 1", "Sub-concept 2", ...],
            "connections": [
                ["Main Concept", "Sub-concept 1", "relationship"],
                ["Main Concept", "Sub-concept 2", "relationship"],
                ["Sub-concept 1", "Sub-concept 1.1", "relationship"],
                ...
            ],
            "layout": "hierarchical"
        }}
        
        Only provide the JSON with no other text or explanation.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=flowchart_prompt)
    result = chain.run(text=combined_text, question=question)
    
    try:
        import json
        flowchart_data = json.loads(result)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in flowchart_data["nodes"]:
            G.add_node(node)
        
        # Add edges
        for connection in flowchart_data["connections"]:
            from_node, to_node, label = connection
            G.add_edge(from_node, to_node, label=label)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.title(flowchart_data["title"], fontsize=16)
        
        # Use hierarchical layout
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, arrowsize=20)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.axis("off")
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode()
        
        return {"visual": img_str, "title": flowchart_data["title"]}
    except Exception as e:
        print(f"Error generating flowchart: {str(e)}")
        return None
# Main Navigation Menu using native Streamlit components
tabs = st.tabs(["📄 Upload", "🔍 Q&A", "📝 Summarize", "📊 Analyze"])

# Upload Documents Page
with tabs[0]:
    st.title("📄 Upload Documents")
    st.markdown("Upload your documents for processing and analysis. Supported formats: PDF, DOCX, TXT")
    
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info(f"Current Settings:\n- Chunk Size: {st.session_state.chunk_size}\n- Chunk Overlap: {st.session_state.chunk_overlap}")
    
    with col2:
        clear_button = st.button("Clear All Documents")
        if clear_button:
            st.session_state.documents = []
            st.session_state.vector_store = None
            st.session_state.document_sources = {}
            st.session_state.processed_docs = []
            st.success("All documents cleared!")
    
    if uploaded_files:
        process_button = st.button("Process Documents")
    
        if process_button:
            with st.spinner("Processing documents..."):
                # Initialize embeddings
                embeddings = initialize_embeddings()
                if not embeddings:
                    st.error("Failed to initialize embeddings. Please check your API key.")
                else:
                    # Process each document
                    new_docs = []
                    
                    # Initialize image storage if not present
                    if "document_images" not in st.session_state:
                        st.session_state.document_images = {}
                    
                    # Set up progress tracking
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    for i, file in enumerate(uploaded_files):
                        # Check if we've already processed this file
                        file_hash = get_document_hash(file.getvalue())
                        
                        if file_hash in st.session_state.processed_docs:
                            progress_text.text(f"Skipping already processed file: {file.name}")
                            continue
                        
                        progress_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})")
                        doc_progress = st.progress(0)
                        
                        chunks, images = process_document_with_images(file, progress_bar=doc_progress)
                        
                        if chunks:
                            new_docs.extend(chunks)
                            st.session_state.document_sources[file.name] = len(chunks)
                            st.session_state.processed_docs.append(file_hash)
                            
                            # Store images if any were found
                            if images:
                                st.session_state.document_images[file.name] = images
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if new_docs:
                        # Generate unique IDs for each document
                        texts = [doc.page_content for doc in new_docs]
                        metadatas = [doc.metadata for doc in new_docs]
                        
                        # Create new vector store if none exists
                        if st.session_state.vector_store is None:
                            # Generate unique IDs for new documents
                            ids = [str(uuid.uuid4()) for _ in range(len(new_docs))]
                            st.session_state.vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas, ids=ids)
                        else:
                            # Add new documents to existing vector store with unique IDs
                            ids = [str(uuid.uuid4()) for _ in range(len(new_docs))]
                            st.session_state.vector_store.add_texts(texts, metadatas=metadatas, ids=ids)
                        
                        progress_text.text("Creating vector index...")
                        
                        # Save vector store
                        save_vector_store(st.session_state.vector_store, "docmind_store")
                        
                        st.session_state.documents.extend(new_docs)
                        st.success(f"Successfully processed {len(new_docs)} new document chunks!")
                    else:
                        st.warning("No new documents to process.")
                    
                    # Clear progress indicators
                    progress_text.empty()
                    progress_bar.empty()

    
    # Display status of processed documents
    if st.session_state.document_sources:
        st.subheader("Processed Documents")
        
        # Create summary table
        document_data = []
        for doc_name, chunk_count in st.session_state.document_sources.items():
            document_data.append({"Document": doc_name, "Chunks": chunk_count})
        
        # Display as DataFrame
        st.dataframe(pd.DataFrame(document_data))
        
        total_chunks = sum(st.session_state.document_sources.values())
        st.info(f"Total document chunks in memory: {total_chunks}")
    
    else:
        st.write("No documents processed yet. Please upload and process some documents.")


# Then in your chat processing, utilize the memory
def process_chat_with_rag(user_input, vector_store, memory):
    llm = initialize_llm()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    result = qa({"question": user_input})
    return result
# Q&A Page
with tabs[1]:
    st.title("🔍 Question & Answer")
    
    if not st.session_state.vector_store:
        # Try to load from cache
        st.session_state.vector_store = load_vector_store("docmind_store")
    
    if not st.session_state.vector_store:
        st.warning("No documents have been processed. Please go to the Upload tab first.")
    else:
        st.markdown("Ask a question about your documents and get answers based on their content.")
        
        # Learning style selector
        learning_styles = ["Standard", "Visual learner", "Auditory learner", "Reading/writing learner", "Kinesthetic learner"]
        selected_style = st.radio("Select your learning style for the answer:", learning_styles, horizontal=True)
        # Replace with enhanced input that encourages natural language questions:
        st.markdown("Ask me anything about your documents.")
        question = st.text_area("Enter your question:", height=100, placeholder="Ask me anything about your documents. I can answer based on the content while providing additional context.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            k_value = st.slider(
                "Number of relevant chunks to retrieve",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                help="Higher values may provide more comprehensive answers but can introduce noise"
            )
        
        with col2:
            submit_button = st.button("Submit Question")
        
        with col3:
            generate_visual = st.checkbox("Generate visual flowchart", value=True)
        
        if question and submit_button:
            with st.spinner("Generating answer..."):
                # Perform Q&A with semantic image search
                result = perform_qa_with_images(question, st.session_state.vector_store, k=k_value, learning_style=selected_style)
                
                if result:
                    # Display answer with better formatting
                    st.markdown("### Answer")
                    st.markdown(result["answer"])
                    
                    # Display sources if available
                    if result["sources"]:
                        st.markdown("### Sources")
                        for source in result["sources"]:
                            st.markdown(f"- {source}")
                    
                    # Display semantically relevant images
                    if "relevant_images" in result and result["relevant_images"]:
                        st.markdown("### Relevant Images")
                        
                        # Create columns for images
                        img_cols = st.columns(min(len(result["relevant_images"]), 2))
                        
                        for i, img in enumerate(result["relevant_images"]):
                            col_idx = i % len(img_cols)
                            with img_cols[col_idx]:
                                source = img.get("source", "Unknown source")
                                similarity = img.get("similarity", None)
                                
                                caption = f"{source} (Page {img['page']})"
                                if similarity:
                                    caption += f" - Relevance: {similarity:.2f}"
                                    
                                st.image(
                                    f"data:image/{img['format']};base64,{img['data']}", 
                                    caption=caption, 
                                    use_column_width=True
                                )

# Use this updated code in the Summary tab display section
def display_summary_results(summary_result, learning_style):
    """Display summary results with appropriate visual elements based on learning style"""
    # Display summary with better formatting
    st.markdown("### Document Summary")
    st.markdown(summary_result["summary"])
    
    # Display sources if available
    if summary_result["sources"]:
        st.markdown("### Sources")
        for source in summary_result["sources"]:
            st.markdown(f"- {source}")
    
    if learning_style == "Visual learner":
        # Check if table data is available
        if "table_data" in summary_result and summary_result["table_data"]:
            print("### Key Topics Table")
            # Code to display the table goes here
        else:
            print("No table found for this document. Please refer to other resources for visual aids.")

            
        # Display flowchart if available
        if "flowchart" in summary_result and summary_result["flowchart"]:
            st.markdown("### Concept Relationship Map")
            st.image(f"data:image/png;base64,{summary_result['flowchart']['visual']}", 
                    caption=summary_result["flowchart"].get("title", "Document Concept Map"), 
                    use_column_width=True)

# Summarize Page
with tabs[2]:
    st.title("📝 Document Summarization")
    
    if not st.session_state.vector_store:
        # Try to load from cache
        st.session_state.vector_store = load_vector_store("docmind_store")
    
    if not st.session_state.vector_store:
        st.warning("No documents have been processed. Please go to the Upload tab first.")
    else:
        st.markdown("Generate a comprehensive summary of all your uploaded documents.")
        
        # Learning style selector
        summary_learning_styles = ["Standard", "Visual learner", "Auditory learner", "Reading/writing learner", "Kinesthetic learner"]
        selected_summary_style = st.radio("Select your learning style for the summary:", summary_learning_styles, horizontal=True)
        
        summary_col1, summary_col2 = st.columns([1, 1])
        
        with summary_col1:
            summary_button = st.button("Generate Summary")
        
        with summary_col2:
            summary_visual = st.checkbox("Include visual representation", value=True)
        
        if summary_button:
            with st.spinner("Generating comprehensive summary..."):
                # Generate summary
                summary_result = generate_summary(st.session_state.vector_store, learning_style=selected_summary_style)
                
                if summary_result:
                    # Use the new display function
                    display_summary_results(summary_result, selected_summary_style)
                else:
                    st.error("Failed to generate summary. Please try again.")
# Analysis Page
with tabs[3]:
    st.title("📊 Document Analysis")
    
    if not st.session_state.vector_store:
        # Try to load from cache
        st.session_state.vector_store = load_vector_store("docmind_store")
    
    if not st.session_state.vector_store:
        st.warning("No documents have been processed. Please go to the Upload tab first.")
    else:
        st.markdown("Analyze your documents to extract key information and insights.")
        
        # Tabs for different analysis types
        analysis_tabs = st.tabs(["📑 Key Topics", "🔍 Sentiment Analysis","📋 Custom Analysis"])
        
        # Key Topics tab
        with analysis_tabs[0]:
            st.subheader("Key Topics & Keywords")
            
            num_keywords = st.slider("Number of keywords to extract", 5, 20, 10)
            
            extract_topics_button = st.button("Extract Key Topics")
            
            if extract_topics_button:
                with st.spinner("Extracting key topics and keywords..."):
                    keywords = extract_keywords(st.session_state.vector_store, num_keywords=num_keywords)
                    
                    if keywords:
                        st.markdown("### Key Topics")
                        
                        keyword_cols = st.columns(3)
                        for i, keyword in enumerate(keywords):
                            col_idx = i % 3
                            with keyword_cols[col_idx]:
                                st.markdown(f"- **{keyword}**")
                        
                        with st.spinner("Generating topic visualization..."):
                            import matplotlib.pyplot as plt
                            
                            keywords_to_plot = keywords[:min(10, len(keywords))]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            y_pos = np.arange(len(keywords_to_plot))
                            importance = np.linspace(1, 0.4, len(keywords_to_plot))
                            
                            ax.barh(y_pos, importance, align='center', color='skyblue')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(keywords_to_plot)
                            ax.invert_yaxis()
                            ax.set_xlabel('Relative Importance')
                            ax.set_title('Key Topics by Relative Importance')
                            
                            buf = BytesIO()
                            plt.savefig(buf, format="png", bbox_inches="tight")
                            plt.close()
                            buf.seek(0)
                            
                            img_str = base64.b64encode(buf.read()).decode()
                            
                            st.image(f"data:image/png;base64,{img_str}", 
                                    caption="Key Topics Visualization", 
                                    use_column_width=True)
                    else:
                        st.error("Failed to extract keywords. Please try again.")
                
        # Sentiment Analysis tab
        with analysis_tabs[1]:
                    st.subheader("Sentiment Analysis")
                    
                    sentiment_button = st.button("Analyze Sentiment")
                    
                    if sentiment_button:
                        with st.spinner("Analyzing document sentiment..."):
                            sentiment_result = analyze_sentiment(st.session_state.vector_store)
                            
                            if sentiment_result:
                                st.markdown("### Sentiment Analysis Results")
                                st.markdown(sentiment_result)
                                
                                # Try to extract the sentiment category for visualization
                                sentiment_categories = ["Strongly Positive", "Positive", "Neutral", "Negative", "Strongly Negative"]
                                detected_sentiment = None
                                
                                for category in sentiment_categories:
                                    if category.lower() in sentiment_result.lower():
                                        detected_sentiment = category
                                        break
                                
                                if detected_sentiment:
                                    # Create a visual representation of sentiment
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    
                                    categories_pos = np.arange(len(sentiment_categories))
                                    sentiment_pos = sentiment_categories.index(detected_sentiment)
                                    
                                    # Create bars
                                    bars = ax.bar(categories_pos, 
                                                [0.2 if i != sentiment_pos else 0.8 for i in range(len(sentiment_categories))], 
                                                color=['lightgray' if i != sentiment_pos else 'skyblue' for i in range(len(sentiment_categories))])
                                    
                                    # Highlight the detected sentiment
                                    bars[sentiment_pos].set_color('blue')
                                    
                                    ax.set_xticks(categories_pos)
                                    ax.set_xticklabels(sentiment_categories)
                                    ax.set_ylabel('Confidence')
                                    ax.set_title('Document Sentiment Analysis')
                                    
                                    # Save to BytesIO
                                    buf = BytesIO()
                                    plt.savefig(buf, format="png", bbox_inches="tight")
                                    plt.close()
                                    buf.seek(0)
                                    
                                    # Convert to base64
                                    img_str = base64.b64encode(buf.read()).decode()
                                    
                                    # Display the chart
                                    st.image(f"data:image/png;base64,{img_str}", 
                                            caption="Sentiment Visualization", 
                                            use_column_width=True)
                            else:
                                st.error("Failed to analyze sentiment. Please try again.")                        
        # Custom Analysis tab
        with analysis_tabs[2]:
            st.subheader("Custom Document Analysis")
            
            custom_analysis_types = [
                "Content Structure Analysis",
                "Main Arguments Extraction",
                "Learning Objectives Identification",
                "Technical Complexity Assessment", 
                "Key Definitions Extraction",
                "Action Items Identification"
            ]
    
    selected_analysis = st.selectbox("Select analysis type:", custom_analysis_types)
    
    custom_learning_styles = ["Standard", "Visual learner", "Auditory learner", "Reading/writing learner", "Kinesthetic learner"]
    custom_selected_style = st.radio("Select your learning style for the analysis:", custom_learning_styles, horizontal=True)
    
    custom_analysis_button = st.button("Run Custom Analysis")
    
    if custom_analysis_button:
        with st.spinner(f"Running {selected_analysis}..."):
            analysis_result = perform_custom_analysis(selected_analysis, st.session_state.vector_store, learning_style=custom_selected_style)
            if analysis_result:
                display_analysis_results(analysis_result, selected_analysis, custom_selected_style)
            else:
                st.error(f"Failed to perform {selected_analysis}. Please try again.")
            llm = initialize_llm()
            if not llm:
                st.error("Failed to initialize language model. Please check your API key.")
            else:
                # Get documents from vector store
                docs = st.session_state.vector_store.similarity_search("", k=5)
                
                # Combine document contents
                combined_text = "\n\n".join([doc.page_content for doc in docs])
                
                learning_style_instructions = get_learning_style_prompt(custom_selected_style)
                
                # Create prompt based on selected analysis type
                analysis_prompts = {
                    "Content Structure Analysis": f"""
                    Analyze the structure of the following content. Identify main sections, subsections, and how
                    information is organized. Evaluate the logical flow of information and make recommendations
                    for improved structure if applicable.
                    
                    CONTENT:
                    {{text}}
                    """,
                    
                    "Main Arguments Extraction": f"""
                    Extract the main arguments or key points from the following content. Identify the central thesis,
                    supporting arguments, evidence provided, and any counterarguments addressed.
                    
                    {learning_style_instructions}
                    
                    CONTENT:
                    {{text}}
                    """,
                    
                    "Learning Objectives Identification": f"""
                    Analyze the following content and identify what appear to be the main learning objectives.
                    What key knowledge, skills, or competencies would someone gain from this material?
                    
                    {learning_style_instructions}
                    
                    CONTENT:
                    {{text}}
                    """,
                    
                    "Technical Complexity Assessment": f"""
                    Assess the technical complexity of the following content. Identify specialized terminology,
                    complex concepts, prerequisites needed to understand this material, and the overall complexity level.
                    
                    {learning_style_instructions}
                    
                    CONTENT:
                    {{text}}
                    """,
                    
                    "Key Definitions Extraction": f"""
                    Extract key terms and their definitions from the following content. Identify important concepts,
                    technical terms, and provide clear definitions based on the context they're used in.
                    
                    {learning_style_instructions}
                    
                    CONTENT:
                    {{text}}
                    """,
                    
                    "Action Items Identification": f"""
                    Analyze the following content and extract any action items, tasks, or next steps mentioned.
                    Identify who should complete them (if specified), deadlines or timeframes, and priority levels.
                    
                    {learning_style_instructions}
                    
                    CONTENT:
                    {{text}}
                    """
                }
                
                selected_prompt = PromptTemplate(
                    input_variables=["text"],
                    template=analysis_prompts[selected_analysis]
                )
                
                chain = LLMChain(llm=llm, prompt=selected_prompt)
                result = chain.run(text=combined_text)
                
                # Display results
                st.markdown(f"### {selected_analysis} Results")
                st.markdown(result)

# CSS Customizations
st.markdown("""
<style>
    /* Remove default Streamlit background gradient */
    .stApp {
        background-color: white;
    }
    
    /* Enhanced headers */
    h1, h2, h3 {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Better spacing for content */
    .stMarkdown {
        line-height: 1.6;
    }
    
    /* Improve button styling */
    .stButton>button {
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    
    /* Style code blocks */
    code {
        padding: 2px 4px;
        background: #f5f5f5;
        border-radius: 3px;
    }
    
    /* Adjust tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
    }

    /* Enhance dataframe styling */
    .dataframe {
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    
    .dataframe thead tr {
        background-color: #f1f1f1;
    }
    
    .dataframe th, .dataframe td {
        padding: 8px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Add explanation for learning styles
with st.sidebar.expander("ℹ️ Learning Styles Explained", expanded=False):
    st.markdown("""
    **Visual Learner**: 
    Prefers information presented with images, charts, and spatial relationships.
    
    **Auditory Learner**: 
    Learns best through listening, discussions, and verbal explanations.
    
    **Reading/Writing Learner**: 
    Prefers information displayed as words, lists, and text-based materials.
    
    **Kinesthetic Learner**: 
    Learns through doing, practical examples, and hands-on experiences.
    """)

# Add additional learning resources section
with st.sidebar.expander("📚 Learning Resources", expanded=False):
    st.markdown("""
    **For Visual Learners:**
    - Use mind maps to connect ideas
    - Convert text to diagrams
    - Watch video explanations
    
    **For Auditory Learners:**
    - Read content aloud
    - Discuss concepts with others
    - Listen to audio versions of materials
    
    **For Reading/Writing Learners:**
    - Take detailed notes
    - Rewrite key concepts in your own words
    - Create word-based outlines
    
    **For Kinesthetic Learners:**
    - Apply concepts to real problems
    - Use physical objects to model ideas
    - Take breaks to move around while studying
    """)

# Add a help section
with st.sidebar.expander("❓ How to Use", expanded=False):
    st.markdown("""
    **1. Upload Documents**
    - Upload your files (PDF, DOCX, TXT)
    - Click "Process Documents" to extract content
    
    **2. Ask Questions**
    - Type your question in the Q&A tab
    - Select your preferred learning style
    - Click "Submit Question" to get answers
    
    **3. Generate Summary**
    - Go to the Summarize tab
    - Select your preferred learning style
    - Click "Generate Summary"
    
    **4. Analyze Content**
    - Use the Analyze tab to extract key information
    - Different analysis types are available
    - Select your preferred learning style for results
    """)
# Add a clear button to the sidebar
with st.sidebar.expander("🔗 Connect With Me", expanded=False):
    st.markdown("""
    <hr>
    <div class="profile-links">
        <a href="" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="20px"> GitHub
        </a><br><br>
        <a href="" target="_blank">
            <img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-512.png" width="20px"> Kaggle
        </a><br><br>
        <a href="">
            <img src="https://cdn-icons-png.flaticon.com/512/561/561127.png" width="20px"> Email
        </a><br><br>
        <a href="https://huggingface.co/maria355" target="_blank">
            <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="20px"> Hugging Face
        </a>
    </div>
    <hr>
    """, unsafe_allow_html=True)

