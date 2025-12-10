import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_pdf_documents(folder_path):
    """
    Load all PDF documents from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing PDF reports
        
    Returns:
        list: List of loaded documents
    """
    # Initialize DirectoryLoader with PyPDFLoader
    loader = DirectoryLoader(
        folder_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    # Load all PDF files from the folder
    documents = loader.load()
    
    print(f"✓ Loaded {len(documents)} documents from {folder_path}")
    
    # Return loaded documents
    return documents


def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents (list): List of loaded documents
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"✓ Created {len(chunks)} chunks")
    
    # Return chunks
    return chunks


def create_vector_db(chunks, db_path="faiss_index"):
    """
    Create and save FAISS vector database from document chunks.
    
    Args:
        chunks (list): List of text chunks to embed
        db_path (str): Path where the FAISS database will be saved
        
    Returns:
        FAISS: The created vector database
    """
    # Initialize HuggingFaceEmbeddings with "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"✓ Embedding model loaded")
    
    # Create FAISS vector store from chunks using embeddings
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    print(f"✓ FAISS vector database created")
    
    # Save the vector database to disk
    vector_db.save_local(db_path)
    
    print(f"✓ Vector database saved to {db_path}")
    print(f"  - Index file: {db_path}/index.faiss")
    print(f"  - Pickle file: {db_path}/index.pkl")
    
    # Return the vector database
    return vector_db


def main(reports_folder_path, vector_db_path="faiss_index"):
    """
    Main function to orchestrate the complete indexing pipeline.
    
    Args:
        reports_folder_path (str): Path to the folder containing laboratory reports
        vector_db_path (str): Path where the FAISS database will be saved
    """
    print("=" * 60)
    print("Starting Pipeline 1: Data Indexing")
    print("=" * 60)
    
    # Step 1: Load PDF documents
    print("\n[1/4] Loading PDF documents...")
    documents = load_pdf_documents(reports_folder_path)
    
    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=100)
    
    # Step 3: Create and save vector database
    print("\n[3/4] Creating embeddings and vector database...")
    vector_db = create_vector_db(chunks, vector_db_path)
    
    # Step 4: Completion
    print("\n[4/4] Pipeline completed successfully!")
    print(f"Vector database saved at: {vector_db_path}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    REPORTS_FOLDER = "Dataset"
    VECTOR_DB_PATH = "faiss_index"
    
    # Call main function with appropriate parameters
    main(REPORTS_FOLDER, VECTOR_DB_PATH)