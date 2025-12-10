from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()




def load_vector_db(db_path):
    """
    Load the saved FAISS vector database from disk.
    
    Args:
        db_path (str): Path to the saved FAISS database
        
    Returns:
        FAISS: Loaded vector database
    """
    # Initialize HuggingFaceEmbeddings with same model used during indexing
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load FAISS database using FAISS.load_local()
    vector_db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print(f"✓ Vector database loaded from {db_path}")
    print(f"  - Total vectors: {vector_db.index.ntotal}")
    
    # Return loaded vector database
    return vector_db


def retrieve_context(vector_db, question, top_k=10):
    """
    Retrieve relevant document chunks using similarity search.
    
    Args:
        vector_db (FAISS): Loaded FAISS database
        question (str): User's question to search for relevant context
        top_k (int): Number of top chunks to retrieve
        
    Returns:
        list: List of retrieved document chunks
    """
    # Perform similarity search using vector_db
    retrieved_chunks = vector_db.similarity_search(question, k=top_k)
    
    print(f"✓ Retrieved {len(retrieved_chunks)} relevant chunks")
    
    # Return top_k retrieved chunks
    return retrieved_chunks


def format_context(docs):
    """
    Formats retrieved chunks with metadata for readability.
    
    Args:
        docs (list): List of retrieved document chunks
        
    Returns:
        str: Formatted context string
    """
    formatted = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source} | Page {page}]\n{d.page_content.strip()}")
    return "\n\n".join(formatted)


def generate_answer(question, retrieved_chunks):
    """
    Generate final answer using LLM with retrieved context.
    
    Args:
        question (str): User question
        retrieved_chunks (list): Retrieved chunks from vector DB
        
    Returns:
        str: Generated answer from LLM
    """
    # Check if we have retrieved chunks
    if not retrieved_chunks:
        return "I'm sorry, but the provided document does not contain enough information to answer that."
    
    # Prepare prompt by combining question + retrieved chunks
    ctx = format_context(retrieved_chunks)
    
    
    SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions based ONLY on the provided context from the PDF documents. 

    Rules:
    - If the answer is in the context, provide a clear and concise response
    - If the answer is NOT in the context, say "I cannot answer this based on the provided documents"
    - Do not make up information
    - Be direct and factual"""

    full_prompt = f"""
{SYSTEM_PROMPT}
---------------------
📘 **Context:**
{ctx}
---------------------
❓ **Question:**
{question}
---------------------
🧩 **Answer:**
""".strip()
    
    # Initialize LLM
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set!")
    
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=api_key,
        temperature=0.7,
        max_tokens=1024
    )
    
    # Pass prompt to LLM
    response = llm.invoke(full_prompt)
    
    # Extract clean text output
    if hasattr(response, "content"):
        answer_text = response.content.strip()
    elif isinstance(response, dict) and "content" in response:
        answer_text = response["content"].strip()
    else:
        answer_text = str(response).strip()
    
    print("✓ Answer generated successfully")
    
    # Return generated answer
    return answer_text


def main(user_question, vector_db_path="faiss_index"):
    """
    Main function to orchestrate the complete retrieval + answer pipeline.
    
    Args:
        user_question (str): Question asked by the user
        vector_db_path (str): Path to the saved FAISS vector database
        
    Returns:
        str: Generated answer
    """
    print("=" * 60)
    print("Starting Pipeline 2: Retrieval and Answer Generation")
    print("=" * 60)
    
    # Step 1: Load vector database
    print("\n[1/3] Loading vector database...")
    vector_db = load_vector_db(vector_db_path)
    
    # Step 2: Retrieve context
    print("\n[2/3] Retrieving relevant chunks...")
    retrieved_chunks = retrieve_context(vector_db, user_question, top_k=10)
    
    # Step 3: Generate answer using LLM
    print("\n[3/3] Generating answer using LLM...")
    answer = generate_answer(user_question, retrieved_chunks)
    
    print("\nPipeline 2 completed successfully!")
    print("=" * 60)
    
    return answer


# if __name__ == "__main__":
#     # Example usage
#     USER_QUESTION = "What is the main topic of the document?"
#     VECTOR_DB_PATH = "faiss_index"
    
#     # Call main function with appropriate parameters
#     answer = main(USER_QUESTION, VECTOR_DB_PATH)
#     print(f"\nAnswer: {answer}")