
import hashlib
from io import BytesIO
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def parse_pdf(file: BytesIO, filename: str):
    pdf = PdfReader(file)
    full_text = ""
    
    # Extract text from each page
    for page in pdf.pages:
        full_text += page.extract_text() or ""
    
    # Set up a recursive character splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    
    # Split the text into chunks
    chunks = text_splitter.split_text(full_text)
    
    return chunks


def create_embeddings(texts: list[str], openai_api_key: str):
    embedding = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
    return embedding.embed_documents(texts)
    

def get_embeddings(pdf_file: BytesIO, file_name: str, openai_api_key: str):
    embeddings = []
    pages = []
    for file, file_name in zip(pdf_file, file_name):
        # pages = parse_pdf(BytesIO(file), file_name)
        page = parse_pdf(BytesIO(file), file_name)
        pages.append(page)
        embeddings.append(create_embeddings(page, openai_api_key))

    return embeddings, pages


def compute_hash(data: bytes) -> str:
    """Compute the hash of the data."""
    return hashlib.sha256(data).hexdigest()


def upsert_data_to_pinecone(embeddings_list, pages_list, pdf_files, index):
    """Upsert data to Pinecone index."""
    upsert_data = []

    for file_embedding, page, pdf_file in zip(embeddings_list, pages_list, pdf_files):
        file_hash = compute_hash(pdf_file.getvalue()) 

        # Query Pinecone to check for duplicate hash
        query_result = index.query(
            vector=[0.0]*1536,
            top_k=1,
            filter={"file_hash": {"$eq": file_hash}}
        )

        # If the hash exists, skip upserting
        if query_result['matches']:
            # st.warning(f"Duplicate PDF detected: {pdf_file.name}")
            continue

        # If the hash is not found, proceed to upsert
        ids = [str(uuid.uuid4()) for _ in file_embedding]

        for i, (embedding, page_text) in enumerate(zip(file_embedding, page)):
            record = {
                "id": ids[i],
                "values": embedding,
                "metadata": {
                    "text": page_text,
                    "filename": pdf_file.name,
                    "file_hash": file_hash
                }
            }
            upsert_data.append(record)

    return upsert_data


def perform_similarity_search(question, openai_api_key, index, top_k=3):
    """Embeds user query and performs similarity search on the Pinecone index."""
    embedding = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
    embedded_question = embedding.embed_query(question)
    return index.query(vector=embedded_question, top_k=top_k, include_metadata=True)

