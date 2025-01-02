
from io import BytesIO
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

# def text_to_docs(text: list[str], filename: str) -> list[Document]:
#     if isinstance(text, str):
#         text = [text]
#     page_docs = [Document(page_content=page) for page in text]
#     for i, doc in enumerate(page_docs):
#         doc.metadata["page"] = i + 1

#     doc_chunks = []
#     for doc in page_docs:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=4000,
#             separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
#             chunk_overlap=0,
#         )
#         chunks = text_splitter.split_text(doc.page_content)
#         for i, chunk in enumerate(chunks):
#             doc = Document(
#                 page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
#             )
#             doc.metadata["filename"] = filename  # Add filename to metadata
            
#             doc_chunks.append(doc)
#     return doc_chunks

# def docs_to_embed(docs: list[Document], openai_api_key):
#     embedding = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
    
#     # Prepare documents for embedding (combine content + metadata)
#     combined_texts = [
#         f"Filename: {doc.metadata.get('filename', 'Unknown')}\n"
#         f"Page: {doc.metadata.get('page', 'N/A')}\n"
#         f"Content: {doc.page_content}"
#         for doc in docs
#     ]
    
#     # Generate embeddings for combined text
#     vector_embeddings = embedding.embed_documents(combined_texts)
    
#     # Attach embeddings to documents' metadata
#     for i, doc in enumerate(docs):
#         doc.metadata['embedding'] = vector_embeddings[i]
    
#     return docs






#test the parse pdf function
# file = open("MAT337.pdf", "rb")
# pages = parse_pdf(BytesIO(file.read()), "MAT337.pdf")

# # print(pages)

# print(embeddings)
# print(len(embeddings))


# docs = text_to_docs(['hello world hi hi bye'], "test.pdf")
# print(docs)


# for doc in embedded_docs:
#     print(doc.metadata['embedding'])
