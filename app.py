import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from brain import get_embeddings, perform_similarity_search, upsert_data_to_pinecone


st.title('EZStudy')

# Load environment variables
load_dotenv(override=True)

# File uploader for PDFs
pdf_files = st.file_uploader('Upload your PDF file', type='pdf', accept_multiple_files=True)

#Load pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "ezstudy"

if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, metric="cosine", dimension=1536, spec = ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(index_name)

if pdf_files:
    pdf_file_names = [pdf_file.name for pdf_file in pdf_files]
    embeddings_list, pages_list = get_embeddings(
        [file.getvalue() for file in pdf_files], pdf_file_names, os.getenv("OPENAI_API_KEY")
    )

    data_upsert = upsert_data_to_pinecone(embeddings_list, pages_list, pdf_files, index)

    if data_upsert:
        index.upsert(vectors=data_upsert)
        st.toast("PDFs uploaded successfully!")

# Initialize the prompt in session state if it doesn't exist
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = [{"role": "system", "content": "none"}]

prompt = st.session_state['prompt']

# Display chat messages from the prompt
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question
question = st.chat_input("Ask Anything")


if question:

    similarity_search = perform_similarity_search(question, os.getenv("OPENAI_API_KEY"), index, top_k=3)

    contexts_extracted = "\n ".join([result["metadata"]["text"] for result in similarity_search["matches"]])
    file_names = ", ".join(set([result["metadata"]["filename"] for result in similarity_search["matches"]]))

    llm_prompt = """You are a helpful Assistant who answers to users questions based on multiple contexts given to you.
    
    The evidence are the context of the pdf extract with metadata. 
     
    The PDF content is:
    {contexts_extracted}

    The file names is/are: {file_names}
    """

    # Append user's question to the prompt
    prompt.append({"role": "user", "content": question})

    prompt.append({"role": "system", "content": llm_prompt.format(contexts_extracted=contexts_extracted, file_names=file_names)})

    # Display the user's question immediately
    with st.chat_message("user"):
        st.write(question)

    # Initialize the assistant's response
    with st.chat_message("assistant"):
        botmsg = st.empty()

        llm = ChatOpenAI(
            model="gpt-4o-2024-08-06",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Get the assistant's response
        response = llm.invoke(prompt)
        result = response.content.strip()

        botmsg.write(result)

        prompt.append({"role": "assistant", "content": result})





