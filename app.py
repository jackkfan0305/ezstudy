import streamlit as st
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from helper import get_embeddings, perform_similarity_search, upsert_data_to_pinecone

st.title('EZStudy')

# Load environment variables
load_dotenv(override=True)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "ezstudy"

#Load Pinecone
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, metric="cosine", dimension=1536, spec = ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(index_name)


with st.sidebar:
    menu = option_menu(
        menu_title="Navigation",
        options=["Chat", "Quiz"],
        default_index=0
    )

if 'pdf_content' not in st.session_state:
    st.session_state['pdf_content'] = ""

if 'generated_quiz' not in st.session_state:
    st.session_state['generated_quiz'] = False

if 'quiz_questions' not in st.session_state:
    st.session_state['quiz_questions'] = []

if 'current_question' not in st.session_state:
    st.session_state['current_question'] = 0

if 'score' not in st.session_state:
    st.session_state['score'] = 0

if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None

def generate_quiz():
    st.session_state['generated_quiz'] = True
    st.session_state['quiz_questions'] = []
    

def submit_answer():
    question_data = st.session_state['quiz_questions'][st.session_state['current_question']]
    
    # Ensure an option is selected
    if st.session_state['selected_option']:
        # Check if the answer is correct
        if st.session_state['selected_option'] == question_data['correct_answer']:
            st.session_state['score'] += 1
            st.success("Correct! 🎉")
        else:
            st.error(f"Incorrect. The correct answer was **{question_data['correct_answer']}**.")
        
        # Move to the next question
        st.session_state['current_question'] += 1
    else:
        st.warning("Please select an option before submitting.")


if menu == "Chat": 
    st.session_state['quiz_questions'] = []
    st.subheader("Upload files and start chatting!")
    
    pdf_files = st.file_uploader('Upload your PDF file', type='pdf', accept_multiple_files=True)

    if pdf_files:
        pdf_file_names = [pdf_file.name for pdf_file in pdf_files]
        embeddings_list, pages_list = get_embeddings(
            [file.getvalue() for file in pdf_files], pdf_file_names, os.getenv("OPENAI_API_KEY")
        )

        st.session_state['pdf_content'] = pages_list

        data_upsert = upsert_data_to_pinecone(embeddings_list, pages_list, pdf_files, index)

        if data_upsert:
            index.upsert(vectors=data_upsert)
            st.toast("PDFs uploaded successfully!")

    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = [{"role": "system", "content": "none"}]

    prompt = st.session_state['prompt']

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

        llm_prompt = """You are an intelligent and knowledgeable assistant designed to help users understand and analyze the contents of PDF documents they upload. Your role is to provide clear, concise, and informative answers to their questions by referencing the most relevant sections of the uploaded PDFs.
        Use the provided PDF extracts as evidence to craft accurate responses. If necessary, synthesize information from multiple parts of the document to ensure comprehensive answers. Where applicable, explain complex concepts in simple terms and provide additional context to enhance user understanding.
        ### PDF Context and Metadata:
        {contexts_extracted}

        ### Source File(s):
        {file_names}

        If the information is not available within the provided context, let the user know that the answer cannot be determined based on the current documents. Encourage users to upload additional relevant files if necessary.

        Always be thorough and structured in your response, breaking down key points step by step if needed. Summarize long explanations at the end for clarity.
        """

        # Append user's question to the prompt
        prompt.append({"role": "user", "content": question})

        prompt.append({"role": "system", "content": llm_prompt.format(contexts_extracted=contexts_extracted, file_names=file_names)})

        # Display the user's question immediately
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            botmsg = st.empty()

            llm = ChatOpenAI(
                model="gpt-4o-2024-08-06",
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            response = llm.invoke(prompt)
            result = response.content.strip()

            botmsg.write(result)

            prompt.append({"role": "assistant", "content": result})

elif menu == "Quiz":

    st.sidebar.title("🧩 Generate Multiple Choice Quiz")

    st.sidebar.subheader("Select how many questions you want to generate:")
    num_questions = st.sidebar.select_slider("Number of questions", options=[5, 10, 15, 20], value=5)
    
    st.sidebar.subheader("Select the difficulty level:")
    difficulty = st.sidebar.selectbox("Difficulty", ["Easy", "Medium", "Hard"])     

    st.sidebar.button("Generate Quiz", on_click=generate_quiz)


    if st.session_state['pdf_content']:

        if st.session_state['generated_quiz']:
            st.toast("📚 Generating quiz questions from the uploaded PDFs...")

            llm = ChatOpenAI(
                model="gpt-4o-2024-08-06",
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            quiz_prompt = f"""
            Generate {num_questions} multiple-choice quiz question based on the following PDF content:
            
            {st.session_state['pdf_content']}

            The difficulty of the quiz should be {difficulty}.
            
            In the generated quiz question, format the question as follows:
            Q: [Question text]
            A. [Option 1]
            B. [Option 2]
            C. [Option 3]
            D. [Option 4]
            Correct Answer: [Correct Option]\n
            """

            response = llm.invoke([{"role": "system", "content": quiz_prompt}])
            question_split = response.content.strip().split("\n\n")

            for question in question_split:
                full_q = question.split("\n")
                q_asked = full_q[0].replace("Q: ", "").strip()
                options = full_q[1:5]
                correct_answer = full_q[-1].replace("Correct Answer: ", "").strip()
                st.session_state['quiz_questions'].append({
                    "question": q_asked,
                    "options": options,
                    "correct_answer": correct_answer
                })
        elif not st.session_state['quiz_questions']:
            st.warning("Please generate quiz questions first.")
        
        st.session_state['generated_quiz'] = False

        if st.session_state['current_question'] < len(st.session_state['quiz_questions']):
            question_data = st.session_state['quiz_questions'][st.session_state['current_question']]
            
            st.write(f"**Question {st.session_state['current_question'] + 1}:** {question_data['question'][3:]}")

            st.session_state['selected_option'] = st.radio(
                "Choose your answer:",
                options=question_data['options'],
                key=f"answer_{st.session_state['current_question']}", 
                index=None
            )
            
            st.button("Submit", on_click=submit_answer)
            print(st.session_state['current_question'])
            print(st.session_state['selected_option'])
        elif st.session_state['quiz_questions']:
            st.write(f"🏁 Quiz complete! Your final score: **{st.session_state['score']}/{len(st.session_state['quiz_questions'])}**")
            st.session_state['current_question'] = 0
            st.session_state['score'] = 0
    else:
        st.warning("Please upload PDF files in the 'Chat' section first.")





