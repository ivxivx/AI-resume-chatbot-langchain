import os
import tempfile

from streamlit_datalist import stDatalist
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from htmlTemplates import css, bot_template, user_template

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama

st.set_page_config(page_title="Chat",
                       page_icon=":page_facing_up:")
st.write(css, unsafe_allow_html=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
def handle_userInput(user_question):
    rag_chain = st.session_state.conversation
    if rag_chain == None: 
        return
    
    response = rag_chain.invoke({
        'input': user_question, 
        'chat_history': st.session_state.chat_history,
    })
    print("response", response)

    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(response['answer'])

    # st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)

def load_documents(file: UploadedFile) -> list[Document]:
    raw_documents = []

    temp_dir = tempfile.mkdtemp()
    temp_full_path = os.path.join(temp_dir, file.name)

    with open(temp_full_path, "wb") as f:
        f.write(file.getvalue())
        f.close()

        # each page is loaded as a separate document
        raw_document_pages = PyPDFLoader(temp_full_path).load()
        print(f"Loaded document {file.name}")

        raw_documents.extend(raw_document_pages)

    return raw_documents

def split_documents(raw_documents: list[Document]) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    print("Split document")
    return documents            

def create_vector_store(documents: list[Document]) -> VectorStore:
    db = FAISS.from_documents(documents, embeddings)

    print("Created vector store")
    return db
    
def create_rag_chain(db: VectorStore):
    retriever = db.as_retriever()
    llm = ChatOllama(model="llama3")
    
    # https://smith.langchain.com/hub/langchain-ai/chat-langchain-rephrase
    contextualize_q_system_prompt = (
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. "
        "Do NOT try to answer the question. "
        "Chat History: {chat_history}"
        "Follow Up Input: {input}"
        "Standalone Question:"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    qa_system_prompt = (
        "You are a candidate who is tasked to answer questions based on the following pieces of retrieved context. "
        "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
        "---Context begins"
        "{context}"
        "---Context ends"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt), 
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}"),
    ])

    # create_stuff_documents_chain: this chain use a variable called "context"
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title(':orange[Chat]')
    user_question = stDatalist("Ask a question", [
            "What is your name", 
            "What are your hobbies", 
            "Do you have email address",
            "Are you now in Malaysia",
            "Have you completed bachelor degree"
        ])
    
    if user_question:
        handle_userInput(user_question)

    with st.sidebar:
        st.subheader(":orange[Your documents]")
        uploaded_files = st.file_uploader(
                ":blue[Upload file here and click on 'Process Document']. Accepts :red[PDF files only.]",
                type = "pdf",
                accept_multiple_files=False)
        
        if st.button("Process Document"):
            if uploaded_files is not None:
                with st.spinner("Processing document(s)"):
                    raw_documents = load_documents(uploaded_files)
                    documents = split_documents(raw_documents)

                    db = create_vector_store(documents)

                    st.session_state.conversation = create_rag_chain(db)

                
if __name__ == '__main__':
    main()