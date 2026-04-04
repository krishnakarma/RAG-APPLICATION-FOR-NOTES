import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ------------------ CONFIG ------------------
load_dotenv()
PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="RAG Book Assistant")
st.title("📚 RAG Book Assistant")
st.write("Upload a text-based PDF and ask questions")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


# ------------------ CREATE VECTOR DB ------------------

if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        if os.path.exists(PERSIST_DIR):
            st.warning("⚠️ Database already exists. Delete it to recreate.")
        else:
            with st.spinner("Processing document..."):

                loader = PyPDFLoader(file_path)
                docs = loader.load()

                st.write(f"📄 Pages loaded: {len(docs)}")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(docs)

                st.write(f"✂️ Chunks created: {len(chunks)}")

                if not chunks:
                    st.error("❌ No text found in PDF. Use a text-based PDF.")
                    st.stop()

                embeddings = MistralAIEmbeddings(model="mistral-embed")

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIR
                )

                vectorstore.persist()

            st.success("✅ Vector database created!")

    # Clean temp file
    os.remove(file_path)


# ------------------ LOAD & QUERY ------------------

if os.path.exists(PERSIST_DIR):

    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(model="mistral-small-latest")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the PDF")

    query = st.text_input("Enter your question")

    if query:

        docs = retriever.invoke(query)

        if not docs:
            st.error("❌ No relevant content found.")
            st.stop()

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        st.write("### 🤖 AI Answer")
        st.write(response.content)