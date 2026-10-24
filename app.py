import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import time
import uuid
import httpx

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ------------------ CONFIG ------------------
load_dotenv()

# Streamlit Cloud: Secrets. Local: .env or .streamlit/secrets.toml
try:
    api_key = st.secrets["MISTRAL_API_KEY"]
except Exception:
    api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("MISTRAL_API_KEY not found. Add it to Streamlit Secrets or .env")
    st.stop()

os.environ["MISTRAL_API_KEY"] = api_key

BATCH_SIZE = 20

st.set_page_config(page_title="RAG Book Assistant")
st.title("📚 RAG Book Assistant")
st.write("Upload a text-based PDF and ask questions")


# ------------------ HELPERS ------------------
def call_with_retry(fn, *args, retries=4, **kwargs):
    """Retry on 429 (rate limit); show the real status code on any other error."""
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 429 and i < retries - 1:
                time.sleep(2 ** i)  # wait 1s, 2s, 4s
                continue
            st.error(f"Mistral API error {code}: {e.response.text[:300]}")
            st.stop()


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


# ------------------ CREATE VECTOR DB (per user session) ------------------
if uploaded_file:

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        with st.spinner("Processing document..."):

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            os.remove(file_path)

            st.write(f"📄 Pages loaded: {len(docs)}")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)

            if not chunks:
                st.error("❌ No text found in PDF. Use a text-based PDF.")
                st.stop()

            st.write(f"✂️ Chunks created: {len(chunks)}")

            embeddings = MistralAIEmbeddings(model="mistral-embed")

            # In-memory store, unique per upload, so users do not clash
            vs = Chroma(
                collection_name=f"c_{uuid.uuid4().hex}",
                embedding_function=embeddings
            )

            progress = st.progress(0.0)
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                call_with_retry(vs.add_documents, batch)
                progress.progress(min((i + BATCH_SIZE) / len(chunks), 1.0))
                time.sleep(1)  # stay under the rate limit

            st.session_state.vectorstore = vs

        st.success("✅ Vector database created!")


# ------------------ LOAD & QUERY ------------------
if st.session_state.get("vectorstore"):

    retriever = st.session_state.vectorstore.as_retriever(
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

        docs = call_with_retry(retriever.invoke, query)

        if not docs:
            st.error("❌ No relevant content found.")
            st.stop()

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = call_with_retry(llm.invoke, final_prompt)

        st.write("### 🤖 AI Answer")
        st.write(response.content)

        st.write("### 📎 Sources")
        for d in docs:
            page = d.metadata.get("page", 0) + 1
            st.caption(f"Page {page}: {d.page_content[:150]}...")

elif uploaded_file:
    st.info("Click 'Create Vector Database' to start asking questions.")