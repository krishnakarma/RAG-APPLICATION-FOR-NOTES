import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import time
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ------------------ CONFIG ------------------
load_dotenv()

# Model names live here so they are easy to change if Google/Mistral retire one
GEMINI_CHAT_MODEL = "gemini-flash-latest"
GEMINI_EMBED_MODEL = "models/gemini-embedding-2"
MISTRAL_CHAT_MODEL = "mistral-small-latest"
MISTRAL_EMBED_MODEL = "mistral-embed"

BATCH_SIZE = 20


def get_secret(name):
    """Streamlit Cloud: Secrets. Local: .env or environment variable."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


google_key = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
mistral_key = get_secret("MISTRAL_API_KEY")

# Gemini is used if its key exists, otherwise Mistral
if google_key:
    PROVIDER = "Gemini"
elif mistral_key:
    PROVIDER = "Mistral"
else:
    st.error("No API key found. Add GOOGLE_API_KEY (or MISTRAL_API_KEY) to Streamlit Secrets or .env")
    st.stop()


def build_models():
    if PROVIDER == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL, google_api_key=google_key)
        llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, google_api_key=google_key)
    else:
        os.environ["MISTRAL_API_KEY"] = mistral_key
        from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        embeddings = MistralAIEmbeddings(model=MISTRAL_EMBED_MODEL)
        llm = ChatMistralAI(model=MISTRAL_CHAT_MODEL)
    return embeddings, llm


st.set_page_config(page_title="RAG Book Assistant")
st.title("📚 RAG Book Assistant")
st.write("Upload a text-based PDF and ask questions")
st.caption(f"Powered by {PROVIDER}")


# ------------------ HELPERS ------------------
def call_with_retry(fn, *args, retries=5, **kwargs):
    """Retry on rate limits (429 / RESOURCE_EXHAUSTED); show the real error otherwise."""
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            rate_limited = (
                "429" in msg
                or "RESOURCE_EXHAUSTED" in msg
                or "rate limit" in msg.lower()
            )
            if rate_limited and i < retries - 1:
                time.sleep(5 * 2 ** i)  # 5s, 10s, 20s, 40s
                continue
            st.error(f"{PROVIDER} API error: {msg[:500]}")
            st.stop()


def get_text(response):
    """Gemini can return content as a list of blocks; Mistral returns a string."""
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


embeddings, llm = build_models()

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
                time.sleep(1.5)  # stay under the rate limit

            st.session_state.vectorstore = vs
            st.session_state.pop("last_query", None)

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

        # Streamlit reruns the script on every interaction, so cache by question
        if st.session_state.get("last_query") != query:

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

            st.session_state.last_query = query
            st.session_state.last_result = (get_text(response), docs)

        answer, docs = st.session_state.last_result

        st.write("### 🤖 AI Answer")
        st.write(answer)

        st.write("### 📎 Sources")
        for d in docs:
            page = d.metadata.get("page", 0) + 1
            st.caption(f"Page {page}: {d.page_content[:150]}...")

elif uploaded_file:
    st.info("Click 'Create Vector Database' to start asking questions.")