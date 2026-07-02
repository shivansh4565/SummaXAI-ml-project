from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

import tempfile
import os

load_dotenv()

# ----------------------- CONFIG -----------------------

st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- CSS -----------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:700;
    color:#4F46E5;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:35px;
}

.metric-card{
    background:#fafafa;
    padding:18px;
    border-radius:12px;
    border:1px solid #ececec;
    text-align:center;
}

.summary-box{
    background:#f8f9fa;
    padding:25px;
    border-radius:15px;
    border-left:6px solid #4F46E5;
    font-size:17px;
    color:#000 !important;
    line-height:1.8;
}

.summary-box *{
    color:#000 !important;
}

.preview-box{
    background:#fcfcfc;
    padding:18px;
    border-radius:10px;
    border:1px solid #ddd;
    max-height:300px;
    overflow-y:auto;
    color:#000 !important;
    line-height:1.7;
}

div[data-testid="stButton"]>button{
    width:100%;
    height:55px;
    font-size:18px;
    border-radius:10px;
    background:#4F46E5;
    color:white;
}

div[data-testid="stButton"]>button:hover{
    background:#3730A3;
}

</style>
""", unsafe_allow_html=True)

# ----------------------- TITLE -----------------------

st.markdown(
    '<div class="main-title">📄 AI PDF Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload any PDF • Select a page • Generate an AI Summary</div>',
    unsafe_allow_html=True
)

# ----------------------- MODEL -----------------------

@st.cache_resource
def load_model():
    # Try Streamlit Secrets first (for deployment)
    api_key = st.secrets.get("MISTRAL_API_KEY")

    # Fallback to .env (for local development)
    if api_key is None:
        api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        st.error(
            "❌ MISTRAL_API_KEY not found.\n\n"
            "Add it to your .env file for local development "
            "or to Streamlit Secrets when deployed."
        )
        st.stop()

    return ChatMistralAI(
        model="mistral-small-2506",
        api_key=api_key,
        temperature=0.3
    )

model = load_model()

# ----------------------- SIDEBAR -----------------------

with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    st.divider()

    st.info(
        """
        **Features**

        ✅ Upload PDF

        ✅ Select any page

        ✅ AI Summary

        ✅ Original Page Preview
        """
    )

# ----------------------- MAIN -----------------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    total_pages = len(docs)

    filesize = uploaded_file.size / (1024 * 1024)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Pages", total_pages)

    with col2:
        st.metric("💾 Size", f"{filesize:.2f} MB")

    with col3:
        st.metric("🤖 Model", "Mistral")

    st.divider()

    page = st.slider(
        "Select Page",
        1,
        total_pages,
        1
    )

    left, right = st.columns([1.2,1])

    with left:

        st.subheader("📖 Page Preview")

        st.markdown(
            f"""
<div class="preview-box">

{docs[page-1].page_content}

</div>
""",
            unsafe_allow_html=True
        )

    with right:

        st.subheader("⚙️ Controls")

        summarize = st.button("✨ Generate Summary")

    if summarize:

        with st.spinner("Generating AI Summary..."):

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are an expert AI assistant.

Summarize this PDF page professionally.

Return:

# Main Idea

# Key Concepts

# Important Points

# Conclusion

Use bullet points wherever possible.
"""
                    ),
                    ("human", "{text}")
                ]
            )

            chain = prompt | model

            response = chain.invoke(
                {"text": docs[page-1].page_content}
            )

        st.divider()

        st.subheader("📝 AI Summary")

        st.markdown(
            f"""
<div class="summary-box">

{response.content}

</div>
""",
            unsafe_allow_html=True
        )

        with st.expander("📄 View Original Text"):
            st.write(docs[page-1].page_content)

    os.remove(temp_path)

else:

    st.markdown(
        """
        <br><br>

        ### 👈 Upload a PDF from the sidebar to get started.

        Once uploaded, you can:

        - 📄 Browse pages
        - 🤖 Generate AI summaries
        - 📖 Read original content
        """,
        unsafe_allow_html=True
    )
