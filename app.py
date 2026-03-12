import streamlit as st
from rag_pipeline import (
    load_and_split_pdfs,
    create_vectorstore,
    load_vectorstore,
    create_rag_chain
)

st.title("📄 Resume RAG Chatbot")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

if st.button("Process Resumes"):

    with st.spinner("Processing resumes..."):

        docs = load_and_split_pdfs(uploaded_files)

        vectorstore = create_vectorstore(docs)

        st.success("Resumes processed successfully!")



question = st.text_input("Ask about a candidate")

if question:

    vectorstore = load_vectorstore()

    qa_chain = create_rag_chain(vectorstore)

    result = qa_chain.run(question)

    st.write("### Answer")
    st.write(result)