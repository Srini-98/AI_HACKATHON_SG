import streamlit as st
import PyPDF2
from splitResume import get_response

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    pdfReader = PyPDF2.PdfReader(uploaded_file)
    pageObj = pdfReader.pages[0]
    file_name = uploaded_file.name
    response = get_response(pageObj.extract_text() , file_name)
    st.write(response)
    
#command to launch streamlit