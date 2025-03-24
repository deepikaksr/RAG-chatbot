import streamlit as st  # for UI
import sqlite3  # for DB
import pandas as pd  # for data handling
import numpy as np  # for numerical ops
import json  # for JSON
import re  # for regex
import os  # for file ops
import PyPDF2  # for PDF
from io import StringIO  # for text reading
from docx import Document  # for DOCX
import google.generativeai as genai  # for Gemini API
from sklearn.feature_extraction.text import TfidfVectorizer  # for TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # for similarity
from PIL import Image  # for images
import pytesseract  # for OCR
from dotenv import load_dotenv  # for loading .env
import cv2  # for video frame sampling
import tempfile  # for creating ephemeral files
import subprocess  # run shell commands
import requests  # make HTTP requests

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Configure Gemini API if available
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Create in-memory sqlite db allowing multi-thread
if "sqlite_connection" not in st.session_state:
    st.session_state.sqlite_connection = sqlite3.connect(":memory:", check_same_thread=False)

# Store CSV info
if "csv_tables" not in st.session_state:
    st.session_state.csv_tables = {}

# Store all text chunks (including from images, docs, video)
if "all_text_chunks" not in st.session_state:
    st.session_state.all_text_chunks = []

# TF-IDF vector DB
if "tfidf_database" not in st.session_state:
    st.session_state.tfidf_database = {}

# Conversation messages
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

def chunk_text(full_text, chunk_size=100, overlap=20):
    words = full_text.split()
    chunks = []
    index = 0
    while index < len(words):
        chunk = " ".join(words[index:index+chunk_size])
        chunks.append(chunk)
        index += (chunk_size - overlap)
    return chunks

def extract_text_from_txt(uploaded_file):
    uploaded_file.seek(0)
    try:
        string_data = StringIO(uploaded_file.read().decode("utf-8"))
        text = string_data.read()
    except:
        text = ""
    return text.split("\n")

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except:
        pass
    return text.split("\n")

def extract_text_from_docx(uploaded_file):
    uploaded_file.seek(0)
    try:
        docx_document = Document(uploaded_file)
        text = "\n".join([para.text for para in docx_document.paragraphs])
    except:
        text = ""
    return text.split("\n")

def extract_text_from_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        csv_dataframe = pd.read_csv(uploaded_file)
        csv_dataframe = csv_dataframe.dropna(axis=1, how='all')
        csv_strings = csv_dataframe.astype(str)
        rows = csv_strings.apply(lambda row: " | ".join(row.values), axis=1).tolist()
        return rows, csv_dataframe
    except:
        return [], None

def extract_text_from_image(uploaded_file):
    uploaded_file.seek(0)
    try:
        opened_image = Image.open(uploaded_file)
        ocr_text = pytesseract.image_to_string(opened_image)
    except:
        ocr_text = ""
    return ocr_text.split("\n")

def transcribe_audio_with_groq(audio_path, api_key):
    if not api_key:
        return "No GROQ_API_KEY provided."
    try:
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {"file": ("audio.wav", audio_data, "audio/wav")}
        data = {"model": "whisper-large-v3"} 
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            return f"Groq API error: {response.text}"
    except:
        return "Error calling Groq Whisper API."


def extract_text_from_video(uploaded_file):
    file_data = uploaded_file.read()
    text_segments = []
    with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
        temp_video.write(file_data)
        temp_video.flush()
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
            ffmpeg_command = [
                "ffmpeg", "-y", "-i", temp_video.name,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                temp_audio.name
            ]
            try:
                subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                transcription = transcribe_audio_with_groq(temp_audio.name, groq_api_key)
                if transcription.strip():
                    text_segments.append(transcription)
            except:
                pass
        video_capture = cv2.VideoCapture(temp_video.name)
        if not video_capture.isOpened():
            return text_segments
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            video_capture.release()
            return text_segments
        duration = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        sample_rate_seconds = 5
        while True:
            time_in_seconds = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if time_in_seconds > duration:
                break
            success, frame = video_capture.read()
            if not success:
                break
            if int(time_in_seconds) % sample_rate_seconds == 0:
                frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_text = pytesseract.image_to_string(frame_image)
                if frame_text.strip():
                    text_segments.append(frame_text)
        video_capture.release()
    joined_text = "\n".join([seg for seg in text_segments if seg.strip()])
    return joined_text.split("\n")

def process_file(uploaded_file):
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()
    csv_dataframe = None
    if extension == ".csv":
        lines, csv_dataframe = extract_text_from_csv(uploaded_file)
        return lines, csv_dataframe
    elif extension == ".pdf":
        lines = extract_text_from_pdf(uploaded_file)
        merged_text = " ".join([line.strip() for line in lines if line.strip()])
        return chunk_text(merged_text), None
    elif extension == ".txt":
        lines = extract_text_from_txt(uploaded_file)
        merged_text = " ".join([line.strip() for line in lines if line.strip()])
        return chunk_text(merged_text), None
    elif extension in [".doc", ".docx"]:
        lines = extract_text_from_docx(uploaded_file)
        merged_text = " ".join([line.strip() for line in lines if line.strip()])
        return chunk_text(merged_text), None
    elif extension in [".png", ".jpg", ".jpeg"]:
        lines = extract_text_from_image(uploaded_file)
        merged_text = " ".join([line.strip() for line in lines if line.strip()])
        return chunk_text(merged_text), None
    elif extension in [".mp4", ".mov", ".avi", ".mkv"]:
        lines = extract_text_from_video(uploaded_file)
        merged_text = " ".join([line.strip() for line in lines if line.strip()])
        return chunk_text(merged_text), None
    else:
        st.warning(f"Unsupported: {extension}")
        return [], None

def build_vector_db(text_chunks):
    tfidf_vectorizer = TfidfVectorizer()
    embeddings = tfidf_vectorizer.fit_transform(text_chunks)
    st.session_state.tfidf_database = {
        "chunks": text_chunks,
        "embeddings": embeddings,
        "vectorizer": tfidf_vectorizer
    }

def retrieve_knowledge(query, top_n=5):
    if not st.session_state.tfidf_database:
        return []
    tfidf_vectorizer = st.session_state.tfidf_database["vectorizer"]
    embeddings = st.session_state.tfidf_database["embeddings"]
    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, embeddings)[0]
    indices = np.argsort(similarities)[::-1][:top_n]
    results = [(st.session_state.tfidf_database["chunks"][i], similarities[i]) for i in indices]
    return results

def generate_sql_query(natural_language_query, csv_dataframe, table_name):
    prompt_text = f"""
You are an AI that converts natural language to SQL for a SQLite database.
Table Name: {table_name}
Columns: {', '.join(csv_dataframe.columns)}
Please respond in this exact JSON format:
{{"sql_query": string, "confidence": float, "explanation": string}}
Natural language question:
"{natural_language_query}"
"""
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(prompt_text)
        raw_output = response.text.strip()
        cleaned_output = re.sub(r'```json\n(.*?)\n```', r'\1', raw_output, flags=re.DOTALL)
        parsed_json = json.loads(cleaned_output)
        return parsed_json.get("sql_query", "")
    except:
        return ""

def generate_csv_answer(natural_language_query, sql_result):
    prompt_text = f"""
You are an intelligent assistant. The following is the result of a database query:
{sql_result}
Based on this structured data, answer the user's question in natural language:
"{natural_language_query}"
"""
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(prompt_text)
        return response.text.strip()
    except:
        return "Error generating CSV answer."

def generate_unstructured_answer(natural_language_query, sql_result, retrieved_chunks):
    knowledge_string = "\n".join([f"- {chunk}" for chunk, _ in retrieved_chunks])
    prompt_text = f"""
You are an intelligent assistant answering questions based on available data.
Structured data (SQL query result): {sql_result}
Unstructured data from files:
{knowledge_string}
DO NOT make up any information. If the data is insufficient, ask for clarification.
User question: {natural_language_query}
"""
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(prompt_text)
        return response.text.strip()
    except:
        return "Error generating final answer."

# Streamlit UI
st.set_page_config(layout="wide") 

with st.sidebar:
    st.title("File Uploader")
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["txt","pdf","csv","docx","doc","png","jpg","jpeg","mp4","mov","avi","mkv"],
        accept_multiple_files=False
    )
    if uploaded_file:
        extracted_chunks, csv_dataframe = process_file(uploaded_file)  # process file
        st.session_state.all_text_chunks.extend(extracted_chunks)  # store all text chunks
        if csv_dataframe is not None: 
            try:
                table_name = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")
                csv_dataframe.columns = [col.strip() for col in csv_dataframe.columns]
                csv_dataframe.to_sql(table_name, st.session_state.sqlite_connection, index=False, if_exists="replace")
                st.session_state.csv_tables[table_name] = csv_dataframe
                st.success(f"Loaded CSV: {table_name}")
            except:
                st.error("Error loading CSV.")
        if st.session_state.all_text_chunks:  # build vector DB
            build_vector_db(st.session_state.all_text_chunks)

st.title("RAG-Based Chatbot")
st.write("This chatbot uses the RAG model to answer questions based on uploaded files.")

user_input = st.chat_input("Ask a question...") 
if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    if not st.session_state.all_text_chunks and not st.session_state.csv_tables:
        final_answer = "No file uploaded."
    else:
        if st.session_state.csv_tables: # check if CSV tables are available
            first_table = list(st.session_state.csv_tables.keys())[0]
            dataframe_for_sql = st.session_state.csv_tables[first_table]
            sql_query = generate_sql_query(user_input, dataframe_for_sql, first_table)
            if sql_query:  # if SQL query generated
                try:
                    cursor = st.session_state.sqlite_connection.cursor()
                    cursor.execute(sql_query)
                    rows = cursor.fetchall()
                    if "count(" in sql_query.lower() and rows and len(rows) == 1 and len(rows[0]) == 1:
                        sql_result = f"There are {rows[0][0]} records."
                    else:
                        sql_result = "\n".join([str(row) for row in rows]) if rows else "No matches."
                    final_answer = generate_csv_answer(user_input, sql_result)
                except Exception as e:
                    final_answer = f"Error executing SQL: {e}"
            else:
                retrieved_chunks = retrieve_knowledge(user_input)  # retrieve knowledge
                final_answer = generate_unstructured_answer(user_input, "", retrieved_chunks)
        else:
            retrieved_chunks = retrieve_knowledge(user_input)  # retrieve knowledge
            final_answer = generate_unstructured_answer(user_input, "", retrieved_chunks)
    st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})

for message in st.session_state.chat_messages:  # display chat messages
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])
