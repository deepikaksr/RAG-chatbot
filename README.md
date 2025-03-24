# RAG-Based Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit. It accepts multiple file types—text documents, CSV, PDF, DOCX, images, and videos—and uses TF-IDF vectorization for unstructured retrieval, plus a Gemini API call for structured CSV queries (SQL) and final answer generation. Audio from videos is transcribed using a Whisper model via a Groq API call (placeholder code can be replaced with a real endpoint).

## Features

### File Ingestion
- Upload TXT, PDF, CSV, DOCX, images (PNG, JPG), or videos (MP4, MOV, AVI, MKV).
- CSV rows are stored in an in-memory SQLite database for SQL-based queries.
- Images undergo OCR with pytesseract.
- Videos are sampled for frames (OCR) and audio is transcribed (Whisper via Groq).

### TF-IDF Vector Database
- All extracted text chunks are embedded with TF-IDF.
- Queries are also embedded, and the most relevant chunks are retrieved by cosine similarity.

### CSV → SQL
- If CSV data is present, natural language queries are converted into SQL by Gemini, run in SQLite, then summarized back into text by Gemini.

### Chat UI
- Uses Streamlit’s chat elements to display a user/assistant conversation.
- Each user query and chatbot response is appended to the conversation history.

## Installation & Setup

### Clone or Download this repository.

### Install Requirements:
```bash
pip install -r requirements.txt
```
Includes streamlit, pytesseract, opencv-python, PyPDF2, docx, google-generativeai, requests, etc.

### Install Dependencies:
- **Tesseract for OCR** (check your OS package manager or Tesseract docs).
- **ffmpeg for audio extraction from videos** (install via your OS package manager).

### Set Up .env:
Create a `.env` file in the project root with:
```ini
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```
If you have a real Groq endpoint, replace the placeholder URL with the actual one.

## Running the App

### Start Streamlit:
```bash
streamlit run app.py
```
(Replace `app.py` with the name of your Python file containing the code.)

### Open Your Browser:
Navigate to the local URL shown in the terminal (usually `http://localhost:8501`).

### Upload Files:
Use the sidebar to upload TXT, PDF, CSV, DOCX, images, or videos.

### Ask Questions:
Type your question in the chat input. The system will either:
- Convert it to SQL if CSV data is present, then retrieve and summarize results.
- Or retrieve relevant chunks via TF-IDF and generate a final answer with Gemini.

## Notes

### OCR:
- The code calls `pytesseract.image_to_string` for images and video frames.
- Ensure Tesseract is installed and on your PATH.

### Video Processing:
- Uses OpenCV for frame sampling and ffmpeg for audio extraction.
- Adjust the sampling rate (default is 5 or 15 seconds) in `extract_text_from_video`.

### Whisper Transcription:
- The code includes a placeholder function `transcribe_audio_with_groq` for Groq’s Whisper API.
- Update the URL, headers, and parameters for your actual endpoint.
