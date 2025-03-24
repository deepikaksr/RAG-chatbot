# RAG-Based Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit. It accepts multiple file types—text documents, CSV, PDF, DOCX, images, and videos—and uses TF-IDF vectorization for unstructured retrieval, plus a Gemini API call for structured CSV queries (SQL) and final answer generation. Audio from videos is transcribed using a Whisper model via a Groq API call.

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


## Project Structure

- **app.py** - Main backend script for the Streamlit chatbot
- **.env** - Environment variables for API keys (GEMINI_API_KEY, GROQ_API_KEY).
- **data/** - Folder for storing uploaded files (TXT, PDF, CSV, DOCX, images, videos).
- **README.md** - Project documentation


## Installation & Setup

### Clone this repository.
git clone https://github.com/deepikaksr/faq-chatbot.git cd faq-chatbot

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

## Running the App

### Start Streamlit:
```bash
streamlit run app.py
```

### Open Your Browser:
Navigate to the local URL shown in the terminal.

### Upload Files:
Use the sidebar to upload TXT, PDF, CSV, DOCX, images, or videos.

### Ask Questions:
Type your question in the chat input. The system will either:
- Convert it to SQL if CSV data is present, then retrieve and summarize results.
- Or retrieve relevant chunks via TF-IDF and generate a final answer with Gemini.

