# AI Document Assistant

An AI-powered document question-answering application built with **Streamlit, LangChain, FAISS, and OpenRouter**.

The application allows users to upload a single document, build a semantic vector index, and ask questions grounded in the uploaded content. It can also optionally enrich answers with live web search while keeping the uploaded document as the primary source of truth.

---

## Overview

**AI Document Assistant** is designed to make information retrieval from documents faster and more interactive.

Instead of manually searching through a document, users can upload a supported file, build an index, and ask questions through a conversational interface.

The system retrieves relevant document chunks and uses an OpenRouter-powered language model to generate grounded responses.

When enabled, live web search can provide additional context beyond the uploaded document.

---

## Key Features

- 📄 Upload a single document
- 🔎 Semantic document search using FAISS
- 🤖 AI-powered document question answering
- 🧩 LangChain-based document processing and retrieval
- 🌐 Optional live web context enrichment
- 🔗 OpenRouter integration for AI models
- 📝 Support for PDF, DOCX, TXT, and Markdown files
- 🔍 Inspect document chunks used to generate an answer
- 🌐 Inspect web snippets used for additional context
- ⚡ Local document index caching for faster repeated usage
- 🛡️ Document-first answering behavior

---

## Supported File Formats

| Format | Supported |
|---|---|
| PDF | ✅ |
| DOCX | ✅ |
| TXT | ✅ |
| Markdown (`.md`) | ✅ |

---

## How It Works

The application follows a document-grounded question-answering workflow:

```text
          ┌─────────────────────┐
          │   Upload Document   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Document Parsing   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   Text Chunking     │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Embedding Generation│
          │     OpenRouter      │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   FAISS Vector      │
          │       Index         │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │     User Query      │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Relevant Chunks     │
          │     Retrieved       │
          └──────────┬──────────┘
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
       Document Context   Web Search
              │             │
              └──────┬──────┘
                     ▼
          ┌─────────────────────┐
          │   AI Answer         │
          │  Generation         │
          └─────────────────────┘
```

The uploaded document remains the primary source of information, while web search can be used when broader context is desired.

---

## Technology Stack

### Frontend

- **Streamlit** — Interactive web application interface

### AI & LLM

- **OpenRouter** — Access to chat and embedding models

### Document Processing

- **LangChain** — Document loading, chunking, retrieval, prompting, and vector-store orchestration

### Vector Search

- **FAISS** — Local semantic similarity search

### Web Search

- **DDGS** — Live web search for optional additional context

---

## Project Workflow

### 1. Upload a Document

Upload one supported document:

- PDF
- DOCX
- TXT
- Markdown

### 2. Build the Index

Click **Build index** to process the uploaded document.

The application:

1. Reads the document
2. Splits the content into chunks
3. Generates embeddings
4. Stores the embeddings in a FAISS vector index

### 3. Ask Questions

Use the chat interface to ask questions about the uploaded document.

The application retrieves relevant document chunks and uses them as context for generating the answer.

### 4. Optional Web Enrichment

The **Enrich answers with live web search** option can be enabled when broader web context is useful.

This allows the response to combine information retrieved from the uploaded document with relevant web search results.

### 5. Inspect Retrieved Context

The application allows users to inspect the document chunks and web snippets used for the latest answer.

This makes the retrieval process more transparent and easier to review.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DarshanS2004/AI-Document-Assistant.git
cd AI-Document-Assistant
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file based on the provided `.env.example` file.

Example:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

The project also supports configurable chat and embedding models.

### Recommended Models

**Chat Model**

```text
openai/gpt-4o-mini
```

**Embedding Model**

```text
openai/text-embedding-3-small
```

These models can be configured through the application's sidebar or environment configuration.

---

## Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Usage

1. Launch the Streamlit application.
2. Provide your OpenRouter API key if it is not already configured through `.env`.
3. Upload one supported document.
4. Click **Build index**.
5. Ask questions using the chat interface.
6. Enable live web enrichment when additional web context is required.
7. Review the retrieved document chunks and web snippets associated with the latest answer.

---

## Document-Grounded Responses

A major design principle of this project is keeping the uploaded document as the **primary source of truth**.

When answering questions, the system prioritizes information retrieved from the uploaded document.

If the requested information cannot be found in the uploaded document, the application indicates that instead of presenting unsupported information as if it came from the document.

This helps reduce misleading document-based responses.

---

## Local Caching

The application stores document index data in:

```text
.cache/
```

Caching allows repeated uploads of the same document to be processed faster.

---

## Project Structure

A typical project structure is:

```text
AI-Document-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
└── .cache/
```

> The exact structure may vary depending on the current project implementation.

---

## Security

### Protect Your API Key

Never commit your `.env` file or API keys to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
.cache/
```

If an API key is accidentally exposed publicly, revoke it immediately and generate a replacement.

### Recommended `.gitignore`

```gitignore
.env
.venv/
__pycache__/
*.pyc
.cache/
```

---

## Privacy Considerations

Uploaded documents are processed by the application for indexing and question answering.

Before deploying this application publicly or using sensitive documents, review how documents, API requests, cached indexes, and external web searches are handled.

For production deployments, additional security and privacy controls should be implemented.

---

## Limitations

- The application processes one document at a time.
- Answer quality depends on the quality and structure of the uploaded document.
- Web-enriched responses depend on the availability and quality of live search results.
- AI-generated responses may contain errors.
- API usage may incur costs depending on the selected OpenRouter models.
- Local vector indexing requires additional storage for cached data.

---

## Future Improvements

Potential improvements include:

- Multi-document support
- Persistent vector databases
- Conversation history
- Advanced document citation
- Improved retrieval and reranking
- Hybrid keyword + semantic search
- Additional document formats
- Authentication and user accounts
- Document management
- Advanced source verification
- Streaming AI responses
- Production-grade observability
- Fine-grained privacy controls

---

## Use Cases

The AI Document Assistant can be adapted for applications such as:

- 📚 Research document analysis
- 📑 Report question answering
- 🏢 Business document analysis
- 🎓 Academic study assistance
- 📋 Policy and guideline exploration
- 📖 Personal knowledge retrieval
- 🔍 Technical document analysis

---

## Why This Project?

This project demonstrates the practical implementation of a modern **Retrieval-Augmented Generation (RAG)** workflow.

It combines:

```text
Document Processing
        +
Semantic Embeddings
        +
Vector Search
        +
LLM Generation
        +
Optional Web Search
        =
AI Document Assistant
```

The project focuses on making AI responses more useful by grounding them in retrieved document content instead of relying only on the language model's general knowledge.

---

## Project Highlights

### AI & Machine Learning

- LLM-powered question answering
- Embedding-based semantic retrieval
- Retrieval-Augmented Generation workflow
- Document-grounded responses

### Backend / Processing

- LangChain document orchestration
- FAISS vector indexing
- Local document caching
- OpenRouter model integration

### User Experience

- Interactive Streamlit dashboard
- Conversational question-answering interface
- Optional live web enrichment
- Transparent retrieval inspection

---

## Responsible AI

This application is intended as a document analysis and information retrieval tool.

AI-generated responses should be reviewed before being used for important decisions. The system should not be treated as an authoritative source simply because an answer is generated by an AI model.

---

## License

This project is available under the license included in this repository.

---

## Author

**Darshan S**

GitHub:

```text
https://github.com/DarshanS2004
```

---

## Conclusion

**AI Document Assistant** demonstrates how modern AI applications can combine document processing, semantic vector search, large language models, and optional web retrieval into a practical interactive application.

The project provides a foundation for building more advanced document intelligence systems while emphasizing document-grounded answers and retrieval transparency.