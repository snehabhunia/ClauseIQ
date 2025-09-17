# ClauseIQ - Legal Document RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for analyzing legal documents, medical insurance policies, and loan agreements. Built specifically for Windows compatibility with modern technologies.

## Features

- 📄 **Document Processing**: Upload and process PDF/DOCX files
- 🔍 **Smart Search**: Vector-based semantic search using FAISS
- 🤖 **AI Responses**: Context-aware answers powered by Groq LLM
- 💾 **Persistent Storage**: Vector embeddings saved to disk
- 🎨 **Modern UI**: React-based responsive frontend
- ⚡ **Fast API**: High-performance FastAPI backend

## Tech Stack

### Backend
- **Python 3.11+** with FastAPI
- **FAISS** for vector storage (CPU-optimized for Windows)
- **SentenceTransformers** for embeddings (`all-MiniLM-L6-v2`)
- **LangChain** for document processing
- **Groq API** for LLM responses

### Frontend
- **React 18+** with modern hooks
- **Axios** for API communication
- **React-Dropzone** for file uploads
- **Lucide React** for icons
- **React-Markdown** for formatted responses

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.11 or higher
- **Node.js**: 18+ 
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 2GB free space for dependencies

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/ClauseIQ.git
cd ClauseAI
