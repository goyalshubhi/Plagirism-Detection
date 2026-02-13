Plagiarism Detection & AI Content Analysis System

A full-stack plagiarism and AI-generated content detection platform built as a university-level project.
The system allows users to upload documents or text, detects plagiarism using semantic embeddings, identifies AI-generated content, and presents results through a web interface.

🚀 Features
🔍 Plagiarism Detection

Chunk-based semantic similarity using Sentence Transformers

Cosine similarity scoring

Highlights matching passages between documents

Batch comparison support

🤖 AI-Generated Content Detection

Local AI detection model (no paid API required)

Confidence score + classification

Provider-based architecture (extensible to OpenAI / Together AI)

🧠 NLP & ML

Transformer-based embeddings

Vector similarity computation

Optimized for CPU (no GPU required)

🗂️ File & Text Analysis

Upload PDF, DOCX, TXT files

Direct text input supported

Text extraction and preprocessing

🧩 Backend Architecture

FastAPI (async)

Modular service-based design

Background processing using Celery

PostgreSQL + pgvector for storage

Redis for task queue

MinIO for object storage

🌐 Frontend

React + Vite

Nginx-served production build

API-driven UI

🐳 Fully Dockerized

One-command startup

Multi-container architecture

Isolated services

🏗️ System Architecture
Frontend (React + Nginx)
        |
        v
FastAPI Backend
  ├── Auth & Users
  ├── Plagiarism Service
  ├── AI Detection Service
  ├── Batch Processing
        |
        v
PostgreSQL + pgvector
Redis (Celery Queue)
MinIO (File Storage)

🛠️ Tech Stack
Backend

FastAPI

Pydantic v2

SQLAlchemy

Celery + Redis

PostgreSQL + pgvector

Sentence-Transformers

PyTorch (CPU)

Frontend

React

Vite

Nginx

DevOps

Docker

Docker Compose

📁 Project Structure
plagiarism-detection/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── models/
│   │   ├── core/
│   │   └── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
└── README.md

⚙️ Installation & Setup
🔹 Prerequisites

Docker Desktop

Docker Compose

At least 8 GB RAM recommended (ML models)

🔹 Run the Project
docker compose build
docker compose up -d

🔹 Verify Services
Service	URL
Frontend	http://localhost

Backend API	http://localhost:8000

Swagger UI	http://localhost:8000/docs

Health Check	http://localhost:8000/health

MinIO	http://localhost:9001
🧪 Testing via Swagger (Important)
Step 1: Open Swagger
http://localhost:8000/docs

Step 2: Analyze Text

Go to POST /api/v1/analyze

Click Try it out

Paste text in the text field

Use default options

Execute

You’ll receive a batch_id.

Step 3: Fetch Results
GET /api/v1/batches/{batch_id}/results


Returns:

Plagiarism similarity score

Matching chunks

AI detection results

☁️ Cloud & Distributed Features Used

Object storage (MinIO – S3 compatible)

Background processing (Celery workers)

Scalable API containers

Stateless backend services

Queue-based batch processing

🎓 Academic Justification

This project demonstrates:

NLP & ML application in real systems

Backend API design

Asynchronous processing

Containerized deployment

Database + vector similarity search

Clean modular architecture

⚠️ This is a university-level project, not a production SaaS.
The focus is learning, correctness, and clarity, not massive scale.

🔮 Future Enhancements

FAISS-based vector search

GPU acceleration

Advanced plagiarism visualization

Multi-language detection

Role-based admin dashboards

👩‍💻 Author

Shubhangi Goyal
University Project – Cloud & NLP Systems

✅ Status

🟢 Backend working
🟢 Frontend working
🟢 Swagger testing enabled
🟢 Fully Dockerized
