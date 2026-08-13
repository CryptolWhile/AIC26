# AIC26 - Advanced Video Retrieval System 🎬🔍

<div align="center">
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
  [![Milvus](https://img.shields.io/badge/Milvus-Vector%20DB-blue.svg)](https://milvus.io/)
  [![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.19-green.svg)](https://www.elastic.co/)
  
  > 🚀 An AI-powered multimodal video retrieval application that extracts, indexes, and semantically searches through large-scale video datasets.
</div>

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technologies](#%EF%B8%8F-technologies)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [Data Pipeline Workflow](#-data-pipeline-workflow)

## 🌟 Overview

**AIC26** is a comprehensive Video Retrieval System that enables users to perform intelligent semantic searches across thousands of videos. By dissecting videos into discrete "shots" and "keyframes", and encoding them using State-of-the-Art Vision-Language models like **CLIP** and **SigLIP**, the system allows you to find exact moments in videos using natural language text queries or image queries.

## ✨ Key Features

- 🧠 **Multimodal Semantic Search**: Search videos using natural language or images via `OpenCLIP` and `HuggingFace SigLIP`.
- ✂️ **Smart Video Processing Pipeline**: Automated shot boundary detection and optimal keyframe distillation.
- 🗄️ **Robust Data Ingestion**: Idempotent database insertion with auto-resume capabilities (avoids duplicating vectors).
- 🚀 **High-Performance Retrieval**: Real-time vector search using **Milvus** coupled with full-text search on **Elasticsearch**.
- 🖥️ **Modern Web Interface**: Clean, responsive frontend built with React and Material UI.

## ⚙️ Technologies

| Layer       | Technologies                                                   |
|-------------|---------------------------------------------------------------|
| **AI Models** | HuggingFace, OpenCLIP (ViT-L/14, ViT-H/14, SigLIP)            |
| **Backend** | Python, Flask, LangChain, PyMilvus, FFmpeg, OpenCV            |
| **Frontend**| React, Vite, Material UI (MUI), Axios                         |
| **Databases**| Milvus (Vector), Elasticsearch (Metadata/Text), MongoDB       |
| **DevOps**  | Docker, Docker Compose, Kaggle (for heavy GPU extraction)     |

## 📁 Project Structure

```text
AIC26/
├── backend/                  # Python Flask API & AI Pipeline
│   ├── src/
│   │   ├── configs/          # YAML configurations for extraction & ingestion
│   │   ├── database/         # Milvus, Elastic, and Mongo DB connectors
│   │   ├── routes/           # Flask API endpoints
│   │   ├── scripts/          # Bash scripts for pipelines (db_ingestion.sh)
│   │   └── services/         # Core logic (Embedding, Processing, Retrieval)
│   ├── processed/            # Extracted data (images, shots, keyframes, embeddings)
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React (Vite) User Interface
│   ├── src/                  # React components and pages
│   └── package.json          # Node dependencies
└── docker-compose.yml        # Infrastructure setup (Milvus, Elastic, Mongo)
```

## 🚀 Installation Guide

### System Requirements
- [Python 3.10+](https://www.python.org) (Conda recommended)
- [Node.js & npm](https://nodejs.org/en/download/)
- [Docker & Docker Compose](https://docs.docker.com/get-docker/)

### 1. Database Infrastructure (Docker)
Start the core databases (Milvus Standalone, Elasticsearch, MongoDB, MinIO, etcd) via Docker:
```bash
cd AIC26
docker-compose up -d
```
*(Wait ~30 seconds for all database services to fully initialize before running ingestion).*

### 2. Backend Setup
```bash
cd AIC26/backend

# Create and activate conda environment
conda create -n dezus_py310 python=3.10
conda activate dezus_py310

# Install dependencies
pip install -r requirements.txt

# Run the API Server
python app.py
```
*The backend API will run on http://localhost:5000 (or as configured).*

### 3. Frontend Setup
```bash
cd AIC26/frontend

# Install dependencies
npm install

# Run Vite development server
npm run dev
```
*The web app will be available at http://localhost:5173*

## 🔄 Data Pipeline Workflow

If you want to extract and ingest new video datasets:

1. **Extraction (Usually on Kaggle/GPU)**:
   - Extract shots: `python -m src.services.processing.pipeline shot_extraction ...`
   - Extract keyframes: `python -m src.services.processing.pipeline keyframe_extraction ...`
   - Generate embeddings: `sh src/scripts/embedding_extraction.sh`
2. **Ingestion (Local)**:
   - Place your extracted data into `backend/processed/`
   - Run the DB ingestion script:
   ```bash
   cd backend
   sh src/scripts/db_ingestion.sh
   ```
   *(Note: Ensure Docker DBs are running. If you want to start completely fresh without duplicating vectors in Milvus, run `docker-compose down` and delete the `volumes/` directory first).*

---
**Maintained by**: [Your Name/Team] 
