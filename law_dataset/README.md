# VietLawBERT

VietLawBERT is an advanced legal language model (LLM) system specializing in Vietnam legal documents, specifically focusing on traffic law interpretation and application. Built with a Hybrid RAG (Retrieval-Augmented Generation) architecture combining vector search (Milvus), graph traversal (Neo4j), and contextual API integration.

## Architecture

### Core Components

**1. Web Crawler**
- `src/crawler/` → Legal document extraction, preprocessing
- Spider to scrape VBPL (Việt Nam Pháp Luật) website
- HTML cleaning, normalization (removing scripts, ads, navigation)

**2. Preprocessing**
- `src/preprocess/` → Chunking & Contextualization
- `legal_chunker.py` → Split legal text into logical chunks
- `contextualizer.py` → Generate contextual embeddings for chunks

**3. Vector Store**
- `milvus_client.py` → Milvus vector database for semantic search
- Stores `finetuned_embeddings` (Matryoshka Layers)

**4. Knowledge Graph**
- `neo4j_client.py` → Neo4j graph database for structured relationships
- Stores legal document hierarchy (Part → Section → Clause → Point)

**5. Retrieval Layer**
- `src/rag/` → Hybrid RAG implementation
- `retriever.py` → Combines vector search (Milvus) and graph traversal (Neo4j)
- `generator.py` → Text generation using external LLM APIs

**6 reformer.py** → Code and infrastructure for bug fixes and iterative improvements

### Pipeline Flow
```
VBPL Website → Crawler → Preprocessing → Milvus(Embedding) + Neo4j(Graph) → RAG Retrieval → Generative Model → Response
```

### Design Philosophy

1. **Hybrid RAG Approach**
   - **Vector Search (Milvus)**: Semantic similarity matching
   - **Graph Traversal (Neo4j)**: Logical relationship extraction
   - **API Integration**: Multiple LLM providers (OpenAI, Ollama, etc.)

2.: **Human-in-the-Loop**
   - Document cleaning methods validated by legal experts
   - Processed legally-relevant content efficiently
   - Combines structured and unstructured data processing

3. **Robust Data Pipeline**
   - Handles large-scale legal document ingestion
   - Supports dynamic updates and modifications to knowledge base

**Strengths:**
- Specialized for Vietnamese legal documents
- Sophisticated vector search capabilities
- Structured relationship handling
- Production-ready architecture ready for scale

**Future Expansion:**
- Integration with other language legal systems
- Real-time legal document updates
- Advanced legal reasoning modules

## Technical Features

### Retrieval Methods
- **Milvus Retrieval**: Quick semantic similarity search for short-term context
- **Neo4j Retrieval**: Deep contextual investigation for long-term context (document hierarchy)
- **Hybrid RRF Fusion**: Combines both retrieval methods for comprehensive context extraction

### Generation Infrastructure
- Multiple LLM providers (OpenAI, Gemini, Ollama, etc.)
- Streaming response generation
- Structured output formatting
- Context-enhanced legal text generation

### Data Management
- Automated pipeline processing
- Robust error handling and logging
- Scalable file management system

## Usage

### Basic Usage (CLI)
```bash
# Start the system
python src/main.py

# Menus:
# 0 - Start Docker containers
# 1 - Crawl data
# 2 - Rescue failed links
# 3 - HTML to Markdown conversion
# 4 - Chunk & Contextualize
# 5 - Ingest data to DB
# 6 - Full pipeline (automatically starts chatbot)
# 7 - Start chatbot interface
# 8 - Generate training triplet data
# 9 - Fine-tune embedding models
# q - Exit
```

### Application
Run the chatbot application:
```bash
python app.py
```

## Development Notes

### Directory Structure
```
VietLawBERT
├── Dockerfile, docker-compose.yml                    # Docker orchestration
├── law_dataset/
│   ├── src/
│   │   ├── crawler/         # Web scraping
│   │   ├── preprocess/      # Text processing
│   │   ├── database/        # Vector & graph databases
│   │   ├── rag/            # Retrieval & generation
│   │   └── streaming/       # Real-time processing
│   ├── app.py              # Main CLI interface
│   ├── README.md           # Project documentation
│   └── requirements.txt    # Dependencies
├── data/
│   ├── raw/               # Raw downloaded files
│   ├── processed/         # Processed text files
│   ├── json/              # Structured metadata
│   └── logs/             # System logs
└── models/               # Model weights (optional)
```

### Configuration
- All environment variables are specified in `.env` file
- Database connection settings in `config.py`
- LLM API endpoints configured in `config.py`

### Dependencies
```
pip install -r requirements.txt
```

### Notes
- Vietnamese text processing requires appropriate encoding support
- External LLM APIs require API keys for operation
- Document cleaning may need periodic updates depending on source website structure changes
```