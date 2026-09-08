# Private Research Agent 🔬

A modular, privacy-first **Autonomous Research Agent** that transforms natural-language research questions into fact-grounded, citation-backed intelligence reports.

The agent dynamically searches the web, scrapes and sanitizes evidence from multiple sources, generates local dense vector embeddings with zero external data leakage, indexes knowledge in a local Qdrant vector database, and synthesizes answers using high-speed LLMs with strict source verification.

---

## Key Highlights

- **Dynamic Web Research**: Automatically searches the live web for relevant sources using Firecrawl rather than relying on stale model training data.
- **Multi-Source Scraping & Sanitization**: Extracts clean Markdown from discovered URLs, strips web boilerplate (cookie banners, navigation links, tracking scripts), and normalizes text structure.
- **Deterministic Provenance & Deduplication**: Computes SHA-256 content hashes and stable document IDs (`doc_<hash>`) to prevent duplicate indexing across research sessions.
- **Structure-Aware Chunking**: Chunks documents along semantic markdown boundaries (headings, lists, code blocks, paragraphs) with configurable token targets and sliding overlap.
- **Privacy-First Local Embeddings**: Generates 1024-dimensional dense vectors locally using `BAAI/bge-m3` via ONNX-accelerated FastEmbed (with SentenceTransformers and Ollama fallbacks). No raw document text or embeddings are sent to third-party embedding APIs.
- **Idempotent Vector Storage**: Indexes chunk vectors, payloads, and provenance metadata in a local Qdrant vector database with automated document version replacement.
- **Dense Vector Retrieval**: Performs cosine similarity retrieval to extract the most relevant evidence chunks for any given research query.
- **Fast Grounded Synthesis**: Generates streaming answers using Groq Cloud (`openai/gpt-oss-20b`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) or a 100% offline local Ollama instance.
- **Verified Citations**: Validates all inline source tags (`[S1]`, `[S2]`) against retrieved evidence chunks, filters out hallucinated citations, and formats a clean bibliography linking back to verified source URLs.

---

## Research Workflow

```text
                     [ User Research Question ]
                                 │
                                 ▼
                    [ 1. Firecrawl Web Search ]
                                 │
                                 ▼
                  [ 2. Discovered Research URLs ]
                                 │
                                 ▼
                 [ 3. Multi-URL Web Scraping ]
                                 │
                                 ▼
             [ 4. Rule-Based Cleaning & Boilerplate Strip ]
                                 │
                                 ▼
           [ 5. Deterministic Document IDs & SHA-256 Hash ]
                                 │
                                 ▼
              [ 6. Structure-Aware Semantic Chunking ]
                                 │
                                 ▼
           [ 7. Local Vector Embeddings (BAAI/bge-m3) ]
                                 │
                                 ▼
            [ 8. Idempotent Indexing in Qdrant Database ]
                                 │
                                 ▼
              [ 9. Dense Vector Evidence Retrieval ]
                                 │
                                 ▼
         [ 10. Grounded Answer Synthesis & Citation Validation ]
                                 │
                                 ▼
               [ Verified Answer + Source Bibliography ]
```

---

## Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (for local Qdrant vector database)
- **Firecrawl API Key** (for web search & scraping — sign up at [firecrawl.dev](https://firecrawl.dev))
- **Groq API Key** (for ultra-fast cloud inference — sign up at [console.groq.com](https://console.groq.com)) or **Ollama** installed locally for 100% offline operation.

---

## Getting Started

### 1. Clone the Repository & Set Up Environment

```bash
# Clone repository
git clone <repository-url>
cd private-research-agent

# Create and activate virtual environment
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory based on `.env.example`:

```env
# Application Settings
APP_NAME=Private Research Agent
ENVIRONMENT=DEVELOPMENT

# LLM Provider ("groq" for cloud speed, "ollama" for 100% local)
LLM_PROVIDER=groq

# Groq Cloud LLM Settings
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

# Ollama Local LLM Settings (if using local inference)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b

# Firecrawl Web Search & Scraping
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=research_documents

# Local Embedding Model
EMBEDDING_MODEL=BAAI/bge-m3

# Chunking Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Persistence Directories
DATA_RAW_DIR=data/raw
DATA_PROCESSED_DIR=data/processed
DATA_CACHE_DIR=data/cache
```

### 3. Start Local Qdrant Instance

Launch Qdrant using Docker Compose:

```bash
docker compose up -d
```

Verify that Qdrant is running at `http://localhost:6333/dashboard`.

---

## Usage

### Interactive Research Session

Launch the interactive research CLI to ask continuous research questions:

```bash
python app/main.py
```

```text
======================================================================
  Private Research Agent - Dynamic Research Engine
  (Type 'quit', 'exit', 'bye', or 'stop' to end the session)
======================================================================

What research topic would you like to investigate? What are the latest developments in agentic RAG?
```

### Direct Single Research Query

Execute a full research run directly from the command line:

```bash
python app/main.py "What are the latest developments in agentic RAG?"
```

Customize the number of search URLs and retrieved chunks:

```bash
python app/main.py "Recent breakthroughs in quantum computing" --urls 5 --top_k 5
```

### Single URL Ingestion

Ingest and index a specific paper or webpage directly into Qdrant:

```bash
python app/main.py --url "https://arxiv.org/abs/2005.11401"
```

### System Health & Database Diagnostics

Check connectivity to Qdrant, active LLM configuration, and existing collections:

```bash
python app/main.py --status
```

---

## Citation Verification & Provenance

To guarantee that answers remain strictly grounded in verified evidence:

1. **Structured Source IDs**: Retrieved context is supplied to the LLM with explicit tags (`[S1]`, `[S2]`).
2. **Strict Grounding Prompting**: The system prompt instructs the model to answer exclusively based on the provided evidence and cite source tags.
3. **Anti-Hallucination Validation**:
   - Matches all citations in the generated response against real source IDs.
   - Discards fabricated or out-of-range citations (e.g., `[S99]`).
   - Suppresses citations if the model determines there is insufficient information.
4. **Source Bibliography**: Prints the final verified list of source titles and actual URLs mapped to the citation tags.

---

## Running Automated Tests

Run the complete test suite:

```bash
pytest tests/ -v
```

---

## Security & Privacy Considerations

- **Zero Embedding Leakage**: Embeddings are computed locally on your CPU/GPU using open-source models (`BAAI/bge-m3`), preventing internal data from leaving your system during vectorization.
- **Untrusted Web Content Isolation**: Scraped web markdown is treated as untrusted text with prompt injection guardrails to prevent untrusted webpage instructions from overriding system behavior.
- **Local Vector Storage**: All vector indexes and chunk texts reside in your local Qdrant container.
