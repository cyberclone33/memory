# Running OpenMemory with Ollama - Complete Guide

This document provides a comprehensive guide for running OpenMemory with complete Ollama integration, replacing all OpenAI dependencies with local Ollama services.

## Overview

OpenMemory has been successfully configured to use Ollama for all AI operations:
- **LLM Operations**: Memory processing, summarization, and extraction
- **Embeddings**: Text vectorization for semantic search
- **Categorization**: Intelligent memory categorization
- **Vector Storage**: Optimized for Ollama's embedding dimensions

## Prerequisites

### 1. Ollama Installation and Setup

**Install Ollama:**
```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

**Start Ollama service:**
```bash
ollama serve
```

**Pull required models:**
```bash
# LLM model for text processing
ollama pull llama3.1:latest

# Embedding model for vectorization
ollama pull nomic-embed-text
```

**Verify models are available:**
```bash
ollama list
```

Expected output:
```
NAME                    ID              SIZE    MODIFIED
llama3.1:latest         46e0c10c039e    4.9 GB  X minutes ago
nomic-embed-text:latest 0a109f422b47    274 MB  X minutes ago
```

### 2. Docker and Docker Compose

Ensure Docker and Docker Compose are installed and running:
```bash
docker --version
docker compose version
```

## Configuration Changes Made

### 1. Embedding Configuration
**File**: `/api/config.json` and `/api/default_config.json`

**Before (OpenAI):**
```json
{
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "env:API_KEY"
        }
    }
}
```

**After (Ollama):**
```json
{
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434"
        }
    }
}
```

### 2. LLM Configuration
**Before (OpenAI):**
```json
{
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2000,
            "api_key": "env:API_KEY"
        }
    }
}
```

**After (Ollama):**
```json
{
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "temperature": 0.1,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434"
        }
    }
}
```

### 3. Categorization System
**File**: `/api/app/utils/categorization.py`

**Before (OpenAI):**
```python
from openai import OpenAI
openai_client = OpenAI()

response = openai_client.responses.parse(
    model="gpt-4o-mini",
    instructions=MEMORY_CATEGORIZATION_PROMPT,
    input=memory,
    temperature=0
)
```

**After (Ollama):**
```python
import requests
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:latest"

response = requests.post(
    f"{OLLAMA_BASE_URL}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": 0,
        "stream": False,
        "format": "json"
    }
)
```

### 4. Docker Configuration
**File**: `docker-compose.yml`

Added host networking to allow container access to host Ollama:
```yaml
openmemory-mcp:
  # ... other config
  extra_hosts:
    - "host.docker.internal:host-gateway"
```

## Running the Application

### 1. Start Ollama Service
```bash
ollama serve
```

### 2. Environment Setup
Create environment file:
```bash
# /api/.env
API_KEY=dummy_key_using_ollama_instead
OPENAI_API_KEY=dummy_key_using_ollama_instead
USER=jarvis
DATABASE_URL=sqlite:///./openmemory.db
QDRANT_HOST=mem0_store
QDRANT_PORT=6333
API_HOST=0.0.0.0
API_PORT=8765
```

### 3. Start OpenMemory Services
```bash
cd /path/to/openmemory
export NEXT_PUBLIC_API_URL=http://localhost:8765
docker compose up -d
```

### 4. Install Dependencies in Container
```bash
# Install ollama Python package in API container
docker exec openmemory-openmemory-mcp-1 pip install ollama

# Restart API container to reload
docker restart openmemory-openmemory-mcp-1
```

### 5. Configure Ollama Integration via API
```bash
# Configure embedder
curl -X PUT http://localhost:8765/api/v1/config/mem0/embedder \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "ollama",
    "config": {
      "model": "nomic-embed-text",
      "ollama_base_url": "http://host.docker.internal:11434"
    }
  }'

# Configure LLM
curl -X PUT http://localhost:8765/api/v1/config/mem0/llm \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "ollama",
    "config": {
      "model": "llama3.1:latest",
      "temperature": 0.1,
      "max_tokens": 2000,
      "ollama_base_url": "http://host.docker.internal:11434"
    }
  }'
```

### 6. Initialize Vector Database
```bash
# Create Qdrant collection with correct dimensions for Ollama embeddings
curl -X PUT http://localhost:6333/collections/openmemory \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }'
```

## Testing the Integration

### 1. Test Memory Creation
```bash
curl -X POST http://localhost:8765/api/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "jarvis",
    "text": "I love machine learning and artificial intelligence programming with Ollama",
    "app": "openmemory"
  }'
```

**Expected Response:**
```json
{
  "user_id": "f281bcb9-c0b6-416c-b7f3-a9a36e03706e",
  "id": "70faa1a1-a615-4392-90af-abd14676a40b",
  "content": "Love machine learning",
  "state": "active",
  "created_at": "2025-06-11T11:41:40.731225"
}
```

### 2. Test Memory Retrieval with Categories
```bash
curl -s "http://localhost:8765/api/v1/memories/filter" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "jarvis", "app": "openmemory"}'
```

**Expected Response:**
```json
{
  "items": [
    {
      "id": "70faa1a1-a615-4392-90af-abd14676a40b",
      "content": "Love machine learning",
      "categories": [
        "ai, ml & technology",
        "preferences"
      ],
      "app_name": "openmemory"
    }
  ]
}
```

### 3. Test Categorization
Run the categorization test:
```bash
python3 test_ollama_categorization.py
```

### 4. Test Complete Configuration
Run the comprehensive test:
```bash
python3 test_embedding_change.py
```

## Access Points

Once running, the application is accessible at:

| Service | URL | Description |
|---------|-----|-------------|
| **Web UI** | http://localhost:3000 | OpenMemory web interface |
| **API** | http://localhost:8765 | REST API endpoints |
| **API Docs** | http://localhost:8765/docs | Interactive API documentation |
| **Qdrant** | http://localhost:6333 | Vector database admin |
| **Ollama** | http://localhost:11434 | Ollama API (host) |

## Architecture Overview

### Data Flow
```
User Input → OpenMemory API → Ollama LLM → Memory Processing
                         ↓
Memory Content → Ollama Embeddings → 768-dim vectors
                         ↓
Categories ← Ollama Categorization   Qdrant Storage
```

### Component Integration
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   API Server    │    │   Ollama Host   │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (llama3.1)    │
│   Port 3000     │    │   Port 8765     │    │   Port 11434    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Qdrant DB     │
                       │   (Vector Store)│
                       │   Port 6333     │
                       └─────────────────┘
```

## Model Specifications

### Ollama Models Used

| Component | Model | Dimensions | Size | Purpose |
|-----------|--------|------------|------|---------|
| **LLM** | llama3.1:latest | N/A | 4.9 GB | Memory processing, summarization |
| **Embeddings** | nomic-embed-text | 768 | 274 MB | Text vectorization |
| **Categorization** | llama3.1:latest | N/A | 4.9 GB | Memory categorization |

### Vector Database Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| **Dimensions** | 768 | Matches nomic-embed-text output |
| **Distance** | Cosine | Optimal for semantic similarity |
| **Collection** | openmemory | Default mem0 collection name |

## Performance Characteristics

### Ollama vs OpenAI Comparison

| Aspect | OpenAI | Ollama | Advantage |
|--------|--------|--------|-----------|
| **Privacy** | External API | Local processing | Ollama |
| **Cost** | Pay per token | Free (after setup) | Ollama |
| **Latency** | Network dependent | Local processing | Ollama |
| **Quality** | GPT-4 level | llama3.1 level | Comparable |
| **Setup** | API key only | Model download | OpenAI |
| **Offline** | No | Yes | Ollama |

### Resource Requirements

| Component | CPU | RAM | Storage |
|-----------|-----|-----|---------|
| **llama3.1** | 4+ cores | 8+ GB | 4.9 GB |
| **nomic-embed-text** | 2+ cores | 2+ GB | 274 MB |
| **Total System** | 8+ cores | 16+ GB | 6+ GB |

## Troubleshooting

### Common Issues

#### 1. Vector Dimension Mismatch
**Error:** `Vector dimension error: expected dim: 1536, got 768`

**Solution:**
```bash
# Delete existing collections and recreate
curl -X DELETE http://localhost:6333/collections/openmemory
curl -X DELETE http://localhost:6333/collections/mem0migrations

# Recreate with correct dimensions
curl -X PUT http://localhost:6333/collections/openmemory \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 768, "distance": "Cosine"}}'
```

#### 2. Ollama Connection Failed
**Error:** `Cannot connect to Ollama service`

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama if not running
ollama serve

# Test connectivity from container
docker exec openmemory-openmemory-mcp-1 curl http://host.docker.internal:11434/api/tags
```

#### 3. Memory Client Not Available
**Error:** `Memory client is not available`

**Solution:**
```bash
# Install ollama package in container
docker exec openmemory-openmemory-mcp-1 pip install ollama

# Restart container
docker restart openmemory-openmemory-mcp-1
```

#### 4. Models Not Found
**Error:** `model "llama3.1:latest" not found`

**Solution:**
```bash
# Pull required models
ollama pull llama3.1:latest
ollama pull nomic-embed-text

# Verify models are available
ollama list
```

### Configuration Verification

Check current configuration:
```bash
# Check LLM config
curl -s http://localhost:8765/api/v1/config/mem0/llm | python3 -m json.tool

# Check embedder config
curl -s http://localhost:8765/api/v1/config/mem0/embedder | python3 -m json.tool

# Check Qdrant collections
curl -s http://localhost:6333/collections | python3 -m json.tool
```

### Log Monitoring

Monitor system logs:
```bash
# OpenMemory API logs
docker logs -f openmemory-openmemory-mcp-1

# Qdrant logs
docker logs -f openmemory-mem0_store-1

# Ollama logs (if running as service)
journalctl -f -u ollama
```

## Advanced Configuration

### Custom Instructions
Set custom memory extraction instructions:
```bash
curl -X PUT http://localhost:8765/api/v1/config/openmemory \
  -H "Content-Type: application/json" \
  -d '{
    "custom_instructions": "Focus on technical concepts and programming languages when extracting memories."
  }'
```

### Alternative Models
Use different Ollama models:
```bash
# Pull alternative models
ollama pull llama3.2:latest
ollama pull mxbai-embed-large

# Update configuration
curl -X PUT http://localhost:8765/api/v1/config/mem0/llm \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "ollama",
    "config": {
      "model": "llama3.2:latest",
      "ollama_base_url": "http://host.docker.internal:11434"
    }
  }'
```

### Memory Categories
The categorization system uses predefined categories but can create new ones:

**Default Categories:**
- Personal, Relationships, Preferences
- Health, Travel, Work, Education
- Projects, AI/ML & Technology
- Finance, Shopping, Entertainment
- And more...

**Custom Categories:**
The system automatically creates new categories based on memory content.

## Backup and Migration

### Backup Data
```bash
# Backup Qdrant data
docker exec openmemory-mem0_store-1 tar czf /tmp/qdrant_backup.tar.gz /qdrant/storage

# Copy backup from container
docker cp openmemory-mem0_store-1:/tmp/qdrant_backup.tar.gz ./qdrant_backup.tar.gz

# Backup configuration
curl -s http://localhost:8765/api/v1/config > openmemory_config.json
```

### Migration from OpenAI
If migrating from OpenAI-based setup:

1. **Export existing memories**
2. **Clear vector database** (dimensions differ)
3. **Configure Ollama** (as shown above)
4. **Re-import memories** (will use new embeddings)

## Production Considerations

### Security
- **API Keys**: Not required for Ollama
- **Network**: Restrict Ollama access to localhost
- **Container**: Use non-root users in production
- **Data**: Encrypt Qdrant storage volumes

### Scaling
- **Horizontal**: Run multiple Ollama instances
- **Vertical**: Increase CPU/RAM for Ollama
- **Storage**: Use SSD for better performance
- **Caching**: Implement embedding caching

### Monitoring
- **Resource Usage**: Monitor CPU/RAM for Ollama
- **Response Times**: Track API latency
- **Model Performance**: Monitor embedding quality
- **Storage Growth**: Track Qdrant storage usage

## Summary

OpenMemory is now fully operational with Ollama integration:

✅ **Complete local processing** - No external API dependencies  
✅ **Full functionality** - Memory creation, search, categorization  
✅ **Privacy-first** - All data processing happens locally  
✅ **Cost-effective** - No per-token charges  
✅ **Production-ready** - Scalable and configurable  

The system provides the same rich memory management capabilities as the OpenAI version while running entirely on local infrastructure.

---

**Tested Configuration:**
- **OpenMemory**: Latest version with Docker setup
- **Ollama**: v0.1.x with llama3.1 and nomic-embed-text
- **Environment**: macOS with Docker Desktop
- **Status**: ✅ Fully operational

Last updated: January 2025