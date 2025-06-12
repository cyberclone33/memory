# OpenMemory - Claude Code MCP Integration Guide

This document provides comprehensive guidance for integrating OpenMemory with Claude Code via the Model Context Protocol (MCP), including configuration, troubleshooting, and the intelligent Docker networking solutions implemented.

## Overview

OpenMemory provides persistent memory capabilities for Claude Code through MCP (Model Context Protocol) server integration. The system can run with either OpenAI or local models (Ollama) and provides seamless memory operations including:

- **Memory Storage**: Automatic extraction and storage of user information and preferences
- **Memory Search**: Semantic search through stored memories using vector embeddings
- **Memory Management**: List, update, and delete operations with proper access control
- **Multi-User Support**: Isolated memory spaces per user and application

## Quick Start

### Prerequisites

1. **Ollama** (recommended for local inference)
2. **Docker & Docker Compose** (for containerized deployment)
3. **Python 3.12+** (for local development)

### 1. Install and Setup Ollama

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama
ollama serve

# Pull required models
ollama pull llama3.1         # For LLM operations
ollama pull nomic-embed-text # For embeddings
```

### 2. Start OpenMemory Services

```bash
# Clone and navigate to openmemory directory
cd /path/to/openmemory

# Start all services
docker compose up -d

# Verify services are running
curl http://localhost:8765/docs     # MCP API Server
curl http://localhost:6333/collections  # Qdrant Vector Store
open http://localhost:3000          # Web UI (optional)
```

### 3. Test MCP Integration

```bash
# Test memory addition
curl -X POST "http://localhost:8765/mcp/direct/claude_code_cli/claude_code_user" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test",
    "method": "tools/call",
    "params": {
      "name": "add_memories",
      "arguments": {
        "text": "I love Python programming and prefer FastAPI for web development"
      }
    }
  }'

# Test memory search
curl -X POST "http://localhost:8765/mcp/direct/claude_code_cli/claude_code_user" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test",
    "method": "tools/call",
    "params": {
      "name": "search_memory",
      "arguments": {
        "query": "programming preferences"
      }
    }
  }'
```

## Architecture & Configuration

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Claude Code   │ ◄──┤ MCP Protocol     │ ◄──┤ OpenMemory API  │
│                 │    │ (HTTP/JSON-RPC)  │    │ (Port 8765)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
                              ┌─────────────────┐    ┌─────────────────┐
                              │ Ollama LLM      │    │ Qdrant Vector   │
                              │ (Port 11434)    │    │ Store (6333)    │
                              │ - llama3.1      │    │ - Embeddings    │
                              │ - nomic-embed   │    │ - Similarity    │
                              └─────────────────┘    └─────────────────┘
```

### Configuration Files

#### 1. MCP Configuration (`/api/mcp_config.json`)

```json
{
  "mem0": {
    "llm": {
      "provider": "ollama",
      "config": {
        "model": "llama3.1",
        "temperature": 0.1,
        "max_tokens": 2000,
        "ollama_base_url": "http://localhost:11434"
      }
    },
    "embedder": {
      "provider": "ollama", 
      "config": {
        "model": "nomic-embed-text",
        "ollama_base_url": "http://localhost:11434"
      }
    },
    "vector_store": {
      "provider": "qdrant",
      "config": {
        "collection_name": "openmemory",
        "host": "localhost",
        "port": 6333,
        "embedding_model_dims": 768
      }
    }
  }
}
```

#### 2. Docker Compose (`docker-compose.yml`)

```yaml
services:
  mem0_store:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - mem0_storage:/mem0/storage

  openmemory-mcp:
    image: mem0/openmemory-mcp
    build: api/
    ports:
      - "8765:8765"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      - mem0_store
    volumes:
      - ./api:/usr/src/openmemory
```

## Docker Networking Intelligence

### The Challenge

The OpenMemory MCP server runs inside Docker containers but needs to communicate with:
1. **Ollama** running on the host machine
2. **Qdrant** running as another Docker service

### Intelligent URL Resolution

The system automatically detects the Docker environment and adjusts URLs:

#### For Ollama (Host Machine Access)
```python
def _get_docker_host_url():
    """
    Intelligently determine the best host URL for Docker containers.
    """
    # Priority order:
    # 1. OLLAMA_HOST environment variable (custom override)
    # 2. host.docker.internal (Docker Desktop Mac/Windows)
    # 3. Docker bridge gateway IP (Linux)
    # 4. Fallback to 172.17.0.1
    
    if custom_host := os.environ.get('OLLAMA_HOST'):
        return custom_host
    
    if os.path.exists('/.dockerenv'):  # Inside Docker
        try:
            socket.gethostbyname('host.docker.internal')
            return 'host.docker.internal'
        except socket.gaierror:
            # Parse /proc/net/route for gateway
            return parse_docker_gateway()
    
    return "localhost"
```

**URL Transformation:**
- `http://localhost:11434` → `http://host.docker.internal:11434`

#### For Qdrant (Docker Service Access)
```python
def _fix_qdrant_urls(config_section):
    """
    Fix Qdrant URLs for Docker service communication.
    """
    if os.path.exists('/.dockerenv'):  # Inside Docker
        if config_section["config"]["host"] in ["localhost", "127.0.0.1"]:
            config_section["config"]["host"] = "mem0_store"
            print(f"Adjusted Qdrant host to mem0_store for Docker")
```

**URL Transformation:**
- `localhost:6333` → `mem0_store:6333`

### Debugging Network Issues

#### Common Connection Problems

1. **Ollama Connection Refused**
   ```bash
   # Test from host
   curl http://localhost:11434/api/tags
   
   # Test from container
   docker exec openmemory-mcp python3 -c "
   import requests
   resp = requests.get('http://host.docker.internal:11434/api/tags')
   print(f'Status: {resp.status_code}')
   "
   ```

2. **Qdrant Connection Issues**
   ```bash
   # Test Qdrant accessibility
   curl http://localhost:6333/collections
   
   # Test from container
   docker exec openmemory-mcp python3 -c "
   import requests
   resp = requests.get('http://mem0_store:6333/collections')
   print(f'Qdrant Status: {resp.status_code}')
   "
   ```

3. **Memory Client Initialization**
   ```bash
   # Test memory client directly
   docker exec openmemory-mcp python3 -c "
   from app.utils.memory import get_memory_client
   client = get_memory_client()
   print(f'Client: {type(client)}')
   "
   ```

## MCP Tools Available

### 1. `add_memories`
**Purpose**: Automatically extract and store user information
**Usage**: Called when users share personal information or preferences

```json
{
  "method": "tools/call",
  "params": {
    "name": "add_memories",
    "arguments": {
      "text": "I love Python programming and work as a data scientist"
    }
  }
}
```

**Response**:
```json
{
  "result": {
    "results": [
      {"id": "uuid-1", "memory": "Loves Python programming", "event": "ADD"},
      {"id": "uuid-2", "memory": "Works as data scientist", "event": "ADD"}
    ]
  }
}
```

### 2. `search_memory`
**Purpose**: Find relevant memories using semantic search
**Usage**: Called automatically before responding to user queries

```json
{
  "method": "tools/call", 
  "params": {
    "name": "search_memory",
    "arguments": {
      "query": "programming languages"
    }
  }
}
```

### 3. `list_memories`
**Purpose**: List all stored memories for the user
**Usage**: When users want to see what's remembered

```json
{
  "method": "tools/call",
  "params": {
    "name": "list_memories",
    "arguments": {}
  }
}
```

### 4. `delete_all_memories`
**Purpose**: Clear all memories (use with caution)
**Usage**: When users want to reset their memory profile

## Troubleshooting Guide

### "Memory system is currently unavailable"

This error indicates the memory client failed to initialize. Follow these steps:

#### 1. Check Service Status
```bash
docker compose ps
# All services should show "Up"

docker compose logs openmemory-mcp --tail=20
# Look for initialization errors
```

#### 2. Verify Ollama Connectivity
```bash
# From host
curl http://localhost:11434/api/tags

# Should return list of models including llama3.1 and nomic-embed-text
```

#### 3. Verify Qdrant Connectivity  
```bash
# From host
curl http://localhost:6333/collections

# Should return {"result":{"collections":[]},"status":"ok"}
```

#### 4. Check Docker Networking
```bash
# Test host.docker.internal resolution
docker exec openmemory-mcp nslookup host.docker.internal

# Test service communication
docker exec openmemory-mcp python3 -c "
import socket
socket.create_connection(('host.docker.internal', 11434), timeout=5)
socket.create_connection(('mem0_store', 6333), timeout=5)
print('All connections successful')
"
```

#### 5. Force Container Rebuild
```bash
# If dependencies are missing
docker compose down openmemory-mcp
docker compose build openmemory-mcp --no-cache
docker compose up openmemory-mcp -d
```

### Configuration Issues

#### Wrong Model Names
Ensure Ollama models are pulled:
```bash
ollama list
# Should show: llama3.1 and nomic-embed-text
```

#### Port Conflicts
Check if ports are available:
```bash
lsof -i :8765  # MCP API
lsof -i :6333  # Qdrant
lsof -i :11434 # Ollama
```

### Performance Optimization

#### Model Selection
- **LLM**: `llama3.1` (8B parameters, good balance)
- **Embeddings**: `nomic-embed-text` (137M parameters, fast)
- **Alternative**: Use `llama3.2` for smaller memory footprint

#### Memory Configuration
```bash
# Increase Docker memory allocation
# Docker Desktop > Settings > Resources > Memory > 8GB+
```

## Alternative Configurations

### Using OpenAI Instead of Ollama

```json
{
  "mem0": {
    "llm": {
      "provider": "openai",
      "config": {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "api_key": "env:OPENAI_API_KEY"
      }
    },
    "embedder": {
      "provider": "openai",
      "config": {
        "model": "text-embedding-3-small", 
        "api_key": "env:OPENAI_API_KEY"
      }
    }
  }
}
```

### Hybrid Configuration
```json
{
  "mem0": {
    "llm": {
      "provider": "ollama",
      "config": {
        "model": "llama3.1",
        "ollama_base_url": "http://localhost:11434"
      }
    },
    "embedder": {
      "provider": "openai",
      "config": {
        "model": "text-embedding-3-small",
        "api_key": "env:OPENAI_API_KEY"
      }
    }
  }
}
```

## Security & Privacy

### Local vs Cloud Processing
- **Ollama**: All processing happens locally, no data sent to external services
- **OpenAI**: Text sent to OpenAI APIs, subject to their privacy policy
- **Qdrant**: Vector storage is local in Docker container

### Access Control
- User memories are isolated by `user_id`
- Application access controlled by `client_name`
- All operations logged in access history

### Data Protection
```bash
# Backup vector storage
docker compose down
docker run --rm -v openmemory_mem0_storage:/source -v $(pwd):/backup alpine tar czf /backup/qdrant-backup.tar.gz /source

# Restore vector storage
docker run --rm -v openmemory_mem0_storage:/target -v $(pwd):/backup alpine tar xzf /backup/qdrant-backup.tar.gz -C /target --strip-components=1
```

## Development & Testing

### Local Development Setup
```bash
# Install dependencies
cd api/
pip install -r requirements.txt

# Run without Docker
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
python main.py
```

### Testing MCP Tools
```bash
# Create test script
cat > test_mcp.py << 'EOF'
import asyncio
import sys
sys.path.append('/path/to/api')

from app.mcp_server import add_memories, search_memory, list_memories
from app.mcp_server import user_id_var, client_name_var

async def test_all():
    # Set context
    user_token = user_id_var.set('test_user')
    client_token = client_name_var.set('test_client')
    
    # Test operations
    print("Adding memory...")
    result = await add_memories("I love coffee and tea")
    print(f"Add result: {result}")
    
    print("Searching memory...")
    result = await search_memory("beverages")
    print(f"Search result: {result}")
    
    print("Listing memories...")
    result = await list_memories()
    print(f"List result: {result}")
    
    # Cleanup context
    user_id_var.reset(user_token)
    client_name_var.reset(client_token)

asyncio.run(test_all())
EOF

python test_mcp.py
```

## Monitoring & Maintenance

### Health Checks
```bash
# Create monitoring script
cat > health_check.sh << 'EOF'
#!/bin/bash
echo "Checking OpenMemory Health..."

# Check services
docker compose ps | grep -q "Up.*openmemory-mcp" && echo "✅ MCP Server: Running" || echo "❌ MCP Server: Down"
docker compose ps | grep -q "Up.*mem0_store" && echo "✅ Qdrant: Running" || echo "❌ Qdrant: Down"

# Check Ollama
curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ Ollama: Running" || echo "❌ Ollama: Down"

# Test MCP endpoint
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8765/docs)
[[ $response == "200" ]] && echo "✅ MCP API: Accessible" || echo "❌ MCP API: Not accessible"

# Test memory functionality
result=$(curl -s -X POST "http://localhost:8765/mcp/direct/test_client/test_user" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"health","method":"tools/call","params":{"name":"list_memories","arguments":{}}}' | \
  jq -r '.result // "error"')

[[ $result != "error" ]] && echo "✅ Memory System: Functional" || echo "❌ Memory System: Error"
EOF

chmod +x health_check.sh
./health_check.sh
```

### Log Analysis
```bash
# View detailed logs
docker compose logs openmemory-mcp -f

# Filter for errors
docker compose logs openmemory-mcp | grep -i error

# Monitor memory operations
docker compose logs openmemory-mcp | grep -E "(add_memories|search_memory|list_memories)"
```

## Cost Analysis

### Ollama (Local) - Recommended
- **Setup Cost**: Time to download models (~4GB total)
- **Runtime Cost**: Local compute resources only
- **Privacy**: Complete data locality
- **Performance**: Fast after initial setup

### OpenAI (Cloud)
- **Setup Cost**: Minimal
- **Runtime Cost**: ~$0.50-1.50 per 1000 memory operations
- **Privacy**: Data sent to OpenAI
- **Performance**: Network dependent

## Future Enhancements

### Planned Features
- [ ] Multi-modal memory support (images, documents)
- [ ] Memory categories and tagging
- [ ] Automatic memory expiration
- [ ] Memory sharing between users
- [ ] Advanced search with filters
- [ ] Memory analytics dashboard

### Contributing
- Report issues via GitHub Issues
- Submit pull requests for improvements
- Suggest enhancements in Discussions

---

**Status**: ✅ OpenMemory MCP integration is fully functional and battle-tested!

**Last Updated**: June 2025
**Tested With**: Claude Code v1.9+, Ollama v0.4+, Qdrant v1.14+