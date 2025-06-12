# OpenMemory - Ollama Embeddings Configuration Guide

This document provides a comprehensive guide on configuring OpenMemory to use Ollama embeddings instead of OpenAI embeddings.

## Overview

OpenMemory has been successfully configured to use Ollama for text embeddings, providing:
- **Local embedding generation** - No external API calls required
- **Privacy** - All embedding processing happens locally
- **Cost efficiency** - No per-token charges for embeddings
- **Performance** - Direct local processing without network latency

## Configuration Changes Made

### Before (OpenAI Embeddings)
```json
{
    "mem0": {
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": "env:API_KEY"
            }
        }
    }
}
```

### After (Ollama Embeddings)
```json
{
    "mem0": {
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "ollama_base_url": "http://localhost:11434"
            }
        }
    }
}
```

## Files Modified

### 1. Main Configuration File
**File:** `/api/config.json`
- **Provider:** `openai` → `ollama`
- **Model:** `text-embedding-3-small` → `nomic-embed-text`
- **URL:** OpenAI API → `http://localhost:11434`

### 2. Default Configuration Template
**File:** `/api/default_config.json`
- Same changes as above
- Used as template for new configurations

## Embedding Model Details

| Aspect | OpenAI | Ollama |
|--------|--------|--------|
| **Provider** | OpenAI | Ollama (Local) |
| **Model** | text-embedding-3-small | nomic-embed-text |
| **Dimensions** | 1536 | 768 |
| **Cost** | $0.00002 per 1K tokens | Free (local) |
| **Privacy** | Data sent to OpenAI | Fully local |
| **Network** | Requires internet | Local only |

## Prerequisites

### 1. Ollama Installation
Ensure Ollama is installed and running on your host machine:

```bash
# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.ai

# Start Ollama service
ollama serve
```

### 2. Required Model
The `nomic-embed-text` model will be automatically pulled when first used, or you can pre-download it:

```bash
ollama pull nomic-embed-text
```

### 3. Python Dependencies
Install the Ollama Python client:

```bash
pip install ollama
```

## Verification and Testing

### Quick Configuration Test
Run the provided test script to verify the configuration:

```bash
cd /path/to/openmemory
python3 test_embedding_change.py
```

Expected output:
```
🎉 SUCCESS: OpenMemory has been successfully configured to use Ollama embeddings!
```

### Manual Verification Steps

1. **Check Ollama Service:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Test Direct Embedding:**
   ```bash
   curl -X POST http://localhost:11434/api/embeddings \
        -H "Content-Type: application/json" \
        -d '{"model": "nomic-embed-text", "prompt": "test text"}'
   ```

3. **Verify Configuration:**
   ```bash
   grep -A 10 "embedder" api/config.json
   ```

## Architecture Changes

### Data Flow Before (OpenAI)
```
OpenMemory → OpenAI API → text-embedding-3-small → 1536-dim vectors
```

### Data Flow After (Ollama)
```
OpenMemory → Local Ollama → nomic-embed-text → 768-dim vectors
```

## Performance Considerations

### Advantages of Ollama Embeddings
- **Privacy:** All data processing happens locally
- **Cost:** No API charges for embedding generation
- **Latency:** No network round-trip time
- **Availability:** Works offline

### Considerations
- **Local Resources:** Uses local CPU/GPU for processing
- **Model Size:** Requires local storage for embedding model (~2GB)
- **Initial Setup:** First-time model download required

## Troubleshooting

### Common Issues

#### 1. Ollama Service Not Running
**Error:** `Cannot connect to Ollama service`

**Solution:**
```bash
# Start Ollama service
ollama serve

# Or check if it's running
ps aux | grep ollama
```

#### 2. Model Not Found
**Error:** `model "nomic-embed-text" not found`

**Solution:**
```bash
# Pull the model manually
ollama pull nomic-embed-text

# List available models
ollama list
```

#### 3. Connection Refused
**Error:** `Connection refused on localhost:11434`

**Solution:**
- Check if Ollama is running: `ollama serve`
- Verify port is not blocked by firewall
- Try alternative URL: `http://127.0.0.1:11434`

#### 4. Python Import Error
**Error:** `ModuleNotFoundError: No module named 'ollama'`

**Solution:**
```bash
pip install ollama
# or
pip3 install ollama --break-system-packages
```

### Configuration Verification

Run this command to verify your configuration:
```bash
python3 -c "
import json
with open('api/config.json') as f:
    config = json.load(f)
    embedder = config['mem0']['embedder']
    print(f'Provider: {embedder[\"provider\"]}')
    print(f'Model: {embedder[\"config\"][\"model\"]}')
    print(f'URL: {embedder[\"config\"][\"ollama_base_url\"]}')
"
```

## Alternative Ollama Models

If you want to use a different embedding model, here are some options:

| Model | Dimensions | Size | Use Case |
|-------|------------|------|----------|
| nomic-embed-text | 768 | ~2GB | General purpose (default) |
| all-minilm | 384 | ~100MB | Lightweight |
| mxbai-embed-large | 1024 | ~1GB | High performance |

To change models, update the configuration:
```json
{
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "all-minilm",  // Change this
            "ollama_base_url": "http://localhost:11434"
        }
    }
}
```

## Docker Considerations

If running OpenMemory in Docker and Ollama on the host:

### Docker Compose Configuration
```yaml
services:
  openmemory:
    # ... other config
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
```

### Alternative Host URLs for Docker
- `http://host.docker.internal:11434` (Docker Desktop)
- `http://172.17.0.1:11434` (Linux Docker)
- `http://192.168.1.100:11434` (Replace with your host IP)

## Migration from OpenAI

### Data Compatibility
- **Existing embeddings:** OpenAI embeddings (1536-dim) are incompatible with Ollama embeddings (768-dim)
- **Migration needed:** Re-generate embeddings for existing memories
- **Vector database:** May need to clear and rebuild vector indices

### Migration Script
If you have existing memories, you may need to re-process them:

```python
# Example migration script
from mem0 import Memory

# Initialize with new Ollama configuration
memory = Memory.from_config(config)

# Re-add existing memories to generate new embeddings
for old_memory in existing_memories:
    memory.add(old_memory.text, user_id=old_memory.user_id)
```

## Monitoring and Maintenance

### Log Monitoring
Check Ollama logs for embedding requests:
```bash
# Ollama typically logs to console when running with 'ollama serve'
ollama serve --log-level debug
```

### Performance Monitoring
Monitor local resource usage:
```bash
# CPU usage
top -p $(pgrep ollama)

# Memory usage
ps aux | grep ollama
```

### Model Updates
Keep your embedding model updated:
```bash
# Check for model updates
ollama pull nomic-embed-text

# List local models with versions
ollama list
```

## Security Considerations

### Benefits
- **Data Privacy:** No data sent to external services
- **API Key Security:** No OpenAI API key required
- **Network Security:** No external network dependencies

### Best Practices
- Keep Ollama service updated
- Restrict network access to Ollama port (11434) if needed
- Monitor local resource usage
- Regular security updates for the host system

## Support and Resources

### Documentation
- [Ollama Official Documentation](https://ollama.ai/docs)
- [nomic-embed-text Model Page](https://ollama.ai/library/nomic-embed-text)
- [mem0 Embedding Documentation](https://docs.mem0.ai/components/embedders/ollama)

### Community
- [Ollama GitHub](https://github.com/ollama/ollama)
- [mem0 GitHub](https://github.com/mem0ai/mem0)

### Testing
Use the provided test scripts:
- `test_embedding_change.py` - Quick configuration verification
- `test_openmemory_ollama_config.py` - Comprehensive testing
- `test_ollama_embeddings.py` - Direct Ollama embedding tests

---

## Summary

✅ **OpenMemory is now configured to use Ollama embeddings**

**Key Changes:**
- Embedder provider: OpenAI → Ollama
- Model: text-embedding-3-small → nomic-embed-text  
- Processing: Cloud API → Local machine
- Cost: Pay-per-use → Free
- Privacy: External → Fully local

**Files Modified:**
- `/api/config.json`
- `/api/default_config.json`

**Verification:** Run `python3 test_embedding_change.py` to confirm everything is working correctly.

---

*Last updated: January 2025*  
*Configuration tested and verified working*