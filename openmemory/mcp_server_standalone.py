#!/usr/bin/env python3
"""
Standalone MCP server for OpenMemory.
This script serves as the entry point for the MCP server integration with Claude Code CLI.
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / "api" / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add the API directory to Python path
api_path = Path(__file__).parent / "api"
sys.path.insert(0, str(api_path))

# Set default environment variables for MCP operation
os.environ.setdefault('DATABASE_URL', 'sqlite:///./api/openmemory.db')
os.environ.setdefault('QDRANT_HOST', 'localhost')
os.environ.setdefault('QDRANT_PORT', '6333')
os.environ.setdefault('API_KEY', 'dummy_key_for_mcp')
os.environ.setdefault('OPENAI_API_KEY', 'dummy_key_for_mcp')

# Override config for MCP standalone mode
import json
config_override = {
    "mem0": {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1:latest",
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

# Write config override for MCP
config_path = Path(__file__).parent / "api" / "mcp_config.json"
with open(config_path, 'w') as f:
    json.dump(config_override, f, indent=2)

# Set environment to use the MCP config
os.environ['CONFIG_PATH'] = str(config_path)

# Patch the memory config loading to use our MCP config
def patch_memory_config():
    import sys
    from app.utils import memory
    
    # Store the original function
    original_get_memory_client = memory.get_memory_client
    
    def get_memory_client_mcp():
        """Get memory client with MCP-specific configuration."""
        try:
            from mem0 import Memory
            # Force reinitialize with our config
            memory._memory_client = None
            memory._config_hash = None
            
            # Use our MCP config directly
            client = Memory.from_config(config_override["mem0"])
            print("Memory client initialized with MCP config", file=sys.stderr)
            return client
        except Exception as e:
            print(f"Warning: Failed to initialize memory client: {e}", file=sys.stderr)
            print("Server will continue running with limited memory functionality", file=sys.stderr)
            return None
    
    # Replace the function
    memory.get_memory_client = get_memory_client_mcp
    return memory

# Apply the patch
memory_module = patch_memory_config()

try:
    from api.app.mcp_server import mcp
    print("MCP server imported successfully", file=sys.stderr)
except Exception as e:
    print(f"Error importing MCP server: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

async def main():
    """Main entry point for the MCP server."""
    try:
        print("Starting OpenMemory MCP server...", file=sys.stderr)
        # Use stdin/stdout for MCP communication
        await mcp.run_stdio_async()
    except Exception as e:
        print(f"Error running MCP server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())