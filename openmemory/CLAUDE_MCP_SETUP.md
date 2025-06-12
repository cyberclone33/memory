# Claude MCP Setup for OpenMemory

This guide shows how to configure Claude to use the OpenMemory MCP server for persistent memory functionality.

## Prerequisites

✅ OpenMemory MCP server running (completed above)
- Server: http://localhost:8765
- Vector Store: http://localhost:6333  
- Web UI: http://localhost:3000

## Configuration Steps

### 1. Create MCP Configuration

Create or update your Claude MCP configuration file:

**Location**: `~/.config/claude-desktop/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "openmemory": {
      "command": "node",
      "args": ["-e", "console.log('OpenMemory MCP Server via SSE')"],
      "env": {
        "USER_ID": "claude_user",
        "CLIENT_NAME": "claude_assistant"
      },
      "transport": {
        "type": "sse", 
        "url": "http://localhost:8765/mcp/claude_assistant/sse/claude_user"
      }
    }
  }
}
```

### 2. Alternative Configuration (if SSE transport not supported)

If your Claude version doesn't support SSE transport, use stdio with a proxy script:

```json
{
  "mcpServers": {
    "openmemory": {
      "command": "python3",
      "args": ["/Users/jarvis/Desktop/FUN/memory/openmemory/mcp_proxy.py"],
      "env": {
        "USER_ID": "claude_user",
        "CLIENT_NAME": "claude_assistant",
        "MCP_SERVER_URL": "http://localhost:8765"
      }
    }
  }
}
```

### 3. MCP Proxy Script (if needed)

Create `mcp_proxy.py` for stdio transport:

```python
#!/usr/bin/env python3
"""
MCP Proxy for OpenMemory - Converts stdio to HTTP/SSE
"""
import json
import sys
import requests
import os
from urllib.parse import urljoin

class MCPProxy:
    def __init__(self):
        self.server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8765')
        self.user_id = os.getenv('USER_ID', 'claude_user')
        self.client_name = os.getenv('CLIENT_NAME', 'claude_assistant')
        self.base_url = f"{self.server_url}/mcp/{self.client_name}/sse/{self.user_id}"
    
    def handle_request(self, request):
        """Forward MCP request to OpenMemory server"""
        try:
            response = requests.post(
                f"{self.base_url}/messages/",
                json=request,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    return response.json()
                except:
                    return {"status": "ok", "result": response.text}
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            return {"error": str(e)}
    
    def run(self):
        """Main stdio loop"""
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = self.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
            except Exception as e:
                error_response = {"error": f"Proxy error: {e}"}
                print(json.dumps(error_response))
                sys.stdout.flush()

if __name__ == "__main__":
    proxy = MCPProxy()
    proxy.run()
```

## Available MCP Tools

Once configured, Claude will have access to these memory tools:

### 1. `add_memories`
**Description**: Add new memories when users share information about themselves, preferences, or anything relevant for future conversations.

**Usage**: Called automatically when users provide personal information.

**Example**:
```
User: "I love playing guitar and have been learning for 5 years"
→ Calls add_memories("I love playing guitar and have been learning for 5 years")
```

### 2. `search_memory` 
**Description**: Search through stored memories. Called EVERY TIME users ask questions.

**Usage**: Automatically searches relevant memories before responding.

**Example**:
```
User: "What instruments do I play?"
→ Calls search_memory("instruments play music")
→ Finds: "I love playing guitar and have been learning for 5 years"
```

### 3. `list_memories`
**Description**: List all stored memories for the user.

**Usage**: When users want to see what Claude remembers about them.

### 4. `delete_all_memories`
**Description**: Delete all memories (use with caution).

**Usage**: When users want to reset their memory profile.

## Testing the Configuration

### 1. Restart Claude Desktop
After saving the configuration, restart Claude Desktop application.

### 2. Test Memory Functions
Try these commands in Claude:

```
"Remember that I prefer Python over JavaScript for backend development"
```

```
"What programming languages do I prefer?"
```

```
"What do you remember about me?"
```

### 3. Verify in OpenMemory Dashboard
Check the web UI at http://localhost:3000 to see if memories are being stored.

## Troubleshooting

### Claude Not Connecting
1. Check if MCP server is running: `curl http://localhost:8765/docs`
2. Verify configuration file location and syntax
3. Check Claude Desktop logs for MCP errors
4. Ensure no firewall blocking localhost connections

### Memory Operations Failing
1. Check server logs: `docker compose logs openmemory-mcp`
2. Verify environment variables (OPENAI_API_KEY)
3. Test direct API: `curl http://localhost:8765/openapi.json`
4. Check Qdrant vector store: `curl http://localhost:6333/collections`

### SSE Transport Issues
1. Use stdio transport with proxy script instead
2. Check if Claude version supports SSE
3. Try alternative MCP client tools for testing

## Advanced Configuration

### Custom User/Client Names
Modify the configuration to use specific identifiers:

```json
{
  "mcpServers": {
    "openmemory": {
      "env": {
        "USER_ID": "your_unique_user_id",
        "CLIENT_NAME": "your_app_name"
      },
      "transport": {
        "url": "http://localhost:8765/mcp/your_app_name/sse/your_unique_user_id"
      }
    }
  }
}
```

### Multiple Memory Profiles
Configure different memory contexts for different use cases:

```json
{
  "mcpServers": {
    "openmemory-personal": {
      "transport": {
        "url": "http://localhost:8765/mcp/personal/sse/user123"
      }
    },
    "openmemory-work": {
      "transport": {
        "url": "http://localhost:8765/mcp/work/sse/user123"
      }
    }
  }
}
```

## Security Notes

- Server runs on localhost only (not exposed externally)
- Each user/client combination has isolated memory space
- Access permissions enforced at database level
- All operations logged for audit trail
- HTTPS recommended for production deployments

## Next Steps

1. Configure Claude with MCP settings above
2. Test memory functionality
3. Monitor memory storage in web dashboard
4. Customize user/client identifiers as needed
5. Consider backup strategies for memory data

---

**Status**: MCP server is running and ready for Claude integration! 🚀