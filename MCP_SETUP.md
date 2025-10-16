# Marimo MCP Server Setup

This document explains how to set up and use the Marimo MCP (Model Context Protocol) server for AI assistant integration with your FPL notebooks.

## What is MCP?

The Model Context Protocol (MCP) enables AI assistants like Claude Code, Cursor, and VS Code Copilot to interact with your marimo notebooks programmatically. This allows AI assistants to:

- Read and understand notebook context
- Access marimo documentation
- Execute code and view results
- Interact with notebook state

## Quick Start

```bash
./start-mcp-server.sh
```

This script will:
1. Check if port 2718 is available
2. Start marimo in edit mode with MCP enabled
3. Verify the server started successfully
4. Display connection information

## Manual Setup

If you prefer to start the server manually:

```bash
uv run marimo edit fpl_team_picker/interfaces/ml_xp_notebook.py \
    --headless \
    --no-token \
    --port 2718 \
    --mcp
```

## Configuration

### Add to Claude Code

```bash
claude mcp add --transport http marimo http://localhost:2718/mcp/server
```

### Check Status

```bash
claude mcp list
```

You should see:
```
marimo: http://localhost:2718/mcp/server (HTTP) - ✓ Connected
```

### Remove Server

```bash
claude mcp remove marimo
```

## Requirements

- `marimo[mcp]>=0.17.0` (installed via pyproject.toml)
- `mcp` package (automatically installed with marimo[mcp])
- Port 2718 available

## Important Notes

1. **Edit Mode Only**: MCP server only works with `marimo edit`, NOT `marimo run`
2. **Hidden Flag**: The `--mcp` flag is hidden in the CLI help but is required
3. **Authentication**: Using `--no-token` for local development (add authentication for production)
4. **Watch Mode**: MCP automatically enables watch mode (file changes auto-reload)

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 2718
lsof -i :2718

# Kill the process
lsof -ti :2718 | xargs kill
```

### MCP Server Not Responding

1. Check if server is running in edit mode:
   ```bash
   ps aux | grep "marimo edit"
   ```

2. Verify --mcp flag was used:
   ```bash
   ps aux | grep "\-\-mcp"
   ```

3. Check server logs:
   ```bash
   tail -f /tmp/marimo-mcp.log
   ```

### Connection Failed

1. Verify server is listening:
   ```bash
   curl -s http://localhost:2718/mcp/server
   ```

   Should return a JSON-RPC error about content type (this is expected)

2. Test with Claude Code:
   ```bash
   claude mcp list
   ```

## Configuration Files

- **Marimo Config**: `~/.config/marimo/marimo.toml`
  - Added: `mcp = true` under `[server]` section

- **Project Config**: `pyproject.toml`
  - Updated: `marimo>=0.17.0` to `marimo[mcp]>=0.17.0`

## Logs

Server logs are written to `/tmp/marimo-mcp.log`

```bash
# View logs
tail -f /tmp/marimo-mcp.log

# Search for errors
grep -i error /tmp/marimo-mcp.log
```

## Background Process

The server runs as a background process. To stop it:

```bash
# Find process
ps aux | grep "marimo edit.*2718"

# Stop by port
lsof -ti :2718 | xargs kill
```

## API Endpoints

Once running, the following endpoints are available:

- Web UI: `http://localhost:2718`
- MCP Server: `http://localhost:2718/mcp/server`
- API Status: `http://localhost:2718/api/status`

## Available MCP Tools

The marimo MCP server exposes various tools for AI assistants:

- Code execution
- Notebook inspection
- Variable access
- Documentation lookup
- State management

See the marimo documentation for a complete list of available tools.

## References

- [Marimo Documentation](https://docs.marimo.io/guides/editor_features/mcp/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Claude Code MCP Guide](https://docs.claude.com/en/docs/claude-code/mcp)
