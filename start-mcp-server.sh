#!/bin/bash
# Start marimo MCP server for Claude Code integration
# This enables AI assistants to interact with your marimo notebooks

# Configuration
PORT=2718
NOTEBOOK="fpl_team_picker/interfaces/ml_xp_notebook.py"
LOG_FILE="/tmp/marimo-mcp.log"

# Check if port is already in use
if lsof -ti :$PORT > /dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use"
    echo "Current process:"
    lsof -i :$PORT
    echo ""
    read -p "Kill existing process and restart? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti :$PORT | xargs kill
        sleep 2
    else
        exit 1
    fi
fi

# Start the MCP server
echo "🚀 Starting Marimo MCP server..."
echo "   Port: $PORT"
echo "   Notebook: $NOTEBOOK"
echo "   Log: $LOG_FILE"
echo ""

nohup uv run marimo edit "$NOTEBOOK" \
    --headless \
    --no-token \
    --port $PORT \
    --mcp \
    > "$LOG_FILE" 2>&1 &

# Wait for server to start
sleep 3

# Check if server started successfully
if lsof -ti :$PORT > /dev/null 2>&1; then
    echo "✅ Marimo MCP server started successfully!"
    echo ""
    echo "   MCP URL: http://localhost:$PORT/mcp/server"
    echo "   Web UI: http://localhost:$PORT"
    echo ""
    echo "📝 The server is running in the background."
    echo "   View logs: tail -f $LOG_FILE"
    echo "   Check status: claude mcp list"
    echo ""
else
    echo "❌ Failed to start server. Check logs:"
    cat "$LOG_FILE"
    exit 1
fi
