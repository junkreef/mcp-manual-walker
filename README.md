# MCP Manual Walker

An MCP server to bridge the information gap between AI agents and PDF manuals.

This server scans a directory for PDF files, extracts their bookmarks, and provides tools for an AI agent to access the content of the manuals in a structured, token-efficient way.

## Features

- Automatic PDF discovery
- Bookmark-based Markdown conversion
- Intelligent caching
- Tools for AI agents (`list_manuals`, `get_manual_metadata`, `get_markdown_content`)

## Tech Stack

- Python 3.11+
- FastMCP
- pypdf
- markitdown
- SQLAlchemy
- Pydantic

## Getting Started

1.  Install dependencies: `uv pip install -r requirements.txt`
2.  Place your PDF manuals in the `data/pdfs` directory.
3.  Run the server: `python src/mcp_manual_walker/main.py`