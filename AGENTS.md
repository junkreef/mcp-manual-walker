# AGENTS.md

This document provides a complete overview of the `mcp-manual-walker` project, including its purpose, architecture, and the tools and guidelines it provides for AI agents.

---

## Product Requirements Document: mcp-manual-walker

### 1. Overview

#### 1.1. Product Purpose

This product, "mcp-manual-walker," is an MCP (Multi-purpose Cooperative Processor) server designed to bridge the information gap between AI agents and PDF manuals stored on a local file system.

Its core purpose is to enable an AI agent to efficiently identify and extract only the truly necessary sections from large PDF files, using bookmarks as a guide, and include them in its limited context window. This prevents context contamination, improves response accuracy, and optimizes token consumption.

#### 1.2. Key Features

*   **Automatic PDF Discovery**: Automatically scans a specified directory for PDF files upon server startup, synchronizing the database by adding new files, updating modified ones, and removing deleted ones.
*   **Bookmark-based Markdown Conversion**: Converts text to Markdown on a per-bookmark basis using the `markitdown` library.
*   **Intelligent Caching Mechanism**: Caches the converted Markdown content locally. It monitors the hash value of the PDF files to reconvert content only when a file has been updated, ensuring high-speed processing.
*   **Tools for AI Agents**: Provides three dedicated tools (APIs) for AI agents to utilize the content of the PDFs.

### 2. System Architecture

The server consists of the following components:

*   **Server Framework**: `FastMCP`, enabling interaction with AI agents via tool calls (function calls).
*   **Data Source**: A collection of PDF files stored in a specific directory on the local file system.
*   **Database**: `SQLite3`, used to centrally manage PDF file metadata, hierarchical bookmark information, and cache status.
*   **Cache Storage**: A designated directory on the local file system for storing the converted Markdown files.
*   **Client**: An AI agent that accesses this server through the `FastMCP` tool-calling interface.

### 3. Functional Requirements

#### 3.1. Server Initialization Process

The server automatically performs the following actions on startup:

1.  **Scan for PDF Files**: Recursively scans the PDF root directory specified in the configuration file to list all available PDF files.
2.  **Update Database**:
    *   Calculates the SHA256 hash value for each PDF file found on the filesystem.
    *   Compares it with the information in the database to identify newly added, updated, or deleted PDFs.
3.  **Extract Metadata and Bookmarks**:
    *   For new or updated PDFs, it uses the `pypdf` library to extract:
        *   The document title from the PDF metadata.
        *   Bookmark information (title, level, page number) while preserving the hierarchical structure.
    *   Saves the extracted information, along with the relative file path from the root directory, into the database.
4.  **Synchronize Deletions**: Removes records from the database that correspond to PDF files no longer present in the filesystem.
5.  **Check Cache Integrity**: Marks related old cache entries as invalid if a PDF has been updated.

#### 3.2. Tools for AI Agents (APIs)

The agent interacts with the system using stable, unique IDs for manuals and bookmarks. The typical workflow is:
1. Call `list_manuals()` to see all available manuals and get their `id`s.
2. Call `get_manual_metadata(manual_id)` to get the table of contents for a specific manual, which includes the `id` for each bookmark.
3. Call `get_markdown_content(bookmark_id)` to retrieve the content of a specific section.

---

**Tool 1: `list_manuals()`**

*   **Function**: Returns a list of all available manuals.
*   **Input**: None
*   **Output**: A list of objects, where each object contains the manual's unique ID, filename, and document title.
*   **Example**:
    ```json
    [
      {
        "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "file_name": "manual_v1.2.pdf",
        "document_title": "Operator's Manual v1.2"
      },
      {
        "id": "98b6f7a8-3c3e-4b4a-9a8f-9a8b7c6d5e4f",
        "file_name": "troubleshooting.pdf",
        "document_title": "Troubleshooting Guide"
      }
    ]
    ```

**Tool 2: `get_manual_metadata(manual_id: str)`**

*   **Function**: Returns metadata and hierarchical bookmark information (table of contents) for a specified manual.
*   **Input**:
    *   `manual_id` (string): The unique ID of the manual to query, obtained from `list_manuals()`.
*   **Output**: A JSON object containing the manual's metadata and a nested table of contents. Each bookmark in the TOC has its own unique ID.
*   **Example**:
    ```json
    {
      "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "file_name": "manual_v1.2.pdf",
      "document_title": "Operator's Manual v1.2",
      "file_hash": "a1b2c3d4...",
      "table_of_contents": [
        {
          "id": "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
          "title": "Chapter 1: Introduction",
          "page": 1,
          "children": [
            {
              "id": "b1c2d3e4-f5a6-b7c8-d9e0-f1a2b3c4d5e6",
              "title": "1.1 Scope",
              "page": 2,
              "children": []
            }
          ]
        }
      ]
    }
    ```

**Tool 3: `get_markdown_content(bookmark_id: str)`**

*   **Function**: Returns the Markdown content for a specific bookmark within a manual.
*   **Input**:
    *   `bookmark_id` (string): The unique ID of the bookmark to retrieve, obtained from `get_manual_metadata()`.
*   **Process Flow**: If a valid cache exists, it returns the cached content. Otherwise, it extracts the relevant page range from the PDF, converts it to Markdown, caches it, and then returns the content.
*   **Output**: A string containing the Markdown text.

### 4. Data Model (SQLite3)

**`manuals` Table**: Manages information about the PDF files.
*   `id` (STRING(36), PK): Unique identifier (UUID).
*   `file_name` (TEXT, UNIQUE): The filename.
*   `document_title` (TEXT): The document title from PDF metadata (can be NULL).
*   `relative_path` (TEXT): The relative path from the root directory.
*   `file_hash` (TEXT): The SHA256 hash of the file.
*   `updated_at` (TIMESTAMP): Last modified timestamp.

**`bookmarks` Table**: Manages hierarchical bookmark data.
*   `id` (STRING(36), PK): Unique identifier (UUID).
*   `manual_id` (STRING(36), FK): Reference to `manuals.id`.
*   `ordering` (INTEGER): The original sort order of the bookmark within the PDF.
*   `title` (TEXT): The bookmark title.
*   `level` (INTEGER): The depth level in the hierarchy.
*   `page_num` (INTEGER): The corresponding page number.
*   `parent_id` (STRING(36), FK): The ID of the parent bookmark (`bookmarks.id`).

**`cache` Table**: Manages the Markdown cache.
*   `id` (STRING(36), PK): Unique identifier (UUID).
*   `bookmark_id` (STRING(36), FK): Reference to `bookmarks.id`.
*   `manual_hash` (TEXT): The PDF hash at the time the cache was created.
*   `markdown_file_path` (TEXT): The path to the cached Markdown file.
*   `created_at` (TIMESTAMP): Cache creation timestamp.

### 5. Technology Stack 🛠️

*   **Programming Language**: Python 3.11+
*   **Virtual Environment**: uv
*   **Server Framework**: FastMCP
*   **PDF to Markdown Conversion**: `markitdown[pdf]`
*   **PDF Metadata/Bookmark Extraction**: `pypdf`
*   **Database**:
    *   Engine: SQLite3 (Python standard library)
    *   ORM: SQLAlchemy
*   **Configuration Management**: Pydantic, `pydantic-settings`
*   **Hashing**: `hashlib` (Python standard library)

---

## Agent Development Guide

This guide provides essential information for an AI agent to effectively contribute to this project.

### 1. Environment Setup

1.  **Install Dependencies**: This project uses `uv`. Install all dependencies from the lock file.
    ```bash
    uv sync
    ```
2.  **Configure Environment**: Create a `.env` file in the project root. The application uses this for configuration. You can usually start with the default settings.
    ```.env
    PDF_ROOT_DIR=./data/pdfs
    DB_FILE_PATH=./data/mcp_manual_walker.db
    CACHE_DIR=./cache
    LOG_LEVEL=INFO
    ```

### 2. Committing Changes

*   **Language**: All commit messages must be in **English**.
*   **Format**: Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This makes the commit history easier to read and allows for automated versioning and changelog generation.
    *   **`feat`**: A new feature
    *   **`fix`**: A bug fix
    *   **`docs`**: Documentation only changes
    *   **`refactor`**: A code change that neither fixes a bug nor adds a feature
    *   **`test`**: Adding missing tests or correcting existing tests
    *   **`chore`**: Changes to the build process or auxiliary tools

### 3. Running the Application

*   To start the server, run the `main.py` script. It will be accessible at `http://0.0.0.0:8000`.
    ```bash
    python src/mcp_manual_walker/main.py
    ```

### 4. Quality & Verification

Before finalizing changes, always run the following commands to ensure code quality and prevent regressions.

1.  **Code Style & Comments**:
    *   All Python code must adhere to [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/). `ruff` is used to enforce this.
    *   All comments must be written in **English**.
    *   Write clear and sufficient comments to explain the *why* behind complex or non-obvious code, not just the *what*.

2.  **Linting & Formatting**: Use `ruff` to check for style issues and automatically fix them.
    ```bash
    # Check for issues
    ruff check .

    # Fix automatically
    ruff check . --fix
    ```
3.  **Type Checking**: Use `mypy` to perform static type analysis.
    ```bash
    mypy src/
    ```
4.  **Testing**: This project uses `pytest` for testing. Run all unit and integration tests.
    ```bash
    pytest
    ```

### 5. Project Structure Overview

*   `src/mcp_manual_walker/main.py`: Main application entry point. Defines the server (`FastMCP`) and the tools available to the agent. Contains the core database synchronization logic.
*   `src/mcp_manual_walker/models.py`: Defines the SQLAlchemy database schema (`Manual`, `Bookmark`, `Cache`). **Crucially, it defines the `cascade="all, delete-orphan"` behavior.**
*   `src/mcp_manual_walker/database.py`: Handles database engine creation and session management.
*   `src/mcp_manual_walker/config.py`: Defines application settings using Pydantic, loaded from the `.env` file.
*   `src/mcp_manual_walker/pdf_utils.py`: Contains all logic for interacting with PDF files (hashing, metadata/bookmark extraction, creating temporary files from page ranges).
*   `src/mcp_manual_walker/cache_utils.py`: Manages the lifecycle of cache entries, including finding valid cache and creating new ones.
*   `data/`: Default directory for the SQLite database and the source PDF manuals.
*   `cache/`: Default directory for storing generated Markdown cache files.

### 6. Key Architectural Decisions

*   **ID-based Referencing**: The system uses UUIDs (`manual_id`, `bookmark_id`) instead of mutable names (`file_name`, `bookmark_title`) for all tool inputs and database relations. This ensures stability and prevents broken references if a file or bookmark is renamed.
*   **Database Cascade Deletes**: The data models are configured with `cascade="all, delete-orphan"`. This means deleting a `Manual` record from the database will automatically trigger the deletion of all its associated `Bookmark` and `Cache` records, ensuring data integrity.
