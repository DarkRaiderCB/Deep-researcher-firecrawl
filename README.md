# Deep Researcher

A sophisticated AI-powered research assistant that leverages the Model Context Protocol (MCP) to provide comprehensive research capabilities through multiple integrated tools and services.

**Note**: This project is in early development stages and may undergo significant changes.

**Credits**: This project uses the [Firecrawl MCP server](https://github.com/mendableai/firecrawl-mcp-server) for web crawling and content extraction.

## Overview

Deep Researcher is an intelligent research assistant that combines the power of Large Language Models with specialized tools and data sources. It uses the Model Context Protocol (MCP) architecture to integrate multiple services including:

- **Vector Database Operations**: Store, index, and semantically search research documents
- **Web Crawling**: Extract and analyze content from web sources using Firecrawl
- **Structured Research**: Conduct structured research with customizable prompts
- **Data/Information Management**: Load and utilize various data resources dynamically

## Architecture

The system consists of three main components:

1. **Client Application** (`client.py`): Interactive CLI interface that orchestrates research workflows
2. **Operations Server** (`op_server.py`): MCP server providing vector database and research tools
3. **External Services**: Integration with Firecrawl MCP server for web content extraction

## Features

### Research Capabilities

- **Semantic Search**: Query indexed documents using vector embeddings
- **Document Management**: Store and organize research materials in vector databases
- **Structured Research**: Execute pre-defined research prompts with customizable parameters
- **Resource Integration**: Load and utilize external data sources in research queries

### Interactive Commands

- `@resource:<uri>` - Load MCP resources into the session
- `@prompt:<name>` - Execute structured research prompts
- `@use_resource:<uri> <query>` - Query specific loaded resources
- Standard chat interface for general research assistance

### Multi-Format Support

- Simple key-value arguments: `key:value`
- Multiple parameters: `key1:value1, key2:value2`
- JSON format: `{"key1": "value1", "key2": "value2"}`

## Quick Start

### Prerequisites

- Python 3.12 or higher
- Node.js (for Firecrawl MCP server)
- OpenAI API key
- Firecrawl API key (for web crawling)

### Installation

1. **Install Firecrawl MCP server globally:**

   ```bash
   npm install -g firecrawl-mcp
   ```

2. **Clone and setup the project:**

   ```bash
   git clone <repository-url>
   cd deep-researcher
   uv sync
   ```

3. **Activate the virtual environment:**

   ```bash
   # On Windows
   .venv\Scripts\activate

   # On Unix/macOS
   source .venv/bin/activate
   ```

### Configuration

1. **Configure API Keys:**

   Edit the API keys in the source files:

   - `client.py`: Update the OpenAI API key in the `ChatLiteLLM` initialization
   - `client.py`: Update the Firecrawl API key in the `MultiServerMCPClient` configuration
   - `op_server.py`: Update the OpenAI API key in the `OpenAIEmbeddings` initialization

2. **Example Configuration:**

   ```python
   # In client.py
   llm = ChatLiteLLM(model="gpt-4o", api_key="your-openai-api-key")

   # Firecrawl configuration
   "env": {
       "FIRECRAWL_API_KEY": "your-firecrawl-api-key"
   }
   ```

**Note**: Alternatively `.env` file approach can be configured for better security.

### Running the Application

Start the Deep Researcher client:

```bash
uv run client.py
```

## Usage Guide

### Basic Research Query

```
User: What are the latest developments in quantum computing?
```

### Using Resources

```
# Load a resource
User: @resource:vector://list

# Query the resource
User: @use_resource:vector://list what are the available databases?
```

### Executing Research Prompts

```
# Simple prompt
User: @prompt:research_prompt

# Prompt with arguments
User: @prompt:research_prompt
Arguments: {"topic": "artificial intelligence ethics"}
```

### Document Management

The system automatically manages vector databases for semantic search:

- Documents are stored in `vector_dbs/` directory
- Each database is a separate folder with FAISS indices
- Use the operations server tools to save and search documents

## MCP Tools from `op_server.py`

### Operations Server Tools

#### `save_embeddings(docs: List[str], path: str = "default")`

Store documents in a vector database for semantic search.

#### `semantic_search(query: str, path: str = "default")`

Perform semantic search across stored documents.

#### `available_prompts()`

List all available research prompts.

### Client Commands

| Command                       | Description             | Example                                            |
| ----------------------------- | ----------------------- | -------------------------------------------------- |
| `@resource:<uri>`             | Load MCP resource       | `@resource:file://data.txt`                        |
| `@prompt:<name>`              | Execute research prompt | `@prompt:research_prompt`                          |
| `@use_resource:<uri> <query>` | Query loaded resource   | `@use_resource:file://data.txt summarize findings` |
| `exit` or `quit`              | Exit the application    | `exit`                                             |

## Project Structure

```
deep-researcher/
├── client.py              # Main client application
├── op_server.py           # MCP operations server
├── pyproject.toml         # Project dependencies
├── uv.lock               # Dependency lock file
├── README.md             # This file
└── vector_dbs/           # Vector database storage (created at runtime)
    └── default/          # Default database
        ├── index.faiss   # FAISS index
        └── index.pkl     # Metadata
```

## Development

### Dependencies

The project uses the following key dependencies:

- **LangChain**: LLM integration and workflow orchestration
- **LangGraph**: Graph-based AI application framework
- **FastMCP**: Model Context Protocol server implementation
- **FAISS**: Vector similarity search
- **Colorama**: Colored terminal output

## Security Notes

- API keys are currently stored in source code - consider using environment variables
- Vector databases use `allow_dangerous_deserialization=True` - review for production use. Dockerization could help.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues, questions, or contributions, please open an issue in the repository.

---
