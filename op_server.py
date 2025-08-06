from fastmcp import FastMCP
from pathlib import Path
from typing import List

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# === Init MCP Server ===
mcp = FastMCP("Research Operations")

# === Constants ===
VECTOR_DB_ROOT = Path("vector_dbs/")

# === Resource: List of all available vector DBs ===


@mcp.resource("vector://list")
def list_vector_databases() -> str:
    """
    Returns a newline-separated list of available vector DBs (paths).
    Each represents a subfolder under vector_dbs/.
    """
    if not VECTOR_DB_ROOT.exists():
        return "(No vector databases found)"

    vector_dbs = [
        f"DB Name: {p.name}"
        for p in VECTOR_DB_ROOT.iterdir()
        if (p / "index.faiss").exists()
    ]
    return "\n".join(vector_dbs)

# === Prompt: Research instructions ===


@mcp.prompt()
def research_prompt(topic: str) -> str:
    """Provides a structured deep research prompt on a given topic."""
    return f"""
You are an AI researcher. Conduct a deep investigation into the topic: **{topic}**.

Provide:
1. Definition and scope
2. Current state of research
3. Relevant papers and datasets
4. Potential future directions

Use advanced terminology and ensure scientific rigor.
"""

# === Tool: Save Documents to Embedding Store ===


@mcp.tool()
def save_embeddings(docs: List[str], path: str = "default") -> str:
    """
    Save documents to FAISS DB under vector_dbs/{path}/.
    Automatically creates the directory if it doesn't exist.
    """
    target_path = VECTOR_DB_ROOT / path
    target_path.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  api_key="sk-...")
    documents = [Document(page_content=doc) for doc in docs]
    index_file = target_path / "index.faiss"

    if index_file.exists():
        vectorstore = FAISS.load_local(
            str(target_path), embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(documents)
        vectorstore.save_local(str(target_path))
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(str(target_path))

    return f"Saved {len(docs)} documents to vector_dbs/{path}/"


@mcp.tool()
def semantic_search(query: str, path: str = "default") -> List[str]:
    """
    Perform semantic search over FAISS DB at vector_dbs/{path}/.
    """
    target_path = VECTOR_DB_ROOT / path
    index_file = target_path / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(f"No index found at vector_dbs/{path}/")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  api_key="sk-...")
    vectorstore = FAISS.load_local(
        str(target_path), embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=5)

    return [r.page_content for r in results]


@mcp.tool()
def available_prompts():
    """Lists all prompts available with the server. Give the EXACT NAME of the prompt in response to the user."""
    # invoked when the user wants to know what prompts are available
    data = mcp.get_prompts()
    return data


# === Run the server ===
if __name__ == "__main__":
    mcp.run()
