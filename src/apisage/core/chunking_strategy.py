"""
Chunking Strategy for OpenAPI Specifications
Implements intelligent chunking of API specs using LangChain's RecursiveJsonSplitter
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveJsonSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class APISpecChunker:
    """
    Advanced chunking strategy for OpenAPI 3.0 specifications.
    Uses LangChain's RecursiveJsonSplitter for semantic chunking.
    """

    def __init__(self, max_chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Initialize the chunker with configuration.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = None

        if LANGCHAIN_AVAILABLE:
            self.splitter = RecursiveJsonSplitter(
                max_chunk_size=max_chunk_size, min_chunk_size=100
            )

    def chunk_api_spec(self, api_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk an OpenAPI specification into semantically meaningful pieces.

        Args:
            api_spec: The OpenAPI specification as a dictionary

        Returns:
            List of chunks with metadata
        """
        if not LANGCHAIN_AVAILABLE:
            return self._fallback_chunking(api_spec)

        try:
            # Convert spec to JSON string for processing
            spec_json = json.dumps(api_spec, indent=2)

            # Use LangChain's recursive splitter with comprehensive error handling
            chunks = []
            try:
                if self.splitter:
                    chunks = self.splitter.split_text(spec_json)
                else:
                    raise RuntimeError("Splitter not initialized")
                    
                # Validate chunks - ensure we got valid results
                if not chunks or len(chunks) == 0:
                    raise ValueError("No chunks generated from JSON splitter")
                    
            except (IndexError, KeyError, ValueError, AttributeError, RuntimeError) as e:
                # Multiple fallback strategies for when RecursiveJsonSplitter fails
                try:
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.max_chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=["\n\n", "\n", "}", "{", ",", " ", ""],
                    )
                    chunks = text_splitter.split_text(spec_json)
                    
                    # If text splitter also fails, use fallback chunking
                    if not chunks:
                        return self._fallback_chunking(api_spec)
                        
                except Exception as fallback_error:
                    # Final fallback to our custom chunking strategy
                    return self._fallback_chunking(api_spec)

            # Process chunks and add metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    # Parse chunk back to dict if possible
                    chunk_data = json.loads(chunk)
                    chunk_type = self._identify_chunk_type(chunk_data)
                except json.JSONDecodeError:
                    # If chunk is not valid JSON, treat as text
                    chunk_data = {"content": chunk}
                    chunk_type = "text_fragment"

                processed_chunks.append(
                    {
                        "id": f"chunk_{i}",
                        "type": chunk_type,
                        "content": chunk_data,
                        "text": chunk,
                        "size": len(chunk),
                        "metadata": {
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "source": "api_specification",
                        },
                    }
                )

            return processed_chunks

        except Exception as e:
            # Final exception handler - use fallback chunking
            try:
                return self._fallback_chunking(api_spec)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"All chunking strategies failed. Primary error: {e}. Fallback error: {fallback_error}. Ensure LangChain is properly installed and configured."
                )

    def _identify_chunk_type(self, chunk_data: Dict[str, Any]) -> str:
        """
        Identify the type of chunk based on its content structure.

        Args:
            chunk_data: The chunk data as a dictionary

        Returns:
            String identifying the chunk type
        """
        if isinstance(chunk_data, dict):
            # Check for common OpenAPI sections
            if "info" in chunk_data:
                return "api_info"
            elif "paths" in chunk_data:
                return "api_paths"
            elif "components" in chunk_data:
                return "api_components"
            elif "servers" in chunk_data:
                return "api_servers"
            elif any(path.startswith("/") for path in chunk_data.keys()):
                return "path_operations"
            elif "schema" in chunk_data or "type" in chunk_data:
                return "schema_definition"
            else:
                return "general_content"
        else:
            return "primitive_data"

    

    def chunk_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and chunk an OpenAPI specification from a file.

        Args:
            file_path: Path to the OpenAPI specification file

        Returns:
            List of chunks with metadata
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                api_spec = json.load(f)
            return self.chunk_api_spec(api_spec)
        except Exception as e:
            raise Exception(f"Error loading API spec from file: {e}")

    def get_chunk_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of the chunking results.

        Args:
            chunks: List of chunks

        Returns:
            Summary statistics
        """
        if not chunks:
            return {"total_chunks": 0, "total_size": 0, "chunk_types": {}}

        total_size = sum(chunk["size"] for chunk in chunks)
        chunk_types = {}

        for chunk in chunks:
            chunk_type = chunk["type"]
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = 0
            chunk_types[chunk_type] += 1

        return {
            "total_chunks": len(chunks),
            "total_size": total_size,
            "average_chunk_size": total_size // len(chunks),
            "chunk_types": chunk_types,
            "langchain_available": LANGCHAIN_AVAILABLE,
        }


# Factory function for easy instantiation
def create_chunker(
    max_chunk_size: int = 2000, chunk_overlap: int = 200
) -> APISpecChunker:
    """
    Factory function to create a configured APISpecChunker.

    Args:
        max_chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Configured APISpecChunker instance
    """
    return APISpecChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
