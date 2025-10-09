#!/usr/bin/env python3
"""
Anthropic's Contextual Retrieval Implementation
Implements the contextual chunk enhancement approach from Anthropic's 2024 research
Reduces retrieval failures by 49-67%
"""

import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class ContextualChunk:
    """Enhanced chunk with contextual information"""
    id: str
    original_content: str
    contextual_content: str  # Original + added context
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    bm25_terms: Optional[List[str]] = None

class ContextualChunkEnhancer:
    """Enhances chunks with contextual information using LLM"""

    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    async def enhance_chunk(self, chunk: Dict[str, Any], document_context: Dict[str, Any]) -> str:
        """
        Enhance a single chunk with contextual information

        Args:
            chunk: The original chunk data
            document_context: Metadata about the overall document

        Returns:
            Enhanced chunk text with prepended context
        """

        # Extract document-level context
        api_title = document_context.get("title", "API")
        api_version = document_context.get("version", "unknown")
        chunk_type = chunk.get("type", "unknown")
        original_text = chunk.get("text", "")

        # Create contextual enhancement prompt
        enhancement_prompt = f"""
You are helping to improve a RAG system by adding context to document chunks.

DOCUMENT CONTEXT:
- API Name: {api_title}
- Version: {api_version}
- Chunk Type: {chunk_type}

ORIGINAL CHUNK:
{original_text}

TASK: Add 1-2 sentences of contextual information at the beginning of this chunk to help it be better retrieved and understood. The context should:
1. Identify what API/document this is from
2. Explain what section/category this chunk represents
3. Provide any important background needed to understand this chunk

EXAMPLE:
Original: "Returns 404 if user not found"
Enhanced: "This error response is from the GET /users/{{id}} endpoint in the {api_title} API. Returns 404 if user not found"

Return ONLY the enhanced chunk text (context + original), no explanation:
"""

        try:
            from .llm_manager import LLMRequest

            # Create LLM request for enhancement
            llm_request = LLMRequest(
                prompt=enhancement_prompt,
                model="gpt-4o-mini",  # Fast model for enhancement
                max_tokens=500,
                temperature=0.3
            )

            # Get enhanced content
            response = await self.llm_manager.generate(llm_request)

            if response and response.content:
                enhanced_content = response.content.strip()

                # Ensure we have the original content
                if original_text.lower() not in enhanced_content.lower():
                    enhanced_content = f"{enhanced_content}\n\n{original_text}"

                logger.info("Enhanced chunk",
                           original_length=len(original_text),
                           enhanced_length=len(enhanced_content))

                return enhanced_content
            else:
                logger.warning("Failed to enhance chunk, using original")
                return original_text

        except Exception as e:
            logger.error("Chunk enhancement failed", error=str(e))
            return original_text

class ContextualRetrieval:
    """
    Anthropic's Contextual Retrieval implementation
    Combines contextual embeddings with contextual BM25
    """

    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.enhancer = ContextualChunkEnhancer(llm_manager)
        self.enhanced_chunks: List[ContextualChunk] = []

        # Initialize vector database
        self._init_vector_db()

        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning("SentenceTransformers not available")
            self.embedding_model = None

        # Initialize BM25
        try:
            from rank_bm25 import BM25Okapi
            self.BM25Okapi = BM25Okapi
            self.bm25_index = None
            logger.info("BM25 available")
        except ImportError:
            logger.warning("BM25 not available")
            self.BM25Okapi = None

    def _init_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        try:
            import chromadb
            from chromadb.config import Settings
            import tempfile

            # Create temporary directory for ChromaDB
            self.db_path = tempfile.mkdtemp(prefix="contextual_rag_")

            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create collection for contextual chunks
            self.collection = self.chroma_client.create_collection(
                name="contextual_chunks",
                metadata={"hnsw:space": "cosine"}
            )

            logger.info("ChromaDB initialized for contextual retrieval")

        except ImportError:
            logger.error("ChromaDB not available")
            self.chroma_client = None
            self.collection = None

    async def process_api_spec(self, api_spec: Dict[str, Any]) -> List[ContextualChunk]:
        """
        Process API specification with contextual enhancement

        Args:
            api_spec: OpenAPI specification

        Returns:
            List of enhanced contextual chunks
        """
        logger.info("Starting contextual processing of API spec")

        # Extract document context
        document_context = {
            "title": api_spec.get("info", {}).get("title", "API"),
            "version": api_spec.get("info", {}).get("version", "unknown"),
            "description": api_spec.get("info", {}).get("description", "")
        }

        # Create chunks using existing chunker
        from .chunking_strategy import APISpecChunker
        chunker = APISpecChunker()
        raw_chunks = chunker.chunk_api_spec(api_spec)

        logger.info(f"Created {len(raw_chunks)} raw chunks")

        # Enhance chunks with context
        enhanced_chunks = []
        for i, chunk in enumerate(raw_chunks):
            try:
                # Enhance chunk content
                enhanced_content = await self.enhancer.enhance_chunk(chunk, document_context)

                # Create contextual chunk
                contextual_chunk = ContextualChunk(
                    id=chunk.get("id", f"chunk_{i}"),
                    original_content=chunk.get("text", ""),
                    contextual_content=enhanced_content,
                    metadata={
                        **chunk.get("metadata", {}),
                        "api_title": document_context["title"],
                        "api_version": document_context["version"],
                        "enhancement_applied": True
                    }
                )

                # Generate embedding for enhanced content
                if self.embedding_model:
                    embedding = self.embedding_model.encode([enhanced_content])[0].tolist()
                    contextual_chunk.embedding = embedding

                # Prepare BM25 terms
                if self.BM25Okapi:
                    # Simple tokenization for BM25
                    terms = enhanced_content.lower().split()
                    contextual_chunk.bm25_terms = terms

                enhanced_chunks.append(contextual_chunk)

                logger.debug(f"Enhanced chunk {i+1}/{len(raw_chunks)}")

            except Exception as e:
                logger.error(f"Failed to enhance chunk {i}", error=str(e))
                # Fallback to original chunk
                fallback_chunk = ContextualChunk(
                    id=chunk.get("id", f"chunk_{i}"),
                    original_content=chunk.get("text", ""),
                    contextual_content=chunk.get("text", ""),
                    metadata=chunk.get("metadata", {})
                )
                enhanced_chunks.append(fallback_chunk)

        # Store enhanced chunks
        self.enhanced_chunks = enhanced_chunks

        # Build indexes
        await self._build_indexes()

        logger.info(f"Contextual processing completed: {len(enhanced_chunks)} enhanced chunks")
        return enhanced_chunks

    async def _build_indexes(self):
        """Build vector and BM25 indexes from enhanced chunks"""
        if not self.enhanced_chunks:
            return

        logger.info("Building contextual indexes")

        # Build vector index (ChromaDB)
        if self.collection and self.enhanced_chunks[0].embedding:
            try:
                documents = [chunk.contextual_content for chunk in self.enhanced_chunks]
                embeddings = [chunk.embedding for chunk in self.enhanced_chunks]
                ids = [chunk.id for chunk in self.enhanced_chunks]
                metadatas = [chunk.metadata for chunk in self.enhanced_chunks]

                # Add to ChromaDB
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )

                logger.info(f"Vector index built with {len(documents)} contextual chunks")

            except Exception as e:
                logger.error("Failed to build vector index", error=str(e))

        # Build BM25 index
        if self.BM25Okapi and all(chunk.bm25_terms for chunk in self.enhanced_chunks):
            try:
                corpus = [chunk.bm25_terms for chunk in self.enhanced_chunks]
                self.bm25_index = self.BM25Okapi(corpus)
                logger.info("BM25 index built successfully")

            except Exception as e:
                logger.error("Failed to build BM25 index", error=str(e))

    async def contextual_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform contextual retrieval with hybrid search

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant contextual chunks with scores
        """
        logger.info(f"Performing contextual search for: {query[:50]}...")

        if not self.enhanced_chunks:
            logger.warning("No enhanced chunks available for search")
            return []

        # Vector search
        vector_results = await self._vector_search(query, top_k * 2)

        # BM25 search
        bm25_results = await self._bm25_search(query, top_k * 2)

        # Combine and rerank results
        combined_results = self._combine_results(vector_results, bm25_results, top_k)

        logger.info(f"Contextual search returned {len(combined_results)} results")
        return combined_results

    async def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        if not self.collection or not self.embedding_model:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, len(self.enhanced_chunks))
            )

            # Format results
            vector_results = []
            for i, (doc, distance, metadata) in enumerate(zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                vector_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'vector_score': 1 - distance,  # Convert distance to similarity
                    'bm25_score': 0.0,
                    'source': 'vector'
                })

            return vector_results

        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return []

    async def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search"""
        if not self.bm25_index:
            return []

        try:
            # Tokenize query
            query_terms = query.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_terms)

            # Get top results
            scored_chunks = [(score, i) for i, score in enumerate(scores)]
            scored_chunks.sort(reverse=True)

            bm25_results = []
            for score, idx in scored_chunks[:top_k]:
                if idx < len(self.enhanced_chunks):
                    chunk = self.enhanced_chunks[idx]
                    bm25_results.append({
                        'content': chunk.contextual_content,
                        'metadata': chunk.metadata,
                        'vector_score': 0.0,
                        'bm25_score': float(score),
                        'source': 'bm25'
                    })

            return bm25_results

        except Exception as e:
            logger.error("BM25 search failed", error=str(e))
            return []

    def _combine_results(self, vector_results: List[Dict], bm25_results: List[Dict], top_k: int) -> List[Dict]:
        """
        Combine vector and BM25 results using Reciprocal Rank Fusion (RRF)
        """
        # Create a map of content to combined scores
        content_scores = {}

        # Add vector results
        for i, result in enumerate(vector_results):
            content = result['content']
            rrf_score = 1.0 / (60 + i + 1)  # RRF formula with k=60

            if content not in content_scores:
                content_scores[content] = result.copy()
                content_scores[content]['rrf_score'] = 0.0
                content_scores[content]['hybrid_score'] = 0.0

            content_scores[content]['rrf_score'] += rrf_score
            content_scores[content]['vector_rank'] = i + 1

        # Add BM25 results
        for i, result in enumerate(bm25_results):
            content = result['content']
            rrf_score = 1.0 / (60 + i + 1)  # RRF formula with k=60

            if content not in content_scores:
                content_scores[content] = result.copy()
                content_scores[content]['rrf_score'] = 0.0
                content_scores[content]['hybrid_score'] = 0.0

            content_scores[content]['rrf_score'] += rrf_score
            content_scores[content]['bm25_rank'] = i + 1
            # Update BM25 score if this result has it
            if result['bm25_score'] > content_scores[content].get('bm25_score', 0):
                content_scores[content]['bm25_score'] = result['bm25_score']

        # Calculate final hybrid scores
        for content, data in content_scores.items():
            # Hybrid score combines RRF with individual scores
            vector_score = data.get('vector_score', 0.0)
            bm25_score = data.get('bm25_score', 0.0)
            rrf_score = data.get('rrf_score', 0.0)

            # Weighted combination
            hybrid_score = (0.5 * vector_score + 0.3 * bm25_score + 0.2 * rrf_score)
            content_scores[content]['hybrid_score'] = hybrid_score

        # Sort by hybrid score and return top_k
        sorted_results = sorted(
            content_scores.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )

        return sorted_results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the contextual retrieval system"""
        return {
            "total_chunks": len(self.enhanced_chunks),
            "enhanced_chunks": sum(1 for c in self.enhanced_chunks if c.metadata.get("enhancement_applied")),
            "vector_index_ready": self.collection is not None and len(self.enhanced_chunks) > 0,
            "bm25_index_ready": self.bm25_index is not None,
            "embedding_model_loaded": self.embedding_model is not None
        }