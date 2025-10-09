"""
Context Assembler for RAG systems
Intelligent assembly and optimization of context from search results
"""

import json
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

try:
    from .hybrid_search import SearchResult

    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False

    # Fallback SearchResult class
    @dataclass
    class SearchResult:
        content: str
        metadata: Dict[str, Any]
        vector_score: float = 0.0
        bm25_score: float = 0.0
        hybrid_score: float = 0.0
        chunk_id: str = ""
        source_type: str = ""


@dataclass
class ContextBlock:
    """Represents a block of context with metadata."""

    content: str
    source: str
    relevance_score: float
    chunk_id: str
    token_count: int
    metadata: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class AssembledContext:
    """Represents the final assembled context."""

    context: str
    blocks: List[ContextBlock]
    total_tokens: int
    truncated: bool
    assembly_metadata: Dict[str, Any]


class ContextAssembler:
    """
    Intelligent context assembly system for RAG applications.
    Optimizes context window usage and applies contextual enrichment techniques.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        tokens_per_char: float = 0.25,  # Rough estimation for token counting
        overlap_threshold: float = 0.8,  # Similarity threshold for deduplication
        prioritize_recent: bool = True,
        enable_contextual_enrichment: bool = True,
    ):
        """
        Initialize the context assembler.

        Args:
            max_tokens: Maximum tokens allowed in final context
            tokens_per_char: Rough estimation of tokens per character
            overlap_threshold: Threshold for detecting overlapping content
            prioritize_recent: Whether to prioritize more recent content
            enable_contextual_enrichment: Enable Anthropic's contextual enrichment
        """
        self.max_tokens = max_tokens
        self.tokens_per_char = tokens_per_char
        self.overlap_threshold = overlap_threshold
        self.prioritize_recent = prioritize_recent
        self.enable_contextual_enrichment = enable_contextual_enrichment

    def assemble_context(
        self,
        search_results: List[SearchResult],
        query: str,
        context_template: Optional[str] = None,
        include_metadata: bool = True,
    ) -> AssembledContext:
        """
        Assemble optimal context from search results.

        Args:
            search_results: List of search results to assemble
            query: Original query for context optimization
            context_template: Template for formatting context
            include_metadata: Whether to include metadata in context

        Returns:
            AssembledContext with optimized content
        """
        if not search_results:
            return self._empty_context()

        # Convert search results to context blocks
        context_blocks = self._create_context_blocks(search_results)

        # Apply contextual enrichment if enabled
        if self.enable_contextual_enrichment:
            context_blocks = self._apply_contextual_enrichment(context_blocks, query)

        # Deduplicate overlapping content
        context_blocks = self._deduplicate_content(context_blocks)

        # Prioritize and rank blocks
        context_blocks = self._prioritize_blocks(context_blocks, query)

        # Fit blocks within token limit
        selected_blocks, truncated = self._fit_token_limit(context_blocks)

        # Generate final context
        final_context = self._format_context(
            selected_blocks, query, context_template, include_metadata
        )

        # Calculate total tokens
        total_tokens = sum(block.token_count for block in selected_blocks)

        return AssembledContext(
            context=final_context,
            blocks=selected_blocks,
            total_tokens=total_tokens,
            truncated=truncated,
            assembly_metadata={
                "query": query,
                "original_blocks": len(context_blocks),
                "selected_blocks": len(selected_blocks),
                "contextual_enrichment": self.enable_contextual_enrichment,
                "assembly_time": datetime.now().isoformat(),
            },
        )

    def _create_context_blocks(
        self, search_results: List[SearchResult]
    ) -> List[ContextBlock]:
        """Convert search results to context blocks with metadata."""
        blocks = []

        for i, result in enumerate(search_results):
            # Estimate token count
            token_count = int(len(result.content) * self.tokens_per_char)

            # Determine priority based on relevance score
            if result.hybrid_score >= 0.8:
                priority = 1  # High
            elif result.hybrid_score >= 0.6:
                priority = 2  # Medium
            else:
                priority = 3  # Low

            block = ContextBlock(
                content=result.content,
                source=result.source_type,
                relevance_score=result.hybrid_score,
                chunk_id=result.chunk_id,
                token_count=token_count,
                metadata=result.metadata,
                priority=priority,
            )

            blocks.append(block)

        return blocks

    def _apply_contextual_enrichment(
        self, blocks: List[ContextBlock], query: str
    ) -> List[ContextBlock]:
        """
        Apply Anthropic's contextual enrichment techniques.
        Adds contextual prefixes and improves content organization.
        """
        enriched_blocks = []

        for block in blocks:
            enriched_content = block.content

            # Add contextual prefix based on content type
            if self._is_api_endpoint_content(block.content):
                enriched_content = f"API Endpoint Information:\n{enriched_content}"
            elif self._is_schema_content(block.content):
                enriched_content = f"Data Schema Definition:\n{enriched_content}"
            elif self._is_security_content(block.content):
                enriched_content = f"Security Configuration:\n{enriched_content}"
            elif self._is_example_content(block.content):
                enriched_content = f"Example Usage:\n{enriched_content}"
            else:
                enriched_content = f"API Documentation:\n{enriched_content}"

            # Add query relevance context
            if self._content_directly_answers_query(block.content, query):
                enriched_content = f"[HIGHLY RELEVANT TO QUERY]\n{enriched_content}"

            # Update block with enriched content
            enriched_block = ContextBlock(
                content=enriched_content,
                source=block.source,
                relevance_score=block.relevance_score,
                chunk_id=block.chunk_id,
                token_count=int(len(enriched_content) * self.tokens_per_char),
                metadata=block.metadata,
                priority=block.priority,
            )

            enriched_blocks.append(enriched_block)

        return enriched_blocks

    def _is_api_endpoint_content(self, content: str) -> bool:
        """Check if content describes an API endpoint."""
        endpoint_indicators = [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "paths",
            "/api/",
            "endpoint",
        ]
        return any(indicator in content for indicator in endpoint_indicators)

    def _is_schema_content(self, content: str) -> bool:
        """Check if content describes a data schema."""
        schema_indicators = ["schema", "properties", "type:", "required", "components"]
        return any(indicator in content for indicator in schema_indicators)

    def _is_security_content(self, content: str) -> bool:
        """Check if content describes security configuration."""
        security_indicators = [
            "security",
            "authentication",
            "authorization",
            "API key",
            "token",
            "oauth",
        ]
        return any(
            indicator.lower() in content.lower() for indicator in security_indicators
        )

    def _is_example_content(self, content: str) -> bool:
        """Check if content contains examples."""
        example_indicators = ["example", "sample", "curl", "request", "response"]
        return any(indicator in content.lower() for indicator in example_indicators)

    def _content_directly_answers_query(self, content: str, query: str) -> bool:
        """Check if content directly answers the query."""
        query_words = query.lower().split()
        content_lower = content.lower()

        # Simple heuristic: if most query words appear in content
        matches = sum(1 for word in query_words if word in content_lower)
        return matches >= len(query_words) * 0.7

    def _deduplicate_content(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Remove blocks with overlapping content."""
        deduplicated = []
        seen_hashes = set()

        for block in blocks:
            # Create a hash of the content
            content_hash = hashlib.md5(block.content.encode()).hexdigest()

            # Check for similar content (simple approach)
            is_duplicate = False
            for seen_hash in seen_hashes:
                if (
                    self._calculate_content_similarity(content_hash, seen_hash)
                    > self.overlap_threshold
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(block)
                seen_hashes.add(content_hash)

        return deduplicated

    def _calculate_content_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two content hashes (simplified)."""
        # Simple hamming distance approximation
        if hash1 == hash2:
            return 1.0

        # For now, return low similarity unless exact match
        return 0.0

    def _prioritize_blocks(
        self, blocks: List[ContextBlock], query: str
    ) -> List[ContextBlock]:
        """Prioritize and rank blocks for inclusion."""
        # Sort by priority (1=high, 2=medium, 3=low) and relevance score
        return sorted(blocks, key=lambda b: (b.priority, -b.relevance_score))

    def _fit_token_limit(
        self, blocks: List[ContextBlock]
    ) -> Tuple[List[ContextBlock], bool]:
        """Select blocks that fit within the token limit."""
        selected_blocks = []
        total_tokens = 0
        truncated = False

        for block in blocks:
            if total_tokens + block.token_count <= self.max_tokens:
                selected_blocks.append(block)
                total_tokens += block.token_count
            else:
                truncated = True
                break

        return selected_blocks, truncated

    def _format_context(
        self,
        blocks: List[ContextBlock],
        query: str,
        template: Optional[str] = None,
        include_metadata: bool = True,
    ) -> str:
        """Format the final context string."""
        if not blocks:
            return ""

        if template:
            # Use custom template
            context_parts = []
            for block in blocks:
                formatted_block = template.format(
                    content=block.content,
                    source=block.source,
                    relevance=block.relevance_score,
                    metadata=json.dumps(block.metadata) if include_metadata else "",
                )
                context_parts.append(formatted_block)
        else:
            # Use default formatting
            context_parts = []
            for i, block in enumerate(blocks, 1):
                if include_metadata and block.metadata:
                    header = f"--- Context Block {i} (Source: {block.source}, Relevance: {block.relevance_score:.2f}) ---"
                else:
                    header = f"--- Context Block {i} ---"

                context_parts.extend([header, block.content, ""])

        return "\n".join(context_parts)

    def _empty_context(self) -> AssembledContext:
        """Return an empty assembled context."""
        return AssembledContext(
            context="",
            blocks=[],
            total_tokens=0,
            truncated=False,
            assembly_metadata={
                "query": "",
                "original_blocks": 0,
                "selected_blocks": 0,
                "contextual_enrichment": self.enable_contextual_enrichment,
                "assembly_time": datetime.now().isoformat(),
            },
        )

    def get_assembly_stats(self, assembled_context: AssembledContext) -> Dict[str, Any]:
        """Get statistics about the context assembly process."""
        if not assembled_context.blocks:
            return {"empty_context": True}

        priority_counts = {1: 0, 2: 0, 3: 0}
        source_counts = {}

        for block in assembled_context.blocks:
            priority_counts[block.priority] += 1
            source_counts[block.source] = source_counts.get(block.source, 0) + 1

        return {
            "total_tokens": assembled_context.total_tokens,
            "token_utilization": assembled_context.total_tokens / self.max_tokens,
            "blocks_included": len(assembled_context.blocks),
            "priority_distribution": priority_counts,
            "source_distribution": source_counts,
            "truncated": assembled_context.truncated,
            "contextual_enrichment_enabled": self.enable_contextual_enrichment,
            "assembly_metadata": assembled_context.assembly_metadata,
        }


# Factory function for easy instantiation
def create_context_assembler(
    max_tokens: int = 8000,
    enable_contextual_enrichment: bool = True,
    overlap_threshold: float = 0.8,
) -> ContextAssembler:
    """
    Factory function to create a configured ContextAssembler.

    Args:
        max_tokens: Maximum tokens for context window
        enable_contextual_enrichment: Enable Anthropic's contextual enrichment
        overlap_threshold: Threshold for content deduplication

    Returns:
        Configured ContextAssembler instance
    """
    return ContextAssembler(
        max_tokens=max_tokens,
        enable_contextual_enrichment=enable_contextual_enrichment,
        overlap_threshold=overlap_threshold,
    )


# Global instance for the main API
_context_assembler = None


def get_context_assembler() -> ContextAssembler:
    """
    Get or create the global context assembler instance.

    Returns:
        Global ContextAssembler instance
    """
    global _context_assembler
    if _context_assembler is None:
        _context_assembler = create_context_assembler()
    return _context_assembler
