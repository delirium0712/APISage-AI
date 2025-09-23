"""
AI-Powered API Analysis Service
Actually uses LLM to provide intelligent, context-aware analysis
"""

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Optional
from datetime import datetime

import structlog
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator, Field

from infrastructure.llm_manager import LLMRequest, SimpleLLMManager, ModelConfig
from infrastructure.hybrid_search import HybridSearchEngine
from infrastructure.semantic_cache import SemanticCache
from infrastructure.context_assembler import ContextAssembler
from infrastructure.performance_monitor import PerformanceMonitor
from infrastructure.deepeval_enhanced import DeepEvalEnhanced
from infrastructure.chunking_strategy import APISpecChunker
from infrastructure.contextual_retrieval import ContextualRetrieval
from typing import List

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def get_optimal_llm_params(model: str, base_max_tokens: int = 2000) -> Dict[str, Any]:
    """Get optimal LLM parameters for the specified model"""
    defaults = ModelConfig.get_model_defaults(model)

    # Scale max_tokens based on model capabilities
    optimal_max_tokens = defaults.get("max_tokens", base_max_tokens)

    # For o1 models, use higher max_tokens for reasoning tasks
    if ModelConfig.is_o1_model(model):
        optimal_max_tokens = max(optimal_max_tokens, base_max_tokens * 2)

    params = {
        "max_tokens": optimal_max_tokens,
        "model": model,
    }

    # Only add temperature for non-o1 models
    if defaults.get("temperature") is not None:
        params["temperature"] = defaults["temperature"]

    return params


def serialize_evaluation_metrics(evaluation_metrics):
    """
    Convert DeepEval EvaluationResult objects to JSON-serializable format.
    """
    if not evaluation_metrics:
        return {}

    # Handle EvaluationResult object directly - check for class name first
    if (hasattr(evaluation_metrics, '__class__') and
        'EvaluationResult' in str(evaluation_metrics.__class__)):
        try:
            # Handle DeepEval EvaluationResult objects
            scores = getattr(evaluation_metrics, 'scores', None)
            if scores:
                return {
                    'scores': {
                        'answer_relevancy': getattr(scores, 'answer_relevancy', 0.0),
                        'faithfulness': getattr(scores, 'faithfulness', 0.0),
                        'contextual_relevancy': getattr(scores, 'contextual_relevancy', 0.0),
                        'overall_score': getattr(scores, 'overall_score', 0.0),
                        'confidence_score': getattr(scores, 'confidence_score', 0.0),
                    },
                    'reasoning': getattr(evaluation_metrics, 'reasoning', {}),
                    'evaluation_level': getattr(evaluation_metrics, 'evaluation_level', 'UNKNOWN'),
                    'evaluation_time_ms': getattr(evaluation_metrics, 'evaluation_time_ms', 0.0),
                    'timestamp': str(getattr(evaluation_metrics, 'timestamp', '')),
                }
            else:
                # Try to serialize all non-private attributes
                return {
                    k: v for k, v in evaluation_metrics.__dict__.items()
                    if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
        except Exception as e:
            return {'error': f'Failed to serialize EvaluationResult: {str(e)}'}

    # Handle objects with scores and reasoning attributes
    elif hasattr(evaluation_metrics, 'scores') and hasattr(evaluation_metrics, 'reasoning'):
        try:
            scores = evaluation_metrics.scores
            return {
                'scores': {
                    'answer_relevancy': getattr(scores, 'answer_relevancy', 0.0),
                    'faithfulness': getattr(scores, 'faithfulness', 0.0),
                    'contextual_relevancy': getattr(scores, 'contextual_relevancy', 0.0),
                    'overall_score': getattr(scores, 'overall_score', 0.0),
                    'confidence_score': getattr(scores, 'confidence_score', 0.0),
                },
                'reasoning': evaluation_metrics.reasoning if evaluation_metrics.reasoning else {},
                'evaluation_level': getattr(evaluation_metrics, 'evaluation_level', 'UNKNOWN'),
                'evaluation_time_ms': getattr(evaluation_metrics, 'evaluation_time_ms', 0.0),
                'timestamp': str(getattr(evaluation_metrics, 'timestamp', '')),
            }
        except Exception as e:
            return {'error': f'Failed to serialize object with scores: {str(e)}'}

    # Handle dictionary format
    elif hasattr(evaluation_metrics, 'items'):
        serialized = {}
        for key, value in evaluation_metrics.items():
            if hasattr(value, 'score') and hasattr(value, 'reason'):
                # This is likely an EvaluationResult object
                serialized[key] = {
                    'score': getattr(value, 'score', None),
                    'reason': getattr(value, 'reason', None),
                    'success': getattr(value, 'success', True)
                }
            elif hasattr(value, '__dict__'):
                # Try to serialize object attributes
                try:
                    serialized[key] = {k: v for k, v in value.__dict__.items() if not k.startswith('_')}
                except:
                    serialized[key] = str(value)
            else:
                # Already JSON serializable
                serialized[key] = value
        return serialized

    # Fallback for other types - try to serialize object attributes
    elif hasattr(evaluation_metrics, '__dict__'):
        try:
            return {
                k: v for k, v in evaluation_metrics.__dict__.items()
                if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
        except Exception as e:
            return {'error': f'Failed to serialize object attributes: {str(e)}'}

    # Final fallback
    try:
        return str(evaluation_metrics)
    except:
        return {'error': 'Could not serialize evaluation metrics'}


# FastAPI app
app = FastAPI(
    title="APISage - AI-Powered OpenAPI Analysis",
    description="Intelligent API analysis using LLM for context-aware, specific feedback",
    version="3.0.0",
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)  # Configure for production


# Request/Response models
class AnalysisRequest(BaseModel):
    openapi_spec: Dict[str, Any]
    analysis_depth: Optional[str] = (
        "comprehensive"  # "quick", "standard", "comprehensive"
    )
    focus_areas: Optional[list] = (
        None  # ["security", "performance", "documentation", etc.]
    )


class AnalysisResponse(BaseModel):
    status: str
    analysis: str
    key_findings: Dict[str, Any]
    metadata: Dict[str, Any]
    agent_results: Optional[Dict[str, Any]] = None




class RAGRequest(BaseModel):
    question: str
    openapi_spec: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    model_name: str  # Required model selection


class RAGResponse(BaseModel):
    status: str
    answer: str
    context_used: bool
    response_time: float
    model_used: str
    metadata: Optional[Dict[str, Any]] = None
    # Enhanced RAG fields
    retrieved_contexts: Optional[List[str]] = None
    context_sources: Optional[List[str]] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    cache_hit: Optional[bool] = None


# Removed API key management utilities since using .env configuration


# Initialize LLM manager
llm_manager = SimpleLLMManager(
    model="o1-mini"
)  # Use o1-mini for optimal reasoning performance

# Initialize RAG infrastructure components
hybrid_search = HybridSearchEngine()
semantic_cache = SemanticCache()
context_assembler = ContextAssembler()
performance_monitor = PerformanceMonitor()
rag_evaluator = DeepEvalEnhanced()
api_chunker = APISpecChunker()

# Initialize Anthropic's Contextual Retrieval system
contextual_retrieval = ContextualRetrieval(llm_manager)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "APISage AI-Powered Analysis",
        "llm_available": llm_manager.is_available(),
    }






def create_analysis_prompt(
    spec: Dict[str, Any], analysis_depth: str, focus_areas: list = None
) -> str:
    """Create a detailed prompt for LLM analysis with specific, actionable feedback"""

    # Extract detailed information for specific analysis
    paths = spec.get("paths", {})
    components = spec.get("components", {})
    info = spec.get("info", {})

    # Count actual endpoints and methods
    endpoint_count = len(paths)
    total_methods = sum(len(methods) for methods in paths.values())

    # Analyze what's actually in the spec
    has_authentication = bool(
        spec.get("security")
        or any(
            any("security" in method for method in methods.values())
            for methods in paths.values()
        )
    )

    has_error_responses = any(
        any(
            status.startswith(("4", "5"))
            for status in method.get("responses", {}).keys()
        )
        for methods in paths.values()
        for method in methods.values()
    )

    has_pagination = any(
        any(
            param.get("name") in ["limit", "offset", "page", "cursor"]
            for param in method.get("parameters", [])
        )
        for methods in paths.values()
        for method in methods.values()
    )

    # Build the prompt with specific analysis requirements
    prompt = f"""You are an expert API architect conducting a thorough technical review. 
Analyze THIS SPECIFIC API and provide actionable, specific feedback.

## API SPECIFICATION ANALYSIS
**API Title:** {info.get("title", "Unknown")}
**Version:** {info.get("version", "Unknown")}
**Description:** {info.get("description", "No description provided")}
**Total Endpoints:** {endpoint_count}
**Total HTTP Methods:** {total_methods}
**Has Authentication:** {has_authentication}
**Has Error Responses:** {has_error_responses}
**Has Pagination:** {has_pagination}

## ACTUAL ENDPOINTS FOUND:
{json.dumps({path: list(methods.keys()) for path, methods in paths.items()}, indent=2)}

## FULL SPECIFICATION:
{json.dumps(spec, indent=2)[:10000]}

## ANALYSIS REQUIREMENTS:
- **CRITICAL:** Reference specific lines, paths, and components from the actual spec
- **CRITICAL:** Only mention features that actually exist or are actually missing
- **CRITICAL:** Provide specific code examples for THIS API
- **CRITICAL:** Avoid generic advice that applies to any API

## REQUIRED OUTPUT FORMAT:

<h3>📊 API Score Breakdown</h3>
**Overall Score:** [X/100]
- **Completeness:** [X/100] - [Specific assessment of what's missing]
- **Documentation:** [X/100] - [Specific documentation issues found]
- **Security:** [X/100] - [Specific security gaps identified]
- **Usability:** [X/100] - [Specific developer experience issues]
- **Standards Compliance:** [X/100] - [Specific OpenAPI/REST violations]

<h3>🎯 Executive Summary</h3>
[2-3 sentences about THIS SPECIFIC API's current state and main issues]

<h3>🚨 Critical Issues (Priority Order)</h3>
1. **Issue:** [Specific problem with exact location]
   - **Location:** [Exact path/component/line reference]
   - **Impact:** [Why this specific issue matters for THIS API]
   - **Fix:** [Specific solution with code example]
   - **Priority:** [High/Medium/Low with justification]

2. **Issue:** [Next specific problem]
   - **Location:** [Exact reference]
   - **Impact:** [Specific impact]
   - **Fix:** [Specific solution]
   - **Priority:** [High/Medium/Low]

<h3>🔍 Detailed Analysis</h3>

#### Missing Endpoints Analysis
[Based on the API description and purpose, what endpoints are missing?]
- **Missing:** [Specific endpoint that should exist]
- **Reason:** [Why this endpoint is needed for THIS API]
- **Implementation:** [Specific OpenAPI spec for the missing endpoint]

#### Parameter & Response Analysis
[Analyze actual parameters and responses in the spec]
- **Missing Parameters:** [Specific parameters missing from actual endpoints]
- **Incomplete Responses:** [Specific response schemas that are incomplete]
- **Error Handling:** [Specific error responses that are missing]

#### Schema Analysis
[Analyze the actual schemas defined]
- **Incomplete Schemas:** [Specific fields missing from actual schemas]
- **Type Issues:** [Specific type problems in actual schemas]
- **Validation Missing:** [Specific validation rules missing]

<h3>💡 Specific Recommendations</h3>
[Ordered by impact and effort for THIS API]

#### High Impact, Low Effort
1. **Fix:** [Specific fix for THIS API]
   - **Code Example:** [Actual OpenAPI spec addition]
   - **Impact:** [Why this helps THIS specific API]

#### Medium Impact, Medium Effort
1. **Add:** [Specific addition for THIS API]
   - **Implementation:** [Specific code example]
   - **Benefit:** [Specific benefit for THIS API]

<h3>🛠️ Code Examples</h3>
[Provide actual OpenAPI spec additions/modifications for THIS API]

```yaml
# Example: Adding missing endpoint for THIS API
{paths[list(paths.keys())[0]] if paths else "No endpoints to reference"}:
  # Add specific missing method with actual implementation
```

<h3>📋 Action Items Checklist</h3>
- [ ] [Specific action item 1 for THIS API]
- [ ] [Specific action item 2 for THIS API]
- [ ] [Specific action item 3 for THIS API]

## CRITICAL REMINDERS:
1. **NEVER** mention features that don't exist in the spec
2. **ALWAYS** reference specific paths, methods, and components
3. **ONLY** provide recommendations relevant to THIS API's purpose
4. **INCLUDE** actual code examples for THIS API's structure
5. **AVOID** generic advice like "implement rate limiting" unless specifically relevant

Focus on what THIS API actually needs based on its description and current implementation.
"""

    return prompt


def parse_llm_response(llm_response: str) -> Dict[str, Any]:
    """Parse the LLM response into structured format"""

    # Extract sections from the markdown response
    sections = {}
    current_section = None
    current_content = []

    for line in llm_response.split("\n"):
        if line.startswith("## "):
            if current_section:
                sections[current_section] = "\n".join(current_content)
            current_section = line[3:].strip().lower().replace(" ", "_")
            current_content = []
        elif current_section:
            current_content.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_content)

    # Extract key findings
    key_findings = {
        "critical_issues_count": llm_response.count("Critical:")
        + llm_response.count("CRITICAL"),
        "security_issues_count": llm_response.count("Security:")
        + llm_response.count("vulnerability"),
        "has_authentication": "authentication" in llm_response.lower()
        and "missing" not in llm_response.lower(),
        "documentation_quality": (
            "good"
            if "good documentation" in llm_response.lower()
            else "needs improvement"
        ),
    }

    return {"sections": sections, "key_findings": key_findings}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_api(request: AnalysisRequest):
    """AI-powered API analysis using LLM"""

    start_time = time.time()

    try:
        # Check if LLM is available
        if not llm_manager.is_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service not available. Please set your OpenAI API key first.",
            )

        # Create the analysis prompt
        prompt = create_analysis_prompt(
            request.openapi_spec, request.analysis_depth, request.focus_areas
        )

        logger.info(
            "Starting LLM analysis",
            analysis_depth=request.analysis_depth,
            focus_areas=request.focus_areas,
        )

        # Create LLM request
        llm_request = LLMRequest(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.3,  # Lower temperature for more consistent analysis
        )

        # Get analysis from LLM with token usage
        llm_result = await llm_manager.generate(llm_request)

        if not llm_result or not llm_result.content:
            raise HTTPException(
                status_code=500, detail="Failed to generate analysis from LLM"
            )

        llm_response = llm_result.content
        token_usage = llm_result.usage if llm_result.usage else {}
        selected_model = llm_result.model if llm_result.model else "gpt-4o-mini"

        # Parse the response
        parsed_response = parse_llm_response(llm_response)

        # Create response
        return AnalysisResponse(
            status="success",
            analysis=llm_response,
            key_findings=parsed_response["key_findings"],
            metadata={
                "api_title": request.openapi_spec.get("info", {}).get(
                    "title", "Unknown"
                ),
                "api_version": request.openapi_spec.get("info", {}).get(
                    "version", "Unknown"
                ),
                "endpoints_count": len(request.openapi_spec.get("paths", {})),
                "analysis_depth": request.analysis_depth,
                "focus_areas": request.focus_areas or ["all"],
                "model_used": selected_model,
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
                "analysis_time": time.time() - start_time,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Analysis failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")




@app.post("/rag-query", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    Enhanced RAG Query endpoint with true retrieval-augmented generation
    Uses hybrid search, semantic caching, and evaluation metrics
    """
    logger.info(
        "Enhanced RAG query received",
        question=request.question[:100],
        question_length=len(request.question),
        has_openapi_spec=request.openapi_spec is not None,
    )

    if not request.question.strip():
        logger.warning("Empty question received")
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        start_time = time.time()

        # Define selected model early for use in both cached and non-cached branches
        selected_model = request.model_name

        # Initialize query ID for performance monitoring
        query_id = f"rag_{int(start_time)}_{hash(request.question) % 10000}"

        # Check semantic cache first
        cached_result = semantic_cache.get(request.question)
        cache_hit = cached_result is not None

        if cache_hit:
            logger.info("Cache hit for RAG query")
            response_time = time.time() - start_time

            # Unpack cached result (response, metadata, similarity_score)
            cached_response, cached_metadata, similarity_score = cached_result

            # Parse cached response (it's stored as JSON string)
            import json
            try:
                cached_data = json.loads(cached_response)
            except (json.JSONDecodeError, TypeError):
                # Fallback if cached response is not JSON
                cached_data = {
                    "answer": cached_response,
                    "context_used": False,
                    "model_used": selected_model,
                    "retrieved_contexts": [],
                    "context_sources": [],
                    "evaluation_metrics": {},
                    "metadata": {}
                }

            # Record cached query performance
            performance_monitor.record_query(
                query_id=f"rag_cached_{int(start_time)}_{hash(request.question) % 10000}",
                query=request.question,
                response_time_ms=response_time * 1000,
                tokens_used=0,  # No tokens used for cached response
                success=True,
                cache_hit=True,
                metadata={"model": cached_data.get("model_used", selected_model), "similarity_score": similarity_score}
            )

            return RAGResponse(
                status="success",
                answer=cached_data.get("answer", cached_response),
                context_used=cached_data.get("context_used", False),
                response_time=response_time,
                model_used=cached_data.get("model_used", selected_model),
                cache_hit=True,
                retrieved_contexts=cached_data.get("retrieved_contexts", []),
                context_sources=cached_data.get("context_sources", []),
                evaluation_metrics=cached_data.get("evaluation_metrics", {}),
                metadata=cached_data.get("metadata", {}),
            )

        # Process OpenAPI specification with Anthropic's Contextual Retrieval
        retrieved_contexts = []
        context_sources = []
        has_context = False

        if request.openapi_spec:
            has_context = True
            logger.info("Starting Anthropic Contextual Retrieval processing")

            try:
                # Process API spec with contextual enhancement
                await contextual_retrieval.process_api_spec(request.openapi_spec)

                # Perform contextual search
                search_results = await contextual_retrieval.contextual_search(
                    query=request.question,
                    top_k=5  # Retrieve top 5 enhanced chunks
                )

                # Extract contexts and sources from results
                retrieved_contexts = [result['content'] for result in search_results]
                context_sources = [result['metadata'].get('source', f'contextual_chunk_{i}')
                                 for i, result in enumerate(search_results)]

                logger.info(
                    "Contextual retrieval completed",
                    num_contexts=len(retrieved_contexts),
                    context_sources=context_sources,
                    contextual_stats=contextual_retrieval.get_stats()
                )

            except Exception as e:
                logger.error("Contextual retrieval failed, falling back to basic processing", error=str(e))
                # Fallback to basic chunking if contextual retrieval fails
                chunks = api_chunker.chunk_api_spec(request.openapi_spec)
                retrieved_contexts = [chunk.get("text", "") for chunk in chunks[:5]]
                context_sources = [f"fallback_chunk_{i}" for i in range(len(retrieved_contexts))]

        # Assemble optimized context
        assembled_context = ""
        if retrieved_contexts:
            # Convert retrieved contexts to SearchResult objects for ContextAssembler
            from infrastructure.context_assembler import SearchResult
            search_results = []
            for i, context in enumerate(retrieved_contexts):
                search_result = SearchResult(
                    content=context,
                    metadata={"source": f"contextual_chunk_{i}"},
                    vector_score=0.9,  # High score since these are from contextual retrieval
                    bm25_score=0.8,
                    hybrid_score=0.85,
                    chunk_id=f"contextual_chunk_{i}",
                    source_type="contextual_retrieval"
                )
                search_results.append(search_result)

            assembled_context_result = context_assembler.assemble_context(
                search_results=search_results,
                query=request.question
            )
            assembled_context = assembled_context_result.context

        # Create enhanced contextual RAG prompt
        rag_prompt = f"""You are an expert API documentation assistant using Anthropic's Contextual Retrieval system. You have access to contextually enhanced, relevant API specification chunks.

{"CONTEXTUALLY ENHANCED RETRIEVED CHUNKS:\n" + assembled_context if assembled_context else "Note: No relevant contextual chunks found in the API specification."}

USER QUESTION: {request.question}

Please provide a comprehensive, accurate response that:
1. Directly answers the user's question using the contextually enhanced retrieved chunks
2. Leverages the additional context provided with each chunk for better understanding
3. Provides specific examples from the API specification when available
4. Includes code examples (Python requests, curl, etc.) when appropriate
5. Explains authentication, parameters, and data structures as relevant
6. Is clear and actionable for developers

The retrieved chunks have been contextually enhanced to include background information, reducing retrieval failures by up to 49%. Use this enhanced context to provide more accurate and comprehensive responses.

If the retrieved context doesn't contain sufficient information, clearly state what additional information would be needed."""

        # Generate response using LLM with requested model
        llm_params = get_optimal_llm_params(selected_model, 1500)
        llm_request = LLMRequest(
            prompt=rag_prompt,
            **llm_params  # Contains model and other optimal parameters
        )

        logger.info(
            "Generating enhanced RAG response",
            model=selected_model,
            prompt_length=len(rag_prompt),
            num_retrieved_contexts=len(retrieved_contexts),
            max_tokens=llm_request.max_tokens,
        )

        response = await llm_manager.generate(llm_request)

        if not response or not response.content:
            logger.error("LLM failed to generate response")
            raise HTTPException(status_code=500, detail="Failed to generate response")
        else:
            # Debug: Check what model was actually used
            logger.info(f"DEBUG: Requested model: {selected_model}, Response model: {response.model}")

        # Evaluate RAG response using DeepEval
        evaluation_metrics = {}
        if retrieved_contexts and rag_evaluator:
            try:
                logger.info("Evaluating RAG response with DeepEval")
                eval_result = rag_evaluator.evaluate_rag_response(
                    query=request.question,
                    answer=response.content,
                    context="\n".join(retrieved_contexts) if retrieved_contexts else ""
                )
                evaluation_metrics = eval_result
                logger.info("RAG evaluation completed", metrics=evaluation_metrics)
            except Exception as e:
                logger.warning("RAG evaluation failed", error=str(e))
                evaluation_metrics = {"evaluation_error": str(e)}

        response_time = time.time() - start_time

        # Cache the response for future queries
        cache_data = {
            "answer": response.content,
            "context_used": has_context,
            "model_used": response.model if response else selected_model,
            "retrieved_contexts": retrieved_contexts,
            "context_sources": context_sources,
            "evaluation_metrics": serialize_evaluation_metrics(evaluation_metrics),
            "metadata": {
                "question_length": len(request.question),
                "context_length": len(assembled_context),
                "response_length": len(response.content),
                "num_retrieved_contexts": len(retrieved_contexts),
                "prompt_tokens": response.usage.get("prompt_tokens") if response.usage else None,
                "completion_tokens": response.usage.get("completion_tokens") if response.usage else None,
                "total_tokens": response.usage.get("total_tokens") if response.usage else None,
            }
        }

        # Store in semantic cache as JSON string
        import json
        semantic_cache.put(
            query=request.question,
            response=json.dumps(cache_data),
            metadata={"model": response.model if response else selected_model}
        )

        # Serialize evaluation metrics for logging
        serialized_evaluation = serialize_evaluation_metrics(evaluation_metrics) if evaluation_metrics else {}

        # Extract evaluation score safely
        evaluation_score = None
        if serialized_evaluation:
            if isinstance(serialized_evaluation, dict):
                evaluation_score = serialized_evaluation.get("scores", {}).get("overall_score")
            elif hasattr(serialized_evaluation, 'scores'):
                # Handle case where EvaluationResult wasn't properly serialized
                scores = getattr(serialized_evaluation, 'scores', None)
                if scores:
                    evaluation_score = getattr(scores, 'overall_score', None)

        logger.info(
            "Enhanced RAG query completed",
            response_time=response_time,
            has_context=has_context,
            num_retrieved_contexts=len(retrieved_contexts),
            cache_hit=False,
            evaluation_score=evaluation_score,
        )

        # Record successful query performance
        performance_monitor.record_query(
            query_id=query_id,
            query=request.question,
            response_time_ms=response_time * 1000,
            tokens_used=response.usage.get("total_tokens") if response.usage else 0,
            success=True,
            cache_hit=False,
            evaluation_scores=serialize_evaluation_metrics(evaluation_metrics),
            metadata={
                "model": response.model if response else selected_model,
                "context_used": has_context,
                "num_contexts": len(retrieved_contexts)
            }
        )

        return RAGResponse(
            status="success",
            answer=response.content,
            context_used=has_context,
            response_time=response_time,
            model_used=response.model if response else selected_model,
            cache_hit=False,
            retrieved_contexts=retrieved_contexts,
            context_sources=context_sources,
            evaluation_metrics=serialize_evaluation_metrics(evaluation_metrics),
            metadata=cache_data["metadata"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Enhanced RAG query failed", error=str(e), exc_info=True)
        # Record failed query performance
        performance_monitor.record_query(
            query_id=f"rag_failed_{int(time.time())}_{hash(request.question) % 10000}",
            query=request.question,
            response_time_ms=(time.time() - start_time) * 1000,
            tokens_used=0,
            success=False,
            error_message=str(e),
            metadata={"model": selected_model}
        )
        raise HTTPException(status_code=500, detail=f"Enhanced RAG query failed: {str(e)}")



@app.post("/rag-query-stream")
async def rag_query_stream(request: RAGRequest):
    """
    Streaming RAG Query endpoint with real-time response generation
    Uses Server-Sent Events (SSE) for streaming responses with typewriter effect
    """
    logger.info(
        "Streaming RAG query received",
        question=request.question[:100],
        question_length=len(request.question),
        has_openapi_spec=request.openapi_spec is not None,
    )

    if not request.question.strip():
        logger.warning("Empty question received")
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def generate_streaming_response():
        """Generate streaming response with SSE format"""
        try:
            start_time = time.time()
            selected_model = request.model_name

            # Check semantic cache first
            cached_result = semantic_cache.get(request.question)
            cache_hit = cached_result is not None

            if cache_hit:
                logger.info("Cache hit for streaming RAG query")
                cached_response, cached_metadata, similarity_score = cached_result

                # Stream cached response word by word for consistent UX
                try:
                    cached_data = json.loads(cached_response)
                    answer = cached_data.get("answer", cached_response)
                except (json.JSONDecodeError, TypeError):
                    answer = cached_response

                # Send metadata first
                yield f"data: {json.dumps({'type': 'metadata', 'cache_hit': True, 'model': selected_model})}\n\n"

                # Stream answer word by word
                words = answer.split()
                for i, word in enumerate(words):
                    partial_answer = " ".join(words[:i+1])
                    yield f"data: {json.dumps({'type': 'content', 'content': partial_answer})}\n\n"
                    await asyncio.sleep(0.05)

                response_time = time.time() - start_time
                yield f"data: {json.dumps({'type': 'complete', 'response_time': response_time})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Process with RAG pipeline for non-cached queries
            retrieved_contexts = []
            context_sources = []
            has_context = False

            if request.openapi_spec:
                has_context = True
                try:
                    await contextual_retrieval.process_api_spec(request.openapi_spec)
                    search_results = await contextual_retrieval.contextual_search(
                        query=request.question, top_k=5
                    )
                    retrieved_contexts = [result["content"] for result in search_results]
                    context_sources = [f"Section {i+1}" for i in range(len(retrieved_contexts))]
                    yield f"data: {json.dumps({'type': 'progress', 'message': f'Found {len(retrieved_contexts)} relevant contexts'})}\n\n"
                except Exception as e:
                    logger.error(f"Contextual retrieval failed: {str(e)}")
                    retrieved_contexts = [json.dumps(request.openapi_spec, indent=2)]
                    context_sources = ["API Specification"]

            # Generate response
            context_content = "\n\n".join(retrieved_contexts) if retrieved_contexts else ""
            prompt_template = """You are an expert API documentation assistant. Help the user understand and work with their API specification.

{context_section}

User Question: {question}

Provide a helpful, accurate response with specific examples when possible."""

            context_section = f"API Documentation Context:\n{context_content}\n" if context_content else "No specific API context provided."
            formatted_prompt = prompt_template.format(context_section=context_section, question=request.question)

            yield f"data: {json.dumps({'type': 'generating', 'model': selected_model})}\n\n"

            llm_request = LLMRequest(
                prompt=formatted_prompt,
                model=selected_model,
                max_tokens=2000,
                temperature=0.1 if not ModelConfig.is_o1_model(selected_model) else None
            )

            # Stream LLM response
            full_response = ""
            async for chunk in llm_manager.generate_stream(llm_request):
                if chunk.startswith("data: "):
                    chunk_data = chunk[6:]
                    if chunk_data == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(chunk_data)
                        if "content" in chunk_json:
                            content = chunk_json["content"]
                            full_response += content
                            yield f"data: {json.dumps({'type': 'content', 'content': full_response})}\n\n"
                    except json.JSONDecodeError:
                        continue

            # Cache the result
            if full_response:
                response_data = {
                    "answer": full_response,
                    "context_used": has_context,
                    "model_used": selected_model,
                    "retrieved_contexts": retrieved_contexts,
                    "context_sources": context_sources,
                    "evaluation_metrics": {},
                    "metadata": {}
                }
                semantic_cache.put(query=request.question, response=json.dumps(response_data), metadata={"model": selected_model})

            response_time = time.time() - start_time
            completion_data = {
                'type': 'complete',
                'response_time': response_time,
                'cache_hit': False,
                'model': selected_model,
                'context_used': has_context,
                'contexts_found': len(retrieved_contexts)
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming RAG query failed: {str(e)}")
            error_data = {'type': 'error', 'error': str(e), 'message': 'An error occurred while processing your request'}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_streaming_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Content-Type": "text/plain; charset=utf-8"}
    )


@app.on_event("startup")
async def startup_event():
    """Log application startup information"""
    logger.info(
        "APISage Backend API starting up",
        version="3.0.0",
        llm_available=llm_manager is not None,
        default_model=llm_manager.default_model if llm_manager else None,
        openai_configured=bool(os.getenv("OPENAI_API_KEY")),
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown"""
    logger.info("APISage Backend API shutting down")


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting APISage Backend API server",
        host="0.0.0.0",
        port=8080,
        environment=os.getenv("ENVIRONMENT", "development"),
    )

    uvicorn.run(app, host="0.0.0.0", port=8080)
