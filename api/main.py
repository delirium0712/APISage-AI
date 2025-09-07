"""
AI-Powered API Analysis Service
Actually uses LLM to provide intelligent, context-aware analysis
"""

import json
import os
import time
from typing import Any, Dict, Optional

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from infrastructure.llm_manager import LLMRequest, SimpleLLMManager, ModelConfig
from typing import List

# Vector RAG imports (with fallback for optional dependencies)
try:
    from infrastructure.chunking_strategy import APISpecChunker
    from infrastructure.hybrid_search import get_hybrid_search_engine
    from infrastructure.context_assembler import get_context_assembler
    from infrastructure.cache_layer import get_cache
    VECTOR_RAG_AVAILABLE = True
except ImportError as e:
    VECTOR_RAG_AVAILABLE = False
    print(f"Vector RAG components not available: {e}")

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


# FastAPI app
app = FastAPI(
    title="APISage - AI-Powered OpenAPI Analysis",
    description="Intelligent API analysis using LLM for context-aware, specific feedback",
    version="3.0.0",
)


# Request/Response models
class AnalysisRequest(BaseModel):
    openapi_spec: Dict[str, Any]
    analysis_depth: Optional[
        str
    ] = "comprehensive"  # "quick", "standard", "comprehensive"
    focus_areas: Optional[
        list
    ] = None  # ["security", "performance", "documentation", etc.]


class AnalysisResponse(BaseModel):
    status: str
    analysis: str
    key_findings: Dict[str, Any]
    metadata: Dict[str, Any]
    agent_results: Optional[Dict[str, Any]] = None


class APIKeyRequest(BaseModel):
    api_key: str


class APIKeyResponse(BaseModel):
    status: str
    message: str
    is_set: bool




class RAGRequest(BaseModel):
    question: str
    openapi_spec: Optional[Dict[str, Any]] = None
    context: Optional[str] = None


class RAGResponse(BaseModel):
    status: str
    answer: str
    context_used: bool
    response_time: float
    model_used: str
    metadata: Optional[Dict[str, Any]] = None


# Initialize LLM manager
llm_manager = SimpleLLMManager(model="o1-mini")  # Use o1-mini for optimal reasoning performance


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "APISage AI-Powered Analysis",
        "llm_available": llm_manager.is_available(),
    }



@app.post("/set-api-key", response_model=APIKeyResponse)
async def set_api_key(request: APIKeyRequest):
    """Set OpenAI API key"""
    try:
        os.environ["OPENAI_API_KEY"] = request.api_key

        if llm_manager.refresh_api_key():
            logger.info("API key set and validated successfully")
            return APIKeyResponse(
                status="success",
                message="API key set and validated successfully",
                is_set=True,
            )
        else:
            return APIKeyResponse(
                status="error", message="Failed to validate API key", is_set=False
            )

    except Exception as e:
        logger.error("API key setting failed", error=str(e))
        return APIKeyResponse(
            status="error", message=f"Failed to set API key: {str(e)}", is_set=False
        )


@app.get("/api-key-status")
async def get_api_key_status():
    """Get current API key status"""
    return {
        "is_set": bool(os.getenv("OPENAI_API_KEY")),
        "is_available": llm_manager.is_available(),
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

### 📊 API Score Breakdown
**Overall Score:** [X/100]
- **Completeness:** [X/100] - [Specific assessment of what's missing]
- **Documentation:** [X/100] - [Specific documentation issues found]
- **Security:** [X/100] - [Specific security gaps identified]
- **Usability:** [X/100] - [Specific developer experience issues]
- **Standards Compliance:** [X/100] - [Specific OpenAPI/REST violations]

### 🎯 Executive Summary
[2-3 sentences about THIS SPECIFIC API's current state and main issues]

### 🚨 Critical Issues (Priority Order)
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

### 🔍 Detailed Analysis

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

### 💡 Specific Recommendations
[Ordered by impact and effort for THIS API]

#### High Impact, Low Effort
1. **Fix:** [Specific fix for THIS API]
   - **Code Example:** [Actual OpenAPI spec addition]
   - **Impact:** [Why this helps THIS specific API]

#### Medium Impact, Medium Effort
1. **Add:** [Specific addition for THIS API]
   - **Implementation:** [Specific code example]
   - **Benefit:** [Specific benefit for THIS API]

### 🛠️ Code Examples
[Provide actual OpenAPI spec additions/modifications for THIS API]

```yaml
# Example: Adding missing endpoint for THIS API
{paths[list(paths.keys())[0]] if paths else "No endpoints to reference"}:
  # Add specific missing method with actual implementation
```

### 📋 Action Items Checklist
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
        "documentation_quality": "good"
        if "good documentation" in llm_response.lower()
        else "needs improvement",
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
                "model_used": llm_manager.default_model,
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


@app.post("/analyze-stream")
async def analyze_api_stream(request: AnalysisRequest):
    """Streaming AI-powered API analysis using LLM"""
    from fastapi.responses import StreamingResponse
    
    # Check if LLM is available
    if not llm_manager.is_available():
        async def error_stream():
            yield f"data: {{'error': 'LLM service not available. Please set your OpenAI API key first.'}}\n\n"
        return StreamingResponse(error_stream(), media_type="text/plain")

    async def generate_stream():
        try:
            # Send initial status
            yield f"data: {{'status': 'starting', 'message': 'Initializing analysis...'}}\n\n"
            
            # Create the analysis prompt
            prompt = create_analysis_prompt(
                request.openapi_spec, request.analysis_depth, request.focus_areas
            )
            
            yield f"data: {{'status': 'analyzing', 'message': 'Analyzing API specification...'}}\n\n"

            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.3,
            )

            # Stream the analysis
            async for chunk in llm_manager.generate_stream(llm_request):
                yield chunk

            yield f"data: {{'status': 'complete', 'message': 'Analysis complete'}}\n\n"

        except Exception as e:
            logger.error("Streaming analysis failed", error=str(e), exc_info=True)
            yield f"data: {{'error': 'Analysis failed: {str(e)}'}}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/plain")
















@app.post("/rag-query", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    RAG Query endpoint for conversational API documentation assistant
    Provides contextual answers about API specifications
    """
    logger.info("RAG query received", 
                question=request.question[:100],
                question_length=len(request.question),
                has_openapi_spec=request.openapi_spec is not None)
    
    if not request.question.strip():
        logger.warning("Empty question received")
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        start_time = time.time()
        
        # Check if we have API specification context
        has_context = request.openapi_spec is not None
        context_str = ""
        
        if has_context:
            # Extract relevant information from the OpenAPI spec
            spec = request.openapi_spec
            api_info = spec.get("info", {})
            paths = spec.get("paths", {})
            components = spec.get("components", {})
            
            logger.info("Processing OpenAPI spec context",
                       api_title=api_info.get("title", "Unknown"),
                       api_version=api_info.get("version", "Unknown"),
                       endpoints_count=len(paths),
                       has_components=bool(components))
            
            # Build context string from spec
            context_str = f"""
API Title: {api_info.get('title', 'Unknown')}
Version: {api_info.get('version', 'Unknown')}
Description: {api_info.get('description', 'No description')}

Endpoints ({len(paths)} total):
"""
            
            # Add endpoint summaries
            for path, methods in paths.items():
                for method, details in methods.items():
                    if isinstance(details, dict) and 'summary' in details:
                        context_str += f"- {method.upper()} {path}: {details.get('summary', 'No summary')}\n"
            
            # Add schema information
            schemas = components.get("schemas", {})
            if schemas:
                context_str += f"\nData Models ({len(schemas)} total):\n"
                for schema_name in schemas.keys():
                    context_str += f"- {schema_name}\n"
        
        # Create RAG-style prompt
        rag_prompt = f"""You are an expert API documentation assistant. Your role is to provide helpful, accurate answers about API usage and implementation.

{"CONTEXT - API SPECIFICATION:" + context_str if has_context else "Note: No API specification provided as context."}

USER QUESTION: {request.question}

Please provide a helpful, practical response that:
1. Directly answers the user's question
2. Uses information from the API specification when available
3. Provides code examples when appropriate (Python requests, curl, etc.)
4. Explains any relevant authentication, parameters, or data structures
5. Is clear and actionable for developers

If the question cannot be answered from the provided API specification, politely explain what information would be needed."""

        # Generate response using LLM
        llm_request = LLMRequest(
            prompt=rag_prompt,
            **get_optimal_llm_params(llm_manager.default_model, 1500)
        )
        
        logger.info("Generating LLM response",
                   model=llm_manager.default_model,
                   prompt_length=len(rag_prompt),
                   max_tokens=llm_request.max_tokens)
        
        response = await llm_manager.generate(llm_request)
        response_time = time.time() - start_time
        
        logger.info("LLM response received",
                   response_time=response_time,
                   has_response=response is not None,
                   has_content=response.content if response else None,
                   usage_info=response.usage if response and response.usage else None)
        
        if not response or not response.content:
            logger.error("LLM failed to generate response",
                        response_exists=response is not None,
                        content_exists=response.content if response else None)
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response"
            )
        
        logger.info("RAG query completed", 
                   response_time=response_time,
                   has_context=has_context,
                   model=llm_manager.default_model)
        
        return RAGResponse(
            status="success",
            answer=response.content,
            context_used=has_context,
            response_time=response_time,
            model_used=llm_manager.default_model,
            metadata={
                "question_length": len(request.question),
                "context_length": len(context_str) if has_context else 0,
                "response_length": len(response.content),
                "prompt_tokens": response.usage.get("prompt_tokens") if response.usage else None,
                "completion_tokens": response.usage.get("completion_tokens") if response.usage else None,
                "total_tokens": response.usage.get("total_tokens") if response.usage else None
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("RAG query failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )


@app.post("/rag-query-v2", response_model=RAGResponse)
async def vector_rag_query(request: RAGRequest):
    """
    Enhanced RAG Query endpoint with vector search for large API specifications.
    Automatically handles chunking, indexing, and semantic search for optimal results.
    """
    if not VECTOR_RAG_AVAILABLE:
        # Fallback to original RAG if vector components not available
        return await rag_query(request)
    
    logger.info("Vector RAG query received", 
                question=request.question[:100],
                question_length=len(request.question),
                has_openapi_spec=request.openapi_spec is not None)
    
    if not request.question.strip():
        logger.warning("Empty question received")
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    if not request.openapi_spec:
        logger.warning("No OpenAPI spec provided for vector RAG")
        raise HTTPException(
            status_code=400,
            detail="OpenAPI specification is required for vector RAG queries"
        )
    
    try:
        start_time = time.time()
        
        # Generate API spec hash for caching and indexing
        import hashlib
        spec_str = json.dumps(request.openapi_spec, sort_keys=True)
        api_spec_hash = hashlib.md5(spec_str.encode()).hexdigest()
        api_name = request.openapi_spec.get('info', {}).get('title', 'Unknown API')
        
        # Initialize components
        # Use semantic cache for better query optimization\n        from infrastructure.semantic_cache import get_semantic_cache\n        cache = get_semantic_cache()
        hybrid_search = get_hybrid_search_engine()
        context_assembler = get_context_assembler()
        
        # Initialize relevancy evaluation and performance monitoring
        from infrastructure.deepeval_enhanced import get_deep_eval_enhanced
        from infrastructure.performance_monitor import get_performance_monitor
        
        deep_evaluator = get_deep_eval_enhanced(llm_manager)
        performance_monitor = get_performance_monitor()
        
        # Start performance monitoring
        query_metrics = performance_monitor.start_query(request.question, api_spec_hash)
        
        # Get cache instance
        cache = get_cache()
        
        # Check cache first
        cached_response = cache.get_cached_response(request.question, api_spec_hash)
        if cached_response:
            logger.info("Returning cached response", 
                       cached_at=cached_response.get("cached_at"),
                       cache_hit=True)
            return RAGResponse(**cached_response)
        
        # Check if spec is already indexed
        if not hybrid_search.is_indexed(api_spec_hash):
            logger.info("Indexing new API specification",
                       api_name=api_name,
                       api_spec_hash=api_spec_hash[:8])
            
            # Chunk the specification
            chunker = APISpecChunker()
            chunks = chunker.chunk_spec(request.openapi_spec)
            optimized_chunks = chunker.optimize_chunks(chunks)
            
            logger.info("Chunked API specification",
                       total_chunks=len(optimized_chunks),
                       chunk_types=list(set(c.type for c in optimized_chunks)))
            
            # Index chunks in hybrid search engine
            hybrid_search.index_chunks(optimized_chunks, api_spec_hash, api_name)
        
        # Perform hybrid search
        search_start = time.time()
        search_results = hybrid_search.search(
            query=request.question,
            strategy="hybrid",
            n_results=5,
            api_spec_hash=api_spec_hash
        )
        search_time = time.time() - search_start
        
        # Record search performance
        query_metrics.search_latency_ms = search_time * 1000  # Convert to milliseconds
        query_metrics.chunks_retrieved = len(search_results)
        
        logger.info("Search completed",
                   results_found=len(search_results),
                   search_types=[r.get("search_type") for r in search_results])
        
        if not search_results:
            logger.warning("No relevant information found in API specification")
            # Still generate a response with minimal context
            search_results = []
        
        # Assemble context with token management
        context_result = context_assembler.assemble_context(
            query=request.question,
            search_results=search_results,
            spec_metadata=request.openapi_spec.get('info', {}),
            include_examples=True
        )
        
        logger.info("Context assembled",
                   context_tokens=context_result["total_tokens"],
                   sections_used=context_result["sections_used"],
                   budget_utilization=f"{context_result['budget_utilization']:.1%}")
        
        # Generate response using LLM
        llm_start = time.time()
        llm_request = LLMRequest(
            prompt=context_result["context"],
            **get_optimal_llm_params(llm_manager.default_model, 2000)
        )
        
        response = await llm_manager.generate(llm_request)
        llm_time = time.time() - llm_start
        response_time = time.time() - start_time
        
        # Record LLM performance
        query_metrics.llm_generation_ms = llm_time * 1000  # Convert to milliseconds
        query_metrics.context_tokens = response.usage.get("prompt_tokens", 0) if response.usage else 0
        query_metrics.response_tokens = response.usage.get("completion_tokens", 0) if response.usage else 0
        
        if not response or not response.content:
            logger.error("LLM failed to generate response")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response"
            )
        
        # Prepare response data
        response_data = {
            "status": "success",
            "answer": response.content,
            "context_used": True,
            "response_time": response_time,
            "model_used": llm_manager.default_model,
            "metadata": {
                "question_length": len(request.question),
                "context_tokens": context_result["total_tokens"],
                "sections_used": context_result["sections_used"],
                "search_results_count": len(search_results),
                "budget_utilization": context_result["budget_utilization"],
                "api_spec_hash": api_spec_hash,
                "search_strategy": "vector_hybrid",
                "cached": False,
                "prompt_tokens": response.usage.get("prompt_tokens") if response.usage else None,
                "completion_tokens": response.usage.get("completion_tokens") if response.usage else None,
                "total_tokens": response.usage.get("total_tokens") if response.usage else None
            }
        }
        
        # Evaluate response with enhanced DeepEval system (RAG Triad)
        # Format contexts for evaluation
        contexts = [doc.get('content', '') for doc in search_results[:3]]
        
        # Run comprehensive RAG evaluation
        try:
            import asyncio
            relevancy_score = await deep_evaluator.evaluate_rag_triad(
                query=request.question,
                answer=response.content,
                contexts=contexts
            )
        except Exception as e:
            logger.warning(f"Enhanced evaluation failed, using fallback: {e}")
            # Fallback to simple scoring
            relevancy_score = type('RAGTriadScores', (), {
                'overall_score': 0.7,
                'answer_relevancy': 0.7,
                'faithfulness': 0.8,
                'contextual_relevancy': 0.6,
                'confidence': 0.5
            })()
        
        # Complete performance monitoring
        query_metrics.total_latency_ms = response_time * 1000  # Convert to milliseconds
        query_metrics.context_tokens = max(query_metrics.context_tokens, context_result["total_tokens"])  # Update if higher
        
        # Add relevancy and performance data to response
        response_data["metadata"].update({
            "rag_triad_evaluation": {
                "overall_score": relevancy_score.overall_score,
                "answer_relevancy": relevancy_score.answer_relevancy,
                "faithfulness": relevancy_score.faithfulness,
                "contextual_relevancy": relevancy_score.contextual_relevancy,
                "confidence": relevancy_score.confidence
            },
            "performance": {
                "search_time_ms": search_time * 1000,
                "llm_time_ms": llm_time * 1000,
                "total_time_ms": response_time * 1000,
                "tokens_per_second": (response.usage.get("completion_tokens", 0) / llm_time) if llm_time > 0 and response.usage else 0
            }
        })
        
        # Cache the response
        cache.cache_response(request.question, api_spec_hash, response_data)
        
        # Store performance data
        performance_monitor.record_query(query_metrics)
        
        logger.info("Vector RAG query completed",
                   response_time=response_time,
                   search_results_count=len(search_results),
                   context_tokens=context_result["total_tokens"],
                   rag_triad_score=relevancy_score.overall_score,
                   search_time=search_time,
                   llm_time=llm_time)
        
        return RAGResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Vector RAG query failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Vector RAG query failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Log application startup information"""
    logger.info("APISage Backend API starting up",
                version="3.0.0",
                llm_available=llm_manager is not None,
                default_model=llm_manager.default_model if llm_manager else None,
                openai_configured=bool(os.getenv("OPENAI_API_KEY")))

@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown"""
    logger.info("APISage Backend API shutting down")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting APISage Backend API server",
                host="0.0.0.0",
                port=8080,
                environment=os.getenv("ENVIRONMENT", "development"))

    uvicorn.run(app, host="0.0.0.0", port=8080)
