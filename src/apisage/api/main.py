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
import uuid
from typing import Any, Dict, Optional, List
from datetime import datetime

import structlog
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator, Field

from apisage.core.llm_manager import LLMRequest, SimpleLLMManager, ModelConfig
from apisage.core.hybrid_search import HybridSearchEngine
from apisage.core.semantic_cache import SemanticCache
from apisage.core.context_assembler import ContextAssembler
from apisage.core.performance_monitor import PerformanceMonitor
from apisage.core.deepeval_enhanced import DeepEvalEnhanced
from apisage.core.chunking_strategy import APISpecChunker
from apisage.core.contextual_retrieval import ContextualRetrieval

# Configure logging first
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

# Enhanced performance monitoring imports
try:
    from apisage.core.enhanced_performance_monitor import (
        EnhancedPerformanceMonitor,
        global_monitor,
    )
    from apisage.core.latency_optimizer import (
        MultiTierCache,
        RequestBatcher,
        ConnectionPool,
    )
    from apisage.core.agentic_orchestrator import AgenticOrchestrator

    ENHANCED_MONITORING = True
except ImportError as e:
    logger.warning(f"Enhanced monitoring not available: {e}")
    ENHANCED_MONITORING = False


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


def serialize_evaluation_metrics(evaluation_metrics: Any) -> Dict[str, Any]:
    """
    Convert DeepEval EvaluationResult objects to JSON-serializable format.

    This function handles various evaluation result formats:
    - DeepEval EvaluationResult objects
    - Objects with scores and reasoning attributes
    - Dictionary-like objects with nested evaluation results
    - Generic objects with serializable attributes

    Args:
        evaluation_metrics: The evaluation metrics object to serialize

    Returns:
        Dict containing serialized evaluation data or error information

    Examples:
        >>> result = EvaluationResult(scores=RAGTriadScores(...), ...)
        >>> serialize_evaluation_metrics(result)
        {'scores': {'answer_relevancy': 0.8, ...}, 'reasoning': {...}, ...}

        >>> serialize_evaluation_metrics(None)
        {}

        >>> serialize_evaluation_metrics("invalid")
        {'error': 'Could not serialize evaluation metrics'}
    """
    if not evaluation_metrics:
        return {}

    def _serialize_scores(scores_obj: Any) -> Dict[str, Any]:
        """Helper function to serialize score objects dynamically."""
        if not scores_obj:
            return {}

        # Try to get all attributes that look like scores
        score_dict = {}
        if hasattr(scores_obj, "__dict__"):
            for attr_name, attr_value in scores_obj.__dict__.items():
                if not attr_name.startswith("_") and isinstance(
                    attr_value, (int, float)
                ):
                    score_dict[attr_name] = attr_value
        else:
            # Fallback to common score field names
            common_scores = [
                "answer_relevancy",
                "faithfulness",
                "contextual_relevancy",
                "overall_score",
                "confidence_score",
            ]
            for score_name in common_scores:
                if hasattr(scores_obj, score_name):
                    score_dict[score_name] = getattr(scores_obj, score_name, 0.0)

        return score_dict

    def _serialize_object_attributes(obj: Any) -> Dict[str, Any]:
        """Helper function to safely serialize object attributes."""
        try:
            return {
                k: v
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
                and isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
        except Exception as e:
            return {"error": f"Failed to serialize object attributes: {str(e)}"}

    def _create_error_response(
        error_msg: str, exception: Exception = None
    ) -> Dict[str, Any]:
        """Create consistent error response format."""
        return {
            "error": f"{error_msg}: {str(exception)}" if exception else error_msg,
            "serialization_failed": True,
        }

    # Handle EvaluationResult object directly - check for class name first
    if hasattr(evaluation_metrics, "__class__") and "EvaluationResult" in str(
        evaluation_metrics.__class__
    ):
        try:
            # Handle DeepEval EvaluationResult objects
            scores = getattr(evaluation_metrics, "scores", None)
            if scores:
                return {
                    "scores": _serialize_scores(scores),
                    "reasoning": getattr(evaluation_metrics, "reasoning", {}),
                    "evaluation_level": getattr(
                        evaluation_metrics, "evaluation_level", "UNKNOWN"
                    ),
                    "evaluation_time_ms": getattr(
                        evaluation_metrics, "evaluation_time_ms", 0.0
                    ),
                    "timestamp": str(getattr(evaluation_metrics, "timestamp", "")),
                    "query": getattr(evaluation_metrics, "query", ""),
                    "answer": getattr(evaluation_metrics, "answer", ""),
                    "context": getattr(evaluation_metrics, "context", ""),
                    "metadata": getattr(evaluation_metrics, "metadata", {}),
                }
            else:
                # Try to serialize all non-private attributes
                return _serialize_object_attributes(evaluation_metrics)
        except Exception as e:
            return _create_error_response("Failed to serialize EvaluationResult", e)

    # Handle objects with scores and reasoning attributes
    elif hasattr(evaluation_metrics, "scores") and hasattr(
        evaluation_metrics, "reasoning"
    ):
        try:
            scores = evaluation_metrics.scores
            return {
                "scores": _serialize_scores(scores),
                "reasoning": evaluation_metrics.reasoning
                if evaluation_metrics.reasoning
                else {},
                "evaluation_level": getattr(
                    evaluation_metrics, "evaluation_level", "UNKNOWN"
                ),
                "evaluation_time_ms": getattr(
                    evaluation_metrics, "evaluation_time_ms", 0.0
                ),
                "timestamp": str(getattr(evaluation_metrics, "timestamp", "")),
                "query": getattr(evaluation_metrics, "query", ""),
                "answer": getattr(evaluation_metrics, "answer", ""),
                "context": getattr(evaluation_metrics, "context", ""),
                "metadata": getattr(evaluation_metrics, "metadata", {}),
            }
        except Exception as e:
            return _create_error_response("Failed to serialize object with scores", e)

    # Handle dictionary format
    elif hasattr(evaluation_metrics, "items"):
        try:
            serialized = {}
            for key, value in evaluation_metrics.items():
                if hasattr(value, "score") and hasattr(value, "reason"):
                    # This is likely an EvaluationResult object
                    serialized[key] = {
                        "score": getattr(value, "score", None),
                        "reason": getattr(value, "reason", None),
                        "success": getattr(value, "success", True),
                    }
                elif hasattr(value, "__dict__"):
                    # Try to serialize object attributes
                    serialized[key] = _serialize_object_attributes(value)
                else:
                    # Already JSON serializable
                    serialized[key] = value
            return serialized
        except Exception as e:
            return _create_error_response("Failed to serialize dictionary format", e)

    # Fallback for other types - try to serialize object attributes
    elif hasattr(evaluation_metrics, "__dict__"):
        return _serialize_object_attributes(evaluation_metrics)

    # Final fallback
    try:
        return str(evaluation_metrics)
    except Exception as e:
        return _create_error_response("Could not serialize evaluation metrics", e)


# FastAPI app with enhanced configuration
app = FastAPI(
    title="APISage - AI-Powered OpenAPI Analysis",
    description="Intelligent API analysis using LLM for context-aware, specific feedback with performance monitoring",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add performance middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)  # Configure for production

# Initialize enhanced monitoring if available
enhanced_monitor = None
multi_tier_cache = None
request_batcher = None

if ENHANCED_MONITORING:
    enhanced_monitor = EnhancedPerformanceMonitor(
        service_name="apisage",
        otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
        enable_tracing=True,
        enable_metrics=True,
        enable_profiling=True,
    )

    multi_tier_cache = MultiTierCache(
        l1_size=100,
        l2_size=1000,
        semantic_cache_size=500,
        compression_threshold=1024,
        ttl_seconds=3600,
    )

    request_batcher = RequestBatcher(
        batch_size=10, batch_timeout_ms=100, max_queue_size=1000
    )


# Performance tracking middleware
@app.middleware("http")
async def add_performance_tracking(request: Request, call_next):
    """Middleware to track request performance"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    # Add request ID to state
    request.state.request_id = request_id

    # Enhanced monitoring if available
    if enhanced_monitor:
        with enhanced_monitor.profile_request(
            request_id=request_id, endpoint=request.url.path, method=request.method
        ) as profile:
            request.state.profile = profile
            response = await call_next(request)
    else:
        response = await call_next(request)

    # Add performance headers
    duration = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(round(duration, 2))

    return response


# Request/Response models
class AnalysisRequest(BaseModel):
    openapi_spec: Dict[str, Any]
    analysis_depth: Optional[str] = (
        "comprehensive"  # "quick", "standard", "comprehensive"
    )
    focus_areas: Optional[list] = (
        None  # ["security", "performance", "documentation", etc.]
    )
    enable_caching: Optional[bool] = True
    enable_monitoring: Optional[bool] = True


class AnalysisResponse(BaseModel):
    status: str
    analysis: str
    key_findings: Dict[str, Any]
    metadata: Dict[str, Any]
    agent_results: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    cache_hit: Optional[bool] = False


class RAGRequest(BaseModel):
    question: str
    openapi_spec: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    model_name: str  # Required model selection
    enable_caching: Optional[bool] = True
    enable_monitoring: Optional[bool] = True


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


class MultiAgentResponse(BaseModel):
    status: str
    consolidated_analysis: str
    agent_reports: Dict[str, Dict[str, Any]]
    response_time: float
    agents_used: List[str]
    metadata: Optional[Dict[str, Any]] = None


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

# Optional: DeepEval for RAG evaluation (gracefully handle if not installed)
try:
    rag_evaluator = DeepEvalEnhanced()
except RuntimeError:
    rag_evaluator = None
    logger.warning("DeepEval not available - RAG evaluation features will be limited")

api_chunker = APISpecChunker()

# Initialize Anthropic's Contextual Retrieval system
contextual_retrieval = ContextualRetrieval(llm_manager)

# Initialize Agentic Orchestrator for multi-agent analysis
agentic_orchestrator = AgenticOrchestrator(llm_manager) if ENHANCED_MONITORING else None


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with performance metrics"""
    health_status = {
        "status": "healthy",
        "service": "APISage AI-Powered Analysis with Performance Monitoring",
        "version": "3.1.0",
        "timestamp": datetime.now().isoformat(),
        "llm_available": llm_manager.is_available(),
        "enhanced_monitoring": ENHANCED_MONITORING,
    }

    # Add performance metrics if available
    if enhanced_monitor:
        try:
            perf_summary = enhanced_monitor.get_performance_summary()
            health_status["performance"] = {
                "active_requests": perf_summary.get("active_requests", 0),
                "completed_requests": perf_summary.get("completed_requests", 0),
                "error_rate": perf_summary.get("error_rate", 0),
            }
        except Exception as e:
            health_status["performance_error"] = str(e)

    # Add cache stats if available
    if multi_tier_cache:
        try:
            cache_stats = multi_tier_cache.get_stats()
            health_status["cache"] = {
                "hit_rate": cache_stats.get("hit_rate", 0),
                "l1_size": cache_stats.get("l1_size", 0),
                "l2_size": cache_stats.get("l2_size", 0),
            }
        except Exception as e:
            health_status["cache_error"] = str(e)

    return health_status


@app.get("/metrics")
async def get_performance_metrics():
    """Get detailed performance metrics"""
    if not ENHANCED_MONITORING:
        return {"error": "Enhanced monitoring not available"}

    metrics = {}

    if enhanced_monitor:
        metrics["performance"] = enhanced_monitor.get_performance_summary()
        metrics["recent_profiles"] = enhanced_monitor.export_profiles(limit=5)

    if multi_tier_cache:
        metrics["cache"] = multi_tier_cache.get_stats()

    if request_batcher:
        metrics["batcher"] = request_batcher.get_stats()

    return metrics


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
async def analyze_api(request: AnalysisRequest, req: Request):
    """Enhanced AI-powered API analysis with performance monitoring and caching"""

    start_time = time.time()
    request_id = getattr(req.state, "request_id", str(uuid.uuid4()))
    cache_hit = False

    try:
        # Check if LLM is available
        if not llm_manager.is_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service not available. Please set your OpenAI API key first.",
            )

        # Step 1: Check cache if enabled
        cached_result = None
        cache_key = None

        if request.enable_caching and multi_tier_cache:
            if enhanced_monitor:
                with enhanced_monitor.time_step(request_id, "cache_lookup"):
                    spec_hash = hashlib.sha256(
                        json.dumps(request.openapi_spec, sort_keys=True).encode()
                    ).hexdigest()
                    cache_key = f"analysis_{request.analysis_depth}_{spec_hash[:32]}"
                    cached_result = multi_tier_cache.get(cache_key)

                    if cached_result:
                        enhanced_monitor.record_cache_hit("analysis")
                        cache_hit = True
            else:
                # Fallback to basic caching
                spec_hash = hashlib.sha256(
                    json.dumps(request.openapi_spec, sort_keys=True).encode()
                ).hexdigest()
                cache_key = f"analysis_{request.analysis_depth}_{spec_hash[:32]}"
                cached_result = (
                    multi_tier_cache.get(cache_key) if multi_tier_cache else None
                )
                cache_hit = cached_result is not None

        if cached_result:
            logger.info("Cache hit for analysis", cache_key=cache_key[:16])
            return AnalysisResponse(
                status="success",
                analysis=cached_result["analysis"],
                key_findings=cached_result["key_findings"],
                metadata=cached_result["metadata"],
                performance_metrics={
                    "response_time_ms": (time.time() - start_time) * 1000
                },
                cache_hit=True,
            )
        elif multi_tier_cache:
            enhanced_monitor.record_cache_miss("analysis") if enhanced_monitor else None

        # Step 2: Create the analysis prompt with timing
        if enhanced_monitor:
            with enhanced_monitor.time_step(request_id, "prompt_creation"):
                prompt = create_analysis_prompt(
                    request.openapi_spec, request.analysis_depth, request.focus_areas
                )
        else:
            prompt = create_analysis_prompt(
                request.openapi_spec, request.analysis_depth, request.focus_areas
            )

        logger.info(
            "Starting LLM analysis",
            request_id=request_id,
            analysis_depth=request.analysis_depth,
            focus_areas=request.focus_areas,
            enhanced_monitoring=ENHANCED_MONITORING,
        )

        # Step 3: LLM request with timing
        llm_request = LLMRequest(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.3,  # Lower temperature for more consistent analysis
        )

        if enhanced_monitor:
            with enhanced_monitor.time_step(
                request_id, "llm_generation", model="o1-mini"
            ):
                llm_result = await llm_manager.generate(llm_request)

                if llm_result and llm_result.usage:
                    enhanced_monitor.record_tokens(
                        llm_result.usage.get("total_tokens", 0), model="o1-mini"
                    )
        else:
            llm_result = await llm_manager.generate(llm_request)

        if not llm_result or not llm_result.content:
            if enhanced_monitor:
                enhanced_monitor.record_error("LLM_GENERATION_FAILED", "/analyze")
            raise HTTPException(
                status_code=500, detail="Failed to generate analysis from LLM"
            )

        llm_response = llm_result.content
        token_usage = llm_result.usage if llm_result.usage else {}
        selected_model = llm_result.model if llm_result.model else "o1-mini"

        # Step 4: Parse response with timing
        if enhanced_monitor:
            with enhanced_monitor.time_step(request_id, "response_parsing"):
                parsed_response = parse_llm_response(llm_response)
        else:
            parsed_response = parse_llm_response(llm_response)

        # Step 5: Prepare response data
        analysis_result = {
            "analysis": llm_response,
            "key_findings": parsed_response["key_findings"],
            "metadata": {
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
                "request_id": request_id,
            },
        }

        # Step 6: Cache result if enabled
        if request.enable_caching and multi_tier_cache and cache_key:
            if enhanced_monitor:
                with enhanced_monitor.time_step(request_id, "cache_write"):
                    multi_tier_cache.put(cache_key, analysis_result, ttl=3600)
            else:
                multi_tier_cache.put(cache_key, analysis_result, ttl=3600)

        # Prepare performance metrics
        performance_metrics = {
            "response_time_ms": (time.time() - start_time) * 1000,
            "enhanced_monitoring": ENHANCED_MONITORING,
        }

        if enhanced_monitor:
            profile = enhanced_monitor.get_request_profile(request_id)
            if profile:
                performance_metrics.update(
                    {
                        "step_count": len(profile.steps),
                        "bottlenecks": profile.bottlenecks[:3]
                        if profile.bottlenecks
                        else [],
                    }
                )

        # Create enhanced response
        return AnalysisResponse(
            status="success",
            analysis=analysis_result["analysis"],
            key_findings=analysis_result["key_findings"],
            metadata=analysis_result["metadata"],
            performance_metrics=performance_metrics,
            cache_hit=cache_hit,
        )

    except HTTPException:
        if enhanced_monitor:
            enhanced_monitor.record_error("HTTP_EXCEPTION", "/analyze")
        raise
    except Exception as e:
        if enhanced_monitor:
            enhanced_monitor.record_error(str(type(e).__name__), "/analyze")
        logger.error(
            "Analysis failed", error=str(e), exc_info=True, request_id=request_id
        )
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
                    "metadata": {},
                }

            # Record cached query performance
            performance_monitor.record_query(
                query_id=f"rag_cached_{int(start_time)}_{hash(request.question) % 10000}",
                query=request.question,
                response_time_ms=response_time * 1000,
                tokens_used=0,  # No tokens used for cached response
                success=True,
                cache_hit=True,
                metadata={
                    "model": cached_data.get("model_used", selected_model),
                    "similarity_score": similarity_score,
                },
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
                    top_k=5,  # Retrieve top 5 enhanced chunks
                )

                # Extract contexts and sources from results
                retrieved_contexts = [result["content"] for result in search_results]
                context_sources = [
                    result["metadata"].get("source", f"contextual_chunk_{i}")
                    for i, result in enumerate(search_results)
                ]

                logger.info(
                    "Contextual retrieval completed",
                    num_contexts=len(retrieved_contexts),
                    context_sources=context_sources,
                    contextual_stats=contextual_retrieval.get_stats(),
                )

            except Exception as e:
                logger.error(
                    "Contextual retrieval failed, falling back to basic processing",
                    error=str(e),
                )
                # Fallback to basic chunking if contextual retrieval fails
                chunks = api_chunker.chunk_api_spec(request.openapi_spec)
                retrieved_contexts = [chunk.get("text", "") for chunk in chunks[:5]]
                context_sources = [
                    f"fallback_chunk_{i}" for i in range(len(retrieved_contexts))
                ]

        # Assemble optimized context
        assembled_context = ""
        if retrieved_contexts:
            # Convert retrieved contexts to SearchResult objects for ContextAssembler
            from apisage.core.context_assembler import SearchResult

            search_results = []
            for i, context in enumerate(retrieved_contexts):
                search_result = SearchResult(
                    content=context,
                    metadata={"source": f"contextual_chunk_{i}"},
                    vector_score=0.9,  # High score since these are from contextual retrieval
                    bm25_score=0.8,
                    hybrid_score=0.85,
                    chunk_id=f"contextual_chunk_{i}",
                    source_type="contextual_retrieval",
                )
                search_results.append(search_result)

            assembled_context_result = context_assembler.assemble_context(
                search_results=search_results, query=request.question
            )
            assembled_context = assembled_context_result.context

        # Create enhanced contextual RAG prompt
        context_section = "CONTEXTUALLY ENHANCED RETRIEVED CHUNKS:\n" + assembled_context if assembled_context else "Note: No relevant contextual chunks found in the API specification."

        rag_prompt = f"""You are an expert API documentation assistant using Anthropic's Contextual Retrieval system. You have access to contextually enhanced, relevant API specification chunks.

{context_section}

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
            **llm_params,  # Contains model and other optimal parameters
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
            logger.info(
                f"DEBUG: Requested model: {selected_model}, Response model: {response.model}"
            )

        # Evaluate RAG response using DeepEval
        evaluation_metrics = {}
        if retrieved_contexts and rag_evaluator:
            try:
                logger.info("Evaluating RAG response with DeepEval")
                eval_result = rag_evaluator.evaluate_rag_response(
                    query=request.question,
                    answer=response.content,
                    context="\n".join(retrieved_contexts) if retrieved_contexts else "",
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
                "prompt_tokens": response.usage.get("prompt_tokens")
                if response.usage
                else None,
                "completion_tokens": response.usage.get("completion_tokens")
                if response.usage
                else None,
                "total_tokens": response.usage.get("total_tokens")
                if response.usage
                else None,
            },
        }

        # Store in semantic cache as JSON string
        import json

        semantic_cache.put(
            query=request.question,
            response=json.dumps(cache_data),
            metadata={"model": response.model if response else selected_model},
        )

        # Serialize evaluation metrics for logging
        serialized_evaluation = (
            serialize_evaluation_metrics(evaluation_metrics)
            if evaluation_metrics
            else {}
        )

        # Extract evaluation score safely
        evaluation_score = None
        if serialized_evaluation:
            if isinstance(serialized_evaluation, dict):
                evaluation_score = serialized_evaluation.get("scores", {}).get(
                    "overall_score"
                )
            elif hasattr(serialized_evaluation, "scores"):
                # Handle case where EvaluationResult wasn't properly serialized
                scores = getattr(serialized_evaluation, "scores", None)
                if scores:
                    evaluation_score = getattr(scores, "overall_score", None)

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
                "num_contexts": len(retrieved_contexts),
            },
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
            metadata={"model": selected_model},
        )
        raise HTTPException(
            status_code=500, detail=f"Enhanced RAG query failed: {str(e)}"
        )


@app.post("/analyze-agentic")
async def analyze_agentic(request: AnalysisRequest):
    """
    Multi-Agent Analysis endpoint using specialized AI agents for comprehensive API review
    Uses SecurityAnalyst, DocumentationReviewer, PerformanceEngineer, and IntegrationSpecialist
    """
    if not agentic_orchestrator:
        logger.warning(
            "Multi-agent analysis requested but agentic orchestrator not available"
        )
        raise HTTPException(
            status_code=503,
            detail="Multi-agent analysis not available. Enhanced monitoring components required.",
        )

    logger.info(
        "Multi-agent analysis requested",
        spec_size=len(json.dumps(request.openapi_spec)) if request.openapi_spec else 0,
        default_model=llm_manager.default_model,
    )

    try:
        start_time = time.time()

        # Run multi-agent analysis
        analysis_result = await agentic_orchestrator.analyze(
            api_spec=request.openapi_spec, request_id=f"multiagent_{int(start_time)}"
        )

        response_time = time.time() - start_time

        # Record performance metrics (placeholder - enhanced_monitor methods may vary)
        # if enhanced_monitor:
        #     enhanced_monitor.record_metric("/analyze-agentic", response_time)

        logger.info(
            "Multi-agent analysis completed successfully",
            response_time=response_time,
            agents_count=len(analysis_result.agent_results),
            consolidated_length=len(analysis_result.consolidated_analysis),
        )

        return MultiAgentResponse(
            status="success",
            consolidated_analysis=analysis_result.consolidated_analysis,
            agent_reports={
                result.agent_name: {
                    "analysis": result.analysis,
                    "findings": result.findings,
                    "recommendations": result.recommendations,
                    "severity_score": result.severity_score,
                    "confidence": result.confidence,
                }
                for result in analysis_result.agent_results
            },
            response_time=response_time,
            agents_used=[result.agent_name for result in analysis_result.agent_results],
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": llm_manager.default_model,
                "spec_size": len(json.dumps(request.openapi_spec))
                if request.openapi_spec
                else 0,
            },
        )

    except Exception as e:
        logger.error("Multi-agent analysis failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Multi-agent analysis failed: {str(e)}"
        )


@app.post("/analyze-agentic-stream")
async def analyze_agentic_stream(request: AnalysisRequest):
    """
    Streaming Multi-Agent Analysis endpoint for real-time progress updates
    Returns Server-Sent Events (SSE) with progressive results
    """
    if not agentic_orchestrator:
        logger.warning(
            "Multi-agent analysis requested but agentic orchestrator not available"
        )
        raise HTTPException(
            status_code=503,
            detail="Multi-agent analysis not available. Enhanced monitoring components required.",
        )

    async def generate_streaming_analysis():
        """Generate streaming analysis with real-time progress updates"""
        try:
            start_time = time.time()
            request_id = f"multiagent_stream_{int(start_time)}"

            # Send initial status
            yield f"data: {json.dumps({'type': 'start', 'message': 'Initializing multi-agent analysis...', 'agents': ['SecurityAnalyst', 'DocumentationReviewer', 'PerformanceEngineer', 'IntegrationSpecialist']})}\n\n"

            # Progress tracking
            agent_statuses = {
                "SecurityAnalyst": "pending",
                "DocumentationReviever": "pending",
                "PerformanceEngineer": "pending",
                "IntegrationSpecialist": "pending",
            }

            # Send agent initialization status
            yield f"data: {json.dumps({'type': 'agents_initialized', 'statuses': agent_statuses})}\n\n"

            # Simulate progressive agent execution updates
            import asyncio

            # Agent 1: Security Analysis
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'SecurityAnalyst', 'message': 'Analyzing security vulnerabilities...'})}\n\n"
            await asyncio.sleep(0.5)  # Small delay for UX

            # Agent 2: Documentation Review
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'DocumentationReviewer', 'message': 'Evaluating API documentation quality...'})}\n\n"
            await asyncio.sleep(0.5)

            # Agent 3: Performance Analysis
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'PerformanceEngineer', 'message': 'Assessing performance implications...'})}\n\n"
            await asyncio.sleep(0.5)

            # Agent 4: Integration Review
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'IntegrationSpecialist', 'message': 'Reviewing integration patterns...'})}\n\n"
            await asyncio.sleep(0.5)

            # Run actual multi-agent analysis
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Running comprehensive analysis...', 'elapsed': round(time.time() - start_time, 1)})}\n\n"

            analysis_result = await agentic_orchestrator.analyze(
                api_spec=request.openapi_spec, request_id=request_id
            )

            # Send agent completion updates
            agents_used = [
                result.agent_name for result in analysis_result.agent_results
            ]
            for agent_name in agents_used:
                yield f"data: {json.dumps({'type': 'agent_complete', 'agent': agent_name, 'message': f'{agent_name} analysis completed'})}\n\n"
                await asyncio.sleep(0.2)

            # Send consolidated analysis in chunks for better UX
            consolidated_chunks = analysis_result.consolidated_analysis.split("\n\n")
            for i, chunk in enumerate(consolidated_chunks):
                if chunk.strip():
                    yield f"data: {json.dumps({'type': 'analysis_chunk', 'chunk_index': i, 'content': chunk, 'total_chunks': len(consolidated_chunks)})}\n\n"
                    await asyncio.sleep(0.1)

            # Send final results
            response_time = time.time() - start_time
            final_result = {
                "type": "complete",
                "status": "success",
                "consolidated_analysis": analysis_result.consolidated_analysis,
                "agent_reports": {
                    result.agent_name: {
                        "analysis": result.analysis,
                        "findings": result.findings,
                        "recommendations": result.recommendations,
                        "severity_score": result.severity_score,
                        "confidence": result.confidence,
                    }
                    for result in analysis_result.agent_results
                },
                "response_time": response_time,
                "agents_used": agents_used,
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "model_used": llm_manager.default_model,
                    "spec_size": len(json.dumps(request.openapi_spec))
                    if request.openapi_spec
                    else 0,
                },
            }

            yield f"data: {json.dumps(final_result)}\n\n"
            yield "data: [DONE]\n\n"

            logger.info(
                "Streaming multi-agent analysis completed",
                response_time=response_time,
                agents_count=len(agents_used),
                request_id=request_id,
            )

        except Exception as e:
            logger.error(
                "Streaming multi-agent analysis failed", error=str(e), exc_info=True
            )
            error_data = {
                "type": "error",
                "error": str(e),
                "message": "Multi-agent analysis failed. Please try again.",
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_streaming_analysis(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


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
                    partial_answer = " ".join(words[: i + 1])
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
                    retrieved_contexts = [
                        result["content"] for result in search_results
                    ]
                    context_sources = [
                        f"Section {i + 1}" for i in range(len(retrieved_contexts))
                    ]
                    yield f"data: {json.dumps({'type': 'progress', 'message': f'Found {len(retrieved_contexts)} relevant contexts'})}\n\n"
                except Exception as e:
                    logger.error(f"Contextual retrieval failed: {str(e)}")
                    retrieved_contexts = [json.dumps(request.openapi_spec, indent=2)]
                    context_sources = ["API Specification"]

            # Generate response
            context_content = (
                "\n\n".join(retrieved_contexts) if retrieved_contexts else ""
            )
            prompt_template = """You are an expert API documentation assistant. Help the user understand and work with their API specification.

{context_section}

User Question: {question}

Provide a helpful, accurate response with specific examples when possible."""

            context_section = (
                f"API Documentation Context:\n{context_content}\n"
                if context_content
                else "No specific API context provided."
            )
            formatted_prompt = prompt_template.format(
                context_section=context_section, question=request.question
            )

            yield f"data: {json.dumps({'type': 'generating', 'model': selected_model})}\n\n"

            llm_request = LLMRequest(
                prompt=formatted_prompt,
                model=selected_model,
                max_tokens=2000,
                temperature=0.1
                if not ModelConfig.is_o1_model(selected_model)
                else None,
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
                    "metadata": {},
                }
                semantic_cache.put(
                    query=request.question,
                    response=json.dumps(response_data),
                    metadata={"model": selected_model},
                )

            response_time = time.time() - start_time
            completion_data = {
                "type": "complete",
                "response_time": response_time,
                "cache_hit": False,
                "model": selected_model,
                "context_used": has_context,
                "contexts_found": len(retrieved_contexts),
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming RAG query failed: {str(e)}")
            error_data = {
                "type": "error",
                "error": str(e),
                "message": "An error occurred while processing your request",
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_streaming_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/plain; charset=utf-8",
        },
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


def main():
    """CLI entry point for the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
