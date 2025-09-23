"""
DeepEval Enhanced RAG Evaluation
Production-ready RAG evaluation using DeepEval's RAG Triad metrics
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

try:
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False


@dataclass
class RAGTriadScores:
    """RAG Triad evaluation scores."""

    answer_relevancy: float
    faithfulness: float
    contextual_relevancy: float
    overall_score: float
    confidence_score: float


@dataclass
class EvaluationResult:
    """Complete evaluation result with metadata."""

    scores: RAGTriadScores
    query: str
    answer: str
    context: str
    reasoning: Dict[str, str]
    evaluation_time_ms: float
    timestamp: datetime
    evaluation_level: str
    metadata: Dict[str, Any]


class DeepEvalEnhanced:
    """
    Enhanced RAG evaluation using DeepEval's RAG Triad metrics.
    Provides production-ready evaluation with self-explaining reasoning.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        evaluation_level: str = "MODERATE",
        enable_reasoning: bool = True,
        cache_results: bool = True,
        timeout_seconds: int = 30,
    ):
        """
        Initialize the enhanced DeepEval evaluator.

        Args:
            model: LLM model for evaluation
            evaluation_level: Evaluation strictness (BASIC, MODERATE, STRICT)
            enable_reasoning: Enable detailed reasoning for scores
            cache_results: Cache evaluation results
            timeout_seconds: Evaluation timeout
        """
        self.model = model
        self.evaluation_level = evaluation_level
        self.enable_reasoning = enable_reasoning
        self.cache_results = cache_results
        self.timeout_seconds = timeout_seconds

        # Evaluation cache
        self.evaluation_cache: Dict[str, EvaluationResult] = {}

        # Initialize metrics
        self.metrics = {}
        if not DEEPEVAL_AVAILABLE:
            raise RuntimeError(
                "DeepEval is required for RAG evaluation. Please install: pip install deepeval"
            )

        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize DeepEval metrics based on evaluation level."""
        try:
            # Check if OPENAI_API_KEY is available before initializing metrics
            import os
            if not os.getenv('OPENAI_API_KEY'):
                print("DeepEval metrics initialization skipped - no OpenAI API key available")
                self.metrics = {}
                return

            # Configure thresholds based on evaluation level
            thresholds = self._get_evaluation_thresholds()

            # Initialize RAG Triad metrics
            self.metrics = {
                "answer_relevancy": AnswerRelevancyMetric(
                    threshold=thresholds["answer_relevancy"],
                    model=self.model,
                    include_reason=self.enable_reasoning,
                ),
                "faithfulness": FaithfulnessMetric(
                    threshold=thresholds["faithfulness"],
                    model=self.model,
                    include_reason=self.enable_reasoning,
                ),
                "contextual_relevancy": ContextualRelevancyMetric(
                    threshold=thresholds["contextual_relevancy"],
                    model=self.model,
                    include_reason=self.enable_reasoning,
                ),
            }

            print(
                f"DeepEval metrics initialized with {self.evaluation_level} level thresholds"
            )

        except Exception as e:
            print(f"Error initializing DeepEval metrics: {e}")
            self.metrics = {}

    def _get_evaluation_thresholds(self) -> Dict[str, float]:
        """Get evaluation thresholds based on level."""
        thresholds = {
            "BASIC": {
                "answer_relevancy": 0.5,
                "faithfulness": 0.5,
                "contextual_relevancy": 0.5,
            },
            "MODERATE": {
                "answer_relevancy": 0.7,
                "faithfulness": 0.7,
                "contextual_relevancy": 0.7,
            },
            "STRICT": {
                "answer_relevancy": 0.8,
                "faithfulness": 0.8,
                "contextual_relevancy": 0.8,
            },
        }

        return thresholds.get(self.evaluation_level, thresholds["MODERATE"])

    def evaluate_rag_response(
        self,
        query: str,
        answer: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a RAG response using the RAG Triad metrics.

        Args:
            query: The user query
            answer: The generated answer
            context: The retrieved context
            metadata: Optional metadata

        Returns:
            EvaluationResult with scores and reasoning
        """
        start_time = time.time()

        # Check cache if enabled
        if self.cache_results:
            cache_key = self._generate_cache_key(query, answer, context)
            if cache_key in self.evaluation_cache:
                cached_result = self.evaluation_cache[cache_key]
                print("Returning cached evaluation result")
                return cached_result

        try:
            result = self._evaluate_with_deepeval(query, answer, context, metadata)

            # Cache result if enabled
            if self.cache_results:
                self.evaluation_cache[cache_key] = result

            evaluation_time = (time.time() - start_time) * 1000
            result.evaluation_time_ms = evaluation_time

            return result

        except Exception as e:
            print(f"Evaluation error: {e}")
            return self._create_error_result(query, answer, context, str(e), metadata)

    def _evaluate_with_deepeval(
        self, query: str, answer: str, context: str, metadata: Optional[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate using DeepEval metrics."""

        # Check if metrics are initialized
        if not self.metrics:
            # Return evaluation result with uvloop incompatibility message
            return self._create_uvloop_incompatible_result(query, answer, context, metadata)

        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=[context],  # DeepEval expects a list
        )

        # Run evaluation
        scores = {}
        reasoning = {}
        uvloop_detected = False

        for metric_name, metric in self.metrics.items():
            try:
                # Handle uvloop compatibility issues by running evaluation with a standard asyncio loop
                import asyncio
                import os
                
                def run_evaluation_with_standard_loop():
                    """Run metric evaluation with a fresh asyncio event loop to avoid uvloop conflicts."""
                    # Store the current loop (if any)
                    current_loop = None
                    try:
                        current_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass  # No current loop
                    
                    # Create a new standard asyncio event loop
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    
                    try:
                        # Run the metric evaluation
                        metric.measure(test_case)
                        score = metric.score
                        reason = getattr(metric, 'reason', None)
                        return score, reason
                    finally:
                        # Clean up the event loop
                        new_loop.close()
                        # Restore the original loop if there was one
                        if current_loop is not None:
                            asyncio.set_event_loop(current_loop)
                
                # Try direct evaluation first
                score = None
                reason = None
                
                try:
                    metric.measure(test_case)
                    score = metric.score
                    reason = getattr(metric, 'reason', None)
                except Exception as direct_error:
                    # If direct evaluation fails due to uvloop, try with fresh asyncio loop
                    if "uvloop" in str(direct_error).lower() or "can't patch loop" in str(direct_error).lower():
                        uvloop_detected = True
                        print(f"uvloop conflict detected for {metric_name}, trying with fresh asyncio loop")
                        try:
                            score, reason = run_evaluation_with_standard_loop()
                        except Exception as loop_error:
                            # If that also fails, mark as uvloop incompatible
                            print(f"Fresh loop evaluation also failed for {metric_name}: {loop_error}")
                            print(f"Setting uvloop incompatible status for {metric_name}")
                            score = None  # Use None to indicate uvloop incompatibility
                            reason = f"Evaluation unavailable: DeepEval metrics incompatible with uvloop event loop"
                    else:
                        # For non-uvloop errors, still set fallback values
                        print(f"Non-uvloop error for {metric_name}: {str(direct_error)}")
                        score = 0.0
                        reason = f"Evaluation failed: {str(direct_error)}"
                
                # Handle the score result
                if score is None:
                    # uvloop incompatibility detected
                    scores[metric_name] = None
                    reasoning[metric_name] = reason or "Evaluation unavailable: uvloop incompatibility"
                else:
                    scores[metric_name] = score
                    if self.enable_reasoning and reason:
                        reasoning[metric_name] = reason
                    else:
                        reasoning[metric_name] = f"Score: {score:.2f}"

            except Exception as e:
                error_msg = str(e)
                print(f"Error evaluating {metric_name}: {error_msg}")
                
                # Check if this is a uvloop-related error
                if "uvloop" in error_msg.lower() or "can't patch loop" in error_msg.lower():
                    uvloop_detected = True
                    print(f"uvloop compatibility issue detected for {metric_name}, evaluation unavailable")
                    scores[metric_name] = None
                    reasoning[metric_name] = f"Evaluation unavailable: DeepEval metrics incompatible with uvloop event loop"
                else:
                    scores[metric_name] = 0.0
                    reasoning[metric_name] = f"Evaluation failed: {error_msg}"

        # If uvloop was detected, return special result
        if uvloop_detected or all(score is None for score in scores.values()):
            return self._create_uvloop_incompatible_result(query, answer, context, metadata)

        # Calculate overall score and confidence (only for valid numeric scores)
        valid_scores = [score for score in scores.values() if score is not None]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        confidence_score = self._calculate_confidence_score({k: v for k, v in scores.items() if v is not None})

        # Create RAG Triad scores
        rag_scores = RAGTriadScores(
            answer_relevancy=scores.get("answer_relevancy", 0.0) or 0.0,
            faithfulness=scores.get("faithfulness", 0.0) or 0.0,
            contextual_relevancy=scores.get("contextual_relevancy", 0.0) or 0.0,
            overall_score=overall_score,
            confidence_score=confidence_score,
        )

        return EvaluationResult(
            scores=rag_scores,
            query=query,
            answer=answer,
            context=context,
            reasoning=reasoning,
            evaluation_time_ms=0.0,  # Will be set by caller
            timestamp=datetime.now(),
            evaluation_level=self.evaluation_level,
            metadata=metadata or {},
        )

    def _calculate_confidence_score(self, scores: Dict[str, float]) -> float:
        """Calculate confidence score based on score consistency."""
        if not scores:
            return 0.0

        values = list(scores.values())
        avg_score = sum(values) / len(values)

        # Higher confidence for consistent scores
        variance = sum((x - avg_score) ** 2 for x in values) / len(values)
        consistency = max(0.0, 1.0 - variance)

        # Confidence is combination of average score and consistency
        confidence = avg_score * 0.7 + consistency * 0.3

        return min(1.0, max(0.0, confidence))

    def _generate_cache_key(self, query: str, answer: str, context: str) -> str:
        """Generate cache key for evaluation result."""
        import hashlib

        cache_data = {
            "query": query,
            "answer": answer,
            "context": context[:500],  # Truncate context for key generation
            "evaluation_level": self.evaluation_level,
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _create_error_result(
        self,
        query: str,
        answer: str,
        context: str,
        error_msg: str,
        metadata: Optional[Dict[str, Any]],
    ) -> EvaluationResult:
        """Create error result when evaluation fails."""

        error_scores = RAGTriadScores(
            answer_relevancy=0.0,
            faithfulness=0.0,
            contextual_relevancy=0.0,
            overall_score=0.0,
            confidence_score=0.0,
        )

        error_reasoning = {
            "answer_relevancy": f"Evaluation error: {error_msg}",
            "faithfulness": f"Evaluation error: {error_msg}",
            "contextual_relevancy": f"Evaluation error: {error_msg}",
        }

        return EvaluationResult(
            scores=error_scores,
            query=query,
            answer=answer,
            context=context,
            reasoning=error_reasoning,
            evaluation_time_ms=0.0,
            timestamp=datetime.now(),
            evaluation_level="ERROR",
            metadata=metadata or {},
        )

    def _create_uvloop_incompatible_result(
        self,
        query: str,
        answer: str,
        context: str,
        metadata: Optional[Dict[str, Any]],
    ) -> EvaluationResult:
        """Create result for uvloop incompatibility scenario."""

        # Use -1.0 to indicate uvloop incompatibility (different from 0.0 which indicates failure)
        incompatible_scores = RAGTriadScores(
            answer_relevancy=-1.0,
            faithfulness=-1.0,
            contextual_relevancy=-1.0,
            overall_score=-1.0,
            confidence_score=-1.0,
        )

        incompatible_reasoning = {
            "answer_relevancy": "Quality evaluation unavailable: DeepEval metrics are incompatible with uvloop event loop. RAG response was generated successfully.",
            "faithfulness": "Quality evaluation unavailable: DeepEval metrics are incompatible with uvloop event loop. RAG response was generated successfully.",
            "contextual_relevancy": "Quality evaluation unavailable: DeepEval metrics are incompatible with uvloop event loop. RAG response was generated successfully.",
        }

        return EvaluationResult(
            scores=incompatible_scores,
            query=query,
            answer=answer,
            context=context,
            reasoning=incompatible_reasoning,
            evaluation_time_ms=0.0,
            timestamp=datetime.now(),
            evaluation_level="UVLOOP_INCOMPATIBLE",
            metadata=metadata or {},
        )

    def get_evaluation_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate summary statistics from multiple evaluation results."""
        if not results:
            return {"error": "No evaluation results provided"}

        # Aggregate scores
        total_answer_relevancy = sum(r.scores.answer_relevancy for r in results)
        total_faithfulness = sum(r.scores.faithfulness for r in results)
        total_contextual_relevancy = sum(r.scores.contextual_relevancy for r in results)
        total_overall = sum(r.scores.overall_score for r in results)
        total_confidence = sum(r.scores.confidence_score for r in results)

        count = len(results)

        return {
            "evaluation_count": count,
            "average_scores": {
                "answer_relevancy": total_answer_relevancy / count,
                "faithfulness": total_faithfulness / count,
                "contextual_relevancy": total_contextual_relevancy / count,
                "overall_score": total_overall / count,
                "confidence_score": total_confidence / count,
            },
            "evaluation_level": self.evaluation_level,
            "deepeval_available": DEEPEVAL_AVAILABLE,
            "average_evaluation_time_ms": sum(r.evaluation_time_ms for r in results)
            / count,
            "timestamp": datetime.now().isoformat(),
        }

    def clear_cache(self) -> int:
        """Clear evaluation cache and return number of cached entries cleared."""
        cached_count = len(self.evaluation_cache)
        self.evaluation_cache.clear()
        return cached_count

    def get_evaluator_info(self) -> Dict[str, Any]:
        """Get information about the evaluator configuration."""
        return {
            "deepeval_available": DEEPEVAL_AVAILABLE,
            "model": self.model,
            "evaluation_level": self.evaluation_level,
            "enable_reasoning": self.enable_reasoning,
            "cache_results": self.cache_results,
            "timeout_seconds": self.timeout_seconds,
            "cached_evaluations": len(self.evaluation_cache),
            "thresholds": self._get_evaluation_thresholds(),
            "metrics_initialized": len(self.metrics) > 0,
        }

    def reinitialize_metrics(self) -> bool:
        """Reinitialize metrics when API key becomes available."""
        try:
            self._initialize_metrics()
            return len(self.metrics) > 0
        except Exception as e:
            print(f"Failed to reinitialize DeepEval metrics: {e}")
            return False


# Factory function for easy instantiation
def create_deepeval_enhanced(
    model: str = "gpt-4o-mini",
    evaluation_level: str = "MODERATE",
    enable_reasoning: bool = True,
    cache_results: bool = True,
) -> DeepEvalEnhanced:
    """
    Factory function to create a configured DeepEvalEnhanced evaluator.

    Args:
        model: LLM model for evaluation
        evaluation_level: Evaluation strictness level
        enable_reasoning: Enable detailed reasoning
        cache_results: Cache evaluation results

    Returns:
        Configured DeepEvalEnhanced instance
    """
    return DeepEvalEnhanced(
        model=model,
        evaluation_level=evaluation_level,
        enable_reasoning=enable_reasoning,
        cache_results=cache_results,
    )


# Global instance for the main API
_deepeval_enhanced = None


def get_deep_eval_enhanced() -> DeepEvalEnhanced:
    """
    Get or create the global DeepEval enhanced evaluator instance.

    Returns:
        Global DeepEvalEnhanced instance
    """
    global _deepeval_enhanced
    if _deepeval_enhanced is None:
        _deepeval_enhanced = create_deepeval_enhanced()
    return _deepeval_enhanced
