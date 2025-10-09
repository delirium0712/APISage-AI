"""
uvloop-Compatible RAG Evaluation System
Replacement for DeepEval that avoids nest_asyncio compatibility issues
Provides RAG Triad metrics: Answer Relevancy, Faithfulness, Contextual Relevancy
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGMetrics:
    """RAG evaluation metrics container."""
    
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    contextual_relevancy: float = 0.0
    overall_score: float = 0.0
    evaluation_time: float = 0.0
    error_message: Optional[str] = None
    reasoning: Dict[str, str] = None


@dataclass
class EvaluationInput:
    """Input data for RAG evaluation."""
    
    query: str
    answer: str
    contexts: List[str]
    expected_answer: Optional[str] = None


class UvloopCompatibleRAGEvaluator:
    """
    RAG evaluation system that works with uvloop by using synchronous OpenAI API calls.
    Provides the same RAG Triad metrics as DeepEval without nest_asyncio dependency.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        evaluation_level: str = "BASIC",
        enable_caching: bool = True,
        timeout: int = 30
    ):
        """
        Initialize the evaluator.
        
        Args:
            openai_api_key: OpenAI API key for evaluation
            model: OpenAI model to use for evaluation
            evaluation_level: Evaluation strictness (BASIC, MODERATE, STRICT)
            enable_caching: Enable response caching for performance
            timeout: Request timeout in seconds
        """
        self.openai_api_key = openai_api_key
        self.model = model
        self.evaluation_level = evaluation_level
        self.enable_caching = enable_caching
        self.timeout = timeout
        
        # Cache for evaluation results
        self.cache = {} if enable_caching else None
        
        # Evaluation prompts for each metric
        self.prompts = self._initialize_prompts()
        
        # OpenAI API endpoint
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        logger.info(f"Initialized uvloop-compatible RAG evaluator with model {model}")
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize evaluation prompts for each RAG metric."""
        
        prompts = {
            "answer_relevancy": """
You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate how relevant the given answer is to the user's query.

Query: {query}
Answer: {answer}

Please evaluate the answer relevancy on a scale of 0.0 to 1.0 where:
- 1.0: Answer directly and completely addresses the query
- 0.7-0.9: Answer addresses most aspects of the query
- 0.4-0.6: Answer partially addresses the query
- 0.1-0.3: Answer barely relates to the query
- 0.0: Answer is completely irrelevant

Return your evaluation as a JSON object with this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your evaluation>"
}}
""",
            
            "faithfulness": """
You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate how faithful the given answer is to the provided contexts.

Contexts: {contexts}
Answer: {answer}

Please evaluate the faithfulness on a scale of 0.0 to 1.0 where:
- 1.0: Answer is completely supported by the contexts with no hallucinations
- 0.7-0.9: Answer is mostly supported with minor unsupported details
- 0.4-0.6: Answer is partially supported but contains some unsupported claims
- 0.1-0.3: Answer contains mostly unsupported or contradictory information
- 0.0: Answer is completely unsupported or contradicts the contexts

Return your evaluation as a JSON object with this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your evaluation>"
}}
""",
            
            "contextual_relevancy": """
You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate how relevant the provided contexts are to answering the user's query.

Query: {query}
Contexts: {contexts}

Please evaluate the contextual relevancy on a scale of 0.0 to 1.0 where:
- 1.0: All contexts are highly relevant and necessary for answering the query
- 0.7-0.9: Most contexts are relevant with some additional useful information
- 0.4-0.6: Some contexts are relevant but there's irrelevant information
- 0.1-0.3: Few contexts are relevant, mostly irrelevant information
- 0.0: Contexts are completely irrelevant to the query

Return your evaluation as a JSON object with this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your evaluation>"
}}
"""
        }
        
        return prompts
    
    def _generate_cache_key(self, eval_input: EvaluationInput, metric: str) -> str:
        """Generate cache key for evaluation result."""
        content = f"{metric}:{eval_input.query}:{eval_input.answer}:{':'.join(eval_input.contexts)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Make synchronous call to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert RAG evaluation system. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected OpenAI API response format: {e}")
            raise
    
    def _parse_evaluation_response(self, response: str) -> Tuple[float, str]:
        """Parse evaluation response and extract score and reasoning."""
        try:
            # Clean up response - remove code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            # Parse JSON
            result = json.loads(cleaned_response.strip())
            
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Clamp score to valid range
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            logger.error(f"Response was: {response}")
            return 0.0, f"Failed to parse response: {str(e)}"
    
    def _evaluate_metric(self, eval_input: EvaluationInput, metric: str) -> Tuple[float, str]:
        """Evaluate a single RAG metric."""
        
        # Check cache first
        if self.cache is not None:
            cache_key = self._generate_cache_key(eval_input, metric)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for {metric} evaluation")
                return self.cache[cache_key]
        
        # Format prompt
        prompt = self.prompts[metric].format(
            query=eval_input.query,
            answer=eval_input.answer,
            contexts="\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(eval_input.contexts)])
        )
        
        try:
            # Call OpenAI API
            response = self._call_openai_api(prompt)
            
            # Parse response
            score, reasoning = self._parse_evaluation_response(response)
            
            # Cache result
            if self.cache is not None:
                self.cache[cache_key] = (score, reasoning)
            
            logger.debug(f"Evaluated {metric}: score={score}")
            return score, reasoning
            
        except Exception as e:
            logger.error(f"Error evaluating {metric}: {e}")
            return 0.0, f"Evaluation failed: {str(e)}"
    
    def evaluate(self, eval_input: EvaluationInput) -> RAGMetrics:
        """
        Evaluate RAG response using all three metrics.
        
        Args:
            eval_input: Input data for evaluation
            
        Returns:
            RAGMetrics object with scores and reasoning
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not eval_input.query or not eval_input.answer or not eval_input.contexts:
                return RAGMetrics(
                    error_message="Invalid input: query, answer, and contexts are required",
                    evaluation_time=time.time() - start_time
                )
            
            # Evaluate each metric
            metrics = {}
            reasoning = {}
            
            for metric in ["answer_relevancy", "faithfulness", "contextual_relevancy"]:
                try:
                    score, reason = self._evaluate_metric(eval_input, metric)
                    metrics[metric] = score
                    reasoning[metric] = reason
                except Exception as e:
                    logger.error(f"Failed to evaluate {metric}: {e}")
                    metrics[metric] = 0.0
                    reasoning[metric] = f"Evaluation failed: {str(e)}"
            
            # Calculate overall score (weighted average)
            overall_score = (
                metrics["answer_relevancy"] * 0.4 +
                metrics["faithfulness"] * 0.4 +
                metrics["contextual_relevancy"] * 0.2
            )
            
            evaluation_time = time.time() - start_time
            
            result = RAGMetrics(
                answer_relevancy=metrics["answer_relevancy"],
                faithfulness=metrics["faithfulness"],
                contextual_relevancy=metrics["contextual_relevancy"],
                overall_score=overall_score,
                evaluation_time=evaluation_time,
                reasoning=reasoning
            )
            
            logger.info(f"RAG evaluation completed in {evaluation_time:.2f}s")
            logger.info(f"Scores - AR: {result.answer_relevancy:.3f}, F: {result.faithfulness:.3f}, CR: {result.contextual_relevancy:.3f}, Overall: {result.overall_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"RAG evaluation failed: {e}")
            return RAGMetrics(
                error_message=str(e),
                evaluation_time=time.time() - start_time
            )
    
    def get_evaluation_info(self) -> Dict[str, Any]:
        """Get information about the evaluator configuration."""
        return {
            "evaluator_type": "uvloop_compatible_rag_eval",
            "model": self.model,
            "evaluation_level": self.evaluation_level,
            "caching_enabled": self.enable_caching,
            "cache_size": len(self.cache) if self.cache else 0,
            "timeout": self.timeout,
            "metrics": ["answer_relevancy", "faithfulness", "contextual_relevancy"]
        }
    
    def clear_cache(self) -> int:
        """Clear evaluation cache and return number of cleared entries."""
        if self.cache is None:
            return 0
        
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {count} cached evaluation results")
        return count


# Factory function for easy instantiation
def create_rag_evaluator(
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    evaluation_level: str = "BASIC",
    enable_caching: bool = True
) -> UvloopCompatibleRAGEvaluator:
    """
    Factory function to create a configured RAG evaluator.
    
    Args:
        openai_api_key: OpenAI API key
        model: Model to use for evaluation
        evaluation_level: Evaluation strictness
        enable_caching: Enable result caching
        
    Returns:
        Configured UvloopCompatibleRAGEvaluator instance
    """
    return UvloopCompatibleRAGEvaluator(
        openai_api_key=openai_api_key,
        model=model,
        evaluation_level=evaluation_level,
        enable_caching=enable_caching
    )


# Global evaluator instance
_rag_evaluator = None


def get_rag_evaluator() -> UvloopCompatibleRAGEvaluator:
    """Get or create the global RAG evaluator instance."""
    global _rag_evaluator
    if _rag_evaluator is None:
        _rag_evaluator = create_rag_evaluator()
    return _rag_evaluator


def evaluate_rag_response(
    query: str,
    answer: str,
    contexts: List[str],
    openai_api_key: Optional[str] = None
) -> RAGMetrics:
    """
    Convenience function to evaluate a RAG response.
    
    Args:
        query: User query
        answer: Generated answer
        contexts: Retrieved contexts
        openai_api_key: OpenAI API key
        
    Returns:
        RAGMetrics with evaluation results
    """
    evaluator = get_rag_evaluator()
    
    # Update API key if provided
    if openai_api_key:
        evaluator.openai_api_key = openai_api_key
    
    eval_input = EvaluationInput(
        query=query,
        answer=answer,
        contexts=contexts
    )
    
    return evaluator.evaluate(eval_input)