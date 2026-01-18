"""TruLens evaluation integration for RAG quality monitoring.

Provides feedback functions for:
- Groundedness: Is the answer supported by retrieved context?
- Relevance: Is the retrieved context relevant to the query?
- Answer Relevance: Does the answer address the query?
- Hallucination detection
"""

import logging
from typing import Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FeedbackResult:
    """Result from a feedback function evaluation."""
    
    name: str
    score: float
    reason: str = ""
    metadata: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.score:.3f}"


@dataclass
class TruLensResult:
    """Complete TruLens evaluation result for a query."""
    
    query: str
    answer: str
    context: list[str]
    feedbacks: list[FeedbackResult] = field(default_factory=list)
    
    @property
    def groundedness_score(self) -> Optional[float]:
        for fb in self.feedbacks:
            if "groundedness" in fb.name.lower():
                return fb.score
        return None
    
    @property
    def relevance_score(self) -> Optional[float]:
        for fb in self.feedbacks:
            if "relevance" in fb.name.lower() and "answer" not in fb.name.lower():
                return fb.score
        return None
    
    @property
    def answer_relevance_score(self) -> Optional[float]:
        for fb in self.feedbacks:
            if "answer_relevance" in fb.name.lower():
                return fb.score
        return None
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "context": self.context,
            "feedbacks": [
                {"name": fb.name, "score": fb.score, "reason": fb.reason}
                for fb in self.feedbacks
            ],
        }


class TruLensEvaluator:
    """TruLens-based evaluation for RAG responses.
    
    Uses local feedback functions (no TruLens cloud required).
    For production, consider integrating with TruLens dashboard.
    """
    
    def __init__(
        self,
        llm=None,
        google_api_key: Optional[str] = None,
    ):
        """Initialize TruLens evaluator.
        
        Args:
            llm: Optional LLM for evaluation (will create one if not provided)
            google_api_key: API key for Gemini
        """
        self.llm = llm
        self.google_api_key = google_api_key
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM for feedback evaluation."""
        if self.llm is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from src.config import config
                
                api_key = self.google_api_key or config.GOOGLE_API_KEY
                if api_key:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=api_key,
                        temperature=0.0,  # Deterministic for evaluation
                    )
            except Exception as e:
                logger.warning(f"Could not initialize LLM for TruLens: {e}")
    
    def evaluate_groundedness(
        self,
        answer: str,
        context: list[str],
    ) -> FeedbackResult:
        """Evaluate if the answer is grounded in the context.
        
        Groundedness = How much of the answer is supported by the context?
        
        Args:
            answer: The generated answer
            context: List of context strings used to generate the answer
            
        Returns:
            FeedbackResult with score 0-1
        """
        if not self.llm:
            return FeedbackResult(
                name="groundedness",
                score=0.5,
                reason="LLM not available for evaluation",
            )
        
        context_text = "\n\n".join(context[:5])  # Limit context
        
        prompt = f"""Evaluate how well the answer is grounded in the provided context.

Context:
{context_text}

Answer:
{answer}

Rate the groundedness on a scale of 0 to 1:
- 1.0: Every claim in the answer is directly supported by the context
- 0.7-0.9: Most claims are supported, minor unsupported details
- 0.4-0.6: Some claims are supported, but significant content is not in context
- 0.1-0.3: Few claims are supported by context
- 0.0: The answer contradicts or is completely unrelated to context

Respond with ONLY a JSON object in this format:
{{"score": <float>, "reason": "<brief explanation>"}}"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = self._parse_json_response(response.content)
            
            return FeedbackResult(
                name="groundedness",
                score=float(result.get("score", 0.5)),
                reason=result.get("reason", ""),
            )
        except Exception as e:
            logger.error(f"Groundedness evaluation failed: {e}")
            return FeedbackResult(
                name="groundedness",
                score=0.5,
                reason=f"Evaluation failed: {str(e)}",
            )
    
    def evaluate_relevance(
        self,
        query: str,
        context: list[str],
    ) -> FeedbackResult:
        """Evaluate if the retrieved context is relevant to the query.
        
        Args:
            query: The user query
            context: List of retrieved context strings
            
        Returns:
            FeedbackResult with score 0-1
        """
        if not self.llm:
            return FeedbackResult(
                name="context_relevance",
                score=0.5,
                reason="LLM not available for evaluation",
            )
        
        context_text = "\n\n".join(context[:5])
        
        prompt = f"""Evaluate how relevant the retrieved context is to answering the query.

Query: {query}

Retrieved Context:
{context_text}

Rate the relevance on a scale of 0 to 1:
- 1.0: Context directly and completely addresses the query
- 0.7-0.9: Context is highly relevant with minor gaps
- 0.4-0.6: Context is partially relevant
- 0.1-0.3: Context has minimal relevance
- 0.0: Context is completely irrelevant

Respond with ONLY a JSON object in this format:
{{"score": <float>, "reason": "<brief explanation>"}}"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = self._parse_json_response(response.content)
            
            return FeedbackResult(
                name="context_relevance",
                score=float(result.get("score", 0.5)),
                reason=result.get("reason", ""),
            )
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            return FeedbackResult(
                name="context_relevance",
                score=0.5,
                reason=f"Evaluation failed: {str(e)}",
            )
    
    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> FeedbackResult:
        """Evaluate if the answer addresses the query.
        
        Args:
            query: The user query
            answer: The generated answer
            
        Returns:
            FeedbackResult with score 0-1
        """
        if not self.llm:
            return FeedbackResult(
                name="answer_relevance",
                score=0.5,
                reason="LLM not available for evaluation",
            )
        
        prompt = f"""Evaluate how well the answer addresses the query.

Query: {query}

Answer: {answer}

Rate the answer relevance on a scale of 0 to 1:
- 1.0: Answer directly and completely addresses the query
- 0.7-0.9: Answer mostly addresses the query with minor gaps
- 0.4-0.6: Answer partially addresses the query
- 0.1-0.3: Answer barely addresses the query
- 0.0: Answer does not address the query at all

Respond with ONLY a JSON object in this format:
{{"score": <float>, "reason": "<brief explanation>"}}"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = self._parse_json_response(response.content)
            
            return FeedbackResult(
                name="answer_relevance",
                score=float(result.get("score", 0.5)),
                reason=result.get("reason", ""),
            )
        except Exception as e:
            logger.error(f"Answer relevance evaluation failed: {e}")
            return FeedbackResult(
                name="answer_relevance",
                score=0.5,
                reason=f"Evaluation failed: {str(e)}",
            )
    
    def evaluate_hallucination(
        self,
        answer: str,
        context: list[str],
    ) -> FeedbackResult:
        """Detect hallucinations in the answer.
        
        Args:
            answer: The generated answer
            context: List of context strings
            
        Returns:
            FeedbackResult with score 0-1 (1 = no hallucination, 0 = hallucination)
        """
        if not self.llm:
            return FeedbackResult(
                name="no_hallucination",
                score=0.5,
                reason="LLM not available for evaluation",
            )
        
        context_text = "\n\n".join(context[:5])
        
        prompt = f"""Analyze the answer for hallucinations (claims not supported by context).

Context:
{context_text}

Answer:
{answer}

Identify any statements in the answer that are NOT supported by the context.
Rate the absence of hallucination on a scale of 0 to 1:
- 1.0: No hallucinations detected - all claims are in context
- 0.7-0.9: Minor unsupported details but no significant hallucinations
- 0.4-0.6: Some claims appear fabricated
- 0.1-0.3: Significant hallucinations present
- 0.0: Answer contains major fabrications

Respond with ONLY a JSON object in this format:
{{"score": <float>, "reason": "<list any hallucinated claims>"}}"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = self._parse_json_response(response.content)
            
            return FeedbackResult(
                name="no_hallucination",
                score=float(result.get("score", 0.5)),
                reason=result.get("reason", ""),
            )
        except Exception as e:
            logger.error(f"Hallucination evaluation failed: {e}")
            return FeedbackResult(
                name="no_hallucination",
                score=0.5,
                reason=f"Evaluation failed: {str(e)}",
            )
    
    def evaluate_response(
        self,
        query: str,
        answer: str,
        context: list[str],
    ) -> TruLensResult:
        """Run all feedback evaluations on a response.
        
        Args:
            query: The user query
            answer: The generated answer
            context: List of context strings used
            
        Returns:
            TruLensResult with all feedback scores
        """
        feedbacks = []
        
        # Run all evaluations
        feedbacks.append(self.evaluate_groundedness(answer, context))
        feedbacks.append(self.evaluate_relevance(query, context))
        feedbacks.append(self.evaluate_answer_relevance(query, answer))
        feedbacks.append(self.evaluate_hallucination(answer, context))
        
        return TruLensResult(
            query=query,
            answer=answer,
            context=context,
            feedbacks=feedbacks,
        )
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        import json
        import re
        
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            response = re.sub(r"```\w*\n?", "", response)
            response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            match = re.search(r'\{[^{}]*\}', response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            
            # Return default
            return {"score": 0.5, "reason": "Could not parse response"}
    
    def get_aggregate_scores(
        self,
        results: list[TruLensResult],
    ) -> dict[str, float]:
        """Aggregate scores across multiple results.
        
        Args:
            results: List of TruLensResult
            
        Returns:
            Dict with average scores for each feedback type
        """
        if not results:
            return {}
        
        scores = {}
        counts = {}
        
        for result in results:
            for fb in result.feedbacks:
                if fb.name not in scores:
                    scores[fb.name] = 0.0
                    counts[fb.name] = 0
                scores[fb.name] += fb.score
                counts[fb.name] += 1
        
        return {
            name: scores[name] / counts[name]
            for name in scores
        }

