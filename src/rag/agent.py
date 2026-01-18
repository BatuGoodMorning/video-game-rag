"""LangGraph Agent for multi-hop reasoning and guardrails.

Implements:
1. Input guardrail: Filter off-topic queries
2. Query routing: Simple vs complex queries
3. Multi-hop reasoning: For comparison/recommendation queries using similarity chunks
4. Output guardrail: Validate responses
"""

import logging
from typing import TypedDict, Literal, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.config import config
from src.rag.retriever import GameRetriever
from src.rag.chain import RAGChain


# State definition for the agent
class AgentState(TypedDict):
    """State passed between nodes in the graph."""
    
    query: str
    query_type: str  # "simple", "complex", "off_topic"
    retrieved_context: list[dict]
    intermediate_results: list[dict]
    final_answer: str
    sources: list[str]
    guardrail_status: str  # "passed", "blocked", "warning"
    guardrail_message: str
    error: Optional[str]


class InputGuardrail:
    """Validates input queries to filter off-topic requests."""
    
    GAMING_KEYWORDS = [
        "game", "play", "gaming", "console", "pc", "ps5", "playstation",
        "nintendo", "switch", "xbox", "rpg", "fps", "action", "adventure",
        "multiplayer", "singleplayer", "story", "gameplay", "graphics",
        "metacritic", "review", "score", "rating", "release", "developer",
        "publisher", "genre", "recommend", "similar", "best", "top",
        "witcher", "zelda", "mario", "elden", "god of war", "horizon",
    ]
    
    OFF_TOPIC_PATTERNS = [
        "weather", "stock", "recipe", "cook", "news", "politics",
        "medical", "health", "doctor", "legal", "lawyer", "code",
        "programming", "python", "javascript",
    ]
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize with optional LLM for complex classification."""
        self.llm = llm
    
    def check(self, query: str) -> tuple[bool, str]:
        """Check if query is gaming-related.
        
        Returns:
            Tuple of (is_valid, message)
        """
        query_lower = query.lower()
        
        # Check for gaming keywords
        has_gaming_context = any(kw in query_lower for kw in self.GAMING_KEYWORDS)
        
        # Check for off-topic patterns
        is_off_topic = any(pattern in query_lower for pattern in self.OFF_TOPIC_PATTERNS)
        
        if is_off_topic and not has_gaming_context:
            return False, "This system is designed to answer questions about video games. Please ask about PC, PS5, or Nintendo Switch games."
        
        if not has_gaming_context:
            if self.llm:
                return self._llm_classify(query)
            return True, "Query accepted (uncertain context)"
        
        return True, "Query is gaming-related"
    
    def _llm_classify(self, query: str) -> tuple[bool, str]:
        """Use LLM to classify ambiguous queries."""
        prompt = f"""Determine if this query is related to video games.
Query: "{query}"

Answer with only "YES" or "NO" followed by a brief reason."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip().upper()
            is_gaming = answer.startswith("YES")
            return is_gaming, response.content
        except Exception:
            return True, "Classification failed, allowing query"


class QueryRouter:
    """Routes queries to appropriate handling path."""
    
    COMPLEX_INDICATORS = [
        "like", "similar", "compare", "versus", "vs", "better than",
        "recommend based on", "if i liked", "alternatives to",
        "difference between", "pros and cons",
    ]
    
    def route(self, query: str) -> Literal["simple", "complex"]:
        """Determine if query needs multi-hop reasoning.
        
        Returns:
            "simple" for factual queries, "complex" for comparative/recommendation
        """
        query_lower = query.lower()
        
        for indicator in self.COMPLEX_INDICATORS:
            if indicator in query_lower:
                return "complex"
        
        return "simple"


class OutputGuardrail:
    """Validates output to check for hallucinations and quality."""
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        self.llm = llm
    
    def validate(
        self,
        answer: str,
        sources: list[str],
        retrieved_games: list[str],
        error: Optional[str] = None,
    ) -> tuple[str, str]:
        """Validate the generated answer.
        
        Returns:
            Tuple of (status, message) where status is "passed", "warning", or "blocked"
        """
        if error:
            return "blocked", f"Processing error: {error}"
        
        error_indicators = [
            "sorry, i encountered an error",
            "error processing your query",
            "something went wrong",
        ]
        if any(indicator in answer.lower() for indicator in error_indicators):
            return "blocked", "Response indicates an error occurred"
        
        warnings = []
        
        if len(answer) < 50:
            warnings.append("Answer seems too short")
        
        if len(answer) > 3000:
            warnings.append("Answer is very long, may contain unnecessary content")
        
        if sources and not any(game.lower() in answer.lower() for game in sources[:3]):
            warnings.append("Answer may not be grounded in retrieved sources")
        
        if warnings:
            return "warning", "; ".join(warnings)
        
        return "passed", "Answer validated successfully"


class GameRAGAgent:
    """LangGraph-based agent for game queries with guardrails and multi-hop reasoning."""
    
    def __init__(
        self,
        retriever: GameRetriever,
        google_api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """Initialize the agent.
        
        Args:
            retriever: GameRetriever instance
            google_api_key: Gemini API key
            model_name: Gemini model to use
        """
        api_key = google_api_key or config.GOOGLE_API_KEY
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )
        
        self.retriever = retriever
        self.rag_chain = RAGChain(retriever, google_api_key=api_key)
        
        # Initialize components
        self.input_guardrail = InputGuardrail(self.llm)
        self.query_router = QueryRouter()
        self.output_guardrail = OutputGuardrail(self.llm)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("input_guardrail", self._input_guardrail_node)
        workflow.add_node("route_query", self._route_query_node)
        workflow.add_node("simple_rag", self._simple_rag_node)
        workflow.add_node("multi_hop", self._multi_hop_node)
        workflow.add_node("output_guardrail", self._output_guardrail_node)
        workflow.add_node("reject", self._reject_node)
        
        # Set entry point
        workflow.set_entry_point("input_guardrail")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "input_guardrail",
            self._should_continue_after_input,
            {
                "continue": "route_query",
                "reject": "reject",
            }
        )
        
        workflow.add_conditional_edges(
            "route_query",
            self._get_route,
            {
                "simple": "simple_rag",
                "complex": "multi_hop",
            }
        )
        
        # All paths lead to output guardrail
        workflow.add_edge("simple_rag", "output_guardrail")
        workflow.add_edge("multi_hop", "output_guardrail")
        
        # Terminal edges
        workflow.add_edge("output_guardrail", END)
        workflow.add_edge("reject", END)
        
        return workflow.compile()
    
    def _input_guardrail_node(self, state: AgentState) -> AgentState:
        """Check if input is valid."""
        is_valid, message = self.input_guardrail.check(state["query"])
        
        return {
            **state,
            "guardrail_status": "passed" if is_valid else "blocked",
            "guardrail_message": message,
        }
    
    def _should_continue_after_input(self, state: AgentState) -> str:
        """Determine if we should continue after input guardrail."""
        return "continue" if state["guardrail_status"] == "passed" else "reject"
    
    def _route_query_node(self, state: AgentState) -> AgentState:
        """Route query to appropriate handler."""
        query_type = self.query_router.route(state["query"])
        return {**state, "query_type": query_type}
    
    def _get_route(self, state: AgentState) -> str:
        """Get the route from state."""
        return state["query_type"]
    
    def _simple_rag_node(self, state: AgentState) -> AgentState:
        """Handle simple factual queries."""
        try:
            logger.info(f"Processing simple query: {state['query'][:100]}...")
            result = self.rag_chain.query(
                question=state["query"],
                top_k=5,
                use_reranker=True,
            )
            logger.info(f"Simple RAG query successful, got {len(result.get('sources', []))} sources")
            
            return {
                **state,
                "final_answer": result["answer"],
                "sources": result["sources"],
                "retrieved_context": result["chunks"],
            }
        except Exception as e:
            import traceback
            error_details = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Simple RAG node error: {error_details}")
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": error_details,
                "final_answer": "Sorry, I encountered an error processing your query.",
            }
    
    def _multi_hop_node(self, state: AgentState) -> AgentState:
        """Handle complex queries requiring multi-hop reasoning with similarity chunks."""
        try:
            logger.info(f"Processing complex query: {state['query'][:100]}...")
            
            # Step 1: Extract the reference game (if any)
            reference_game = self._extract_reference_game(state["query"])
            logger.info(f"Extracted reference game: {reference_game}")
            
            intermediate_results = []
            all_chunks = []
            
            if reference_game:
                # Step 2: Get info about the reference game
                logger.info(f"Hop 1: Fetching info about reference game: {reference_game}")
                ref_result = self.retriever.retrieve(
                    query=f"Tell me about {reference_game}",
                    top_k=3,
                    use_reranker=True,
                )
                all_chunks.extend(ref_result.chunks)
                
                intermediate_results.append({
                    "step": "reference_game_info",
                    "game": reference_game,
                    "chunks_retrieved": len(ref_result.chunks),
                })
                
                # Step 3: Get similarity chunks for the reference game
                logger.info(f"Hop 2: Fetching similarity chunks for: {reference_game}")
                similarity_result = self.retriever.retrieve(
                    query=f"Games similar to {reference_game}",
                    top_k=3,
                    chunk_type="similarity",
                    use_reranker=False,  # Similarity chunks don't need reranking
                )
                all_chunks.extend(similarity_result.chunks)
                
                # Extract similar game names from similarity chunks
                similar_games = self._extract_similar_games_from_chunks(similarity_result.chunks)
                intermediate_results.append({
                    "step": "similarity_lookup",
                    "similar_games_found": similar_games,
                })
                
                # Step 4: Get details about similar games
                logger.info(f"Hop 3: Fetching details for {len(similar_games)} similar games")
                for similar_game in similar_games[:3]:  # Limit to top 3
                    game_result = self.retriever.retrieve(
                        query=f"Tell me about {similar_game}",
                        top_k=2,
                        use_reranker=True,
                    )
                    all_chunks.extend(game_result.chunks)
                
                intermediate_results.append({
                    "step": "similar_game_details",
                    "games_queried": similar_games[:3],
                })
            
            # Step 5: Also do a direct query for the original question
            logger.info("Hop 4: Direct query for original question")
            main_result = self.rag_chain.query(
                question=state["query"],
                top_k=5,
                use_reranker=True,
            )
            all_chunks.extend(main_result.get("chunks", []))
            
            # Step 6: Synthesize final answer with all collected context
            if intermediate_results:
                logger.info("Synthesizing final answer with multi-hop context...")
                
                # Build context from all chunks
                context_parts = []
                seen_texts = set()
                for chunk in all_chunks:
                    text = chunk.get("metadata", {}).get("text", "")
                    if text and text not in seen_texts:
                        game_name = chunk.get("metadata", {}).get("game_name", "Unknown")
                        chunk_type = chunk.get("metadata", {}).get("chunk_type", "detail")
                        context_parts.append(f"[{game_name}] ({chunk_type}): {text[:500]}")
                        seen_texts.add(text)
                
                context_text = "\n\n".join(context_parts[:15])  # Limit context size
                
                synthesis_prompt = f"""Based on the user's query and the retrieved information about games, 
provide a comprehensive answer. Use the similarity information to make relevant recommendations.

User Query: {state["query"]}

Retrieved Game Information:
{context_text}

Provide a helpful, detailed response that:
1. Answers the user's question directly
2. References specific games from the retrieved information
3. Explains why games are similar if making recommendations
4. Stays grounded in the provided context"""
                
                synthesis = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
                final_answer = synthesis.content
            else:
                final_answer = main_result["answer"]
            
            # Collect all unique game sources
            all_sources = list(set(
                chunk.get("metadata", {}).get("game_name", "")
                for chunk in all_chunks
                if chunk.get("metadata", {}).get("game_name")
            ))
            
            logger.info("Multi-hop query completed successfully")
            return {
                **state,
                "final_answer": final_answer,
                "sources": all_sources,
                "retrieved_context": all_chunks,
                "intermediate_results": intermediate_results,
            }
            
        except Exception as e:
            import traceback
            error_details = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Multi-hop node error: {error_details}")
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": error_details,
                "final_answer": "Sorry, I encountered an error processing your complex query.",
            }
    
    def _extract_reference_game(self, query: str) -> Optional[str]:
        """Extract a reference game from the query if present."""
        patterns = [
            "like ", "similar to ", "compared to ", "instead of ",
            "better than ", "alternatives to ",
        ]
        
        query_lower = query.lower()
        for pattern in patterns:
            if pattern in query_lower:
                idx = query_lower.find(pattern)
                after = query[idx + len(pattern):].strip()
                words = after.split()[:5]
                potential_game = " ".join(words).rstrip("?,.")
                if len(potential_game) > 2:
                    return potential_game
        
        return None
    
    def _extract_similar_games_from_chunks(self, chunks: list[dict]) -> list[str]:
        """Extract similar game names from similarity chunks."""
        similar_games = []
        
        for chunk in chunks:
            related = chunk.get("metadata", {}).get("related_games", [])
            if related:
                similar_games.extend(related)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_games = []
        for game in similar_games:
            if game not in seen:
                seen.add(game)
                unique_games.append(game)
        
        return unique_games
    
    def _output_guardrail_node(self, state: AgentState) -> AgentState:
        """Validate output."""
        retrieved_games = [
            c["metadata"].get("game_name", "") 
            for c in state.get("retrieved_context", [])
        ]
        
        status, message = self.output_guardrail.validate(
            answer=state.get("final_answer", ""),
            sources=state.get("sources", []),
            retrieved_games=retrieved_games,
            error=state.get("error"),
        )
        
        return {
            **state,
            "guardrail_status": status,
            "guardrail_message": message,
        }
    
    def _reject_node(self, state: AgentState) -> AgentState:
        """Handle rejected queries."""
        return {
            **state,
            "final_answer": state["guardrail_message"],
        }
    
    def query(self, question: str) -> dict:
        """Process a query through the agent.
        
        Args:
            question: User question
            
        Returns:
            Dict with answer, sources, and metadata
        """
        initial_state: AgentState = {
            "query": question,
            "query_type": "",
            "retrieved_context": [],
            "intermediate_results": [],
            "final_answer": "",
            "sources": [],
            "guardrail_status": "",
            "guardrail_message": "",
            "error": None,
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state["final_answer"],
            "sources": final_state["sources"],
            "query_type": final_state["query_type"],
            "guardrail_status": final_state["guardrail_status"],
            "guardrail_message": final_state["guardrail_message"],
            "intermediate_results": final_state["intermediate_results"],
            "error": final_state["error"],
        }
