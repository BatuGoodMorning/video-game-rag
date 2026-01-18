"""RAG Chain using LangChain and Gemini.

Implements a RAG chain for game queries with
context from vector store retrieval.
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import config
from src.rag.retriever import GameRetriever, RetrievalResult
from src.llm.factory import create_llm


# System prompt for the RAG chain
SYSTEM_PROMPT = """You are a knowledgeable video game expert assistant. Your role is to help users 
discover and learn about video games across PC, PlayStation 5, and Nintendo Switch platforms.

You have access to a database of game information including descriptions, gameplay details, 
reviews, and metadata. Use the provided context to answer questions accurately.

Guidelines:
- Be helpful and informative about video games
- Cite specific games when making recommendations
- Include relevant details like sales figures, genres, and release dates when available
- If the context doesn't contain enough information, say so honestly
- Stay focused on video game topics
- When comparing games or recommending similar games, use the similarity information provided

If asked about something unrelated to video games, politely redirect the conversation back to games."""


RAG_PROMPT_TEMPLATE = """Based on the following context about video games, please answer the user's question.

Context:
{context}

User Question: {question}

Provide a helpful, detailed answer based on the context. If the context doesn't fully answer the question, 
say what you can based on available information and note any limitations."""


class RAGChain:
    """RAG chain for game queries."""
    
    def __init__(
        self,
        retriever: GameRetriever,
        google_api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
    ):
        """Initialize RAG chain.
        
        Args:
            retriever: GameRetriever instance
            google_api_key: Google API key (for Gemini API mode)
            model_name: Model name to use
            temperature: Generation temperature
        """
        self.retriever = retriever
        
        # Create LLM using factory (supports both Gemini API and Vertex AI)
        self.llm = create_llm(
            model_name=model_name,
            temperature=temperature,
            google_api_key=google_api_key,
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", RAG_PROMPT_TEMPLATE),
        ])
        
        # Output parser
        self.output_parser = StrOutputParser()
        
        # Build chain
        self.chain = self.prompt | self.llm | self.output_parser
    
    def _format_context(self, retrieval_result: RetrievalResult) -> str:
        """Format retrieved chunks as context string."""
        context_parts = []
        
        for i, chunk in enumerate(retrieval_result.chunks, 1):
            metadata = chunk["metadata"]
            text = metadata.get("text", "")
            game_name = metadata.get("game_name", "Unknown")
            platform = metadata.get("platform", "")
            chunk_type = metadata.get("chunk_type", "detail")
            sales = metadata.get("sales_millions")
            
            # Format sales, handling both string and numeric types
            sales_str = ""
            if sales is not None and sales != "":
                try:
                    sales_float = float(sales)
                    if isinstance(sales_float, (int, float)) and not isinstance(sales_float, bool):
                        sales_str = f" - Sales: {sales_float:.1f}M"
                    else:
                        sales_str = f" - Sales: {sales}M"
                except (ValueError, TypeError, AttributeError):
                    sales_str = f" - Sales: {sales}M"
            
            # Add chunk type indicator for similarity chunks
            type_indicator = ""
            if chunk_type == "similarity":
                type_indicator = " [SIMILAR GAMES]"
            
            context_parts.append(
                f"[{i}] {game_name} ({platform}){sales_str}{type_indicator}\n"
                f"{text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        platform: Optional[str] = None,
        genre: Optional[str] = None,
        use_reranker: bool = True,
    ) -> dict:
        """Process a query through the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            platform: Filter by platform
            genre: Filter by genre
            use_reranker: Whether to use reranker
            
        Returns:
            Dict with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            platform=platform,
            genre=genre,
            use_reranker=use_reranker,
        )
        
        # Format context
        context = self._format_context(retrieval_result)
        
        # Generate answer
        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })
        
        return {
            "answer": answer,
            "sources": retrieval_result.get_game_names(),
            "chunks": retrieval_result.chunks,
            "retrieval_latency_ms": retrieval_result.latency_ms,
            "reranked": retrieval_result.reranked,
        }
