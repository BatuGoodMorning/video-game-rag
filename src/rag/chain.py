"""RAG Chain using LangChain and Gemini.

Implements a simple RAG chain for game queries with
context from vector store retrieval.
"""

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from src.config import config
from src.rag.retriever import GameRetriever, RetrievalResult


# System prompt for the RAG chain
SYSTEM_PROMPT = """You are a knowledgeable video game expert assistant. Your role is to help users 
discover and learn about video games across PC, PlayStation 5, and Nintendo Switch platforms.

You have access to a database of game information including descriptions, gameplay details, 
reviews, and metadata. Use the provided context to answer questions accurately.

Guidelines:
- Be helpful and informative about video games
- Cite specific games when making recommendations
- Include relevant details like Metacritic scores, genres, and release dates when available
- If the context doesn't contain enough information, say so honestly
- Stay focused on video game topics

If asked about something unrelated to video games, politely redirect the conversation back to games."""


RAG_PROMPT_TEMPLATE = """Based on the following context about video games, please answer the user's question.

Context:
{context}

User Question: {question}

Provide a helpful, detailed answer based on the context. If the context doesn't fully answer the question, 
say what you can based on available information and note any limitations."""


class RAGChain:
    """Simple RAG chain for game queries."""
    
    def __init__(
        self,
        retriever: GameRetriever,
        google_api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
    ):
        """Initialize RAG chain.
        
        Args:
            retriever: GameRetriever instance
            google_api_key: Gemini API key
            model_name: Gemini model to use
            temperature: Generation temperature
        """
        self.retriever = retriever
        api_key = google_api_key or config.GOOGLE_API_KEY
        
        if not api_key:
            raise ValueError("Google API key is required")
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
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
            sales = metadata.get("sales_millions")
            
            sales_str = f" - Sales: {sales:.1f}M" if sales else ""
            context_parts.append(
                f"[{i}] {game_name} ({platform}){sales_str}\n"
                f"{text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def query(
        self,
        question: str,
        store: str = "pinecone",
        top_k: int = 5,
        platform: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> dict:
        """Process a query through the RAG pipeline.
        
        Args:
            question: User question
            store: Vector store to use
            top_k: Number of chunks to retrieve
            platform: Filter by platform
            genre: Filter by genre
            
        Returns:
            Dict with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieval_result = self.retriever.retrieve(
            query=question,
            store=store,
            top_k=top_k,
            platform=platform,
            genre=genre,
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
            "store_used": store,
        }
    
    def query_with_comparison(
        self,
        question: str,
        top_k: int = 5,
        platform: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> dict:
        """Query both stores and compare results.
        
        Returns:
            Dict with answers from both stores and comparison
        """
        # Get results from both stores
        pinecone_result, qdrant_result = self.retriever.retrieve_from_both(
            query=question,
            top_k=top_k,
            platform=platform,
            genre=genre,
        )
        
        # Generate answers using each store's context
        pinecone_context = self._format_context(pinecone_result)
        qdrant_context = self._format_context(qdrant_result)
        
        pinecone_answer = self.chain.invoke({
            "context": pinecone_context,
            "question": question,
        })
        
        qdrant_answer = self.chain.invoke({
            "context": qdrant_context,
            "question": question,
        })
        
        # Compare
        comparison = self.retriever.compare_results(pinecone_result, qdrant_result)
        
        return {
            "pinecone": {
                "answer": pinecone_answer,
                "sources": pinecone_result.get_game_names(),
                "latency_ms": pinecone_result.latency_ms,
            },
            "qdrant": {
                "answer": qdrant_answer,
                "sources": qdrant_result.get_game_names(),
                "latency_ms": qdrant_result.latency_ms,
            },
            "comparison": comparison,
        }

