"""Streamlit UI for Video Game RAG.

Features:
- Chat interface for game queries
- Platform/genre filters
- Source attribution
- Latency metrics
- Reranker toggle
"""

import streamlit as st
from typing import Optional

from src.config import config
from src.embeddings.embed import EmbeddingGenerator
from src.vectorstores.pinecone_store import PineconeStore
from src.rag.retriever import GameRetriever
from src.rag.chain import RAGChain
from src.rag.agent import GameRAGAgent
from src.rag.reranker import Reranker


# Page config
st.set_page_config(
    page_title="Video Game RAG",
    page_icon="🎮",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .source-chip {
        display: inline-block;
        background: rgba(233, 69, 96, 0.2);
        color: #e94560;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
    }
    
    .chunk-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #e94560;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .guardrail-passed {
        color: #4ade80;
    }
    
    .guardrail-warning {
        color: #fbbf24;
    }
    
    .guardrail-blocked {
        color: #ef4444;
    }
    
    .reranked-badge {
        background: rgba(74, 222, 128, 0.2);
        color: #4ade80;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_components():
    """Initialize RAG components (cached)."""
    # Check config
    missing = config.validate()
    if missing:
        return None, f"Missing API keys: {', '.join(missing)}"
    
    try:
        # Initialize embedder
        embedder = EmbeddingGenerator(
            model_key="mpnet",
            use_gpu=True,
            google_api_key=config.GOOGLE_API_KEY,
        )
        
        # Initialize Pinecone store
        pinecone_store = None
        if config.PINECONE_API_KEY:
            try:
                pinecone_store = PineconeStore(
                    api_key=config.PINECONE_API_KEY,
                    index_name=config.PINECONE_INDEX_NAME,
                    dimension=embedder.dimension,
                )
            except Exception as e:
                return None, f"Pinecone init failed: {e}"
        
        if not pinecone_store:
            return None, "Pinecone store not available"
        
        # Initialize reranker
        reranker = Reranker()
        
        # Initialize retriever with reranker
        retriever = GameRetriever(
            embedding_generator=embedder,
            pinecone_store=pinecone_store,
            reranker=reranker,
        )
        
        # Initialize agent
        agent = GameRAGAgent(
            retriever=retriever,
            google_api_key=config.GOOGLE_API_KEY,
        )
        
        return {
            "embedder": embedder,
            "pinecone": pinecone_store,
            "reranker": reranker,
            "retriever": retriever,
            "agent": agent,
            "chain": RAGChain(retriever, google_api_key=config.GOOGLE_API_KEY),
        }, None
        
    except Exception as e:
        return None, str(e)


def render_header():
    """Render the app header."""
    st.markdown('<h1 class="main-header">Video Game RAG</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask me anything about PC, PS5, and Nintendo Switch games</p>',
        unsafe_allow_html=True
    )


def render_sidebar(components: dict):
    """Render the sidebar with settings."""
    with st.sidebar:
        st.header("Settings")
        
        st.divider()
        
        # Filters
        st.subheader("Filters")
        
        platform = st.selectbox(
            "Platform",
            ["All", "PC", "PlayStation", "Switch"],
            index=0,
        )
        
        genre = st.selectbox(
            "Genre",
            ["All", "RPG", "Action", "Adventure", "Shooter", "Strategy", "Puzzle"],
            index=0,
        )
        
        st.divider()
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            top_k = st.slider("Results to retrieve", 3, 15, 5)
            use_agent = st.checkbox("Use LangGraph Agent", value=True)
            use_reranker = st.checkbox("Use Reranker", value=True)
        
        st.divider()
        
        # Stats
        st.subheader("Vector Store Stats")
        
        if components.get("pinecone"):
            try:
                stats = components["pinecone"].get_stats()
                st.metric("Pinecone Vectors", stats["total_vectors"])
            except:
                st.text("Pinecone: Not available")
        
        return {
            "platform": None if platform == "All" else platform,
            "genre": None if genre == "All" else genre,
            "top_k": top_k if 'top_k' in dir() else 5,
            "use_agent": use_agent if 'use_agent' in dir() else True,
            "use_reranker": use_reranker if 'use_reranker' in dir() else True,
        }


def render_sources(sources: list[str]):
    """Render source attribution chips."""
    if not sources:
        return
    
    st.markdown("**Sources:**")
    chips_html = "".join([f'<span class="source-chip">{s}</span>' for s in sources[:8]])
    st.markdown(chips_html, unsafe_allow_html=True)


def render_chunks(chunks: list[dict], expanded: bool = False):
    """Render retrieved chunks."""
    with st.expander("Retrieved Chunks", expanded=expanded):
        for i, chunk in enumerate(chunks[:10], 1):
            metadata = chunk.get("metadata", {})
            score = chunk.get("score", 0)
            text = metadata.get("text", "")[:500]
            game = metadata.get("game_name", "Unknown")
            chunk_type = metadata.get("chunk_type", "detail")
            
            type_badge = ""
            if chunk_type == "similarity":
                type_badge = " 🔗"
            elif chunk_type == "summary":
                type_badge = " 📋"
            
            st.markdown(f"""
            <div class="chunk-box">
                <strong>{i}. {game}{type_badge}</strong> (Score: {score:.3f})<br>
                <small>{text}...</small>
            </div>
            """, unsafe_allow_html=True)


def render_guardrail_status(status: str, message: str):
    """Render guardrail status indicator."""
    if status == "passed":
        st.markdown(f'<span class="guardrail-passed">✓ Guardrail: {message}</span>', unsafe_allow_html=True)
    elif status == "warning":
        st.markdown(f'<span class="guardrail-warning">⚠ Guardrail: {message}</span>', unsafe_allow_html=True)
    elif status == "blocked":
        st.markdown(f'<span class="guardrail-blocked">✗ Guardrail: {message}</span>', unsafe_allow_html=True)


def main():
    """Main app function."""
    render_header()
    
    # Initialize components
    components, error = init_components()
    
    if error:
        st.error(f"Initialization Error: {error}")
        st.info("Please check your .env file and ensure API keys are set.")
        st.code("""
# Required in .env:
PINECONE_API_KEY=your_key
GOOGLE_API_KEY=your_key
        """)
        return
    
    # Render sidebar and get settings
    settings = render_sidebar(components)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                render_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask about video games..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching game database..."):
                try:
                    if settings.get("use_agent", True):
                        # Use LangGraph agent
                        result = components["agent"].query(prompt)
                        response = result["answer"]
                        sources = result.get("sources", [])
                        
                        st.markdown(response)
                        render_sources(sources)
                        render_guardrail_status(
                            result.get("guardrail_status", ""),
                            result.get("guardrail_message", "")
                        )
                        
                        # Display actual error if present
                        if result.get("error"):
                            with st.expander("Error Details", expanded=True):
                                st.error(f"**Error:** {result['error']}")
                        
                        if result.get("intermediate_results"):
                            with st.expander("Multi-hop Reasoning Steps"):
                                for step in result["intermediate_results"]:
                                    st.json(step)
                    else:
                        # Simple RAG chain
                        result = components["chain"].query(
                            question=prompt,
                            top_k=settings.get("top_k", 5),
                            platform=settings["platform"],
                            genre=settings["genre"],
                            use_reranker=settings.get("use_reranker", True),
                        )
                        response = result["answer"]
                        sources = result.get("sources", [])
                        
                        st.markdown(response)
                        render_sources(sources)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Retrieval Latency", f"{result['retrieval_latency_ms']:.1f}ms")
                        if result.get("reranked"):
                            col2.markdown('<span class="reranked-badge">Reranked ✓</span>', unsafe_allow_html=True)
                        
                        render_chunks(result.get("chunks", []))
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)


if __name__ == "__main__":
    main()
