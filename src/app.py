"""Streamlit UI for Video Game RAG.

Features:
- Chat interface for game queries
- Vector DB comparison mode
- Platform/genre filters
- Source attribution
- Latency metrics
"""

import streamlit as st
from typing import Optional

from src.config import config
from src.embeddings.embed import EmbeddingGenerator
from src.vectorstores.pinecone_store import PineconeStore
from src.vectorstores.qdrant_store import QdrantStore
from src.rag.retriever import GameRetriever
from src.rag.chain import RAGChain
from src.rag.agent import GameRAGAgent


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
    
    .comparison-win {
        border: 2px solid #4ade80;
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
        
        # Initialize vector stores
        pinecone_store = None
        qdrant_store = None
        
        if config.PINECONE_API_KEY:
            try:
                pinecone_store = PineconeStore(
                    api_key=config.PINECONE_API_KEY,
                    index_name=config.PINECONE_INDEX_NAME,
                    dimension=embedder.dimension,
                )
            except Exception as e:
                st.warning(f"Pinecone init failed: {e}")
        
        try:
            qdrant_store = QdrantStore(
                url=config.QDRANT_URL if config.QDRANT_API_KEY else None,
                api_key=config.QDRANT_API_KEY if config.QDRANT_API_KEY else None,
                collection_name="video_games",
                dimension=embedder.dimension,
                use_local=not config.QDRANT_API_KEY,
            )
        except Exception as e:
            st.warning(f"Qdrant init failed: {e}")
        
        if not pinecone_store and not qdrant_store:
            return None, "No vector stores available"
        
        # Initialize retriever
        retriever = GameRetriever(
            embedding_generator=embedder,
            pinecone_store=pinecone_store,
            qdrant_store=qdrant_store,
        )
        
        # Initialize agent
        agent = GameRAGAgent(
            retriever=retriever,
            google_api_key=config.GOOGLE_API_KEY,
        )
        
        return {
            "embedder": embedder,
            "pinecone": pinecone_store,
            "qdrant": qdrant_store,
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
        
        # Vector DB selection
        available_stores = []
        if components.get("pinecone"):
            available_stores.append("Pinecone (HNSW)")
        if components.get("qdrant"):
            available_stores.append("Qdrant (IVF+PQ)")
        if len(available_stores) == 2:
            available_stores.append("Compare Both")
        
        store_mode = st.selectbox(
            "Vector Database",
            available_stores,
            index=0,
        )
        
        st.divider()
        
        # Filters
        st.subheader("Filters")
        
        platform = st.selectbox(
            "Platform",
            ["All", "PC", "PS5", "Switch"],
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
        
        st.divider()
        
        # Stats
        st.subheader("Vector Store Stats")
        
        if components.get("pinecone"):
            try:
                stats = components["pinecone"].get_stats()
                st.metric("Pinecone Vectors", stats["total_vectors"])
            except:
                st.text("Pinecone: Not available")
        
        if components.get("qdrant"):
            try:
                stats = components["qdrant"].get_stats()
                st.metric("Qdrant Vectors", stats["total_vectors"])
            except:
                st.text("Qdrant: Not available")
        
        return {
            "store_mode": store_mode,
            "platform": None if platform == "All" else platform,
            "genre": None if genre == "All" else genre,
            "top_k": top_k if 'top_k' in dir() else 5,
            "use_agent": use_agent if 'use_agent' in dir() else True,
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
            
            st.markdown(f"""
            <div class="chunk-box">
                <strong>{i}. {game}</strong> (Score: {score:.3f})<br>
                <small>{text}...</small>
            </div>
            """, unsafe_allow_html=True)


def render_comparison(result: dict):
    """Render comparison between vector stores."""
    col1, col2 = st.columns(2)
    
    with col1:
        latency = result["pinecone"]["latency_ms"]
        is_faster = latency <= result["qdrant"]["latency_ms"]
        
        st.markdown(f"""
        ### Pinecone (HNSW)
        **Latency:** {latency:.1f}ms {"✓" if is_faster else ""}
        """)
        st.markdown(result["pinecone"]["answer"])
        render_sources(result["pinecone"]["sources"])
    
    with col2:
        latency = result["qdrant"]["latency_ms"]
        is_faster = latency < result["pinecone"]["latency_ms"]
        
        st.markdown(f"""
        ### Qdrant (IVF+PQ)
        **Latency:** {latency:.1f}ms {"✓" if is_faster else ""}
        """)
        st.markdown(result["qdrant"]["answer"])
        render_sources(result["qdrant"]["sources"])
    
    # Comparison stats
    st.divider()
    comp = result["comparison"]
    
    cols = st.columns(4)
    cols[0].metric("Latency Diff", f"{abs(comp['latency_diff_ms']):.1f}ms")
    cols[1].metric("Overlap Ratio", f"{comp['overlap_ratio']:.0%}")
    cols[2].metric("Common Games", len(comp["common_games"]))
    cols[3].metric("Unique Results", len(comp["only_pinecone"]) + len(comp["only_qdrant"]))


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
                    if settings["store_mode"] == "Compare Both":
                        # Comparison mode
                        result = components["chain"].query_with_comparison(
                            question=prompt,
                            top_k=settings.get("top_k", 5),
                            platform=settings["platform"],
                            genre=settings["genre"],
                        )
                        render_comparison(result)
                        response = "See comparison above"
                        sources = result["comparison"]["common_games"]
                        
                    elif settings.get("use_agent", True):
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
                        
                        if result.get("intermediate_results"):
                            with st.expander("Multi-hop Reasoning Steps"):
                                for step in result["intermediate_results"]:
                                    st.json(step)
                    else:
                        # Simple RAG chain
                        store = "pinecone" if "Pinecone" in settings["store_mode"] else "qdrant"
                        result = components["chain"].query(
                            question=prompt,
                            store=store,
                            top_k=settings.get("top_k", 5),
                            platform=settings["platform"],
                            genre=settings["genre"],
                        )
                        response = result["answer"]
                        sources = result.get("sources", [])
                        
                        st.markdown(response)
                        render_sources(sources)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Retrieval Latency", f"{result['retrieval_latency_ms']:.1f}ms")
                        col2.metric("Store Used", result["store_used"])
                        
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

