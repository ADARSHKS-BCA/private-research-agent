import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path when running as a script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.agent.runner import run_agent_research
from app.ingestion.pipeline import ingest_url
from app.ingestion.qdrant_store import QdrantStore
from app.research_pipeline import run_research


def show_status():
    """Display connection status to Qdrant, models, and existing collections."""
    print(f"=== {settings.app_name} ===")
    print(f"Environment:       {settings.environment}")
    print(f"LLM Provider:      {settings.llm_provider.upper()} ({settings.groq_model if settings.llm_provider == 'groq' else settings.ollama_model})")
    print(f"Embedding Model:   {settings.embedding_model}")
    print(f"Qdrant Host:       {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"Target Collection: {settings.qdrant_collection}")

    try:
        store = QdrantStore()
        collections_resp = store.client.get_collections()
        print("\nConnected to Qdrant successfully!")
        print("Existing collections:")
        if not collections_resp.collections:
            print("  (None)")
        for col in collections_resp.collections:
            print(f"  - {col.name}")
    except Exception as e:
        print(f"\nWarning: Could not connect to Qdrant ({e})")


def main():
    parser = argparse.ArgumentParser(description="Private Research Agent - Dynamic Autonomous Research Engine")
    parser.add_argument("question", nargs="?", type=str, help="Research question to investigate", default=None)
    parser.add_argument("--research", type=str, help="Direct research question flag", default=None)
    parser.add_argument("--url", type=str, help="Ingest a specific single URL directly into Qdrant", default=None)
    parser.add_argument("--status", action="store_true", help="Show system and database status")
    parser.add_argument("--urls", type=int, help="Number of URLs to search and scrape", default=4)
    parser.add_argument("--top_k", type=int, help="Number of chunks to retrieve for synthesis", default=4)
    parser.add_argument("--max-iterations", type=int, help="Maximum research iterations", default=3)
    parser.add_argument("--linear", action="store_true", help="Use linear 1-pass pipeline instead of autonomous agent")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.url:
        try:
            result = ingest_url(args.url, verbose=True)
            print(f"\nDocument ID: {result['document_id']}")
            print(f"Chunks created: {result['chunks_created']}")
            print(f"Qdrant collection: {result['collection']}")
            print("\nIngestion completed successfully.")
        except Exception as e:
            print(f"\n[Error] Ingestion failed: {e}\n", file=sys.stderr)
            sys.exit(1)
    elif args.research or args.question:
        q = args.research or args.question
        try:
            if args.linear:
                run_research(question=q, num_urls=args.urls, top_k_retrieval=args.top_k, stream=True)
            else:
                run_agent_research(question=q, max_iterations=args.max_iterations)
        except Exception as e:
            print(f"\n[Error] Research execution failed: {e}\n", file=sys.stderr)
            sys.exit(1)
    else:
        # Default: Interactive Research Session
        print("=" * 70)
        print(f"  {settings.app_name} - Autonomous LangGraph Research Engine")
        print("  (Type 'quit', 'exit', 'bye', or 'stop' to end the session)")
        print("=" * 70)

        while True:
            try:
                user_q = input("\nWhat research topic would you like to investigate? ").strip()
                if not user_q:
                    continue
                if user_q.lower() in {"quit", "exit", "bye", "stop", "q"}:
                    print("\nEnding session. Goodbye!\n")
                    break

                if args.linear:
                    run_research(question=user_q, num_urls=args.urls, top_k_retrieval=args.top_k, stream=True)
                else:
                    run_agent_research(question=user_q, max_iterations=args.max_iterations)
            except (KeyboardInterrupt, EOFError):
                print("\n\nSession interrupted. Goodbye!\n")
                break
            except Exception as e:
                print(f"\n[Error] Research encountered an error: {e}\n")


if __name__ == "__main__":
    main()