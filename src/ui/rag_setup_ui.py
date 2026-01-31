"""
Research Navigator - RAG Setup UI
Week 2: Core RAG Infrastructure

Gradio interface for setting up the RAG system:
- Select dataset (papers JSON file)
- Configure embedding generation
- Process papers (chunk → embed → store)
- View statistics and progress
"""

import gradio as gr
import json
import time
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    CHROMADB_DIR,
    get_latest_papers_file,
    estimate_cost,
    print_config_summary
)
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore


class RAGSetupInterface:
    """
    Gradio interface for RAG system setup.
    """
    
    def __init__(self):
        """Initialize the RAG setup interface."""
        self.embedding_generator = None
        self.vector_store = None
        self.current_stats = {}
        
    def get_available_datasets(self) -> list:
        """
        Find all available papers JSON files.
        
        Returns:
            List of dataset file paths
        """
        # Primary location: data/processed/
        data_dir = Path("data/processed")
        
        if not data_dir.exists():
            # Fallback: check data/ folder
            data_dir = Path("data")
            if not data_dir.exists():
                return []
        
        # Find all papers_with_fulltext_*.json files
        json_files = sorted(
            data_dir.glob("papers_with_fulltext_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Most recent first
        )
        
        return [str(f) for f in json_files]
    
    def load_dataset_info(self, dataset_path: str) -> str:
        """
        Load and display information about selected dataset.
        
        Args:
            dataset_path: Path to papers JSON file
            
        Returns:
            Formatted dataset information
        """
        if not dataset_path or dataset_path == "No datasets found":
            return "⚠️ No dataset selected"
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            # Calculate statistics
            total_papers = len(papers)
            has_fulltext = sum(1 for p in papers if p.get('full_text') or p.get('fulltext'))
            
            # Sample metadata
            sample = papers[0] if papers else {}
            
            # Handle authors field (might be list of dicts or list of strings)
            authors = sample.get('authors', [])
            if authors:
                # Check if authors are dicts or strings
                if isinstance(authors[0], dict):
                    # Extract names from dicts
                    author_names = [a.get('name', str(a)) for a in authors[:3]]
                else:
                    # Already strings
                    author_names = authors[:3]
                authors_str = ', '.join(author_names)
            else:
                authors_str = 'N/A'
            
            info = f"""
📊 **Dataset Information**

**File:** `{Path(dataset_path).name}`
**Total Papers:** {total_papers:,}
**Papers with Full Text:** {has_fulltext:,} ({has_fulltext/total_papers*100:.1f}%)

**Sample Paper:**
- Title: {sample.get('title', 'N/A')[:80]}...
- Authors: {authors_str}
- Institution: {sample.get('institution', 'N/A')}
- Date: {sample.get('publication_date', sample.get('published', 'N/A'))}

**Estimated Processing:**
- Chunks: ~{has_fulltext * 15:,} (assuming 15 chunks/paper)
- Tokens: ~{has_fulltext * 15 * 500:,} (assuming 500 tokens/chunk)
- Cost: ~${has_fulltext * 15 * 500 * 0.00000002:.4f} (with {EMBEDDING_MODEL})
- Time: ~{has_fulltext * 15 * 0.34 / 60:.1f} minutes

✅ Ready to process!
"""
            return info
            
        except Exception as e:
            return f"❌ Error loading dataset: {e}"
    
    def generate_embeddings(
        self,
        dataset_path: str,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Generate embeddings for selected dataset.
        
        Args:
            dataset_path: Path to papers JSON file
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (status_message, statistics)
        """
        if not dataset_path or dataset_path == "No datasets found":
            return "❌ Please select a dataset first", ""
        
        try:
            # Initialize components
            progress(0, desc="Initializing...")
            self.embedding_generator = EmbeddingGenerator()
            self.vector_store = VectorStore()
            
            # Step 1: Load papers
            progress(0.1, desc="Loading papers...")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            status = f"📂 Loaded {len(papers)} papers\n"
            
            # Step 2: Chunk papers
            progress(0.2, desc="Chunking papers...")
            chunks = self.embedding_generator.chunker.chunk_papers(papers)
            
            status += f"📝 Created {len(chunks)} chunks\n"
            
            # Step 3: Estimate cost
            total_tokens = sum(c['tokens'] for c in chunks)
            cost = estimate_cost(total_tokens, EMBEDDING_MODEL)
            
            status += f"💰 Estimated cost: ${cost:.4f}\n"
            status += f"⏱️ Estimated time: ~{len(chunks) * 0.34 / 60:.1f} minutes\n\n"
            
            # Step 4: Generate embeddings
            progress(0.3, desc=f"Generating embeddings for {len(chunks)} chunks...")
            
            start_time = time.time()
            embedded_chunks = []
            
            for i, chunk in enumerate(chunks):
                # Update progress
                progress_pct = 0.3 + (0.5 * (i / len(chunks)))
                progress(
                    progress_pct,
                    desc=f"Embedding chunk {i+1}/{len(chunks)}..."
                )
                
                # Generate embedding
                embedding = self.embedding_generator.generate_embedding(chunk['text'])
                
                if embedding:
                    chunk['embedding'] = embedding
                    embedded_chunks.append(chunk)
                
                # Rate limiting
                if i < len(chunks) - 1:
                    time.sleep(0.1)
            
            elapsed = time.time() - start_time
            
            status += f"🔢 Generated {len(embedded_chunks)} embeddings in {elapsed:.1f}s\n"
            status += f"   Avg: {elapsed/len(embedded_chunks):.2f}s per embedding\n\n"
            
            # Step 5: Store in vector database
            progress(0.8, desc="Storing in ChromaDB...")
            
            stored_count = self.vector_store.add_chunks(
                embedded_chunks,
                show_progress=False
            )
            
            status += f"💾 Stored {stored_count} chunks in ChromaDB\n\n"
            
            # Step 6: Get statistics
            progress(0.95, desc="Generating statistics...")
            stats = self.vector_store.get_stats()
            
            stats_text = f"""
📊 **Vector Store Statistics**

**Collection:** {stats['collection_name']}
**Total Chunks:** {stats['total_chunks']:,}
**Embedding Dimensions:** {stats['embedding_dimensions']}
**Distance Metric:** {stats['distance_metric']}
**Storage Location:** `{stats['persist_directory']}`

**Processing Summary:**
- Papers processed: {len(papers):,}
- Chunks created: {len(chunks):,}
- Embeddings generated: {len(embedded_chunks):,}
- Successfully stored: {stored_count:,}
- Total time: {elapsed:.1f}s
- Actual cost: ${cost:.4f}

✅ **RAG system is ready!**

You can now query your papers using similarity search.
"""
            
            progress(1.0, desc="Complete!")
            
            status += "✅ Processing complete!\n"
            
            return status, stats_text
            
        except Exception as e:
            error_msg = f"❌ Error during processing: {e}\n\n"
            error_msg += "Common issues:\n"
            error_msg += "- Check OPENAI_API_KEY is set\n"
            error_msg += "- Verify you have API credits\n"
            error_msg += "- Ensure dataset has 'full_text' field\n"
            return error_msg, ""
    
    def get_current_stats(self) -> str:
        """
        Get current vector store statistics.
        
        Returns:
            Formatted statistics string
        """
        try:
            # Initialize vector store (will create collection if doesn't exist)
            store = VectorStore()
            stats = store.get_stats()
            
            if stats['total_chunks'] == 0:
                return f"""
ℹ️ **No data in vector store yet**

**Vector Store Location:** `{stats['persist_directory']}`
**Collection:** `{stats['collection_name']}`

Process a dataset to get started!
"""
            
            return f"""
📊 **Current Vector Store Status**

**Collection:** {stats['collection_name']}
**Total Chunks:** {stats['total_chunks']:,}
**Embedding Dimensions:** {stats['embedding_dimensions']}
**Storage:** `{stats['persist_directory']}`

**Available Metadata Fields:**
{', '.join(stats.get('sample_metadata_keys', []))}

✅ Ready for queries!
"""
        except Exception as e:
            return f"""
⚠️ **Error accessing vector store**

{str(e)}

**Expected location:** `{CHROMADB_DIR}`

This is normal if you haven't processed any datasets yet.
"""
    
    def build_interface(self) -> gr.Blocks:
        """
        Build the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Research Navigator - RAG Setup",
            theme=gr.themes.Soft()
        ) as interface:
            
            gr.Markdown("""
# 🚀 Research Navigator - RAG Setup

Set up your Retrieval-Augmented Generation system for academic literature analysis.

## Process:
1. **Select Dataset** - Choose your papers JSON file
2. **Review Settings** - Check configuration
3. **Generate Embeddings** - Process papers (chunk → embed → store)
4. **View Statistics** - Check results

---
""")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuration")
                    
                    # Configuration display
                    config_info = gr.Markdown(f"""
**Current Settings:**
- **Embedding Model:** `{EMBEDDING_MODEL}`
- **Chunk Size:** {CHUNK_SIZE} tokens
- **Chunk Overlap:** {CHUNK_OVERLAP} tokens
- **Collection:** `{COLLECTION_NAME}`
- **Storage:** `{CHROMADB_DIR}`

💡 *To change settings, edit `src/rag/config.py`*
""")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Current Status")
                    
                    # Current statistics
                    current_status = gr.Markdown(
                        self.get_current_stats()
                    )
                    
                    refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            
            gr.Markdown("---")
            
            # Dataset selection
            gr.Markdown("### 📂 Step 1: Select Dataset")
            
            datasets = self.get_available_datasets()
            
            if not datasets:
                dataset_dropdown = gr.Dropdown(
                    choices=["No datasets found"],
                    value="No datasets found",
                    label="Available Datasets",
                    interactive=False
                )
                gr.Markdown("⚠️ **No datasets found!** Run the Data Collector first to create papers JSON files.")
            else:
                dataset_dropdown = gr.Dropdown(
                    choices=datasets,
                    value=datasets[0] if datasets else None,
                    label="Select Papers JSON File",
                    info="Most recent file is selected by default"
                )
            
            dataset_info = gr.Markdown("ℹ️ Select a dataset to see details")
            
            gr.Markdown("---")
            
            # Processing section
            gr.Markdown("### 🔄 Step 2: Generate Embeddings")
            
            process_btn = gr.Button(
                "🚀 Generate Embeddings & Store in Vector DB",
                variant="primary",
                size="lg"
            )
            
            with gr.Row():
                with gr.Column():
                    processing_status = gr.Textbox(
                        label="Processing Status",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
                
                with gr.Column():
                    statistics_output = gr.Markdown(
                        "ℹ️ Statistics will appear here after processing"
                    )
            
            gr.Markdown("---")
            
            # Help section
            with gr.Accordion("📖 Help & Information", open=False):
                gr.Markdown("""
### How to Use This Interface:

1. **Select Dataset:** Choose your `papers_with_fulltext_*.json` file from the dropdown
2. **Review Info:** Check the dataset information (paper count, cost estimate)
3. **Generate Embeddings:** Click the big button to start processing
4. **Wait:** Processing takes ~25 minutes for 300 papers (~$0.05)
5. **View Results:** Check statistics when complete

### What Happens During Processing:

1. **Loading** - Reads your papers JSON file
2. **Chunking** - Splits papers into 800-token chunks with 100-token overlap
3. **Embedding** - Generates 1536-dimensional vectors using OpenAI API
4. **Storage** - Saves to ChromaDB vector database for fast retrieval

### Troubleshooting:

- **"No datasets found"** → Run Data Collector first
- **"API key not set"** → Check `.env` file has `OPENAI_API_KEY`
- **"No full_text field"** → Re-run Data Collector with PDF extraction
- **Rate limit errors** → Wait a moment and try again

### Next Steps After Setup:

Once processing completes, you can:
- Query papers using similarity search
- Test retrieval with evaluation queries
- Build the query interface
- Compare single-agent vs multi-agent RAG
""")
            
            # Event handlers
            dataset_dropdown.change(
                fn=self.load_dataset_info,
                inputs=[dataset_dropdown],
                outputs=[dataset_info]
            )
            
            process_btn.click(
                fn=self.generate_embeddings,
                inputs=[dataset_dropdown],
                outputs=[processing_status, statistics_output]
            )
            
            refresh_btn.click(
                fn=self.get_current_stats,
                outputs=[current_status]
            )
            
            # Auto-load first dataset info
            interface.load(
                fn=self.load_dataset_info,
                inputs=[dataset_dropdown],
                outputs=[dataset_info]
            )
        
        return interface


def launch_rag_setup_ui(share: bool = False):
    """
    Launch the RAG setup interface.
    
    Args:
        share: Whether to create a public share link
    """
    print("\n" + "="*70)
    print("🚀 RESEARCH NAVIGATOR - RAG SETUP UI")
    print("="*70)
    print_config_summary()
    
    interface = RAGSetupInterface()
    app = interface.build_interface()
    
    print("\n🌐 Launching Gradio interface...")
    print("="*70 + "\n")
    
    app.launch(
        share=share,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    """Launch the UI when run directly"""
    launch_rag_setup_ui(share=False)