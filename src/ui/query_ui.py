"""
Query Interface UI for Research Navigator
Gradio-based web interface for querying the RAG system.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from typing import Tuple
import json
from pathlib import Path
from src.pipeline.rag_pipeline import SimpleRAGPipeline


class QueryInterfaceUI:
    """Gradio UI for querying Research Navigator."""
    
    def __init__(self):
        """Initialize UI with RAG pipeline."""
        self.pipeline = SimpleRAGPipeline()
        self.stats = self.pipeline.get_stats()
        
        # Load example queries if available
        self.example_queries = self._load_example_queries()
        
    def _load_example_queries(self):
        """Load example queries from evaluation file."""
        try:
            eval_path = Path("tests/evaluation_queries.json")
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    data = json.load(f)
                    # Extract just the query text
                    return [q["query"] for q in data.get("queries", [])]
        except Exception as e:
            print(f"Could not load example queries: {e}")
        
        # Fallback examples
        return [
            "What are the main challenges in multi-agent AI systems?",
            "How does retrieval-augmented generation improve language models?",
            "What are recent advances in federated learning?",
            "Explain the concept of agentic AI workflows",
            "What methods are used for citation validation in RAG systems?"
        ]
    
    def process_query(
        self,
        query: str,
        n_chunks: int,
        temperature: float
    ) -> Tuple[str, str, str]:
        """
        Process a query and return formatted results.
        
        Returns:
            (answer, sources, metadata)
        """
        if not query.strip():
            return "Please enter a query.", "", ""
        
        try:
            # Run the RAG pipeline
            response = self.pipeline.query(
                question=query,
                n_chunks=n_chunks,
                max_tokens=10000,  # Required for gpt-5-nano reasoning model
                temperature=temperature
            )
            
            # Format answer
            answer = response.answer
            
            # Format sources
            sources_lines = []
            for i, source in enumerate(response.sources, 1):
                sources_lines.append(
                    f"**[{i}] {source.paper_title}**\n"
                    f"- Authors: {source.authors}\n"
                    f"- Year: {source.year}\n"
                    f"- arXiv: {source.arxiv_id}\n"
                    f"- Relevance Score: {1 - source.distance:.3f}\n"
                )
            sources = "\n".join(sources_lines)
            
            # Format metadata
            metadata = (
                f"⏱️ Response Time: {response.response_time:.2f} seconds\n"
                f"📚 Chunks Retrieved: {response.n_chunks_retrieved}\n"
                f"🔬 Generation Model: {self.stats['generation_model']}\n"
                f"🧮 Embedding Model: {self.stats['embedding_model']}"
            )
            
            return answer, sources, metadata
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            return error_msg, "", ""
    
    def create_interface(self):
        """Create and return the Gradio interface."""
        
        with gr.Blocks(title="Research Navigator - Query Interface") as interface:
            gr.Markdown("# 🔍 Research Navigator - Query Interface")
            gr.Markdown(
                f"**System Status:** {self.stats['total_chunks']:,} chunks indexed | "
                f"Collection: {self.stats['collection_name']}"
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Query input
                    query_input = gr.Textbox(
                        label="Research Query",
                        placeholder="Enter your research question...",
                        lines=3
                    )
                    
                    # Example queries dropdown
                    example_dropdown = gr.Dropdown(
                        choices=self.example_queries,
                        label="Example Queries (click to use)",
                        interactive=True
                    )
                    
                    # Settings
                    with gr.Row():
                        n_chunks_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Chunks to Retrieve"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Temperature (creativity)"
                        )
                    
                    # Submit button
                    submit_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    # Results
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=10
                    )
                    
                    with gr.Accordion("📚 Sources", open=True):
                        sources_output = gr.Markdown()
                    
                    with gr.Accordion("⚙️ Metadata", open=False):
                        metadata_output = gr.Textbox(lines=4)
            
            # Examples section
            gr.Markdown("## 💡 Tips")
            gr.Markdown("""
            - **Specific queries** work better than broad questions
            - Increase **chunks** for more comprehensive answers
            - Higher **temperature** = more creative responses (but potentially less accurate)
            - Check **sources** to verify information and find relevant papers
            """)
            
            # Event handlers
            def use_example(example):
                return example
            
            example_dropdown.change(
                fn=use_example,
                inputs=[example_dropdown],
                outputs=[query_input]
            )
            
            submit_btn.click(
                fn=self.process_query,
                inputs=[query_input, n_chunks_slider, temperature_slider],
                outputs=[answer_output, sources_output, metadata_output]
            )
            
            # Also trigger on Enter key
            query_input.submit(
                fn=self.process_query,
                inputs=[query_input, n_chunks_slider, temperature_slider],
                outputs=[answer_output, sources_output, metadata_output]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(**kwargs)


def main():
    """Main entry point."""
    ui = QueryInterfaceUI()
    ui.launch(share=False, server_port=7860)


if __name__ == "__main__":
    main()