"""
Gradio UI for Research Navigator - Data Collection Module
"""

import gradio as gr
import json
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data.collectors.pdf_extractor import PDFExtractor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.collectors.arxiv_collector import ArxivCollector
from src.data.collectors.semantic_scholar import SemanticScholarEnricher


# Global variables
current_papers = []
current_timestamp = None
confirmed_selection = []


def collect_papers(
    swiss_inst,
    intl_inst,
    categories,
    start_year,
    enable_enrichment,
    progress=gr.Progress()
):
    """Main collection function"""
    global current_papers, current_timestamp, confirmed_selection
    
    # Reset selection
    confirmed_selection = []
    
    try:
        all_institutions = swiss_inst + intl_inst
        
        if not all_institutions:
            return None, "", gr.update(visible=False), None, None
        
        if not categories:
            return None, "", gr.update(visible=False), None, None
        
        category_codes = [cat.split(" - ")[0] for cat in categories]
        
        progress(0, desc="Initializing...")
        
        # Collect all papers
        collector = ArxivCollector(max_results_per_institution=100)
        collector.SWISS_INSTITUTIONS = all_institutions
        collector.AI_CATEGORIES = category_codes
        
        progress(0.2, desc="Collecting from arXiv...")
        papers = collector.collect_papers(start_year=start_year)
        
        if not papers:
            return None, "", gr.update(visible=False), None, None
        
        if enable_enrichment:
            progress(0.6, desc="Enriching with Semantic Scholar...")
            enricher = SemanticScholarEnricher()
            papers = enricher.enrich_papers(papers, delay=0.5)
        
        current_papers = papers.copy()
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        progress(0.9, desc="Generating preview...")
        
        # Generate chart
        chart = generate_summary_chart(papers)
        
        # Generate table
        html_table = generate_interactive_table(papers)
        
        progress(1.0, desc="Complete!")
        
        return chart, html_table, gr.update(visible=True), None, None
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, gr.update(visible=False), None, None


def generate_summary_chart(papers):
    """Generate summary with bar charts"""
    total = len(papers)
    
    # Institution counts
    inst_counts = {}
    for p in papers:
        inst = p['institution']
        inst_counts[inst] = inst_counts.get(inst, 0) + 1
    
    # Category counts (top 8)
    cat_counts = {}
    for p in papers:
        for cat in p.get('categories', []):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    top_cats = dict(sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:8])
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Papers by Institution', 'Top Categories'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Institution chart
    inst_sorted = sorted(inst_counts.items(), key=lambda x: x[1], reverse=True)
    fig.add_trace(
        go.Bar(
            y=[inst for inst, _ in inst_sorted],
            x=[count for _, count in inst_sorted],
            orientation='h',
            marker=dict(color='#3498db'),
            text=[count for _, count in inst_sorted],
            textposition='auto',
            name='Institutions',
            hovertemplate='<b>%{y}</b><br>Papers: %{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Category chart
    fig.add_trace(
        go.Bar(
            y=list(top_cats.keys()),
            x=list(top_cats.values()),
            orientation='h',
            marker=dict(color='#2ecc71'),
            text=list(top_cats.values()),
            textposition='auto',
            name='Categories',
            hovertemplate='<b>%{y}</b><br>Papers: %{x}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Paper Count", row=1, col=1)
    fig.update_xaxes(title_text="Paper Count", row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text=f"📊 Collection Summary: {total} Papers Collected",
        title_x=0.5,
        title_font_size=18
    )
    
    return fig


def generate_interactive_table(papers):
    """Generate HTML table with checkboxes and clickable links"""
    
    papers_count = len(papers)
    
    html = f"""
    <style>
        .paper-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 10px;
        }}
        .paper-table th {{
            background-color: #2c3e50;
            color: white;
            padding: 12px 8px;
            text-align: left;
            border: 1px solid #34495e;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .paper-table td {{
            padding: 10px 8px;
            border: 1px solid #ddd;
            vertical-align: middle;
        }}
        .paper-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .paper-table tr:hover {{
            background-color: #e8f4f8;
        }}
        .paper-table input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        .link-btn {{
            display: inline-block;
            padding: 6px 12px;
            margin: 2px;
            background-color: #3498db;
            color: white;
            border-radius: 4px;
            text-decoration: none;
            font-size: 12px;
            font-weight: bold;
            transition: background-color 0.2s;
        }}
        .link-btn:hover {{
            background-color: #2980b9;
        }}
        .pdf-btn {{
            background-color: #27ae60;
        }}
        .pdf-btn:hover {{
            background-color: #229954;
        }}
        .paper-title {{
            font-weight: 500;
            color: #2c3e50;
        }}
        .table-container {{
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
    </style>
    
    <script>
        window.selectedPaperIndices = [];
        
        function updateSelection() {{
            const checkboxes = document.querySelectorAll('.paper-checkbox');
            window.selectedPaperIndices = [];
            
            checkboxes.forEach((cb, idx) => {{
                if (cb.checked) {{
                    window.selectedPaperIndices.push(idx);
                }}
            }});
            
            const selected = window.selectedPaperIndices.length;
            const total = checkboxes.length;
            
            const countSpan = document.querySelector('#selection_count');
            if (countSpan) {{
                countSpan.textContent = ` `;
            }}
        }}
        
        function selectAll() {{
            document.querySelectorAll('.paper-checkbox').forEach(cb => cb.checked = true);
            updateSelection();
        }}
        
        function deselectAll() {{
            document.querySelectorAll('.paper-checkbox').forEach(cb => cb.checked = false);
            updateSelection();
        }}
        
        // Initialize
        setTimeout(updateSelection, 500);
    </script>
    
    <div class="table-container">
    <table class="paper-table" id="paperTable">
        <thead>
            <tr>
                <th style="width: 4%; text-align: center;">✓</th>
                <th style="width: 3%;">#</th>
                <th style="width: 35%;">Title</th>
                <th style="width: 18%;">Institution</th>
                <th style="width: 6%;">Year</th>
                <th style="width: 6%;">Cites</th>
                <th style="width: 15%;">Categories</th>
                <th style="width: 13%;">Links</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for i, p in enumerate(papers):
        categories_display = ", ".join(p.get('categories', [])[:3])
        
        html += f"""
            <tr>
                <td style="text-align: center;">
                    <input type="checkbox" id="paper_{i}" class="paper-checkbox" checked onchange="updateSelection()">
                </td>
                <td>{i + 1}</td>
                <td><span class="paper-title">{p['title']}</span></td>
                <td>{p['institution']}</td>
                <td>{p['published'].year}</td>
                <td>{p.get('citation_count', 0)}</td>
                <td style="font-size: 11px;">{categories_display}</td>
                <td>
                    <a href="{p['arxiv_url']}" target="_blank" class="link-btn">📄 arXiv</a>
                    <a href="{p['pdf_url']}" target="_blank" class="link-btn pdf-btn">📥 PDF</a>
                </td>
            </tr>
        """
    
    html += f"""
        </tbody>
    </table>
    </div>
    
    <div style="margin-top: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 4px;">
        <button onclick="selectAll()" style="padding: 8px 16px; margin-right: 10px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer;">Select All</button>
        <button onclick="deselectAll()" style="padding: 8px 16px; margin-right: 20px; background-color: #95a5a6; color: white; border: none; border-radius: 4px; cursor: pointer;">Deselect All</button>
        <span style="font-weight: bold;" id="selection_count">📊 Selected: {papers_count}/{papers_count} papers</span>
    </div>
    """
    
    return html


def process_selection(selection_json_str=""):
    """Process the selection JSON from JavaScript"""
    global confirmed_selection, current_papers
    
    try:
        if not selection_json_str or selection_json_str.strip() == "":
            return "❌ No selection data received. Please try clicking the button again.", gr.update(visible=False)
        
        selection_data = json.loads(selection_json_str)
        confirmed_selection = selection_data.get('indices', [])
        total_collected = selection_data.get('total_papers', len(current_papers))
        
        if not confirmed_selection:
            return "❌ No papers selected. Please check at least one paper in the table above.", gr.update(visible=False)
        
        # Show only first 5 papers
        confirmation_msg = f"""
## ✅ Selection Confirmed!

**Selected:** {len(confirmed_selection)} out of {total_collected} papers

**Preview (first 5):**
"""
        
        # Show first 5 selected papers only
        for idx in confirmed_selection[:5]:
            if idx < len(current_papers):
                p = current_papers[idx]
                confirmation_msg += f"- Paper #{idx+1}: {p['title'][:70]}...\n"
        
        if len(confirmed_selection) > 5:
            confirmation_msg += f"\n*...and {len(confirmed_selection) - 5} more papers*\n"
        
        confirmation_msg += "\n**✅ Ready to save!** Click 'Save Papers & Extract PDFs' below."
        
        return confirmation_msg, gr.update(visible=True)
        
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", gr.update(visible=False)


def save_selected_papers(extract_pdfs, progress=gr.Progress()):
    """Save confirmed papers and optionally extract PDFs"""
    global current_papers, current_timestamp, confirmed_selection
    
    if not current_papers:
        return "❌ No papers to save. Collect papers first.", None, None
    
    if not confirmed_selection:
        return "❌ No selection confirmed. Click 'Confirm My Selection' first.", None, None
    
    try:
        # Get selected papers
        papers_to_save = [current_papers[i] for i in confirmed_selection if i < len(current_papers)]
        
        if not papers_to_save:
            return "❌ No valid papers in selection.", None, None
        
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON (metadata only)
        json_file = output_dir / f"papers_final_{current_timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(papers_to_save, f, indent=2, default=str)
        
        # Save CSV
        csv_file = output_dir / f"papers_export_{current_timestamp}.csv"
        csv_data = []
        for p in papers_to_save:
            csv_data.append({
                "Title": p['title'],
                "Authors": "; ".join([a['name'] for a in p['authors']]),
                "Institution": p['institution'],
                "Year": p['published'].year,
                "Citations": p.get('citation_count', 0),
                "Categories": ", ".join(p.get('categories', [])),
                "Abstract": p['abstract'],
                "arXiv_URL": p['arxiv_url'],
                "PDF_URL": p['pdf_url']
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_file, index=False)
        
        file_size_mb = json_file.stat().st_size / (1024 * 1024)
        
        save_msg = f"""
## ✅ Step 1/2: Papers Saved!

**Saved {len(papers_to_save)} selected papers**

**Files in `data/raw/`:**
- **JSON** (metadata): `papers_final_{current_timestamp}.json` ({file_size_mb:.2f} MB)
- **CSV** (review): `papers_export_{current_timestamp}.csv`
"""
        
        # PDF EXTRACTION
        if extract_pdfs:
            save_msg += "\n---\n## 🔄 Step 2/2: Extracting PDFs...\n\n"
            
            progress(0, desc="Initializing PDF extractor...")
            extractor = PDFExtractor()
            
            successful = 0
            failed = 0
            
            for i, paper in enumerate(papers_to_save):
                progress(
                    (i + 1) / len(papers_to_save),
                    desc=f"Extracting {i+1}/{len(papers_to_save)}: {paper['title'][:40]}..."
                )
                
                result = extractor.extract_paper(paper)
                
                if result['extraction_status'] == 'success':
                    successful += 1
                else:
                    failed += 1
                
                papers_to_save[i] = result
            
            # Save enriched version
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            enriched_file = processed_dir / f"papers_with_fulltext_{current_timestamp}.json"
            with open(enriched_file, 'w', encoding='utf-8') as f:
                json.dump(papers_to_save, f, indent=2, default=str)
            
            enriched_size_mb = enriched_file.stat().st_size / (1024 * 1024)
            
            save_msg += f"""
### ✅ PDF Extraction Complete!

**Processed:** {len(papers_to_save)} papers
- **Successful:** {successful} ✅
- **Failed:** {failed} ❌
- **Success Rate:** {(successful/len(papers_to_save)*100):.1f}%

**Enriched file in `data/processed/`:**
- `papers_with_fulltext_{current_timestamp}.json` ({enriched_size_mb:.2f} MB)

**✅ Complete! Ready for ChromaDB ingestion!**
"""
        
        return save_msg, str(json_file), str(csv_file)
        
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None, None


SWISS_INSTITUTIONS = [
    "University of Zurich", "ETH Zurich", "EPFL", "IDSIA",
    "USI", "University of Bern", "University of Basel",
    "University of Geneva", "ZHAW"
]

INTERNATIONAL_INSTITUTIONS = [
    "University of São Paulo", "MIT", "Stanford University",
    "University of Toronto", "University of Oxford",
    "University of Cambridge", "Max Planck Institute"
]

ALL_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision",
    "cs.CL": "Natural Language Processing",
    "cs.RO": "Robotics",
    "cs.NE": "Neural Computing",
    "stat.ML": "Machine Learning (Statistics)",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval"
}


def create_ui():
    """Create interface"""
    
    with gr.Blocks(title="Research Navigator", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("# Research Navigator - Data Collection")
        
        # Row 1: Institutions (left) and Categories (right)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Institutions")
                swiss_inst = gr.CheckboxGroup(
                    choices=SWISS_INSTITUTIONS,
                    value=["ETH Zurich", "EPFL", "University of Zurich", "IDSIA"],
                    label="Swiss Institutions"
                )
                intl_inst = gr.CheckboxGroup(
                    choices=INTERNATIONAL_INSTITUTIONS,
                    value=[],
                    label="International Institutions"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Categories")
                categories = gr.CheckboxGroup(
                    choices=[f"{c} - {n}" for c, n in ALL_CATEGORIES.items()],
                    value=[
                        "cs.AI - Artificial Intelligence",
                        "cs.LG - Machine Learning",
                        "cs.CV - Computer Vision",
                        "cs.CL - Natural Language Processing"
                    ],
                    label="arXiv Categories"
                )
        
        # Row 2: Settings
        gr.Markdown("### Settings")
        with gr.Row():
            start_year = gr.Slider(2015, 2025, 2020, step=1, label="From Year (to Present)")
            enable_enrichment = gr.Checkbox(True, label="Enable Citation Enrichment (Semantic Scholar)")
        
        collect_btn = gr.Button("Collect Papers", variant="primary", size="lg")
        
        gr.Markdown("---")
        
        # Chart summary
        summary_chart = gr.Plot(label="Collection Summary")
        
        gr.Markdown("""
        ### Papers Review
        
        **Review and select papers:**
        - ✅ **Uncheck** papers you don't want
        - 📄 Click **arXiv** to read abstract
        - 📥 Click **PDF** to download paper
        """)
        
        papers_table = gr.HTML(label="Papers")
        
        # Hidden textbox to receive JavaScript selection data
        selection_json = gr.Textbox(visible=False, elem_id="selection_json")
        
        with gr.Group(visible=False) as confirm_section:
            gr.Markdown("---")
            gr.Markdown("### Step 2: Confirm Selection")
            
            # Real Gradio button with JavaScript
            confirm_btn = gr.Button(
                "Confirm My Selection",
                variant="primary",
                size="lg"
            )
            
            confirmation_status = gr.Markdown()
        
        with gr.Group(visible=False) as save_section:
            gr.Markdown("---")
            gr.Markdown("### Step 3: Save & Extract")
            
            extract_checkbox = gr.Checkbox(
                value=True,
                label="Extract full PDF text (~15-20 min for 150-200 papers)",
                info="Required"
            )
            
            save_btn = gr.Button("💾 Save Papers & Extract PDFs", variant="primary", size="lg")
            save_status = gr.Markdown()
            
            gr.Markdown("### Download Files")
            with gr.Row():
                json_download = gr.File(label="Download JSON")
                csv_download = gr.File(label="Export as CSV")
        
        # Event handlers
        collect_btn.click(
            fn=collect_papers,
            inputs=[swiss_inst, intl_inst, categories, start_year, enable_enrichment],
            outputs=[summary_chart, papers_table, confirm_section, json_download, csv_download]
        )
        
        # Confirmation button with proper JavaScript bridge
        confirm_btn.click(
            fn=process_selection,
            inputs=[selection_json],
            outputs=[confirmation_status, save_section],
            js="""() => {
                const checkboxes = document.querySelectorAll('.paper-checkbox');
                const selected = [];
                checkboxes.forEach((cb, idx) => {
                    if (cb.checked) selected.push(idx);
                });
                return JSON.stringify({
                    indices: selected,
                    timestamp: new Date().toISOString(),
                    total_papers: checkboxes.length
                });
            }"""
        )
        
        save_btn.click(
            fn=save_selected_papers,
            inputs=[extract_checkbox],
            outputs=[save_status, json_download, csv_download]
        )
        
        gr.Markdown("""
        ---
        ### Workflow:
        1. **Configure filters** → Select institutions, categories, year range
        2. **Collect papers** → Review summary charts and table
        3. **Review & select** → Uncheck papers you don't want
        4. **Confirm selection** → Click "Confirm My Selection"
        5. **Save & extract** → Click "Save Papers & Extract PDFs"
        
        """)
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)