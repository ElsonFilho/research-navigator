"""
Evaluation UI for Research Navigator
Implements the complete 7-step evaluation workflow with Gradio
ENHANCED with detailed query-by-query results
"""

import gradio as gr
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
import plotly.graph_objects as go
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import Evaluator 
from src.evaluation.results_analyzer import ResultsAnalyzer
from src.evaluation.statistical_tests import StatisticalTests


class EvaluationUI:
    """Gradio UI for Multi-Agent RAG vs Baseline Evaluation"""
    
    def __init__(self):
        self.uploaded_file: Optional[str] = None
        self.queries_df: Optional[pd.DataFrame] = None
        self.validation_errors: list = []
        self.evaluation_results: List[Dict[str, Any]] = []
        
    def validate_csv(self, file) -> tuple:
        """Validate uploaded CSV file"""
        if file is None:
            return (
                gr.update(visible=False),
                gr.update(visible=True, value="⚠️ Please upload a CSV file"),
                gr.update(interactive=False),
                gr.update(visible=False)
            )
        
        try:
            df = pd.read_csv(file.name)
            
            required_cols = ['query_id', 'query_text', 'topic_category', 
                           'expected_coverage', 'segment']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                error_msg = f"❌ Missing required columns: {', '.join(missing_cols)}"
                return (
                    gr.update(visible=False),
                    gr.update(visible=True, value=error_msg),
                    gr.update(interactive=False),
                    gr.update(visible=False)
                )
            
            empty_queries = df[df['query_text'].isna() | (df['query_text'] == '')]
            if not empty_queries.empty:
                error_msg = f"❌ Empty query_text in rows: {empty_queries.index.tolist()}"
                return (
                    gr.update(visible=False),
                    gr.update(visible=True, value=error_msg),
                    gr.update(interactive=False),
                    gr.update(visible=False)
                )
            
            duplicate_ids = df[df['query_id'].duplicated()]['query_id'].unique()
            if len(duplicate_ids) > 0:
                error_msg = f"❌ Duplicate query_ids found: {', '.join(duplicate_ids)}"
                return (
                    gr.update(visible=False),
                    gr.update(visible=True, value=error_msg),
                    gr.update(interactive=False),
                    gr.update(visible=False)
                )
            
            self.uploaded_file = file.name
            self.queries_df = df
            
            success_msg = f"""✅ File loaded: {Path(file.name).name}

Validating...
✅ CSV format valid
✅ All required columns present
✅ {len(df)} queries found
✅ All query IDs unique
✅ No empty fields"""
            
            return (
                gr.update(visible=True, value=success_msg),
                gr.update(visible=False),
                gr.update(interactive=True),
                gr.update(visible=True)
            )
            
        except Exception as e:
            error_msg = f"❌ Error reading CSV: {str(e)}"
            return (
                gr.update(visible=False),
                gr.update(visible=True, value=error_msg),
                gr.update(interactive=False),
                gr.update(visible=False)
            )
    
    def generate_metadata_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for Step 2"""
        if self.queries_df is None:
            return {}
        
        df = self.queries_df
        
        total_queries = len(df)
        unique_topics = df['topic_category'].nunique()
        
        coverage_counts = df['expected_coverage'].value_counts()
        coverage_breakdown = {
            'high': coverage_counts.get('high', 0),
            'medium': coverage_counts.get('medium', 0),
            'low': coverage_counts.get('low', 0)
        }
        
        topic_dist = df['topic_category'].value_counts().to_dict()
        segment_dist = df['segment'].value_counts().to_dict()
        
        return {
            'total_queries': total_queries,
            'unique_topics': unique_topics,
            'coverage_breakdown': coverage_breakdown,
            'topic_distribution': topic_dist,
            'segment_distribution': segment_dist,
            'sample_queries': df.head(10).to_dict('records')
        }
    
    def show_step2(self):
        """Generate Step 2 visualizations"""
        metadata = self.generate_metadata_summary()
        
        summary = f"""
**File:** {Path(self.uploaded_file).name if self.uploaded_file else 'N/A'}  
**Total Queries:** {metadata.get('total_queries', 0)}  
**Unique Topics:** {metadata.get('unique_topics', 0)} categories

**Coverage Breakdown:**
- High coverage: {metadata['coverage_breakdown']['high']} queries ({metadata['coverage_breakdown']['high']/metadata['total_queries']*100:.0f}%)
- Medium coverage: {metadata['coverage_breakdown']['medium']} queries ({metadata['coverage_breakdown']['medium']/metadata['total_queries']*100:.0f}%)
- Low coverage: {metadata['coverage_breakdown']['low']} queries ({metadata['coverage_breakdown']['low']/metadata['total_queries']*100:.0f}%)
"""
        
        # Topic Distribution - Horizontal bar (interactive)
        topics = metadata['topic_distribution']
        fig1 = go.Figure(data=[
            go.Bar(
                y=list(topics.keys()),
                x=list(topics.values()),
                orientation='h',
                marker=dict(
                    color=list(topics.values()),
                    colorscale='Viridis'
                ),
                text=list(topics.values()),
                textposition='auto'
            )
        ])
        fig1.update_layout(
            title='Topic Distribution',
            xaxis_title='Number of Queries',
            yaxis_title='Topic',
            height=400,
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        # Coverage Distribution - Pie chart (interactive)
        coverage = metadata['coverage_breakdown']
        coverage_filtered = {k: v for k, v in coverage.items() if v > 0}
        
        fig2 = go.Figure(data=[
            go.Pie(
                labels=[k.title() for k in coverage_filtered.keys()],
                values=list(coverage_filtered.values()),
                marker=dict(colors=['#2ecc71', '#f39c12', '#e74c3c']),
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>Queries: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        ])
        fig2.update_layout(
            title='Coverage Distribution',
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        sample_df = pd.DataFrame(metadata['sample_queries'])
        
        return (
            gr.update(open=False),
            gr.update(open=True, visible=True),
            summary,
            fig1,
            fig2,
            sample_df
        )
    
    def back_to_step1(self):
        """Navigate back to Step 1"""
        return (
            gr.update(open=True),
            gr.update(open=False)
        )
    
    def format_query_detail(self, result: Dict[str, Any]) -> str:
        """Format a single query result for expandable display"""
        
        # Extract data with CORRECT field names from evaluator.py
        query_id = result['query_id']
        query_text = result['query_text']
        winner = result['winner']
        
        # Judge scores
        overall_a = result.get('overall_quality_a', 'N/A')
        overall_b = result.get('overall_quality_b', 'N/A')
        citation_a = result.get('citation_quality_a', 'N/A')
        citation_b = result.get('citation_quality_b', 'N/A')
        grounding_a = result.get('factual_grounding_a', 'N/A')
        grounding_b = result.get('factual_grounding_b', 'N/A')
        
        # Responses - FIXED field names (response_a, response_b from evaluator)
        response_a = result.get('response_a', 'No response available')
        response_b = result.get('response_b', 'No response available')
        
        # Judge reasoning - FIXED field name
        reasoning = result.get('judge_reasoning', 'No reasoning provided')
        
        # Metadata - FIXED field names
        citations_a = result.get('response_a_citations', 0)
        citations_b = result.get('response_b_citations', 0)
        corpus_cites = result.get('response_b_corpus_citations', 0)
        llm_cites = result.get('response_b_llm_citations', 0)
        
        # Winner display
        if winner == 'A':
            winner_display = "🏆 **Winner: System A (Baseline - GPT-5.2)**"
        elif winner == 'B':
            winner_display = "🏆 **Winner: System B (Multi-Agent RAG)**"
        else:
            winner_display = "🤝 **Result: Tie**"
        
        # Build formatted output - RESPONSES FIRST, side by side, FULL TEXT
        detail = f"""
---

## {query_id}: {query_text}

### 📝 Responses

<table>
<tr>
<td width="50%" valign="top">

**Response A (Baseline - GPT-5.2)**  
*Citations: {citations_a}*

<details open>
<summary><b>View response</b></summary>

{response_a}

</details>

</td>
<td width="50%" valign="top">

**Response B (Multi-Agent RAG)**  
*Citations: {citations_b} ({corpus_cites} corpus, {llm_cites} LLM)*

<details open>
<summary><b>View response</b></summary>

{response_b}

</details>

</td>
</tr>
</table>

---

### 🧑‍⚖️ Judge Evaluation

{winner_display}

<table>
<tr>
<td width="50%" valign="top">

#### 📊 Judge Scores (1-5 scale)

| Metric | System A | System B | Δ |
|--------|----------|----------|---|
| **Overall Quality** | {overall_a} | {overall_b} | {self._format_delta(overall_b, overall_a)} |
| **Citation Quality** | {citation_a} | {citation_b} | {self._format_delta(citation_b, citation_a)} |
| **Factual Grounding** | {grounding_a} | {grounding_b} | {self._format_delta(grounding_b, grounding_a)} |

</td>
<td width="50%" valign="top">

#### 💭 Judge Reasoning

{reasoning}

</td>
</tr>
</table>

---
"""
        return detail
    
    def _format_delta(self, val_b, val_a) -> str:
        """Format delta with + or - sign"""
        if val_b == 'N/A' or val_a == 'N/A':
            return 'N/A'
        try:
            delta = float(val_b) - float(val_a)
            if delta > 0:
                return f"+{delta:.0f}"
            elif delta < 0:
                return f"{delta:.0f}"
            else:
                return "0"
        except:
            return "N/A"
    
    def generate_detailed_results(self) -> str:
        """Generate detailed query-by-query results for Step 4"""
        if not self.evaluation_results:
            return "No results available"
        
        details = "### INDIVIDUAL QUERY RESULTS\n\n"
        details += f"Showing detailed results for {len(self.evaluation_results)} queries\n\n"
        
        for result in self.evaluation_results:
            details += self.format_query_detail(result)
        
        return details
    
    async def run_evaluation_with_results(self, progress=gr.Progress()):
        """Run evaluation and return results for Step 4"""
        
        print("\n" + "="*60)
        print("DEBUG: Starting evaluation...")
        print("="*60)
        
        if self.queries_df is None:
            return (
                "❌ No queries loaded",
                gr.update(visible=True, open=True),
                gr.update(visible=False, open=False),
                "", None, None, None, pd.DataFrame(), ""
            )
        
        queries = self.queries_df.to_dict('records')
        total = len(queries)
        
        evaluator = Evaluator()
        self.evaluation_results = []
        
        # Evaluate all queries
        for i, query in enumerate(queries, 1):
            progress((i-1)/total, desc=f"Evaluating query {i}/{total}...")
            print(f"\nEvaluating query {i}/{total}: {query['query_id']}")
            
            result = await evaluator.evaluate_single_query(query)
            self.evaluation_results.append(result)
        
        # Save results
        print("\nSaving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"data/evaluation/results_{timestamp}.csv"
        Path("data/evaluation").mkdir(parents=True, exist_ok=True)
        evaluator.save_results(self.evaluation_results, results_path)
        
        # Generate Step 4 content
        print("\nGenerating Step 4 content...")
        analyzer = ResultsAnalyzer(self.evaluation_results)
        stats = analyzer.get_summary_stats()
        
        # Summary
        summary = f"""
📁 **Results saved to:** `{results_path}`

---
#### OVERALL PERFORMANCE

**Total Queries Evaluated:** {stats['total_queries']}  
**Evaluation Duration:** ~{stats['avg_latency_b'] * stats['total_queries'] / 60:.1f} minutes

**Win Distribution (LLM-as-Judge):**
- **Baseline (System A) wins:** {stats['system_a_wins']} ({stats['system_a_win_rate']:.1f}%)
- **Multi-Agent (System B) wins:** {stats['system_b_wins']} ({stats['system_b_win_rate']:.1f}%)
- **Ties:** {stats['ties']}
---
<table>
<tr>
<td width="50%" valign="top">

#### AVERAGE QUALITY SCORES (1-5 scale)

**Overall Quality:**
- Baseline: {stats.get('avg_overall_quality_a', 0):.1f} / 5.0
- Multi-Agent: {stats.get('avg_overall_quality_b', 0):.1f} / 5.0
- **Difference: {stats.get('avg_overall_quality_b', 0) - stats.get('avg_overall_quality_a', 0):+.1f}**

**Citation Quality:**
- Baseline: {stats.get('avg_citation_quality_a', 0):.1f} / 5.0
- Multi-Agent: {stats.get('avg_citation_quality_b', 0):.1f} / 5.0
- **Difference: {stats.get('avg_citation_quality_b', 0) - stats.get('avg_citation_quality_a', 0):+.1f}**

**Factual Grounding:**
- Baseline: {stats.get('avg_factual_grounding_a', 0):.1f} / 5.0
- Multi-Agent: {stats.get('avg_factual_grounding_b', 0):.1f} / 5.0
- **Difference: {stats.get('avg_factual_grounding_b', 0) - stats.get('avg_factual_grounding_a', 0):+.1f}**

</td>
<td width="50%" valign="top">

#### PERFORMANCE METRICS

**Response Time:**
- Baseline Average: {stats['avg_latency_a']:.1f}s
- Multi-Agent Average: {stats['avg_latency_b']:.1f}s
- **Speed Ratio: {stats['avg_latency_b'] / stats['avg_latency_a']:.1f}x slower**

**Citation Counts:**
- Baseline Average: {stats.get('avg_citations_a', 0):.1f}
- Multi-Agent Average: {stats.get('avg_citations_b', 0):.1f}
- **Difference: {stats.get('avg_citations_b', 0) - stats.get('avg_citations_a', 0):+.1f}**

**Citation Validation:**
- Baseline: N/A (Level 0 - no validation)
- Multi-Agent: {stats['avg_validation_accuracy_b']:.1%} (Level 1 + 2)

</td>
</tr>
</table>
"""
        
        # Create charts
        print("Creating charts...")
        win_chart = analyzer.create_win_distribution_chart(stats)
        latency_chart = analyzer.create_latency_comparison_chart(stats)
        topic_chart = analyzer.create_topic_performance_chart()
        winners_topic_chart = analyzer.create_winners_by_topic_chart()  # ✨ NEW
        
        # Get individual results and filter columns
        results_data = analyzer.get_individual_results()
        results_df = pd.DataFrame(results_data)
        
        # Remove overall_quality_a and overall_quality_b columns if they exist
        columns_to_remove = ['overall_quality_a', 'overall_quality_b']
        
        # Debug: Print columns before filtering
        print(f"\n📋 DataFrame columns BEFORE filtering: {list(results_df.columns)}")
        
        # Filter out unwanted columns
        if len(results_df.columns) > 0:
            results_df = results_df[[col for col in results_df.columns if col not in columns_to_remove]]
        
        # Debug: Print columns after filtering
        print(f"📋 DataFrame columns AFTER filtering: {list(results_df.columns)}\n")
        
        # Generate detailed results
        detailed_results = self.generate_detailed_results()
        
        final_status = f"🎉 Evaluation complete! {total} queries evaluated successfully"
        
        print("\nReturning results to UI...")
        
        return (
            final_status,
            gr.update(open=False, visible=True),
            gr.update(open=True, visible=True),
            summary,
            win_chart,
            latency_chart,
            topic_chart,
            winners_topic_chart,  # ✨ NEW
            results_df,
            detailed_results
        )

    def build_ui(self):
        """Build the complete Gradio UI"""
        
        with gr.Blocks(title="Research Navigator - Evaluation") as app:
            
            gr.Markdown("""
            # RESEARCH NAVIGATOR - EVALUATION INTERFACE
            ## Multi-Agent RAG vs GPT-5.2 Baseline Comparison
            """)
            
            # ================================================================
            # STEP 1: SELECT QUERY FILE
            # ================================================================
            
            with gr.Accordion("📁 STEP 1: SELECT QUERY FILE", open=True) as step1:
                
                gr.Markdown("""
                Upload your test queries CSV file:
                
                **Expected format:** query_id, query_text, topic_category, expected_coverage, segment
                """)
                
                file_upload = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    type="filepath"
                )
                
                success_msg = gr.Markdown(visible=False)
                error_msg = gr.Markdown(visible=False)
                
                with gr.Row():
                    proceed_step2_btn = gr.Button(
                        "Proceed to Step 2 →",
                        variant="primary",
                        interactive=False
                    )
            
            # ================================================================
            # STEP 2: QUERY SET OVERVIEW
            # ================================================================
            
            with gr.Accordion("📊 STEP 2: QUERY SET OVERVIEW", open=False, visible=False) as step2:
                
                gr.Markdown("#### DATASET SUMMARY")
                summary_text = gr.Markdown()
                
                # Charts side-by-side in 2 columns
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### TOPIC DISTRIBUTION")
                        topic_chart = gr.Plot(label="Topic Distribution")
                    
                    with gr.Column():
                        gr.Markdown("#### COVERAGE DISTRIBUTION")
                        coverage_chart = gr.Plot(label="Coverage Distribution")
                
                gr.Markdown("#### QUERY PREVIEW")
                query_preview = gr.Dataframe(
                    label="Sample Queries",
                    interactive=False
                )
                
                with gr.Row():
                    back_step1_btn = gr.Button("← Change File")
                    proceed_step3_btn = gr.Button(
                        "Proceed to Evaluation →",
                        variant="primary"
                    )
            
            # ================================================================
            # STEP 3: AUTOMATED EVALUATION
            # ================================================================
            
            with gr.Accordion("🔍 STEP 3: AUTOMATED EVALUATION", open=False, visible=False) as step3:
                
                gr.Markdown("""
                #### CONFIGURATION
                
                **System A (Baseline):** GPT-5.2  
                **System B (Test):** Multi-Agent RAG  
                **Judge Model:** GPT-5.2 (Temperature 0.0)
                
                **Estimated time:** ~120-140 seconds per query (includes judging)
                
                ⚠️ **Note:** Progress updates will appear in the terminal. Please wait for evaluation to complete.
                """)
                
                eval_status = gr.Markdown("Ready to begin evaluation...")
                
                with gr.Row():
                    start_eval_btn = gr.Button(
                        "🚀 Start Evaluation",
                        variant="primary",
                        size="lg"
                    )
            
            # ================================================================
            # STEP 4: RESULTS
            # ================================================================
            
            with gr.Accordion("📊 STEP 4: EVALUATION RESULTS", open=False, visible=False) as step4:
                
                step4_summary = gr.Markdown("Loading results...")
                
                # Charts in 2 rows
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### WIN DISTRIBUTION")
                        step4_win_chart = gr.Plot()
                    
                    with gr.Column():
                        gr.Markdown("###  LATENCY COMPARISON")
                        step4_latency_chart = gr.Plot()
                
                gr.Markdown("###  PERFORMANCE BY TOPIC")
                
                with gr.Row():
                    with gr.Column():
                        step4_topic_chart = gr.Plot()
                    
                    with gr.Column():
                        step4_winners_topic_chart = gr.Plot()  # ✨ NEW
                
                gr.Markdown("###  SUMMARY TABLE")
                step4_results_table = gr.Dataframe(
                    label="Overview",
                    interactive=False,
                    wrap=True
                )
                
                #  Detailed query-by-query results
                gr.Markdown("###  DETAILED QUERY-BY-QUERY RESULTS")
                                
                step4_detailed_results = gr.Markdown(
                    value="Detailed results will appear here after evaluation...",
                    elem_id="detailed_results"
                )
                
                with gr.Row():
                    export_csv_btn = gr.Button("📥 Download Results CSV", variant="secondary")
                    proceed_step5_btn = gr.Button("Proceed to Statistical Tests →", variant="primary")
            
            # ================================================================
            # STEP 5: STATISTICAL TESTS
            # ================================================================
            
            with gr.Accordion(" STEP 5: STATISTICAL SIGNIFICANCE TESTS", open=False, visible=False) as step5:
                
                gr.Markdown("""
                ###  Hypothesis Testing
                
                Statistical tests to validate whether observed differences are significant or due to chance.
                
                **Tests performed:**
                - Wilcoxon Signed-Rank Test (for paired comparisons)
                - Cohen's d (effect size)
                - Binomial Test (win rate significance)
                
                **Significance level:** α = 0.05
                """)
                
                step5_results = gr.Markdown("Statistical tests will run automatically...")
                
                with gr.Row():
                    back_step4_btn = gr.Button("← Back to Results")
                    export_stats_btn = gr.Button("📥 Download Statistical Report", variant="secondary")
            
            # ================================================================
            # EVENT HANDLERS
            # ================================================================
            
            # Step 1: File upload validation
            file_upload.change(
                fn=self.validate_csv,
                inputs=[file_upload],
                outputs=[success_msg, error_msg, proceed_step2_btn, step2]
            )
            
            # Step 1 → Step 2
            proceed_step2_btn.click(
                fn=self.show_step2,
                outputs=[step1, step2, summary_text, topic_chart, coverage_chart, query_preview]
            )
            
            # Step 2 back button
            back_step1_btn.click(
                fn=self.back_to_step1,
                outputs=[step1, step2]
            )

            # Step 2 → Step 3
            def show_step3():
                return (
                    gr.update(open=False),
                    gr.update(open=True, visible=True),
                )
            
            proceed_step3_btn.click(
                fn=show_step3,
                outputs=[step2, step3]
            )
            
            # Step 3: Run evaluation → shows Step 4 when done
            start_eval_btn.click(
                fn=self.run_evaluation_with_results,
                outputs=[
                    eval_status,
                    step3,
                    step4,
                    step4_summary,
                    step4_win_chart,
                    step4_latency_chart,
                    step4_topic_chart,
                    step4_winners_topic_chart,  # ✨ NEW
                    step4_results_table,
                    step4_detailed_results
                ]
            )
            
            # Step 4 → Step 5: Run statistical tests
            def run_statistical_tests():
                """Run statistical tests on evaluation results"""
                if not self.evaluation_results:
                    return (
                        gr.update(open=True),
                        gr.update(open=False, visible=False),
                        "⚠️ No evaluation results available. Please run evaluation first."
                    )
                
                # Run statistical tests
                print("\n" + "="*60)
                print("Running Statistical Tests...")
                print("="*60)
                
                tests = StatisticalTests(self.evaluation_results)
                formatted_results = tests.format_results_for_ui()
                
                print("✅ Statistical tests complete!")
                
                return (
                    gr.update(open=False),
                    gr.update(open=True, visible=True),
                    formatted_results
                )
            
            proceed_step5_btn.click(
                fn=run_statistical_tests,
                outputs=[step4, step5, step5_results]
            )
            
            # Step 5 back button
            def back_to_step4():
                return (
                    gr.update(open=True),
                    gr.update(open=False)
                )
            
            back_step4_btn.click(
                fn=back_to_step4,
                outputs=[step4, step5]
            )
        
        return app


def main():
    """Launch the evaluation UI"""
    ui = EvaluationUI()
    app = ui.build_ui()
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()