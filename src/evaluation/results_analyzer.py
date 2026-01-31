"""
Results Analyzer for Evaluation
Generates statistics and visualizations from evaluation results
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List


class ResultsAnalyzer:
    """Analyzes evaluation results and generates visualizations"""
    
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.df = pd.DataFrame(results)
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        
        total_queries = len(self.df)
        
        # Count wins from judge results
        system_a_wins = sum(1 for r in self.results if r.get('winner') == 'A')
        system_b_wins = sum(1 for r in self.results if r.get('winner') == 'B')
        ties = sum(1 for r in self.results if r.get('winner') == 'Tie')
        
        return {
            'total_queries': total_queries,
            'system_a_wins': system_a_wins,
            'system_b_wins': system_b_wins,
            'ties': ties,
            'system_a_win_rate': (system_a_wins / total_queries * 100) if total_queries > 0 else 0,
            'system_b_win_rate': (system_b_wins / total_queries * 100) if total_queries > 0 else 0,
            'avg_latency_a': self.df['latency_a'].mean(),
            'avg_latency_b': self.df['latency_b'].mean(),
            'avg_citations_a': self.df.get('response_a_citations', pd.Series([0]*len(self.df))).mean(),
            'avg_citations_b': self.df['response_b_citations'].mean(),
            'avg_validation_accuracy_b': self.df['response_b_validation_accuracy'].mean(),
            # Judge metrics (1-5 scale)
            'avg_overall_quality_a': self.df.get('overall_quality_a', pd.Series([0]*len(self.df))).mean(),
            'avg_overall_quality_b': self.df.get('overall_quality_b', pd.Series([0]*len(self.df))).mean(),
            'avg_citation_quality_a': self.df.get('citation_quality_a', pd.Series([0]*len(self.df))).mean(),
            'avg_citation_quality_b': self.df.get('citation_quality_b', pd.Series([0]*len(self.df))).mean(),
            'avg_factual_grounding_a': self.df.get('factual_grounding_a', pd.Series([0]*len(self.df))).mean(),
            'avg_factual_grounding_b': self.df.get('factual_grounding_b', pd.Series([0]*len(self.df))).mean(),
        }
    
    def create_win_distribution_chart(self, stats: Dict[str, Any]) -> go.Figure:
        """Create pie chart of win distribution"""
        
        fig = go.Figure(data=[
            go.Pie(
                labels=['Multi-Agent', 'Baseline', 'Ties'],
                values=[stats['system_b_wins'], stats['system_a_wins'], stats['ties']],
                marker=dict(colors=['#2ecc71', '#e74c3c', '#95a5a6']),
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>Wins: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Overall Win Distribution',
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    def create_latency_comparison_chart(self, stats: Dict[str, Any]) -> go.Figure:
        """Create bar chart comparing latency"""
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Baseline', 'Multi-Agent'],
                y=[stats['avg_latency_a'], stats['avg_latency_b']],
                marker=dict(color=['#e74c3c', '#2ecc71']),
                text=[f"{stats['avg_latency_a']:.1f}s", f"{stats['avg_latency_b']:.1f}s"],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Average Response Latency Comparison',
            yaxis_title='Seconds',
            height=350,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_topic_performance_chart(self) -> go.Figure:
        """Create horizontal bar chart of performance by topic"""
        
        # Group by topic
        topic_stats = self.df.groupby('topic_category').agg({
            'query_id': 'count',
            'response_b_citations': 'mean'
        }).reset_index()
        
        fig = go.Figure(data=[
            go.Bar(
                y=topic_stats['topic_category'],
                x=topic_stats['query_id'],
                orientation='h',
                marker=dict(
                    color=topic_stats['response_b_citations'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Citations")
                ),
                text=topic_stats['query_id'],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Queries: %{x}<br>Avg Citations: %{marker.color:.1f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Performance by Topic',
            xaxis_title='Number of Queries',
            yaxis_title='Topic',
            height=300,
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_winners_by_topic_chart(self) -> go.Figure:
        """Create stacked horizontal bar chart of winners by topic"""
        
        # Group by topic and winner
        topic_winners = self.df.groupby(['topic_category', 'winner']).size().reset_index(name='count')
        
        # Pivot to get counts for each winner type
        pivot_data = topic_winners.pivot(index='topic_category', columns='winner', values='count').fillna(0)
        
        # Ensure all winner types exist
        for winner_type in ['A', 'B', 'Tie']:
            if winner_type not in pivot_data.columns:
                pivot_data[winner_type] = 0
        
        # Create stacked horizontal bar chart
        fig = go.Figure()
        
        # Add Baseline (System A) wins
        fig.add_trace(go.Bar(
            name='Baseline Wins',
            y=pivot_data.index,
            x=pivot_data['A'],
            orientation='h',
            marker=dict(color='#e74c3c'),
            text=pivot_data['A'].astype(int),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Baseline Wins: %{x}<extra></extra>'
        ))
        
        # Add Multi-Agent (System B) wins
        fig.add_trace(go.Bar(
            name='Multi-Agent Wins',
            y=pivot_data.index,
            x=pivot_data['B'],
            orientation='h',
            marker=dict(color='#2ecc71'),
            text=pivot_data['B'].astype(int),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Multi-Agent Wins: %{x}<extra></extra>'
        ))
        
        # Add Ties
        fig.add_trace(go.Bar(
            name='Ties',
            y=pivot_data.index,
            x=pivot_data['Tie'],
            orientation='h',
            marker=dict(color='#95a5a6'),
            text=pivot_data['Tie'].astype(int),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Ties: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Winners by Topic',
            xaxis_title='Number of Queries',
            yaxis_title='Topic',
            barmode='stack',
            height=300,
            margin=dict(l=150, r=50, t=50, b=50),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def create_winners_by_topic_chart(self) -> go.Figure:
        """Create grouped bar chart showing winners by topic"""
        
        # Group by topic and count winners
        topic_winners = []
        
        for topic in self.df['topic_category'].unique():
            topic_data = self.df[self.df['topic_category'] == topic]
            
            a_wins = len(topic_data[topic_data['winner'] == 'A'])
            b_wins = len(topic_data[topic_data['winner'] == 'B'])
            ties = len(topic_data[topic_data['winner'] == 'Tie'])
            
            topic_winners.append({
                'topic': topic,
                'baseline_wins': a_wins,
                'multiagent_wins': b_wins,
                'ties': ties,
                'total': len(topic_data)
            })
        
        # Convert to dataframe
        winners_df = pd.DataFrame(topic_winners)
        winners_df = winners_df.sort_values('total', ascending=True)  # Match other chart order
        
        # Create grouped bar chart
        fig = go.Figure()
        
        # Add baseline wins
        fig.add_trace(go.Bar(
            y=winners_df['topic'],
            x=winners_df['baseline_wins'],
            name='Baseline',
            orientation='h',
            marker=dict(color='#e74c3c'),
            text=winners_df['baseline_wins'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Baseline Wins: %{x}<extra></extra>'
        ))
        
        # Add multi-agent wins
        fig.add_trace(go.Bar(
            y=winners_df['topic'],
            x=winners_df['multiagent_wins'],
            name='Multi-Agent',
            orientation='h',
            marker=dict(color='#2ecc71'),
            text=winners_df['multiagent_wins'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Multi-Agent Wins: %{x}<extra></extra>'
        ))
        
        # Add ties (optional - can remove if clutters)
        fig.add_trace(go.Bar(
            y=winners_df['topic'],
            x=winners_df['ties'],
            name='Ties',
            orientation='h',
            marker=dict(color='#95a5a6'),
            text=winners_df['ties'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Ties: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Winners by Topic',
            xaxis_title='Number of Wins',
            yaxis_title='Topic',
            barmode='group',  # Side-by-side bars
            height=300,
            margin=dict(l=150, r=50, t=50, b=50),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def get_individual_results(self) -> List[Dict[str, Any]]:
        """Get formatted individual query results for summary table"""
        
        formatted_results = []
        
        for r in self.results:
            formatted_results.append({
                'query_id': r['query_id'],
                'query_text': r['query_text'][:100] + '...' if len(r['query_text']) > 100 else r['query_text'],
                'topic': r['topic_category'],
                'winner': r.get('winner', 'N/A'),
                'baseline_latency': f"{r['latency_a']:.1f}s",
                'baseline_citations': r.get('response_a_citations', 0),
                'multiagent_latency': f"{r['latency_b']:.1f}s",
                'multiagent_citations': r['response_b_citations'],
                'validation_%': f"{r['response_b_validation_accuracy']*100:.0f}%",
                'winner': r.get('winner', 'N/A'),
            })
        
        return formatted_results