"""
Evaluation Engine for Research Navigator
Runs Multi-Agent RAG vs Baseline comparison
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime
import time

from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.baseline_agent import BaselineAgent
from src.evaluation.citation_extractor import extract_baseline_citations
from src.evaluation.judge import Judge


class Evaluator:
    """Runs evaluation comparing Multi-Agent RAG vs Baseline"""
    
    def __init__(self):
        # Initialize both systems
        self.multi_agent = CoordinatorAgent(config={
            "top_k": 3,
            "threshold": 0.4,
            "use_web_research": True,
            "use_query_decomposition": False,  # Disabled for faster evaluation
            "validate_citations": True
        })
        
        self.baseline = BaselineAgent()

        self.judge = Judge(model="gpt-5.2", temperature=0.0)
        
    async def evaluate_single_query(self, query: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate a single query on both systems"""
        
        query_id = query['query_id']
        query_text = query['query_text']
        
        print(f"\n{'='*80}")
        print(f"Evaluating {query_id}: {query_text[:60]}...")
        print(f"{'='*80}")
        
        # Run Multi-Agent RAG
        print(f"\n Running Multi-Agent RAG...")
        start_time = time.time()
        
        response_b = await self.multi_agent.process(query_text)
        
        latency_b = time.time() - start_time
        
        # Run Baseline
        print(f"\n Running Baseline (GPT-5.2)...")
        start_time = time.time()
        
        response_a = await self.baseline.process(query_text)
        
        latency_a = time.time() - start_time
        
        # Extract baseline response and count citations
        response_a_text = response_a.content.get('answer', '') if response_a.success else ''
        response_a_citations = extract_baseline_citations(response_a_text) if response_a.success else 0
        
        # Extract multi-agent response
        response_b_text = response_b.content.get('synthesis', '') if response_b.success else ''
        response_b_citations = response_b.metadata.get('total_citations', 0) if response_b.success else 0
        response_b_corpus_citations = response_b.metadata.get('corpus_citations', 0) if response_b.success else 0
        response_b_llm_citations = response_b.metadata.get('llm_citations', 0) if response_b.success else 0
        response_b_arxiv_citations = response_b.metadata.get('arxiv_citations', 0) if response_b.success else 0
        response_b_validation = response_b.content.get('validation_accuracy', 0.0) if response_b.success else 0.0

        # ✨ NEW: Call Judge to evaluate both responses
        print(f"\n 🏆 Running Judge Evaluation...")
        judge_start = time.time()
        
        judge_result = await self.judge.evaluate_responses(
            query=query_text,
            response_a=response_a_text,
            response_b=response_b_text,
            citations_a=response_a_citations,
            citations_b=response_b_citations,
            validation_accuracy_b=response_b_validation
        )
        
        judge_latency = time.time() - judge_start
        print(f"   Judge complete: {judge_latency:.1f}s, Winner={judge_result['winner']}")
        
        # Extract results
        result = {
            'query_id': query_id,
            'query_text': query_text,
            'topic_category': query.get('topic_category', 'Unknown'),
            'expected_coverage': query.get('expected_coverage', 'Unknown'),
            'segment': query.get('segment', 'Unknown'),
            
            # System A (Baseline)
            'response_a': response_a_text,
            'response_a_success': response_a.success,
            'response_a_citations': response_a_citations,
            'latency_a': latency_a,
            
            # System B (Multi-Agent) 
            'response_b': response_b_text,
            'response_b_success': response_b.success,
            'response_b_citations': response_b_citations,
            'response_b_corpus_citations': response_b_corpus_citations,
            'response_b_llm_citations': response_b_llm_citations,
            'response_b_arxiv_citations': response_b_arxiv_citations,
            'response_b_validation_accuracy': response_b_validation,
            'response_b_corpus_papers': response_b.content.get('corpus_papers_used', 0) if response_b.success else 0,
            'latency_b': latency_b,
            
            # ✨ NEW: Judge Evaluation Results
            'overall_quality_a': judge_result.get('overall_quality_a', 0),
            'overall_quality_b': judge_result.get('overall_quality_b', 0),
            'citation_quality_a': judge_result.get('citation_quality_a', 0),
            'citation_quality_b': judge_result.get('citation_quality_b', 0),
            'factual_grounding_a': judge_result.get('factual_grounding_a', 0),
            'factual_grounding_b': judge_result.get('factual_grounding_b', 0),
            'winner': judge_result.get('winner', 'Error'),
            'judge_reasoning': judge_result.get('reasoning', ''),
            'judge_latency': judge_latency,
            
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ {query_id} Complete:")
        print(f"   System A (Baseline): {latency_a:.1f}s, {response_a_citations} citations")
        print(f"   System B (Multi-Agent): {latency_b:.1f}s, {response_b_citations} citations ({response_b_corpus_citations} corpus, {response_b_llm_citations} LLM)")
        print(f"   Judge: Winner={result['winner']}, Quality A={result['overall_quality_a']}, B={result['overall_quality_b']}")
        
        return result
    
    async def evaluate_queries(self, queries: List[Dict[str, str]], 
                              progress_callback=None) -> List[Dict[str, Any]]:
        """Evaluate all queries"""
        
        results = []
        total = len(queries)
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'#'*80}")
            print(f"# Query {i}/{total}")
            print(f"{'#'*80}")
            
            # Evaluate single query
            result = await self.evaluate_single_query(query)
            results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(i, total, result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filepath: str):
        """Save evaluation results to CSV"""
        import pandas as pd
        
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
        print(f"✅ Results saved to {filepath}")
        
        return filepath