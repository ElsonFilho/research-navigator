"""
Query Decomposition Agent - Enhanced version with better complexity detection

This agent:
1. Analyzes query complexity using multiple criteria
2. Detects temporal, comparison, and multi-aspect queries
3. Limits sub-query generation to 3-5 max
4. Avoids decomposing simple "what/how" questions
5. Uses GPT-5.2 only when decomposition is truly needed
"""

import re
from typing import Dict, List, Optional
from openai import OpenAI

from src.agents.base_agent import BaseAgent, AgentResponse
from src.rag.config import RAGConfig


class QueryDecompositionAgent(BaseAgent):
    """
    Agent that intelligently decomposes complex queries into sub-questions.
    
    Enhanced with better complexity detection to avoid over-decomposition.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Query Decomposition Agent.
        
        Args:
            config: Configuration dictionary with:
                - max_sub_queries: int, maximum number of sub-queries (default: 5)
                - min_sub_queries: int, minimum for decomposition (default: 2)
        """
        super().__init__(agent_name="query_decomposition_agent", config=config)
        
        # Load OpenAI config
        self.rag_config = RAGConfig()
        self.client = OpenAI(api_key=self.rag_config.openai_api_key)
        
        # Agent configuration
        self.max_sub_queries = self.config.get("max_sub_queries", 5)
        self.min_sub_queries = self.config.get("min_sub_queries", 2)
        
        self.logger.info(
            f"Query Decomposition Agent initialized: "
            f"max_sub_queries={self.max_sub_queries}"
        )
    
    async def process(self, query: str) -> AgentResponse:
        """
        Analyze query and decompose if complex.
        
        Args:
            query: User's research question
            
        Returns:
            AgentResponse with:
                - needs_decomposition: bool
                - sub_queries: list (if decomposed)
                - complexity_score: float (0-1)
                - decomposition_reason: str
        """
        self.logger.info(f"Analyzing query complexity: {query[:60]}...")
        
        try:
            # Step 1: Fast heuristic check
            heuristic_result = self._heuristic_complexity_check(query)
            
            self.logger.info(
                f"Heuristic check: complex={heuristic_result['is_complex']}, "
                f"score={heuristic_result['complexity_score']:.2f}"
            )
            
            # Step 2: If heuristic says simple, skip LLM decomposition
            if not heuristic_result['is_complex']:
                return self._create_response(
                    content={
                        "needs_decomposition": False,
                        "sub_queries": [],
                        "complexity_score": heuristic_result['complexity_score'],
                        "decomposition_reason": heuristic_result['reason'],
                        "heuristic_only": True
                    },
                    metadata={
                        "source": "query_decomposition",
                        "method": "heuristic_only",
                        "complexity_score": heuristic_result['complexity_score']
                    },
                    success=True
                )
            
            # Step 3: If heuristic says complex, use LLM for decomposition
            self.logger.info("Query appears complex - using LLM decomposition...")
            
            sub_queries = self._llm_decompose(query)
            
            # Validate sub-queries
            if len(sub_queries) < self.min_sub_queries:
                self.logger.info(
                    f"Too few sub-queries ({len(sub_queries)}) - treating as simple"
                )
                return self._create_response(
                    content={
                        "needs_decomposition": False,
                        "sub_queries": [],
                        "complexity_score": heuristic_result['complexity_score'],
                        "decomposition_reason": "LLM generated too few sub-queries",
                        "heuristic_only": False
                    },
                    metadata={
                        "source": "query_decomposition",
                        "method": "llm_rejected",
                        "sub_queries_generated": len(sub_queries)
                    },
                    success=True
                )
            
            # Cap sub-queries at max
            if len(sub_queries) > self.max_sub_queries:
                self.logger.warning(
                    f"Capping sub-queries from {len(sub_queries)} to {self.max_sub_queries}"
                )
                sub_queries = sub_queries[:self.max_sub_queries]
            
            self.logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            
            return self._create_response(
                content={
                    "needs_decomposition": True,
                    "sub_queries": sub_queries,
                    "complexity_score": heuristic_result['complexity_score'],
                    "decomposition_reason": heuristic_result['reason'],
                    "heuristic_only": False
                },
                metadata={
                    "source": "query_decomposition",
                    "method": "llm_decomposition",
                    "sub_queries_count": len(sub_queries)
                },
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in query decomposition: {e}")
            # Fail gracefully - treat as simple query
            return self._create_response(
                content={
                    "needs_decomposition": False,
                    "sub_queries": [],
                    "complexity_score": 0.0,
                    "decomposition_reason": f"Error during decomposition: {str(e)}"
                },
                metadata={"error": str(e)},
                success=True  # Still successful (defaults to no decomposition)
            )
    
    def _heuristic_complexity_check(self, query: str) -> Dict:
        """
        Fast heuristic-based complexity check.
        
        Checks for:
        - Temporal comparisons ("evolution from X to Y", "changes between")
        - Explicit comparisons ("compare A vs B", "difference between")
        - Multi-aspect questions (multiple "and", multiple topics)
        - Causal chains ("how does X affect Y")
        
        Avoids decomposing:
        - Simple "what/how" questions
        - Single-aspect queries
        - Definition questions
        
        Args:
            query: User's query
            
        Returns:
            Dict with is_complex, complexity_score, and reason
        """
        query_lower = query.lower()
        complexity_score = 0.0
        reasons = []
        
        # ========== STRONG COMPLEXITY INDICATORS (High Priority) ==========
        
        # Temporal comparison patterns
        temporal_patterns = [
            r'evolution (?:from|between|over)',
            r'changes? (?:from|between|over|across)',
            r'development (?:from|between|since)',
            r'progress (?:from|between|since)',
            r'trends? (?:from|between|over)',
            r'advances? (?:from|between|since)',
            r'\d{4}\s*(?:to|vs|versus)\s*\d{4}',  # Year ranges: "2020 to 2025"
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, query_lower):
                complexity_score += 0.4
                reasons.append("temporal_comparison")
                break
        
        # Explicit comparison patterns
        comparison_patterns = [
            r'compar(?:e|ison) (?:between|of|vs)',
            r'difference(?:s)? between',
            r'(?:vs|versus)\s+\w+',
            r'contrast(?:ing)?\s+\w+\s+(?:and|with)',
            r'A\s+(?:vs|versus)\s+B',
        ]
        
        for pattern in comparison_patterns:
            if re.search(pattern, query_lower):
                complexity_score += 0.4
                reasons.append("explicit_comparison")
                break
        
        # Multi-aspect indicators (multiple topics)
        and_count = query_lower.count(' and ')
        comma_count = query_lower.count(',')
        
        if and_count >= 2 or (and_count >= 1 and comma_count >= 2):
            complexity_score += 0.3
            reasons.append("multi_aspect")
        
        # Causal chain patterns
        causal_patterns = [
            r'how (?:does|do) \w+ (?:affect|impact|influence) \w+',
            r'(?:effect|impact) of \w+ on \w+',
            r'relationship between \w+ and \w+',
        ]
        
        for pattern in causal_patterns:
            if re.search(pattern, query_lower):
                complexity_score += 0.3
                reasons.append("causal_chain")
                break
        
        # ========== SIMPLICITY INDICATORS (Negative Scoring) ==========
        
        # Simple "what/how" questions without complexity markers
        simple_patterns = [
            r'^what (?:is|are) (?:the )?(?:main |key |primary )?(?:approaches?|methods?|techniques?)',
            r'^how (?:does|do|can) (?:\w+ )+(?:work|function|operate)',
            r'^what (?:is|are) (?:the )?definition',
            r'^explain \w+',
            r'^describe \w+',
        ]
        
        is_simple_question = False
        for pattern in simple_patterns:
            if re.search(pattern, query_lower):
                is_simple_question = True
                break
        
        if is_simple_question and complexity_score < 0.3:
            # Simple question without strong complexity markers
            return {
                "is_complex": False,
                "complexity_score": complexity_score,
                "reason": "Simple what/how question without complexity markers"
            }
        
        # ========== LENGTH-BASED INDICATORS ==========
        
        # Very short queries are usually simple
        word_count = len(query.split())
        if word_count <= 8 and complexity_score < 0.3:
            return {
                "is_complex": False,
                "complexity_score": complexity_score,
                "reason": "Short query without complexity markers"
            }
        
        # Very long queries might be complex
        if word_count > 20:
            complexity_score += 0.1
        
        # ========== FINAL DECISION ==========
        
        # Threshold: 0.4 or higher = complex
        is_complex = complexity_score >= 0.4
        
        if not reasons:
            reasons.append("no_complexity_markers")
        
        return {
            "is_complex": is_complex,
            "complexity_score": min(complexity_score, 1.0),
            "reason": ", ".join(reasons)
        }
    
    def _llm_decompose(self, query: str) -> List[str]:
        """
        Use GPT-5.2 to decompose complex query into sub-questions.
        
        Args:
            query: Complex query to decompose
            
        Returns:
            List of sub-queries
        """
        system_prompt = f"""You are a research query analyzer. Your task is to decompose complex research queries into focused sub-questions.

RULES:
1. Generate ONLY 3-5 sub-questions (never more than 5)
2. Each sub-question should be independently answerable
3. Sub-questions should cover different aspects of the main query
4. Keep sub-questions focused and specific
5. Output ONLY the sub-questions, one per line, numbered
6. Do NOT include explanations or preamble

EXAMPLES:

Query: "How has federated learning evolved from 2020 to 2025?"
Output:
1. What were the main federated learning approaches in 2020-2021?
2. What key improvements emerged in federated learning during 2022-2023?
3. What are the latest federated learning developments in 2024-2025?

Query: "Compare centralized vs decentralized approaches for privacy-preserving machine learning"
Output:
1. What are the main privacy-preserving techniques in centralized machine learning?
2. What are the main privacy-preserving techniques in decentralized machine learning?
3. What are the key trade-offs between centralized and decentralized approaches?

Query: "How do transformers compare to CNNs for computer vision tasks in terms of accuracy and efficiency?"
Output:
1. What is the accuracy of transformer-based models on computer vision tasks?
2. What is the accuracy of CNN-based models on computer vision tasks?
3. How do transformers and CNNs compare in computational efficiency for vision tasks?
4. What are the main trade-offs between transformers and CNNs for computer vision?"""

        user_prompt = f"""Decompose this query into 3-5 focused sub-questions:

Query: {query}

Output (numbered sub-questions only):"""

        response = self.client.chat.completions.create(
            model="gpt-5.2",
            temperature=0.2,  # Lower temperature for consistency
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Parse response
        text = response.choices[0].message.content.strip()
        
        # Extract numbered lines
        lines = text.split('\n')
        sub_queries = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 2., etc.)
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            
            if line and len(line) > 10:  # Ignore very short lines
                sub_queries.append(line)
        
        return sub_queries