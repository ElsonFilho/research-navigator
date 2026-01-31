"""
LLM-as-Judge Evaluator
Uses GPT-5.2 to evaluate and compare system responses
"""

import logging
import json
from typing import Dict, Any, Optional
from openai import OpenAI

from src.rag.config import RAGConfig


class Judge:
    """
    LLM-as-Judge for evaluating response quality.
    
    Compares two responses (baseline vs multi-agent) on:
    - Overall Quality (1-5)
    - Citation Quality (1-5)
    - Factual Grounding (1-5)
    - Winner determination (A/B/Tie)
    """
    
    def __init__(self, model: str = "gpt-5.2", temperature: float = 0.0):
        """
        Initialize the Judge.
        
        Args:
            model: OpenAI model to use (default: gpt-5.2)
            temperature: Temperature for evaluation (0.0 for consistency)
        """
        self.config = RAGConfig()
        self.client = OpenAI(api_key=self.config.openai_api_key)
        self.model = model
        self.temperature = temperature
        
        self.logger = logging.getLogger("judge")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Judge initialized with model={model}, temperature={temperature}")
    
    async def evaluate_responses(
        self,
        query: str,
        response_a: str,
        response_b: str,
        citations_a: int = 0,
        citations_b: int = 0,
        validation_accuracy_b: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluate and compare two responses.
        
        Args:
            query: The original query
            response_a: System A (Baseline) response
            response_b: System B (Multi-Agent) response
            citations_a: Number of citations in response A
            citations_b: Number of citations in response B
            validation_accuracy_b: Validation accuracy for response B citations
            
        Returns:
            Dict with evaluation results
        """
        self.logger.info(f"Evaluating responses for query: {query[:60]}...")
        
        # Build the evaluation prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            query=query,
            response_a=response_a,
            response_b=response_b,
            citations_a=citations_a,
            citations_b=citations_b,
            validation_accuracy_b=validation_accuracy_b
        )
        
        try:
            # Call GPT-5.2 to evaluate
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            evaluation_text = response.choices[0].message.content
            
            # Parse the structured evaluation
            result = self._parse_evaluation(evaluation_text)
            
            # Add metadata
            result['raw_evaluation'] = evaluation_text
            result['citations_a'] = citations_a
            result['citations_b'] = citations_b
            result['validation_accuracy_b'] = validation_accuracy_b
            
            self.logger.info(
                f"Evaluation complete: Winner={result['winner']}, "
                f"Quality A={result['overall_quality_a']}, B={result['overall_quality_b']}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {e}")
            return self._create_error_result(str(e))
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the judge.
        """
        return """You are an impartial evaluator comparing two academic literature review responses.

CONTEXT:
- Response A: Baseline LLM (parametric knowledge only, no validation)
- Response B: Multi-Agent RAG (retrieved papers + parametric knowledge, validated citations)
- Response B uses [CORPUS]/[LLM] tags to show source (ignore these tags when scoring)

SCORING SCALE (1-5):
Use the full scale. Don't cluster around 3-4. Be discriminating.

1 = Poor: Major issues, fails to address query, factually wrong, no useful citations
2 = Below Average: Addresses query but superficial, weak citations, significant gaps
3 = Adequate: Adequate response, covers main points, adequate citations, generally accurate
4 = Good: Comprehensive coverage, strong citations, well-structured, accurate
5 = Excellent: Exceptional synthesis, excellent citations, deep insights, precise

EVALUATION CRITERIA:

1. OVERALL QUALITY (1-5)
   - Comprehensiveness: Does it cover the main approaches, including current/recent developments?
   - Structure: Is it well-organized and coherent?
   - Depth: Does it go beyond surface-level description?
   - Synthesis: Does it compare/contrast, not just list?
   
   Score 5: Comprehensive, well-structured synthesis that covers the major approaches and recent 
     developments (2022+), compares trade-offs between methods, integrates evidence with sources, 
     good as a draft section of a survey paper
   Score 4: Good coverage of the main points, with some comparisons, acceptable as a draft section
     of a survey paper
   Score 3: Adequate coverage of main points, mostly descriptive
   Score 2: Superficial or incomplete coverage
   Score 1: Fails to meaningfully address query

2. CITATION QUALITY (1-5)
   - Relevance: Do citations match the specific claims?
   - Recency: For fast-moving fields, prefer relevant recent papers
   - Balance: Mix of foundational and recent work is ideal
   - Specificity: Author names, years, specific papers vs vague "studies show"
   - Quantity: Appropriate number for query scope
   
   Score 5: Highly specific, relevant citations, with recent papers (2023+) for 
     fast-moving AI topics, older papers are used as foundational context, 
     and claims are directly supported by cited work
   Score 4: Good citations, mix of recent/foundational, mostly specific
   Score 3: Adequate citations, some older papers, somewhat specific
   Score 2: Few/vague citations, mostly old papers (pre-2022)
   Score 1: No citations or completely irrelevant


3. FACTUAL GROUNDING (1-5)
   - Specificity: Concrete claims with details vs vague generalizations
   - Attribution: Claims tied to known or validated sources
   - Accuracy: Technically correct information
   - Evidence: Verifiable statements
   - Validation: Validated citations reduce hallucination risk (favor when present)
   
   Score 5: All claims specific, accurate, and sourced, preferably validated
   Score 4: Most claims well-grounded, minor unsourced details
   Score 3: Main claims supported, some generalizations
   Score 2: Many unsupported generalizations or vague claims
   Score 1: Mostly unsupported or factually incorrect

DECISION PROCESS:
1. Score each response independently (use full 1-5 scale)
2. Compare: Which would you prefer for an academic literature review?
3. Winner = A, B, or Tie (Tie only if truly comparable in value)
4. Reasoning = 2-3 sentences that explicitly reference:
- at least one concrete claim
- at least one citation or lack thereof
- one specific strength or weakness

OUTPUT FORMAT (JSON only, no markdown):
{
  "overall_quality_a": <1-5>,
  "overall_quality_b": <1-5>,
  "citation_quality_a": <1-5>,
  "citation_quality_b": <1-5>,
  "factual_grounding_a": <1-5>,
  "factual_grounding_b": <1-5>,
  "winner": "<A|B|Tie>",
  "reasoning": "<specific 2-3 sentence explanation citing examples>"
}"""
    
    def _build_user_prompt(
        self,
        query: str,
        response_a: str,
        response_b: str,
        citations_a: int,
        citations_b: int,
        validation_accuracy_b: float
    ) -> str:
        """Build the user prompt with query and both responses."""
        
        return f"""QUERY: {query}

─── RESPONSE A (Baseline LLM) ───
{response_a}

Metadata: {citations_a} citations, no validation

─── RESPONSE B (Multi-Agent RAG) ───
{response_b}

Metadata: {citations_b} citations, {validation_accuracy_b:.0%} validated

─── TASK ───
Evaluate both responses using the 1-5 scale for each criterion.
Use the full scale - don't default to 3-4.
Provide JSON output only:"""
    
    def _parse_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        """
        Parse the JSON evaluation from the judge.
        
        Args:
            evaluation_text: Raw text from judge (should be JSON)
            
        Returns:
            Parsed evaluation dictionary
        """
        try:
            # Remove markdown code blocks if present
            text = evaluation_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # Parse JSON
            result = json.loads(text)
            
            # Validate required fields
            required_fields = [
                'overall_quality_a', 'overall_quality_b',
                'citation_quality_a', 'citation_quality_b',
                'factual_grounding_a', 'factual_grounding_b',
                'winner', 'reasoning'
            ]
            
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate score ranges (1-5)
            for field in required_fields[:6]:
                score = result[field]
                if not isinstance(score, (int, float)) or score < 1 or score > 5:
                    raise ValueError(f"Invalid score for {field}: {score}")
            
            # Validate winner
            if result['winner'] not in ['A', 'B', 'Tie']:
                raise ValueError(f"Invalid winner: {result['winner']}")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Raw text: {evaluation_text[:200]}...")
            return self._create_error_result(f"JSON parsing failed: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error parsing evaluation: {e}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result when evaluation fails."""
        return {
            'overall_quality_a': 0,
            'overall_quality_b': 0,
            'citation_quality_a': 0,
            'citation_quality_b': 0,
            'factual_grounding_a': 0,
            'factual_grounding_b': 0,
            'winner': 'Error',
            'reasoning': f"Evaluation failed: {error_message}",
            'error': True
        }