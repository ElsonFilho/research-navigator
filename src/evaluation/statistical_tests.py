"""
Statistical Tests Module for Research Navigator Evaluation
Implements hypothesis testing for Multi-Agent RAG vs Baseline comparison
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple
import pandas as pd


class StatisticalTests:
    """Perform statistical significance tests on evaluation results"""
    
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize with evaluation results
        
        Args:
            results: List of evaluation result dictionaries from Evaluator
        """
        self.results = results
        self.df = pd.DataFrame(results)
        self.alpha = 0.05  # Significance level
    
    def wilcoxon_test(self, metric_a: str, metric_b: str) -> Dict[str, Any]:
        """
        Perform Wilcoxon Signed-Rank Test for paired samples
        
        Args:
            metric_a: Column name for System A metric
            metric_b: Column name for System B metric
        
        Returns:
            Dictionary with test results
        """
        # Extract paired data
        data_a = self.df[metric_a].values
        data_b = self.df[metric_b].values
        
        # Remove pairs where either value is NaN
        valid_mask = ~(np.isnan(data_a) | np.isnan(data_b))
        data_a_clean = data_a[valid_mask]
        data_b_clean = data_b[valid_mask]
        
        # Check if we have enough data
        if len(data_a_clean) < 3:
            return {
                'test': 'Wilcoxon Signed-Rank',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'error': 'Insufficient data (n < 3)'
            }
        
        # Perform test
        try:
            statistic, p_value = stats.wilcoxon(data_a_clean, data_b_clean, 
                                                alternative='two-sided')
            
            return {
                'test': 'Wilcoxon Signed-Rank',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'n_pairs': len(data_a_clean),
                'mean_a': float(np.mean(data_a_clean)),
                'mean_b': float(np.mean(data_b_clean)),
                'mean_diff': float(np.mean(data_b_clean - data_a_clean))
            }
        except Exception as e:
            return {
                'test': 'Wilcoxon Signed-Rank',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'error': str(e)
            }
    
    def cohens_d(self, metric_a: str, metric_b: str) -> float:
        """
        Calculate Cohen's d effect size for paired samples
        
        Args:
            metric_a: Column name for System A metric
            metric_b: Column name for System B metric
        
        Returns:
            Cohen's d value
        """
        data_a = self.df[metric_a].values
        data_b = self.df[metric_b].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(data_a) | np.isnan(data_b))
        data_a_clean = data_a[valid_mask]
        data_b_clean = data_b[valid_mask]
        
        if len(data_a_clean) < 2:
            return 0.0
        
        # Calculate differences
        diff = data_b_clean - data_a_clean
        
        # Calculate Cohen's d for paired samples
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        if std_diff == 0:
            return 0.0
        
        d = mean_diff / std_diff
        return float(d)
    
    def binomial_test(self, n_wins_b: int, n_total: int) -> Dict[str, Any]:
        """
        Perform Binomial Test for win rate
        Tests null hypothesis: win rate = 0.5 (50/50 distribution)
        
        Args:
            n_wins_b: Number of wins for System B
            n_total: Total number of comparisons
        
        Returns:
            Dictionary with test results
        """
        if n_total == 0:
            return {
                'test': 'Binomial',
                'n_wins_b': 0,
                'n_total': 0,
                'win_rate': 0.0,
                'p_value': None,
                'significant': False,
                'error': 'No comparisons'
            }
        
        # Perform two-sided binomial test (null hypothesis: p=0.5)
        p_value = stats.binomtest(n_wins_b, n_total, p=0.5, alternative='two-sided').pvalue
        
        return {
            'test': 'Binomial',
            'n_wins_b': n_wins_b,
            'n_total': n_total,
            'win_rate': n_wins_b / n_total,
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'null_hypothesis': 'Win rate = 50%'
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all statistical tests on evaluation results
        
        Returns:
            Dictionary with all test results
        """
        results = {}
        
        # H1: Overall Quality
        print("\n Running H1: Overall Quality Test...")
        h1 = self.wilcoxon_test('overall_quality_a', 'overall_quality_b')
        h1['effect_size'] = self.cohens_d('overall_quality_a', 'overall_quality_b')
        results['h1_overall_quality'] = h1
        
        # H2: Citation Quality
        print(" Running H2: Citation Quality Test...")
        h2 = self.wilcoxon_test('citation_quality_a', 'citation_quality_b')
        h2['effect_size'] = self.cohens_d('citation_quality_a', 'citation_quality_b')
        results['h2_citation_quality'] = h2
        
        # H3: Factual Grounding
        print(" Running H3: Factual Grounding Test...")
        h3 = self.wilcoxon_test('factual_grounding_a', 'factual_grounding_b')
        h3['effect_size'] = self.cohens_d('factual_grounding_a', 'factual_grounding_b')
        results['h3_factual_grounding'] = h3
        
        # H4: Win Rate
        print(" Running H4: Win Rate Test...")
        n_wins_b = len(self.df[self.df['winner'] == 'B'])
        n_total = len(self.df[self.df['winner'].isin(['A', 'B'])])  # Exclude ties
        h4 = self.binomial_test(n_wins_b, n_total)
        results['h4_win_rate'] = h4
        
        # Summary statistics
        results['summary'] = {
            'total_queries': len(self.df),
            'system_a_wins': len(self.df[self.df['winner'] == 'A']),
            'system_b_wins': len(self.df[self.df['winner'] == 'B']),
            'ties': len(self.df[self.df['winner'] == 'Tie']),
            'alpha': self.alpha
        }
        
        print("✅ All tests complete!\n")
        return results
    
    def interpret_effect_size(self, d: float) -> str:
        """
        Interpret Cohen's d effect size
        
        Args:
            d: Cohen's d value
        
        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def format_results_for_ui(self) -> str:
        """
        Format test results for display in Gradio UI
        
        Returns:
            Markdown-formatted string with all test results
        """
        test_results = self.run_all_tests()
        
        # Helper function to safely format values
        def safe_format(value, format_str=".4f", default="N/A"):
            """Safely format a value that might be None"""
            if value is None:
                return default
            try:
                return f"{value:{format_str}}"
            except:
                return default
        
        # Build markdown output
        h1 = test_results['h1_overall_quality']
        
        output = f"""
####  HYPOTHESIS TESTING (α = {self.alpha})

**Sample Size:** {test_results['summary']['total_queries']} queries

---

### H1: OVERALL QUALITY

**Test:** Wilcoxon Signed-Rank Test  
**Null Hypothesis:** No difference in overall quality between systems

**Results:**
- **Statistic:** W = {safe_format(h1.get('statistic'), '.1f')}
- **p-value:** {safe_format(h1.get('p_value'), '.4f')}
- **Effect Size (Cohen's d):** {safe_format(h1.get('effect_size'), '.2f')} ({self.interpret_effect_size(h1.get('effect_size', 0))})
- **Mean Difference:** {safe_format(h1.get('mean_diff'), '+.2f', '0.00')}

**Result:** {'✅ SIGNIFICANT' if h1.get('significant') else '❌ NOT SIGNIFICANT'} (p {'<' if h1.get('p_value', 1) < self.alpha else '≥'} 0.05)

**Interpretation:** {"System B produces significantly higher quality responses." if h1.get('significant') and h1.get('mean_diff', 0) > 0 else "No significant difference in overall quality." if not h1.get('significant') else "System A produces significantly higher quality responses."}

---

### H2: CITATION QUALITY

**Test:** Wilcoxon Signed-Rank Test  
**Null Hypothesis:** No difference in citation quality between systems

**Results:**
- **Statistic:** W = {safe_format(test_results['h2_citation_quality'].get('statistic'), '.1f')}
- **p-value:** {safe_format(test_results['h2_citation_quality'].get('p_value'), '.4f')}
- **Effect Size (Cohen's d):** {safe_format(test_results['h2_citation_quality'].get('effect_size'), '.2f')} ({self.interpret_effect_size(test_results['h2_citation_quality'].get('effect_size', 0))})
- **Mean Difference:** {safe_format(test_results['h2_citation_quality'].get('mean_diff'), '+.2f', '0.00')}

**Result:** {'✅ SIGNIFICANT' if test_results['h2_citation_quality'].get('significant') else '❌ NOT SIGNIFICANT'} (p {'<' if test_results['h2_citation_quality'].get('p_value', 1) < self.alpha else '≥'} 0.05)

**Interpretation:** {"System B shows significantly better citation quality." if test_results['h2_citation_quality'].get('significant') and test_results['h2_citation_quality'].get('mean_diff', 0) > 0 else "No significant difference in citation quality." if not test_results['h2_citation_quality'].get('significant') else "System A shows significantly better citation quality."}

---

### H3: FACTUAL GROUNDING

**Test:** Wilcoxon Signed-Rank Test  
**Null Hypothesis:** No difference in factual grounding between systems

**Results:**
- **Statistic:** W = {safe_format(test_results['h3_factual_grounding'].get('statistic'), '.1f')}
- **p-value:** {safe_format(test_results['h3_factual_grounding'].get('p_value'), '.4f')}
- **Effect Size (Cohen's d):** {safe_format(test_results['h3_factual_grounding'].get('effect_size'), '.2f')} ({self.interpret_effect_size(test_results['h3_factual_grounding'].get('effect_size', 0))})
- **Mean Difference:** {safe_format(test_results['h3_factual_grounding'].get('mean_diff'), '+.2f', '0.00')}

**Result:** {'✅ SIGNIFICANT' if test_results['h3_factual_grounding'].get('significant') else '❌ NOT SIGNIFICANT'} (p {'<' if test_results['h3_factual_grounding'].get('p_value', 1) < self.alpha else '≥'} 0.05)

**Interpretation:** {"System B provides significantly better factual grounding." if test_results['h3_factual_grounding'].get('significant') and test_results['h3_factual_grounding'].get('mean_diff', 0) > 0 else "No significant difference in factual grounding." if not test_results['h3_factual_grounding'].get('significant') else "System A provides significantly better factual grounding."}

---

### H4: WIN RATE

**Test:** Binomial Test  
**Null Hypothesis:** 50/50 win distribution (no preference)

**Results:**
- **System B wins:** {test_results['h4_win_rate']['n_wins_b']}/{test_results['h4_win_rate']['n_total']} ({test_results['h4_win_rate']['win_rate']:.1%})
- **p-value:** {safe_format(test_results['h4_win_rate'].get('p_value'), '.4f')}

**Result:** {'✅ SIGNIFICANT' if test_results['h4_win_rate'].get('significant') else '❌ NOT SIGNIFICANT'} (p {'<' if test_results['h4_win_rate'].get('p_value', 1) < self.alpha else '≥'} 0.05)

**Interpretation:** {"System B wins significantly more than expected by chance." if test_results['h4_win_rate'].get('significant') and test_results['h4_win_rate']['win_rate'] > 0.5 else "Win rate not significantly different from 50/50." if not test_results['h4_win_rate'].get('significant') else "System A wins significantly more than expected by chance."}

{'✅ **SUCCESS CRITERION MET:** ≥60% win rate achieved' if test_results['h4_win_rate']['win_rate'] >= 0.6 else '⚠️ **SUCCESS CRITERION NOT MET:** Win rate below 60% threshold'}

---

###  SUMMARY OF FINDINGS

**Win Distribution:**
- System A wins: {test_results['summary']['system_a_wins']}
- System B wins: {test_results['summary']['system_b_wins']}
- Ties: {test_results['summary']['ties']}

**Statistical Significance:**
- Overall Quality: {'✅ Significant' if test_results['h1_overall_quality'].get('significant') else '❌ Not significant'}
- Citation Quality: {'✅ Significant' if test_results['h2_citation_quality'].get('significant') else '❌ Not significant'}
- Factual Grounding: {'✅ Significant' if test_results['h3_factual_grounding'].get('significant') else '❌ Not significant'}
- Win Rate: {'✅ Significant' if test_results['h4_win_rate'].get('significant') else '❌ Not significant'}

**Effect Sizes:**
- Overall Quality: {self.interpret_effect_size(test_results['h1_overall_quality'].get('effect_size', 0))} (d={test_results['h1_overall_quality'].get('effect_size', 0):.2f})
- Citation Quality: {self.interpret_effect_size(test_results['h2_citation_quality'].get('effect_size', 0))} (d={test_results['h2_citation_quality'].get('effect_size', 0):.2f})
- Factual Grounding: {self.interpret_effect_size(test_results['h3_factual_grounding'].get('effect_size', 0))} (d={test_results['h3_factual_grounding'].get('effect_size', 0):.2f})

---

**Conclusion:** {self._generate_conclusion(test_results)}
"""
        return output
    
    def _generate_conclusion(self, test_results: Dict[str, Any]) -> str:
        """Generate overall conclusion based on test results"""
        
        sig_count = sum([
            test_results['h1_overall_quality'].get('significant', False),
            test_results['h2_citation_quality'].get('significant', False),
            test_results['h3_factual_grounding'].get('significant', False),
            test_results['h4_win_rate'].get('significant', False)
        ])
        
        if sig_count >= 3:
            if test_results['h4_win_rate']['win_rate'] > 0.5:
                return "Multi-Agent RAG significantly outperforms the baseline across multiple metrics. The quality improvement justifies the increased latency."
            else:
                return "Baseline significantly outperforms Multi-Agent RAG across multiple metrics. Further optimization needed."
        elif sig_count >= 2:
            return "Mixed results with some significant differences. System choice depends on specific use case requirements."
        else:
            return "No strong evidence of significant difference between systems. Both perform comparably on most metrics."


# Test function
if __name__ == "__main__":
    # Example usage
    print("Statistical Tests Module")
    print("=" * 60)
    print("This module provides statistical testing for evaluation results.")
    print("\nUsage:")
    print("  from statistical_tests import StatisticalTests")
    print("  tests = StatisticalTests(evaluation_results)")
    print("  results = tests.run_all_tests()")
    print("  formatted = tests.format_results_for_ui()")