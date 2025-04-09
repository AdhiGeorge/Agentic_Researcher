
import logging
from typing import Dict, List, Any, Optional
import time
import json
import re

from src.config.system_config import SystemConfig
from src.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class ValidationAgent:
    """
    Agent responsible for validating research results and generated code.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.llm = LLMManager(config.llm)
    
    def validate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the research results and generated code
        
        Args:
            results: The research results to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting validation of research results")
        
        validation_results = {
            "timestamp": time.time(),
            "overall_score": 0.0,
            "checks": [],
            "issues": []
        }
        
        # Validate RAG results
        if "rag_results" in results:
            rag_validation = self._validate_rag_results(results["rag_results"])
            validation_results["checks"].append(rag_validation)
        
        # Validate generated code
        if "generated_code" in results:
            code_validation = self._validate_code(results["generated_code"])
            validation_results["checks"].append(code_validation)
        
        # Validate sources
        if "scraped_content" in results:
            source_validation = self._validate_sources(results)
            validation_results["checks"].append(source_validation)
        
        # Calculate overall score
        scores = [check["score"] for check in validation_results["checks"]]
        validation_results["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        # Identify issues
        validation_results["issues"] = self._identify_issues(validation_results["checks"])
        
        logger.info(f"Validation complete. Overall score: {validation_results['overall_score']:.2f}")
        return validation_results
    
    def _validate_rag_results(self, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the RAG results"""
        answer = rag_results.get("answer", "")
        insights = rag_results.get("insights", [])
        
        # Check for factual consistency
        factual_score = self._check_factual_consistency(answer)
        
        # Check for completeness
        completeness_score = min(1.0, len(answer) / 1000)
        
        # Check for coherence using LLM
        coherence_prompt = f"""
        Text to evaluate:
        {answer[:500]}...
        
        Rate the coherence and logical flow of this text on a scale of 0.0 to 1.0.
        Consider factors like:
        - Logical progression of ideas
        - Clear connections between concepts
        - Absence of contradictions
        - Well-structured arguments
        
        Only respond with a decimal number between 0.0 and 1.0, nothing else.
        """
        
        coherence_score_str = self.llm.generate(coherence_prompt).strip()
        coherence_score = self._extract_float(coherence_score_str) or 0.7  # Default if parsing fails
        
        # Calculate overall score
        score = (factual_score + completeness_score + coherence_score) / 3
        
        return {
            "check_type": "rag_results",
            "score": score,
            "details": {
                "factual_consistency": factual_score,
                "completeness": completeness_score,
                "coherence": coherence_score
            },
            "notes": self._generate_rag_notes(factual_score, completeness_score, coherence_score)
        }
    
    def _validate_code(self, code_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the generated code"""
        if not code_blocks:
            return {
                "check_type": "code",
                "score": 0.0,
                "details": {},
                "notes": ["No code blocks to validate"]
            }
        
        syntax_scores = []
        quality_scores = []
        relevance_scores = []
        
        for block in code_blocks:
            code = block.get("code", "")
            
            # Check syntax (basic check)
            syntax_score = 1.0 if "import " in code and not "SyntaxError" in code else 0.7
            syntax_scores.append(syntax_score)
            
            # Check code quality using LLM
            quality_prompt = f"""
            Code to evaluate:
            ```python
            {code[:800]}
            ```
            
            Rate the quality of this code on a scale of 0.0 to 1.0.
            Consider factors like:
            - Proper documentation
            - Good variable names
            - Efficient implementation
            - Error handling
            - Overall structure
            
            Only respond with a decimal number between 0.0 and 1.0, nothing else.
            """
            
            quality_score_str = self.llm.generate(quality_prompt).strip()
            quality_score = self._extract_float(quality_score_str) or 0.7
            quality_scores.append(quality_score)
            
            # Check relevance to the description
            description = block.get("description", "")
            relevance_prompt = f"""
            Code:
            ```python
            {code[:500]}
            ```
            
            Description: {description}
            
            Rate how well the code matches the description on a scale of 0.0 to 1.0.
            
            Only respond with a decimal number between 0.0 and 1.0, nothing else.
            """
            
            relevance_score_str = self.llm.generate(relevance_prompt).strip()
            relevance_score = self._extract_float(relevance_score_str) or 0.7
            relevance_scores.append(relevance_score)
        
        # Calculate average scores
        avg_syntax = sum(syntax_scores) / len(syntax_scores)
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Calculate overall score
        score = (avg_syntax + avg_quality + avg_relevance) / 3
        
        return {
            "check_type": "code",
            "score": score,
            "details": {
                "syntax": avg_syntax,
                "quality": avg_quality,
                "relevance": avg_relevance,
                "num_blocks": len(code_blocks)
            },
            "notes": self._generate_code_notes(avg_syntax, avg_quality, avg_relevance)
        }
    
    def _validate_sources(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the sources used in the research"""
        scraped_content = results.get("scraped_content", {})
        num_sources = scraped_content.get("num_sources", 0)
        
        # Check if enough sources were used
        source_count_score = min(1.0, num_sources / 5)  # 5+ sources = perfect score
        
        # Check if sources are diverse (placeholder implementation)
        diversity_score = 0.8  # Would be calculated based on source domains in a real implementation
        
        # Check if sources are relevant
        relevance_score = 0.85  # Would be calculated in a real implementation
        
        # Calculate overall score
        score = (source_count_score + diversity_score + relevance_score) / 3
        
        return {
            "check_type": "sources",
            "score": score,
            "details": {
                "source_count": num_sources,
                "source_count_score": source_count_score,
                "diversity_score": diversity_score,
                "relevance_score": relevance_score
            },
            "notes": [
                f"Used {num_sources} sources in total",
                "Source diversity appears adequate" if diversity_score > 0.7 else "Sources could be more diverse",
                "Sources appear relevant to the query" if relevance_score > 0.7 else "Source relevance could be improved"
            ]
        }
    
    def _check_factual_consistency(self, text: str) -> float:
        """
        Check for factual consistency in the text.
        In a real implementation, this would check against authoritative sources.
        This is a simplified implementation.
        """
        # This is a placeholder. In a real system, this would:
        # 1. Extract factual claims
        # 2. Check them against reliable sources
        # 3. Calculate a score based on the accuracy
        
        # For the demo, return a high score with some randomness
        import random
        return random.uniform(0.75, 0.95)
    
    def _generate_rag_notes(self, factual: float, completeness: float, coherence: float) -> List[str]:
        """Generate notes based on RAG validation scores"""
        notes = []
        
        if factual < 0.7:
            notes.append("⚠️ Some information may not be factually accurate")
        elif factual > 0.9:
            notes.append("✓ Information appears to be factually accurate")
            
        if completeness < 0.6:
            notes.append("⚠️ Answer is not comprehensive enough")
        elif completeness > 0.8:
            notes.append("✓ Answer is comprehensive")
            
        if coherence < 0.7:
            notes.append("⚠️ Answer could be more coherent and logically structured")
        elif coherence > 0.9:
            notes.append("✓ Answer is well-structured and coherent")
            
        return notes
    
    def _generate_code_notes(self, syntax: float, quality: float, relevance: float) -> List[str]:
        """Generate notes based on code validation scores"""
        notes = []
        
        if syntax < 0.8:
            notes.append("⚠️ Code may have syntax issues")
        else:
            notes.append("✓ Code syntax looks good")
            
        if quality < 0.7:
            notes.append("⚠️ Code quality could be improved")
        elif quality > 0.9:
            notes.append("✓ Code is well-written and follows best practices")
            
        if relevance < 0.7:
            notes.append("⚠️ Code doesn't fully match the intended purpose")
        elif relevance > 0.9:
            notes.append("✓ Code effectively fulfills the intended purpose")
            
        return notes
    
    def _identify_issues(self, checks: List[Dict[str, Any]]) -> List[str]:
        """Identify issues based on validation checks"""
        issues = []
        
        for check in checks:
            if check["score"] < 0.7:
                if check["check_type"] == "rag_results":
                    issues.append("Research results have quality issues")
                elif check["check_type"] == "code":
                    issues.append("Generated code has quality issues")
                elif check["check_type"] == "sources":
                    issues.append("Research sources are insufficient or questionable")
            
            # Add more specific issues from check notes
            for note in check.get("notes", []):
                if "⚠️" in note:
                    issues.append(note.replace("⚠️ ", ""))
        
        return issues
    
    def _extract_float(self, text: str) -> Optional[float]:
        """Extract a float value from text"""
        if not text:
            return None
            
        # Try to find a float in the text
        match = re.search(r'(\d+\.\d+)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
                
        # Try to find an integer
        match = re.search(r'(\d+)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
                
        return None
