"""
Citation Management for Agentic Researcher

This module provides utilities for tracking, formatting, and managing
citations and sources during the research process.
"""

import re
import logging
import json
import urllib.parse
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Data class for citation information"""
    title: str
    url: str
    authors: Optional[List[str]] = field(default_factory=list)
    publication_date: Optional[str] = None
    access_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    source_type: str = "webpage"  # webpage, book, journal, etc.
    content_excerpt: Optional[str] = None
    relevance_score: float = 0.0
    citation_id: str = None
    
    def __post_init__(self):
        """Generate citation ID if not provided"""
        if not self.citation_id:
            # Create a unique ID based on URL and title
            id_string = f"{self.url}_{self.title}"
            self.citation_id = hashlib.md5(id_string.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def to_apa(self) -> str:
        """Format citation in APA style"""
        try:
            # Format authors
            author_text = ""
            if self.authors and len(self.authors) > 0:
                if len(self.authors) == 1:
                    author_text = f"{self.authors[0]}. "
                elif len(self.authors) == 2:
                    author_text = f"{self.authors[0]} & {self.authors[1]}. "
                else:
                    author_text = f"{self.authors[0]} et al. "
            
            # Format date
            date_text = ""
            if self.publication_date:
                # Try to extract year
                year_match = re.search(r'\b(19|20)\d{2}\b', self.publication_date)
                if year_match:
                    date_text = f"({year_match.group(0)}). "
                else:
                    date_text = f"({self.publication_date}). "
            else:
                date_text = "(n.d.). "
            
            # Format title
            title_text = f"{self.title}. "
            
            # Format according to source type
            if self.source_type.lower() == "webpage":
                # For webpage, include "Retrieved from URL"
                return f"{author_text}{date_text}{title_text}Retrieved from {self.url}"
            
            elif self.source_type.lower() == "journal":
                # For journal articles
                return f"{author_text}{date_text}{title_text}Retrieved from {self.url}"
            
            elif self.source_type.lower() == "book":
                # For books
                return f"{author_text}{date_text}{title_text}"
            
            else:
                # Default format
                return f"{author_text}{date_text}{title_text}Retrieved from {self.url}"
                
        except Exception as e:
            logger.error(f"Error formatting citation: {str(e)}")
            # Return basic citation if error occurs
            return f"{self.title}. Retrieved from {self.url}"
    
    def to_mla(self) -> str:
        """Format citation in MLA style"""
        try:
            # Format authors
            author_text = ""
            if self.authors and len(self.authors) > 0:
                if len(self.authors) == 1:
                    author_text = f"{self.authors[0]}. "
                elif len(self.authors) == 2:
                    author_text = f"{self.authors[0]} and {self.authors[1]}. "
                else:
                    author_text = f"{self.authors[0]} et al. "
            
            # Format title with quotes for articles/webpages
            if self.source_type.lower() in ["webpage", "journal"]:
                title_text = f"\"{self.title}.\" "
            else:
                title_text = f"{self.title}. "
            
            # Format date
            date_text = ""
            if self.publication_date:
                date_text = f"{self.publication_date}, "
            
            # Format according to source type
            if self.source_type.lower() == "webpage":
                # For webpage, include access date
                return f"{author_text}{title_text}{date_text}Accessed {self.access_date}, {self.url}"
            
            elif self.source_type.lower() == "journal":
                # For journal articles
                return f"{author_text}{title_text}{date_text}Accessed {self.access_date}, {self.url}"
            
            elif self.source_type.lower() == "book":
                # For books
                return f"{author_text}{title_text}{date_text}"
            
            else:
                # Default format
                return f"{author_text}{title_text}{date_text}Accessed {self.access_date}, {self.url}"
                
        except Exception as e:
            logger.error(f"Error formatting citation: {str(e)}")
            # Return basic citation if error occurs
            return f"\"{self.title}.\" {self.url}. Accessed {self.access_date}."


class CitationManager:
    """
    Manages citations for research sources
    
    Handles:
    - Tracking sources
    - Generating formatted citations
    - Assessing source quality
    - Exporting references lists
    """
    
    def __init__(self):
        """Initialize citation manager"""
        self.citations: Dict[str, Citation] = {}
        
    def add_citation(self, citation: Union[Citation, Dict[str, Any]]) -> str:
        """
        Add a citation to the manager
        
        Args:
            citation: Citation object or dictionary with citation data
            
        Returns:
            Citation ID
        """
        try:
            # Convert dict to Citation if needed
            if isinstance(citation, dict):
                citation = Citation(**citation)
            
            # Store citation
            self.citations[citation.citation_id] = citation
            logger.info(f"Added citation: {citation.title} [{citation.citation_id}]")
            
            return citation.citation_id
        
        except Exception as e:
            logger.error(f"Error adding citation: {str(e)}")
            return None
    
    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """
        Retrieve a citation by ID
        
        Args:
            citation_id: Citation ID
            
        Returns:
            Citation object or None if not found
        """
        return self.citations.get(citation_id)
    
    def remove_citation(self, citation_id: str) -> bool:
        """
        Remove a citation by ID
        
        Args:
            citation_id: Citation ID
            
        Returns:
            True if removed, False if not found
        """
        if citation_id in self.citations:
            del self.citations[citation_id]
            logger.info(f"Removed citation: {citation_id}")
            return True
        return False
    
    def get_all_citations(self) -> List[Citation]:
        """
        Get all citations
        
        Returns:
            List of Citation objects
        """
        return list(self.citations.values())
    
    def extract_domain(self, url: str) -> str:
        """
        Extract domain from URL
        
        Args:
            url: URL string
            
        Returns:
            Domain string
        """
        try:
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc
            return domain
        except Exception:
            return url
    
    def assess_source_quality(self, citation: Citation) -> Dict[str, Any]:
        """
        Assess the quality of a source based on various factors
        
        Args:
            citation: Citation object
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            "domain_quality": 0.0,
            "content_quality": 0.0,
            "recency": 0.0,
            "overall_score": 0.0
        }
        
        try:
            # Domain quality check
            domain = self.extract_domain(citation.url)
            
            # Academic domains get high scores
            if domain.endswith(('.edu', '.gov', '.org')):
                quality_metrics["domain_quality"] = 0.9
            # Known reputable domains
            elif any(trusted in domain for trusted in ['wikipedia', 'britannica', 'sciencedirect', 'springer', 'nature']):
                quality_metrics["domain_quality"] = 0.8
            # News sites
            elif any(news in domain for news in ['news', 'bbc', 'cnn', 'reuters', 'apnews']):
                quality_metrics["domain_quality"] = 0.7
            # Default for unknown domains
            else:
                quality_metrics["domain_quality"] = 0.5
            
            # Content quality based on excerpt length (basic heuristic)
            if citation.content_excerpt:
                excerpt_length = len(citation.content_excerpt)
                quality_metrics["content_quality"] = min(0.9, excerpt_length / 1000)
            
            # Recency check
            if citation.publication_date:
                # Try to extract year
                year_match = re.search(r'\b(19|20)\d{2}\b', citation.publication_date)
                if year_match:
                    year = int(year_match.group(0))
                    current_year = datetime.now().year
                    age = current_year - year
                    
                    # More recent sources get higher scores
                    if age <= 1:
                        quality_metrics["recency"] = 0.95
                    elif age <= 3:
                        quality_metrics["recency"] = 0.85
                    elif age <= 5:
                        quality_metrics["recency"] = 0.75
                    elif age <= 10:
                        quality_metrics["recency"] = 0.6
                    else:
                        quality_metrics["recency"] = 0.4
            
            # Calculate overall score (weighted average)
            weights = {
                "domain_quality": 0.4,
                "content_quality": 0.4,
                "recency": 0.2
            }
            
            quality_metrics["overall_score"] = sum(
                quality_metrics[metric] * weight 
                for metric, weight in weights.items()
            )
            
            # Round scores to 2 decimal places
            for key in quality_metrics:
                quality_metrics[key] = round(quality_metrics[key], 2)
            
            return quality_metrics
        
        except Exception as e:
            logger.error(f"Error assessing source quality: {str(e)}")
            return quality_metrics
    
    def generate_bibliography(self, style: str = "apa") -> Dict[str, Any]:
        """
        Generate a formatted bibliography
        
        Args:
            style: Citation style ("apa", "mla")
            
        Returns:
            Dictionary with formatted citations
        """
        try:
            style = style.lower()
            
            if style not in ["apa", "mla"]:
                style = "apa"  # Default to APA
            
            formatted_citations = []
            
            # Sort citations by relevance score (descending)
            sorted_citations = sorted(
                self.citations.values(),
                key=lambda c: c.relevance_score,
                reverse=True
            )
            
            for citation in sorted_citations:
                if style == "apa":
                    formatted = citation.to_apa()
                else:
                    formatted = citation.to_mla()
                
                formatted_citations.append({
                    "citation_id": citation.citation_id,
                    "formatted": formatted,
                    "url": citation.url,
                    "relevance_score": citation.relevance_score
                })
            
            return {
                "style": style,
                "citations": formatted_citations,
                "count": len(formatted_citations),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating bibliography: {str(e)}")
            return {
                "style": style,
                "citations": [],
                "count": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def export_citations(self, format: str = "json") -> Dict[str, Any]:
        """
        Export all citations in specified format
        
        Args:
            format: Export format ("json", "text")
            
        Returns:
            Dictionary with export data
        """
        try:
            format = format.lower()
            
            if format == "json":
                export_data = {
                    "citations": [c.to_dict() for c in self.citations.values()],
                    "count": len(self.citations),
                    "timestamp": datetime.now().isoformat()
                }
                return {
                    "format": "json",
                    "data": json.dumps(export_data, ensure_ascii=False),
                    "success": True
                }
                
            elif format == "text":
                # Generate plain text list
                text_content = "Citations:\n\n"
                
                for i, citation in enumerate(self.citations.values(), 1):
                    text_content += f"{i}. {citation.to_apa()}\n\n"
                
                return {
                    "format": "text",
                    "data": text_content,
                    "success": True
                }
                
            else:
                return {
                    "format": format,
                    "success": False,
                    "error": f"Unsupported format: {format}"
                }
                
        except Exception as e:
            logger.error(f"Error exporting citations: {str(e)}")
            return {
                "format": format,
                "success": False,
                "error": str(e)
            }
    
    def clear_citations(self) -> None:
        """Clear all citations"""
        citation_count = len(self.citations)
        self.citations = {}
        logger.info(f"Cleared {citation_count} citations")
        
    def update_relevance_scores(self, query: str) -> None:
        """
        Update relevance scores for all citations based on query
        
        Args:
            query: Search query
        """
        try:
            query = query.lower()
            query_terms = set(re.findall(r'\b\w+\b', query))
            
            for citation_id, citation in self.citations.items():
                # Extract terms from title and excerpt
                title = citation.title.lower()
                excerpt = citation.content_excerpt.lower() if citation.content_excerpt else ""
                
                title_terms = set(re.findall(r'\b\w+\b', title))
                excerpt_terms = set(re.findall(r'\b\w+\b', excerpt))
                
                # Calculate overlap ratio with query
                title_overlap = len(title_terms.intersection(query_terms)) / max(1, len(query_terms))
                excerpt_overlap = len(excerpt_terms.intersection(query_terms)) / max(1, len(query_terms))
                
                # Weighted relevance score (title matches are more important)
                relevance = (0.7 * title_overlap) + (0.3 * excerpt_overlap)
                
                # Update relevance score
                citation.relevance_score = round(relevance, 2)
            
            logger.info(f"Updated relevance scores for {len(self.citations)} citations")
            
        except Exception as e:
            logger.error(f"Error updating relevance scores: {str(e)}")
