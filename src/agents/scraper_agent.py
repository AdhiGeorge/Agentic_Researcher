
import logging
import time
import random
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import robotexclusionrulesparser

from src.config.system_config import SystemConfig

logger = logging.getLogger(__name__)

class ScraperAgent:
    """
    Agent responsible for scraping web content from search results.
    Implements various scraping hygiene practices.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.user_agent = UserAgent()
        self.robots_parser = robotexclusionrulesparser.RobotExclusionRulesParser()
        self.session = requests.Session()
    
    def scrape(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scrape content from the provided search results
        
        Args:
            search_results: List of search result dictionaries with urls
            
        Returns:
            List of dictionaries with scraped content
        """
        logger.info(f"Starting scraping of {len(search_results)} results")
        
        scraped_results = []
        
        for result in search_results:
            url = result.get("url")
            if not url:
                continue
                
            try:
                # Check if scraping is allowed by robots.txt
                if self.config.scraper.respect_robots_txt and not self._is_scraping_allowed(url):
                    logger.info(f"Skipping {url} due to robots.txt restrictions")
                    continue
                
                # Add random delay to avoid overloading servers
                self._random_delay()
                
                # Scrape the content
                content = self._scrape_url(url)
                
                if content:
                    # Calculate token count (approximation)
                    token_count = len(content.split()) * 1.3  # Rough approximation
                    
                    scraped_results.append({
                        "url": url,
                        "title": result.get("title", ""),
                        "content": content,
                        "source": result.get("source", "web"),
                        "token_count": int(token_count),
                        "relevance_score": result.get("relevance_score", 0.0),
                        "scraped_at": time.time()
                    })
                    
                    logger.info(f"Successfully scraped: {url}")
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
        
        logger.info(f"Completed scraping. Successfully scraped {len(scraped_results)} URLs")
        return scraped_results
    
    def _scrape_url(self, url: str) -> Optional[str]:
        """Scrape content from a URL"""
        headers = {"User-Agent": self.user_agent.random}
        
        try:
            response = self.session.get(
                url, 
                headers=headers, 
                timeout=self.config.scraper.timeout
            )
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(["script", "style", "iframe", "nav", "footer"]):
                element.extract()
            
            # Extract main content
            main_content = ""
            
            # Try to find main content container
            main_elements = soup.select("main, article, .content, #content, .post, .article")
            
            if main_elements:
                main_content = main_elements[0].get_text(separator="\n", strip=True)
            else:
                # Fallback to body if no main content container found
                main_content = soup.body.get_text(separator="\n", strip=True)
            
            # Clean up content
            lines = [line.strip() for line in main_content.splitlines() if line.strip()]
            main_content = "\n".join(lines)
            
            return main_content
            
        except Exception as e:
            logger.error(f"Error in _scrape_url for {url}: {str(e)}")
            return None
    
    def _is_scraping_allowed(self, url: str) -> bool:
        """Check if scraping is allowed by robots.txt"""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = f"{base_url}/robots.txt"
        
        try:
            # Fetch robots.txt
            response = self.session.get(robots_url, timeout=5)
            
            if response.status_code == 200:
                self.robots_parser.parse(response.text)
                return self.robots_parser.is_allowed("*", parsed_url.path)
            
            # If robots.txt doesn't exist or can't be accessed, assume scraping is allowed
            return True
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {base_url}: {str(e)}")
            return True  # Assume allowed if check fails
    
    def _random_delay(self):
        """Add a random delay between requests"""
        delay = random.uniform(
            self.config.scraper.delay_min,
            self.config.scraper.delay_max
        )
        time.sleep(delay)
