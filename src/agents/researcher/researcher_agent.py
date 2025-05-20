"""Researcher Agent for Agentic Researcher

This module implements the Researcher agent that performs web searches
and content scraping to gather information for research queries.
"""

import os
import sys
import logging
import json
import time
import random
import concurrent.futures
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Use imports that work for both direct execution and when imported as a module
from src.agents.base_agent import BaseAgent
from src.utils.config import config
from src.utils.openai_client import AzureOpenAIClient
from src.db.sqlite_manager import SQLiteManager
from src.search.unified_scraper import UnifiedScraper
from src.search.duckduckgo import DuckDuckGoSearch
from src.search.google import GoogleSearch
from src.search.tavily import TavilySearch

class ResearcherAgent(BaseAgent):
    """Researcher agent that performs web searches and content scraping.
    
    The Researcher agent is responsible for gathering information from various
    sources on the web based on research plans and keywords.
    
    Attributes:
        config (ConfigLoader): Configuration loader
        search_engines (Dict[str, Any]): Configured search engines
    """
    
    def __init__(self):
        """Initialize the ResearcherAgent."""
        super().__init__(name="Researcher", description="Performs web searches and content scraping")
        
        # Use the singleton config instance
        self.config = config
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAIClient()
        
        # Initialize search engines based on configuration
        self.search_engines = self._initialize_search_engines()
        
        self.logger.info(f"ResearcherAgent initialized with {self.config.azure_openai_deployment}")
    
    def _initialize_search_engines(self) -> Dict[str, Any]:
        """Initialize search engine configurations.
        
        Returns:
            Dict[str, Any]: Configured search engines
        """
        search_engines = {}
        
        # Read search engine priority from config
        try:
            # Read the search engine priorities from config
            primary = self.config.get_value('search', 'PRIMARY_SEARCH_ENGINE', 'duckduckgo')
            secondary = self.config.get_value('search', 'SECONDARY_SEARCH_ENGINE', 'tavily')
            tertiary = self.config.get_value('search', 'TERTIARY_SEARCH_ENGINE', 'google')
            
            # Set the priority list based on available engines
            search_engines["priority"] = []
            for engine in [primary, secondary, tertiary]:
                if engine.lower() in ['duckduckgo', 'google', 'tavily']:
                    search_engines["priority"].append(engine.lower())
            
            # If no valid engines were found, set defaults
            if not search_engines["priority"]:
                search_engines["priority"] = ["duckduckgo", "tavily", "google"]
                
            self.logger.info(f"Using search engine priority: {search_engines['priority']}")
        except Exception as e:
            self.logger.warning(f"Error reading search engine config: {e}. Using defaults.")
            search_engines["priority"] = ["duckduckgo", "tavily", "google"]
        
        # Configure maximum results per search
        try:
            max_results = int(self.config.get_value('search', 'RESULTS_PER_SEARCH', 10))
        except Exception:
            max_results = 10
        
        # Configure maximum retries
        try:
            max_retries = int(self.config.get_value('search', 'MAX_SEARCH_RETRIES', 3))
        except Exception:
            max_retries = 3
            
        # Initialize search engine clients
        # DuckDuckGo configuration
        try:
            self.duckduckgo_client = DuckDuckGoSearch()
            search_engines["duckduckgo"] = {
                "enabled": True,  # DuckDuckGo doesn't require API key
                "client": self.duckduckgo_client,
                "max_results": max_results
            }
            self.logger.info("DuckDuckGo search engine initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize DuckDuckGo search engine: {e}")
            search_engines["duckduckgo"] = {"enabled": False}
        
        # Google search configuration
        try:
            self.google_client = GoogleSearch()
            google_enabled = hasattr(self.google_client, 'api_key') and hasattr(self.google_client, 'cse_id') and \
                          self.google_client.api_key and self.google_client.cse_id
            
            search_engines["google"] = {
                "enabled": google_enabled,
                "client": self.google_client if google_enabled else None,
                "max_results": max_results
            }
            
            if google_enabled:
                self.logger.info("Google search engine initialized")
            else:
                self.logger.warning("Google search engine disabled (missing API key or CSE ID)")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Google search engine: {e}")
            search_engines["google"] = {"enabled": False}
        
        # Tavily search configuration
        try:
            self.tavily_client = TavilySearch()
            tavily_enabled = hasattr(self.tavily_client, 'api_key') and self.tavily_client.api_key
            
            search_engines["tavily"] = {
                "enabled": tavily_enabled,
                "client": self.tavily_client if tavily_enabled else None,
                "max_results": max_results
            }
            
            if tavily_enabled:
                self.logger.info("Tavily search engine initialized")
            else:
                self.logger.warning("Tavily search engine disabled (missing API key)")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Tavily search engine: {e}")
            search_engines["tavily"] = {"enabled": False}
            
        # Check if any search engines are enabled
        enabled_engines = [name for name, config in search_engines.items() 
                          if name != "priority" and config.get("enabled", False)]
        if not enabled_engines:
            self.logger.warning("No search engines are enabled! Search functionality will be limited.")
            
        return search_engines
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process research request and gather information.
        
        Args:
            input_data (Dict[str, Any]): Input data containing the query and/or keywords
            
        Returns:
            Dict[str, Any]: The search and scraping results
        """
        query = input_data.get("query", "")
        keywords = input_data.get("keywords", [])
        plan = input_data.get("plan", "")
        project_id = input_data.get("project_id", None)
        
        self.logger.info(f"Processing research request for project_id: {project_id}")
        project_id = input_data.get("project_id", 0)
        
        if not query and not keywords:
            self.logger.error("No query or keywords provided to ResearcherAgent")
            raise ValueError("No query or keywords provided to ResearcherAgent")
        
        self.log_activity(f"Researching [{project_id}]: {query if query else ', '.join(keywords)}")
        
        try:
            start_time = time.time()
            
            # Generate search queries from the plan and keywords
            search_queries = self._generate_search_queries(query, keywords, plan)
            
            # Perform web searches using the appropriate search engines
            search_results = self.search_web(search_queries)
            
            # Scrape content from search results (pass project_id to store in SQLite)
            scraped_content = self.scrape_content(search_results, project_id)
            
            # Prepare the final results
            results = {
                "query": query,
                "project_id": project_id,
                "search_queries": search_queries,
                "search_results": search_results,
                "web_content": scraped_content,
                "processing_time": time.time() - start_time,
                "status": "completed"
            }
            
            self.logger.info(f"Research completed in {results['processing_time']:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in research process: {str(e)}")
            return {
                "query": query,
                "project_id": project_id,
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _generate_search_queries(self, query: str, keywords: List[str], plan: str) -> List[str]:
        """Generate effective search queries based on the query, keywords, and plan.
        
        Args:
            query (str): The original research query
            keywords (List[str]): Extracted keywords
            plan (str): Research plan
            
        Returns:
            List[str]: List of search queries
        """
        # Use Azure OpenAI to generate more effective search queries
        try:
            # If we have keywords and plan, use them to create better queries
            if keywords or plan:
                prompt = f"""
Based on the following research query, generate 3-5 effective search engine queries:

RESEARCH QUERY: {query}

{"Keywords: " + ", ".join(keywords) if keywords else ""}
{"Research Plan: " + plan if plan else ""}

Create search queries that will help find comprehensive and accurate information.
Format each query on a new line prefixed with QUERY:  
"""
                
                # Generate search queries using Azure OpenAI
                messages = [
                    {"role": "system", "content": "You are a research assistant that creates effective search engine queries from a research question."},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.openai_client.generate_completion(messages, temperature=0.7)
                
                # Extract the queries from the response
                search_queries = []
                for line in response.strip().split("\n"):
                    if line.strip().startswith("QUERY:"):
                        query_text = line.replace("QUERY:", "").strip()
                        if query_text:
                            search_queries.append(query_text)
                    
                # Ensure we have at least the original query if nothing was generated
                if not search_queries and query:
                    search_queries.append(query)
                
                self.logger.info(f"Generated {len(search_queries)} search queries using Azure OpenAI")
                return search_queries
            else:
                # Just use the original query if we don't have keywords or plan
                return [query] if query else []
                
        except Exception as e:
            self.logger.error(f"Error generating search queries with Azure OpenAI: {str(e)}. Using fallback method.")
            
            # Fallback to simple approach if OpenAI generation fails
            search_queries = []
            
            # Start with the original query if it exists
            if query:
                search_queries.append(query)
            
            # Add individual high-priority keywords with the query
            if keywords:
                for keyword in keywords[:3]:  # Use top 3 keywords
                    if query:
                        search_queries.append(f"{query} {keyword}")
                    else:
                        search_queries.append(keyword)
            
            # Filter out duplicates and limit to 5 queries
            search_queries = list(dict.fromkeys(search_queries))[:5]
            
            self.logger.info(f"Generated {len(search_queries)} search queries using fallback method")
            return search_queries
    
    def search_web(self, search_queries: List[str]) -> List[Dict[str, Any]]:
        """Perform web searches using the configured search engines.
        
        Args:
            search_queries (List[str]): List of search queries to use
            
        Returns:
            List[Dict[str, Any]]: Search results containing URLs and snippets
        """
        self.log_activity(f"Performing web searches for {len(search_queries)} queries")
        
        all_results = []
        
        # Try search engines in priority order
        for engine_name in self.search_engines["priority"]:
            engine_config = self.search_engines.get(engine_name)
            
            if not engine_config or not engine_config.get("enabled"):
                self.logger.info(f"Search engine {engine_name} not enabled, skipping")
                continue
            
            self.logger.info(f"Using search engine: {engine_name}")
            
            try:
                # Perform searches for each query
                for query in search_queries:
                    results = self._search_with_engine(engine_name, query, engine_config)
                    if results:
                        all_results.extend(results)
                        self.logger.info(f"Found {len(results)} results for query: {query}")
                
                # If we have enough results, stop trying other engines
                if len(all_results) >= 10:
                    self.logger.info(f"Found sufficient results ({len(all_results)}), stopping search")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error using search engine {engine_name}: {str(e)}")
                # Continue with next engine on error
        
        # Deduplicate results by URL
        unique_results = []
        seen_urls = set()
        
        for result in all_results:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append(result)
        
        self.logger.info(f"Found {len(unique_results)} unique search results")
        return unique_results[:15]  # Limit to 15 results
    
    def _search_with_engine(self, engine_name: str, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform a search with the specified engine using the dedicated search module.
        
        Args:
            engine_name (str): Name of the search engine to use
            query (str): Search query
            config (Dict[str, Any]): Engine configuration
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        self.log_activity(f"Searching with {engine_name} for: {query}")
        
        if not config.get("enabled", False):
            self.logger.warning(f"Search engine {engine_name} is not enabled")
            return []
            
        try:
            # Get the client for the specified engine
            client = config.get("client")
            if not client:
                self.logger.warning(f"No client available for search engine {engine_name}")
                return []
                
            # Get max results
            max_results = config.get("max_results", 10)
            
            # Perform the search using the engine's client
            raw_results = client.search(query, max_results=max_results)
            
            # Format results consistently
            results = []
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("body", ""),
                    "engine": engine_name
                })
                
            self.logger.info(f"{engine_name.capitalize()} search returned {len(results)} results")
            return results
                
        except Exception as e:
            self.logger.error(f"Error using search engine {engine_name}: {e}")
            return []
    
    def scrape_content(self, search_results: List[Dict[str, Any]], project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape content from the search result URLs using parallel processing.
        
        Args:
            search_results (List[Dict[str, Any]]): Search results containing URLs
            project_id (Optional[int]): Project ID for storing in SQLite
            
        Returns:
            List[Dict[str, Any]]): Scraped content
        """
        self.log_activity(f"Scraping content from {len(search_results)} URLs using unified scraper")
        
        # Get scraping configuration
        headless = self.config.scraping_headless
        respect_robots = self.config.scraping_respect_robots
        user_agent_rotation = self.config.scraping_user_agent_rotation
        max_workers = min(8, len(search_results))  # Limit to 8 parallel workers max
        
        # Create a list of URLs from search results
        urls_to_scrape = []
        for result in search_results:
            url = result.get("url")
            # Skip if not http or https
            if not url or not url.startswith("http"):
                self.logger.warning(f"Skipping invalid or non-HTTP URL: {url}")
                continue
            urls_to_scrape.append({
                "url": url,
                "engine": result.get("engine", ""),
                "title": result.get("title", "")
            })
        
        # Initialize the UnifiedScraper with our configuration
        scraper = UnifiedScraper(
            headless=headless,
            rotate_user_agents=user_agent_rotation,
            honor_robots_txt=respect_robots,
            use_stealth_mode=True  # Enable stealth mode for better scraping
        )
        
        # Define function to process a single URL
        def process_url(result_item):
            try:
                url = result_item["url"]                
                # Parse URL to get domain for save path
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                
                # Create save path
                save_dir = os.path.join(
                    self.config.scraped_data_dir,
                    domain
                )
                os.makedirs(save_dir, exist_ok=True)
                
                # Generate filename from URL path
                path_parts = parsed_url.path.strip("/").split("/")
                filename = "_".join(path_parts) if path_parts and path_parts[0] else "index"
                filename = filename.replace(".", "_")[:100] + ".html"  # Limit length and replace dots
                
                save_path = os.path.join(save_dir, filename)
                
                # Check if already scraped
                if os.path.exists(save_path):
                    self.logger.info(f"Already scraped: {url}, using cached version")
                    with open(save_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                        
                    # Process the cached content
                    soup = BeautifulSoup(html_content, "html.parser")
                    title = soup.title.string if soup.title else result_item.get("title", "")
                    text = self._extract_text_from_html(html_content)
                    
                    self.logger.info(f"Processed cached content for {url}")
                else:
                    # Use UnifiedScraper to scrape the URL
                    self.logger.info(f"Scraping URL: {url}")
                    content, metadata = scraper.scrape(url)
                    
                    # If content is HTML and needs saving
                    if metadata.get("content_type", "").startswith("text/html"):
                        # Save HTML to file
                        html_content = metadata.get("raw_html", "") 
                        if not html_content and "raw_html" not in metadata:
                            # Use BeautifulSoup to get HTML from content
                            soup = BeautifulSoup(content, "html.parser")
                            html_content = str(soup)
                        
                        # Save to file
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(html_content)
                        
                        self.logger.info(f"Saved HTML content to {save_path}")
                        
                        # Get title and content
                        title = metadata.get("title", result_item.get("title", ""))
                        text = content
                    else:
                        # For non-HTML content (like PDFs)
                        title = result_item.get("title", "")
                        text = content
                        html_content = ""  # No HTML for non-HTML content
                        
                        # For document files that were saved to temp files
                        if "temp_file" in metadata:
                            # Copy the temp file to our storage location with proper extension
                            import shutil
                            temp_file = metadata["temp_file"]
                            file_ext = os.path.splitext(url)[1] or ".bin"
                            save_file_path = os.path.join(save_dir, filename.replace(".html", file_ext))
                            shutil.copy2(temp_file, save_file_path)
                            save_path = save_file_path
                
                # Truncate if too long
                if len(text) > 50000:
                    text = text[:50000] + "... (truncated)"
                
                # Store in SQLite if project_id is provided
                if project_id is not None:
                    # Create metadata for storage
                    storage_metadata = {
                        "source": result_item.get("engine", ""),
                        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "file_path": save_path
                    }
                    
                    # Initialize SQLite manager
                    db = SQLiteManager()
                    
                    # Store raw content in SQLite
                    db.store_scraped_data(
                        project_id=project_id,
                        url=url,
                        title=title,
                        content=html_content if html_content else text,
                        metadata=storage_metadata
                    )
                    
                    self.logger.info(f"Stored content for {url} in SQLite database")
                
                # Return the processed result
                return {
                    "url": url,
                    "title": title,
                    "text": text,
                    "html_path": save_path,
                    "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                self.logger.error(f"Error processing {result_item.get('url')}: {str(e)}")
                return None
        
        try:
            scraped_data = []
            
            # Process URLs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit processing tasks
                future_to_url = {executor.submit(process_url, item): item for item in urls_to_scrape}
                
                self.logger.info(f"Submitted {len(future_to_url)} processing tasks to thread pool")
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_url):
                    result = future.result()
                    if result:
                        scraped_data.append(result)
            
            self.logger.info(f"Successfully processed {len(scraped_data)} URLs")
            
            # Close the scraper to clean up resources
            scraper.close()
            
            return scraped_data
            
        except Exception as e:
            self.logger.error(f"Error in parallel processing: {str(e)}")
            
            # Make sure to close the scraper
            try:
                scraper.close()
            except:
                pass
                
            return []
            
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content.
        
        Args:
            html_content: HTML content string
            
        Returns:
            str: Cleaned text content
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text and normalize whitespace
            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())
            
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from HTML: {str(e)}")
            return "Error extracting text content"


# Example usage when run directly
if __name__ == "__main__":
    import os
    import sys
    import json
    import time
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n===== ResearcherAgent - Search and Scraping =====\n")
    
    try:
        # Create the ResearcherAgent
        researcher = ResearcherAgent()
        print(f"Created ResearcherAgent with search engines: {researcher.search_engines['priority']}\n")
        
        # Choose a research query about quantum computing
        research_query = "What are the latest advancements in quantum error correction?"
        print(f"Research Query: \"{research_query}\"\n")
        
        # Generate search queries based on the research query
        keywords = ["quantum error correction", "advancements", "recent research", "quantum computing"]
        research_plan = "Find information about the latest research in quantum error correction, focusing on papers and announcements from the last 2 years."
        
        print("Step 1: Generating optimized search queries")
        search_queries = researcher._generate_search_queries(
            query=research_query,
            keywords=keywords,
            plan=research_plan
        )
        
        print(f"Generated {len(search_queries)} optimized search queries:")
        for i, query in enumerate(search_queries[:3], 1):
            print(f"  {i}. {query}")
        if len(search_queries) > 3:
            print(f"  ... and {len(search_queries) - 3} more queries")
        
        # Perform web searches using the generated queries
        print("\nStep 2: Performing web searches")
        search_start_time = time.time()
        
        # Use just the first 2 queries for the example to save time
        example_queries = search_queries[:2]
        search_results = researcher.search_web(example_queries)
        
        search_duration = time.time() - search_start_time
        print(f"Search completed in {search_duration:.2f} seconds")
        print(f"Found {len(search_results)} search results")
        
        # Display a few search results
        if search_results:
            print("\nTop search results:")
            for i, result in enumerate(search_results[:3], 1):
                print(f"  {i}. {result['title']}")
                print(f"     URL: {result['url']}")
                if 'snippet' in result:
                    snippet = result['snippet']
                    if len(snippet) > 100:
                        snippet = snippet[:100] + "..."
                    print(f"     Snippet: \"{snippet}\"")
                print()
        
        # Scrape content from search results (only first 2 for example)
        print("Step 3: Scraping content from search results")
        scrape_start_time = time.time()
        
        # Only scrape the first 2 results for the example
        example_results = search_results[:2] if len(search_results) >= 2 else search_results
        scraped_content = researcher.scrape_content(example_results)
        
        scrape_duration = time.time() - scrape_start_time
        print(f"Scraping completed in {scrape_duration:.2f} seconds")
        print(f"Successfully scraped {len(scraped_content)} web pages")
        
        # Display scraped content preview
        if scraped_content:
            for i, content in enumerate(scraped_content, 1):
                text = content.get('text', '')
                preview = text[:150].replace('\n', ' ')
                print(f"\nScraped Content {i}:")
                print(f"  URL: {content.get('url', 'N/A')}")
                print(f"  Title: {content.get('title', 'N/A')}")
                print(f"  Content Length: {len(text)} characters")
                print(f"  Preview: \"{preview}...\"")
        
        print("\nResearch process complete! The scraped content can now be processed by")
        print("other agents in the system, such as the AnswerAgent for synthesis.")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback
        traceback.print_exc()