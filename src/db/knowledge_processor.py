"""
Knowledge Processor for Agentic Researcher
Processes raw scraped data from SQLite and stores clean data in Qdrant
"""
import os
import sys
import time
import logging
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple

# Add project root to Python path to allow direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from bs4 import BeautifulSoup
from src.db.sqlite_manager import SQLiteManager
from src.db.qdrant_manager import QdrantManager

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeProcessor:
    """
    Knowledge Processor for creating a clean knowledge base from raw scraped data
    
    This class:
    1. Retrieves raw HTML data from SQLite
    2. Cleans and processes the HTML to extract meaningful text
    3. Stores the processed text in Qdrant for vector search
    4. Updates SQLite to track which data has been processed
    """
    
    def __init__(self, collection_name: str = "research_data"):
        """
        Initialize the KnowledgeProcessor
        
        Args:
            collection_name: Name of the Qdrant collection
        """
        # Initialize database managers
        self.sqlite_manager = SQLiteManager()
        self.qdrant_manager = QdrantManager(collection_name=collection_name)
        
        logger.info("KnowledgeProcessor initialized")
    
    def process_project_data(self, project_id: int, batch_size: int = 20, max_workers: int = 4) -> Dict[str, Any]:
        """
        Process all unprocessed data for a project
        
        Args:
            project_id: Project ID
            batch_size: Number of items to process in each batch
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dict: Statistics about the processing
        """
        start_time = time.time()
        logger.info(f"Starting knowledge processing for project {project_id}")
        
        # Get all unprocessed data
        unprocessed_data = self.sqlite_manager.get_unprocessed_scraped_data(project_id, limit=batch_size)
        
        if not unprocessed_data:
            logger.info(f"No unprocessed data found for project {project_id}")
            return {
                "processed_count": 0,
                "error_count": 0,
                "processing_time": time.time() - start_time
            }
        
        logger.info(f"Found {len(unprocessed_data)} unprocessed items to process")
        
        # Process data in parallel
        processed_count = 0
        error_count = 0
        
        # Define the processing function for a single item
        def process_item(item):
            try:
                # Extract raw HTML and metadata
                content = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", "")
                item_id = item.get("id")
                
                if not content or not url:
                    logger.warning(f"Skipping item {item_id}: Missing content or URL")
                    return {"success": False, "error": "Missing content or URL", "id": item_id}
                
                # Clean HTML and extract text
                clean_text = self._clean_html(content)
                
                if not clean_text:
                    logger.warning(f"Skipping item {item_id}: No meaningful text extracted")
                    return {"success": False, "error": "No meaningful text extracted", "id": item_id}
                
                # Create document for Qdrant
                document = {
                    "url": url,
                    "title": title,
                    "text": clean_text,
                    "source": "web",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Store in Qdrant
                point_ids = self.qdrant_manager.store_document(document, project_id)
                
                if not point_ids:
                    logger.warning(f"Failed to store document in Qdrant for item {item_id}")
                    return {"success": False, "error": "Failed to store in Qdrant", "id": item_id}
                
                # Update SQLite with processed status and vector ID
                vector_id = str(point_ids[0]) if point_ids else ""
                self.sqlite_manager.mark_scraped_data_processed(item_id, vector_id)
                
                logger.info(f"Successfully processed item {item_id}")
                return {"success": True, "id": item_id, "vector_id": vector_id}
                
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                return {"success": False, "error": str(e), "id": item.get("id")}
        
        # Process items in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in unprocessed_data}
            
            for future in concurrent.futures.as_completed(future_to_item):
                result = future.result()
                if result.get("success"):
                    processed_count += 1
                else:
                    error_count += 1
        
        processing_time = time.time() - start_time
        logger.info(f"Completed knowledge processing: {processed_count} processed, {error_count} errors in {processing_time:.2f} seconds")
        
        return {
            "processed_count": processed_count,
            "error_count": error_count,
            "processing_time": processing_time
        }
    
    def _clean_html(self, html_content: str) -> str:
        """
        Clean HTML content and extract meaningful text
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            str: Cleaned text content
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
                element.extract()
            
            # Remove hidden elements
            for element in soup.find_all(style=lambda value: value and "display:none" in value):
                element.extract()
            
            # Get text with proper spacing
            lines = [line.strip() for line in soup.get_text(separator="\n").split("\n") if line.strip()]
            text = "\n".join(lines)
            
            # Normalize whitespace
            text = " ".join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {str(e)}")
            return ""
    
    def get_processing_stats(self, project_id: int) -> Dict[str, Any]:
        """
        Get statistics about processed and unprocessed data
        
        Args:
            project_id: Project ID
            
        Returns:
            Dict: Statistics about processed and unprocessed data
        """
        try:
            # Get counts from SQLite
            with self.sqlite_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Count all scraped data for the project
                cursor.execute("SELECT COUNT(*) FROM scraped_data WHERE project_id = ?", (project_id,))
                total_count = cursor.fetchone()[0]
                
                # Count processed data
                cursor.execute("SELECT COUNT(*) FROM scraped_data WHERE project_id = ? AND processed = 1", (project_id,))
                processed_count = cursor.fetchone()[0]
                
                # Count unprocessed data
                cursor.execute("SELECT COUNT(*) FROM scraped_data WHERE project_id = ? AND processed = 0", (project_id,))
                unprocessed_count = cursor.fetchone()[0]
            
            return {
                "total_count": total_count,
                "processed_count": processed_count,
                "unprocessed_count": unprocessed_count,
                "completion_percentage": processed_count / total_count * 100 if total_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {
                "total_count": 0,
                "processed_count": 0,
                "unprocessed_count": 0,
                "completion_percentage": 0,
                "error": str(e)
            }

# Example usage when run directly
if __name__ == "__main__":
    # Add project root to path for imports to work correctly
    import os
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*80)
    print("KNOWLEDGE PROCESSOR EXAMPLE USAGE")
    print("="*80)
    print("This example demonstrates how to use KnowledgeProcessor to process raw HTML data")
    print()
    
    # Create a unique collection name for this test
    import time
    collection_name = f"test_collection_{int(time.time())}"  
    print(f"Creating new Qdrant collection: {collection_name}")
    
    # Initialize the KnowledgeProcessor
    processor = KnowledgeProcessor(collection_name=collection_name)
    
    # Define a project ID for our test data
    project_id = 12345
    
    # Create SQLite schema if it doesn't exist
    print("\n1. PREPARING DATABASE\n" + "-"*30)
    sqlite_manager = SQLiteManager()
    
    # Ensure the scraped_data table exists with required schema
    print("Ensuring database schema is ready...")
    with sqlite_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_data (
            id INTEGER PRIMARY KEY,
            project_id INTEGER,
            url TEXT,
            title TEXT,
            content TEXT,
            processed INTEGER DEFAULT 0,
            processing_time REAL DEFAULT 0
        )
        """)
        conn.commit()
        print("Schema preparation complete")
    
    # Insert some test data
    print("\n2. CREATING TEST DATA\n" + "-"*30)
    test_data = [
        {
            "url": "https://example.com/quantum-computing",
            "title": "Introduction to Quantum Computing",
            "content": """
            <html>
                <head><title>Introduction to Quantum Computing</title></head>
                <body>
                    <header><nav>Menu items</nav></header>
                    <main>
                        <h1>Quantum Computing Fundamentals</h1>
                        <p>Quantum computing is a type of computing that uses quantum mechanics phenomena such as 
                        superposition and entanglement to perform operations on data.</p>
                        <p>Quantum computers use qubits instead of classical bits, allowing them to solve 
                        certain problems much faster than classical computers.</p>
                        <div class="important-concept">
                            <h2>Key Quantum Concepts</h2>
                            <ul>
                                <li>Superposition</li>
                                <li>Entanglement</li>
                                <li>Quantum Gates</li>
                            </ul>
                        </div>
                    </main>
                    <footer>Copyright 2025</footer>
                </body>
            </html>
            """,
        },
        {
            "url": "https://example.com/machine-learning",
            "title": "Machine Learning Fundamentals",
            "content": """
            <html>
                <head><title>Machine Learning Fundamentals</title></head>
                <body>
                    <header><nav>Menu items</nav></header>
                    <main>
                        <h1>Introduction to Machine Learning</h1>
                        <p>Machine learning is a branch of artificial intelligence that focuses on building 
                        systems that learn from data.</p>
                        <p>These systems automatically improve with experience without being explicitly programmed. 
                        Common algorithms include neural networks, decision trees, and support vector machines.</p>
                        <script>alert('This should be removed');</script>
                        <div style="display:none">Hidden text that should be removed</div>
                    </main>
                    <footer>Copyright 2025</footer>
                </body>
            </html>
            """,
        },
        {
            "url": "https://example.com/blockchain",
            "title": "Understanding Blockchain",
            "content": """
            <html>
                <head><title>Understanding Blockchain Technology</title></head>
                <body>
                    <header><nav>Menu items</nav></header>
                    <main>
                        <h1>Blockchain Technology Explained</h1>
                        <p>Blockchain is a distributed ledger technology that maintains a continuously 
                        growing list of records called blocks.</p>
                        <p>Each block contains a timestamp and a link to the previous block, making the 
                        data tamper-resistant. It forms the foundation for cryptocurrencies like 
                        Bitcoin and Ethereum.</p>
                    </main>
                    <footer>Copyright 2025</footer>
                </body>
            </html>
            """,
        }
    ]
    
    # First, clear any existing test data for this project
    with sqlite_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scraped_data WHERE project_id = ?", (project_id,))
        conn.commit()
        print(f"Cleared existing test data for project {project_id}")
    
    # Insert test data
    insert_count = 0
    with sqlite_manager.get_connection() as conn:
        cursor = conn.cursor()
        for item in test_data:
            cursor.execute(
                "INSERT INTO scraped_data (project_id, url, title, content, processed) VALUES (?, ?, ?, ?, 0)",
                (project_id, item["url"], item["title"], item["content"])
            )
            insert_count += 1
        conn.commit()
    
    print(f"Inserted {insert_count} test HTML documents into SQLite")
    
    # Get processing stats before processing
    stats_before = processor.get_processing_stats(project_id)
    print(f"\nBefore processing: Total: {stats_before['total_count']}, "
          f"Processed: {stats_before['processed_count']}, "
          f"Unprocessed: {stats_before['unprocessed_count']}, "
          f"Completion: {stats_before['completion_percentage']:.1f}%")
    
    # Process the data
    print("\n3. PROCESSING DATA\n" + "-"*30)
    print("Starting data processing...")
    result = processor.process_project_data(project_id, batch_size=10, max_workers=2)
    
    print(f"Processing completed with {result['processed_count']} items processed "
          f"and {result['error_count']} errors in {result['processing_time']:.2f} seconds")
    
    # Get processing stats after processing
    stats_after = processor.get_processing_stats(project_id)
    print(f"\nAfter processing: Total: {stats_after['total_count']}, "
          f"Processed: {stats_after['processed_count']}, "
          f"Unprocessed: {stats_after['unprocessed_count']}, "
          f"Completion: {stats_after['completion_percentage']:.1f}%")
    
    # Show example of cleaned text
    print("\n4. EXAMPLE OF CLEANED TEXT\n" + "-"*30)
    with sqlite_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT url, title FROM scraped_data WHERE project_id = ? AND processed = 1 LIMIT 1", 
            (project_id,)
        )
        processed_item = cursor.fetchone()
        
        if processed_item:
            url, title = processed_item
            print(f"Processed document: {title}")
            print(f"URL: {url}")
            
            # Search in Qdrant to retrieve the processed text
            search_results = processor.qdrant_manager.search_similar(
                query=title,  # Search using the title
                project_id=project_id,
                limit=1,
                threshold=0.5
            )
            
            if search_results:
                print("\nCleaned and processed text:")
                print("-" * 50)
                print(search_results[0]['text'][:500] + "..." if len(search_results[0]['text']) > 500 else search_results[0]['text'])
                print("-" * 50)
    
    # Clean up
    print("\n5. CLEAN UP\n" + "-"*30)
    
    # Delete test data from SQLite
    with sqlite_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scraped_data WHERE project_id = ?", (project_id,))
        deleted_count = cursor.rowcount
        conn.commit()
    
    print(f"Deleted {deleted_count} test items from SQLite")
    
    # Delete data from Qdrant
    deleted = processor.qdrant_manager.delete_project_data(project_id)
    print(f"Deleted project data from Qdrant: {deleted}")
    
    # Close connections
    processor.qdrant_manager.close()
    print("\nKnowledge Processor example completed successfully!")
    print("="*80)
