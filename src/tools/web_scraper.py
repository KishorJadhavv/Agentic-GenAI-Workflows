import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class ScrapingResult(BaseModel):
    """Schema for scraped and summarized content."""
    url: str
    title: str
    summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProfessionalWebScraper:
    """
    Advanced Web Scraper tool with LLM-based summarization and extraction.
    Designed for high-reliability agentic tool usage.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.timeout = 15

    def scrape(self, url: str) -> Optional[str]:
        """Fetches raw HTML content from a URL."""
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Scraping error for {url}: {e}")
            return None

    def parse_and_summarize(self, html: str, url: str) -> ScrapingResult:
        """Extracts text content and uses LLM to generate a concise summary."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        title = soup.title.string if soup.title else "No Title"
        text = soup.get_text(separator=' ', strip=True)[:4000]  # Truncate for LLM context

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following web content into a concise, professional paragraph. "
                       "Focus on key insights relevant for an AI researcher."),
            ("user", "{content}")
        ])

        chain = prompt | self.llm
        summary_response = chain.invoke({"content": text})

        return ScrapingResult(
            url=url,
            title=title,
            summary=summary_response.content,
            metadata={"content_length": len(text)}
        )

    def run(self, url: str) -> Dict[str, Any]:
        """Main execution method for agent tool calling."""
        logger.info(f"Tool execution started for: {url}")
        html = self.scrape(url)
        if not html:
            return {"error": "Failed to retrieve content."}
        
        result = self.parse_and_summarize(html, url)
        return result.dict()

if __name__ == "__main__":
    scraper = ProfessionalWebScraper()
    # result = scraper.run("https://en.wikipedia.org/wiki/Artificial_intelligence")
    # print(result)
