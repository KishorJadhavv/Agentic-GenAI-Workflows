import pytest
from unittest.mock import MagicMock, patch
from src.tools.web_scraper import ProfessionalWebScraper, ScrapingResult

@pytest.fixture
def mock_html_content():
    return """
    <html>
        <head><title>AI Research Test</title></head>
        <body>
            <h1>Exploring Agentic Workflows</h1>
            <p>Agents are transforming how we interact with LLMs.</p>
        </body>
    </html>
    """

@patch('requests.get')
def test_web_scraper_retrieval(mock_get, mock_html_content):
    """Validates the raw scraping functionality."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = mock_html_content
    mock_get.return_value = mock_response

    scraper = ProfessionalWebScraper()
    content = scraper.scrape("http://test-ai-url.com")
    
    assert content is not None
    assert "AI Research Test" in content
    assert "Agentic Workflows" in content

@patch('src.tools.web_scraper.ChatOpenAI')
def test_web_scraper_parsing_and_summary(mock_llm_class, mock_html_content):
    """Validates the LLM extraction and summarization logic."""
    # Setup mock LLM response
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Mocked summary of AI workflows.")
    mock_llm_class.return_value = mock_llm

    scraper = ProfessionalWebScraper()
    result = scraper.parse_and_summarize(mock_html_content, "http://test-ai-url.com")

    assert isinstance(result, ScrapingResult)
    assert result.title == "AI Research Test"
    assert "Mocked summary" in result.summary
    assert result.url == "http://test-ai-url.com"

def test_web_scraper_error_handling():
    """Ensures the scraper handles connection failures gracefully."""
    with patch('requests.get', side_effect=Exception("Connection Failed")):
        scraper = ProfessionalWebScraper()
        result = scraper.run("http://broken-url.com")
        
        assert "error" in result
        assert result["error"] == "Failed to retrieve content."
