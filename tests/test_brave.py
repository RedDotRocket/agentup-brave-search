"""Tests for the Brave plugin."""

import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from brave import Plugin
from agent.plugins import CapabilityContext, CapabilityType


class TestBravePlugin:
    """Test cases for the Brave plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = Plugin()
        self.mock_context = Mock(spec=CapabilityContext)
        self.mock_context.config = {"api_key": "test_key", "default_count": 10}
        self.mock_context.task = Mock()
        self.mock_context.task.history = []

    async def test_validate_config_with_api_key(self):
        """Test config validation with API key."""
        config = {"api_key": "test_key"}
        result = await self.plugin.validate_config(config)
        
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_config_without_api_key(self):
        """Test config validation without API key."""
        config = {}
        result = self.plugin.validate_config(config)
        
        assert result.valid is False
        assert len(result.errors) > 0
        assert "Brave API key is required" in result.errors[0]

    def test_validate_config_with_env_api_key(self):
        """Test config validation with environment variable."""
        config = {}
        with patch.dict(os.environ, {"BRAVE_API_KEY": "env_test_key"}):
            result = self.plugin.validate_config(config)
            assert result.valid is True

    def test_validate_config_invalid_count(self):
        """Test config validation with invalid default_count."""
        config = {"api_key": "test_key", "default_count": 25}
        result = self.plugin.validate_config(config)
        
        assert result.valid is True  # Still valid, just a warning
        assert len(result.warnings) > 0

    def test_can_handle_task_search_keywords(self, context):
        """Test task handling with search keywords."""
        # Mock user input
        context.task = self.mock_context
        message = Mock()
        message.parts = [Mock(text="search for python tutorials")]
        self.mock_context.task.history = [message]
        
        confidence = self.plugin.can_handle_task(self.mock_context)
        assert confidence == 1.0

    def test_can_handle_task_question_keywords(self, context):
        """Test task handling with question keywords."""
        message = Mock()
        message.parts = [Mock(text="what is machine learning")]
        self.mock_context.task.history = [message]
        
        confidence = self.plugin.can_handle_task(self.mock_context)
        assert confidence == 0.6

    def test_can_handle_task_no_keywords(self):
        """Test task handling without search keywords."""
        message = Mock()
        message.parts = [Mock(text="hello there")]
        self.mock_context.task.history = [message]
        
        confidence = self.plugin.can_handle_task(self.mock_context)
        assert confidence == 0.0

    @patch("brave.plugin.Brave")
    def test_execute_capability_success(self, mock_brave_class):
        """Test successful search execution."""
        # Mock search results
        mock_search_results = Mock()
        mock_search_results.web_results = [
            Mock(title="Test Title", url="http://test.com", description="Test description")
        ]
        mock_search_results.news_results = []
        mock_search_results.video_results = []
        
        mock_brave_instance = Mock()
        mock_brave_instance.search.return_value = mock_search_results
        mock_brave_class.return_value = mock_brave_instance
        
        # Mock user input
        message = Mock()
        message.parts = [Mock(text="search for python tutorials")]
        self.mock_context.task.history = [message]
        
        result = self.plugin.execute_capability(self.mock_context)
        
        assert result.success is True
        assert "Web Results" in result.content
        assert "Test Title" in result.content
        assert result.metadata["query"] == "python tutorials"

    def test_execute_capability_no_api_key(self):
        """Test execution without API key."""
        self.mock_context.config = {}
        
        result = self.plugin.execute_capability(self.mock_context)
        
        assert result.success is False
        assert "API key not configured" in result.content

    @patch("brave.plugin.Brave")
    def test_execute_capability_search_error(self, mock_brave_class):
        """Test execution with search error."""
        mock_brave_instance = Mock()
        mock_brave_instance.search.side_effect = Exception("Search API error")
        mock_brave_class.return_value = mock_brave_instance
        
        message = Mock()
        message.parts = [Mock(text="search for something")]
        self.mock_context.task.history = [message]
        
        result = self.plugin.execute_capability(self.mock_context)
        
        assert result.success is False
        assert "Error performing search" in result.content

    def test_get_ai_functions(self):
        """Test AI function definitions."""
        functions = self.plugin.get_ai_functions()
        
        assert len(functions) == 2
        assert functions[0].name == "brave_web_search"
        assert functions[1].name == "brave_news_search"
        assert "query" in functions[0].parameters["required"]


    async def test_extract_search_query(self):
        """Test search query extraction."""
        test_cases = [
            ("search for python tutorials", "python tutorials"),
            ("look up machine learning", "machine learning"),
            ("find restaurants near me", "restaurants near me"),
            ("what is artificial intelligence", "what is artificial intelligence"),
            ("brave search for weather", "weather"),
        ]
        
        for input_text, expected_query in test_cases:
            query = await self.plugin._extract_search_query(input_text)
            assert query == expected_query

    def test_format_search_results_empty(self):
        """Test formatting empty search results."""
        mock_results = Mock()
        mock_results.web_results = []
        mock_results.news_results = []
        mock_results.video_results = []
        
        formatted = self.plugin._format_search_results(mock_results)
        assert formatted == "No results found."

    async def test_format_web_results(self):
        """Test formatting web results."""
        web_results = [
            Mock(title="Title 1", url="http://test1.com", description="Desc 1"),
            Mock(title="Title 2", url="http://test2.com", description="Desc 2"),
        ]
        
        formatted = await self.plugin._format_web_results(web_results)
        
        assert len(formatted) == 2
        assert formatted[0]["title"] == "Title 1"
        assert formatted[1]["url"] == "http://test2.com"

    async def test_format_news_results(self):
        """Test formatting news results."""
        news_results = [
            Mock(title="News 1", url="http://news1.com", source="Source 1", age="1 hour ago"),
        ]
        
        formatted = await self.plugin._format_news_results(news_results)
        
        assert len(formatted) == 1
        assert formatted[0]["source"] == "Source 1"
        assert formatted[0]["age"] == "1 hour ago"

    async def test_format_video_results(self):
        """Test formatting video results."""
        video_results = [
            Mock(title="Video 1", url="http://video1.com", creator="Creator 1", duration="10:30"),
        ]
        
        formatted = await self.plugin._format_video_results(video_results)
        
        assert len(formatted) == 1
        assert formatted[0]["creator"] == "Creator 1"
        assert formatted[0]["duration"] == "10:30"