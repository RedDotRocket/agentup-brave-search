"""
Brave Search plugin for AgentUp with AI capabilities.

A plugin that provides Brave Search functionality
"""

import datetime
import os
from typing import Any

import httpx
import structlog
from agent.plugins.base import Plugin
from agent.plugins.decorators import capability
from agent.plugins.models import CapabilityContext


class BraveSearchResult:
    """Simple data class for Brave search results."""

    def __init__(self, data: dict):
        self.title = data.get("title", "")
        self.url = data.get("url", "")
        self.description = data.get("description", "")


class BraveNewsResult:
    """Simple data class for Brave news results."""

    def __init__(self, data: dict):
        self.title = data.get("title", "")
        self.url = data.get("url", "")
        self.source = data.get("source", "")
        self.age = data.get("age", "")


class BraveVideoResult:
    """Simple data class for Brave video results."""

    def __init__(self, data: dict):
        self.title = data.get("title", "")
        self.url = data.get("url", "")
        self.creator = data.get("creator", "")
        self.duration = data.get("duration", "")


class BraveSearchResponse:
    """Container for Brave search API response."""

    def __init__(self, data: dict):
        self.web_results = [
            BraveSearchResult(result) for result in data.get("web", {}).get("results", [])
        ]
        self.news_results = [
            BraveNewsResult(result) for result in data.get("news", {}).get("results", [])
        ]
        self.video_results = [
            BraveVideoResult(result) for result in data.get("videos", {}).get("results", [])
        ]


class BraveClient:
    """Custom Brave Search API client."""

    def __init__(self, api_key: str, logger=None):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.logger = logger or structlog.get_logger(__name__)

    async def search(self, q: str, count: int = 10, country: str | None = None) -> BraveSearchResponse:
        """Perform a search using the Brave Search API."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }

        params = {"q": q, "count": count}
        if country:
            params["country"] = country

        response = httpx.get(self.base_url, headers=headers, params=params)
        response.raise_for_status()

        # Remove the logging that was causing errors since status check happens after raise_for_status
        # If we get here, the status is already 200 (raise_for_status would have thrown otherwise)

        return BraveSearchResponse(response.json())


class BraveSearchPlugin(Plugin):
    """AI-enabled plugin class for Brave Search."""

    def __init__(self):
        """Initialize the plugin."""
        super().__init__()
        self.config = None
        self.http_client = None
        self.name = "brave_search"
        self.version = "1.0.0"
        self.llm_service = None
        self.api_key = None
        self.brave_client = None
        self.async_brave_client = None

    async def initialize(self, config: dict[str, Any], services: dict[str, Any]):
        """Initialize plugin with configuration and services."""
        self.logger.info("Initializing Brave Search plugin")
        # Store config for later use
        self.config = config
        
        # Try to get API key from config first, then environment variable
        self.api_key = config.get("api_key") or os.environ.get("BRAVE_API_KEY")
        
        # Don't fail during initialization if key is missing - let it be handled at runtime
        # This allows the plugin to be loaded even if the key isn't set yet
        if not self.api_key:
            self.logger.warning("Brave API key not found during initialization - will check again at runtime")
        else:
            # Initialize the client if we have the key
            self.brave_client = BraveClient(api_key=self.api_key, logger=self.logger)

        # Store LLM service for AI operations
        self.llm_service = services.get("llm")

        # Setup other services
        if "http_client" in services:
            self.http_client = services["http_client"]

    @capability(
        id="search_internet",
        name="Brave Search",
        description="A plugin that provides Brave Search functionality",
        scopes=["api:read"],
        ai_function=True,
        ai_parameters={
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input to process with Brave Search"
                },
            "mode": {
                    "type": "string",
                    "enum": ["fast", "accurate", "balanced"],
                    "description": "Processing mode",
                    "default": "balanced"
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "json", "markdown"],
                    "description": "Output format",
                    "default": "text"
                }            },
            "required": ["input"]
        }
    )
    async def search_internet(self, context: CapabilityContext) -> dict[str, Any]:
        """Execute the brave search capability."""
        self.logger.info("Executing Brave Search capability")
        try:
            # Initialize Brave client if not already done
            if not self.brave_client:
                config = context.config or {}
                self.api_key = config.get("api_key") or os.environ.get("BRAVE_API_KEY")
                if not self.api_key:
                    return {
                        "success": False,
                        "error": "Missing API key",
                        "content": "Brave API key not configured"
                    }
                self.brave_client = BraveClient(api_key=self.api_key, logger=self.logger)

            # Extract search query from AI function parameters or task content
            params = context.metadata.get("parameters", {})
            self.logger.info("AI function parameters", params=params)

            query = params.get("input", "")
            self.logger.info("Query from AI parameters", query=query)

            # If no input parameter, try extracting from user message
            if not query:
                user_input = await self._extract_user_input(context)
                self.logger.info("User input from context", user_input=user_input)
                query = await self._extract_search_query(user_input)
                self.logger.info("Query after processing user input", query=query)

            # If still no query, try task content
            if not query:
                input_text = await self._extract_task_content(context)
                self.logger.info("Task content", task_content=input_text)
                query = await self._extract_search_query(input_text)
                self.logger.info("Query after processing task content", query=query)

            self.logger.info("Extracted search query", query=query)

            # Validate query
            if not query or not query.strip():
                self.logger.warning("Empty search query provided")
                return {
                    "success": False,
                    "error": "Empty search query",
                    "content": "Please provide a search query"
                }

            input_text = query  # Use the extracted query as input_text

            # Log the start of processing (demonstrates structured logging)
            self.logger.info(
                "Starting capability execution",
                capability_id="search_internet",
                input_length=len(input_text)
            )

            # Perform the search
            result = await self.brave_client.search(q=query, count=10)
            self.logger.info("Search completed", query=query, result_length=len(result.web_results))

            if not result.web_results:
                self.logger.warning("No results found for the query", query=query)
                return {
                    "success": False,
                    "error": "No results found",
                    "content": "No results found for the given query"
                }

            # Format results - pass the full response object, not just web_results
            formatted_results = await self._format_search_results(result)

            # Extract parameters for AI functions
            params = context.metadata.get("parameters", {})
            mode = params.get("mode", "balanced")
            format = params.get("format", "text")
            input_text = params.get("input", input_text)

            # Return the search results for the main LLM to interpret
            # The plugin doesn't need its own LLM - the main orchestrating LLM will handle interpretation
            formatted_result = await self._format_output(formatted_results, format)

            # Log successful completion
            self.logger.info("Capability execution completed",
                           capability_id="search_internet",
                           mode=mode,
                           format=format,
                           result_length=len(formatted_result))

            return {
                "success": True,
                "content": formatted_result,
                "metadata": {
                    "capability": "search_internet",
                    "mode": mode,
                    "format": format,
                    "processed_at": datetime.datetime.now().isoformat()
                }
            }

        except Exception as e:
            # Log the error with structured data
            self.logger.error("Error in capability execution",
                            capability_id="search_internet",
                            error=str(e),
                            exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "content": f"Error in Brave Search: {str(e)}"
            }

    async def _extract_user_input(self, context: CapabilityContext) -> str:
        """Extract user input from the task context."""
        if hasattr(context.task, "history") and context.task.history:
            # Get the first user message (not the last, as that might be agent response)
            for msg in context.task.history:
                if hasattr(msg, "role") and msg.role.value == "user":
                    if hasattr(msg, "parts") and msg.parts:
                        for part in msg.parts:
                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                return part.root.text
                            elif hasattr(part, "text"):
                                return part.text
        return ""

    async def _extract_task_content(self, context: CapabilityContext) -> str:
        """Extract text content from task context"""
        task = context.task
        if hasattr(task, "content"):
            return task.content
        elif hasattr(task, "messages") and task.messages:
            return task.messages[0].content
        elif hasattr(task, "message"):
            return task.message
        else:
            return str(task)

    async def _format_output(self, result: str, format: str) -> str:
        """Format the result according to the specified format."""
        if format == "json":
            import json
            return json.dumps({
                "result": result,
                "plugin": "brave-search",
                "timestamp": datetime.datetime.now().isoformat()
            }, indent=2)
        elif format == "markdown":
            return f"## Brave Search Result\n\n{result}\n\n*Processed by brave-search*"
        else:  # text
            return result

    async def get_config_schema(self) -> dict[str, Any]:
        """Define configuration schema for AI plugin."""
        return {
            "type": "object",
            "properties": {
                "llm_model": {
                    "type": "string",
                    "description": "LLM model to use",
                    "default": "gpt-4o-mini"
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 4000,
                    "default": 200,
                    "description": "Maximum tokens for LLM responses"
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.7,
                    "description": "LLM temperature setting"
                }
            },
            "additionalProperties": False
        }

    async def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate AI plugin configuration."""
        return {
            "valid": True,
            "errors": [],
            "warnings": []
        }

    async def cleanup(self):
        """Cleanup resources when plugin is destroyed."""
        # Clear LLM service reference
        self.llm_service = None

        # Close HTTP client if available
        if hasattr(self, 'http_client') and hasattr(self.http_client, 'close'):
            await self.http_client.close()

    async def _extract_search_query(self, user_input: str) -> str:
        """Extract the search query from user input."""
        # Remove common search prefixes
        prefixes = [
            "search for",
            "look up",
            "find",
            "search",
            "brave search for",
            "web search for",
            "please search for",
            "can you search for",
            "i want to search for",
        ]

        query = user_input.lower()
        for prefix in prefixes:
            if query.startswith(prefix):
                query = user_input[len(prefix) :].strip()
                break

        # If no prefix found, use the entire input
        if query == user_input.lower():
            query = user_input

        return query

    async def _format_search_results(self, search_results) -> str:
        """Format search results for display."""
        sections = []

        # Web results
        if hasattr(search_results, 'web_results') and search_results.web_results:
            web_section = "## Web Results\n\n"
            for i, result in enumerate(search_results.web_results[:10], 1):
                title = getattr(result, "title", "No title")
                url = getattr(result, "url", "")
                description = getattr(result, "description", "No description")
                web_section += f"{i}. **{title}**\n   {url}\n   {description}\n\n"
            sections.append(web_section)

        # News results
        if hasattr(search_results, 'news_results') and search_results.news_results:
            news_section = "## News Results\n\n"
            for i, result in enumerate(search_results.news_results[:5], 1):
                title = getattr(result, "title", "No title")
                url = getattr(result, "url", "")
                age = getattr(result, "age", "")
                source = getattr(result, "source", "")
                news_section += f"{i}. **{title}**\n   {source} - {age}\n   {url}\n\n"
            sections.append(news_section)

        # Video results
        if hasattr(search_results, 'video_results') and search_results.video_results:
            video_section = "## Video Results\n\n"
            for i, result in enumerate(search_results.video_results[:5], 1):
                title = getattr(result, "title", "No title")
                url = getattr(result, "url", "")
                creator = getattr(result, "creator", "")
                duration = getattr(result, "duration", "")
                video_section += f"{i}. **{title}**\n   By {creator} - {duration}\n   {url}\n\n"
            sections.append(video_section)

        if not sections:
            return "No results found."

        return "\n".join(sections)

    async def _format_web_results(self, web_results) -> list[dict]:
        """Format web results for JSON response."""
        if not web_results:
            return []

        formatted = []
        for result in web_results[:10]:
            formatted.append(
                {
                    "title": getattr(result, "title", ""),
                    "url": getattr(result, "url", ""),
                    "description": getattr(result, "description", ""),
                }
            )
        return formatted

    async def _format_news_results(self, news_results) -> list[dict]:
        """Format news results for JSON response."""
        if not news_results:
            return []

        formatted = []
        for result in news_results[:10]:
            formatted.append(
                {
                    "title": getattr(result, "title", ""),
                    "url": getattr(result, "url", ""),
                    "source": getattr(result, "source", ""),
                    "age": getattr(result, "age", ""),
                }
            )
        return formatted

    async def _format_video_results(self, video_results) -> list[dict]:
        """Format video results for JSON response."""
        if not video_results:
            return []

        formatted = []
        for result in video_results[:10]:
            formatted.append(
                {
                    "title": getattr(result, "title", ""),
                    "url": getattr(result, "url", ""),
                    "creator": getattr(result, "creator", ""),
                    "duration": getattr(result, "duration", ""),
                }
            )
        return formatted
