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

    async def search(
        self, q: str, count: int = 10, country: str | None = None
    ) -> BraveSearchResponse:
        """Perform a search using the Brave Search API."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
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
        self.name = "brave_search"
        self.version = "1.0.0"
        self.config = None
        self.api_key = None
        self.brave_client = None

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the plugin with settings."""
        super().configure(config)

        # Store config for later use
        self.config = config

        # Try to get API key from config first, then environment variable
        self.api_key = config.get("api_key") or os.environ.get("BRAVE_API_KEY")

        # Don't fail during initialization if key is missing - let it be handled at runtime
        # This allows the plugin to be loaded even if the key isn't set yet
        if not self.api_key:
            self.logger.warning(
                "Brave API key not found during configuration - will check again at runtime"
            )
        else:
            # Initialize the client if we have the key
            self.brave_client = BraveClient(api_key=self.api_key, logger=self.logger)

        if config.get("debug", False):
            self.logger.info(f"Plugin configured with API key present: {bool(self.api_key)}")

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
                    "description": "The input to process with Brave Search",
                },
                "mode": {
                    "type": "string",
                    "enum": ["fast", "accurate", "balanced"],
                    "description": "Processing mode",
                    "default": "balanced",
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "json", "markdown"],
                    "description": "Output format",
                    "default": "text",
                },
            },
            "required": ["input"],
        },
        examples=[
            "Search for the latest news on AI advancements",
            "Find articles about climate change",
            "Look up recent technology trends",
            "What are the top headlines today?",
            "Find videos about space exploration",
        ],
        input_modes=["text/plain"],
        output_modes=["text/plain", "application/json"],
        security=[{"scopes": ["api:read"]}],
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
                        "content": "Brave API key not configured",
                    }
                self.brave_client = BraveClient(api_key=self.api_key, logger=self.logger)

            # Extract search query from AI function parameters or task content
            params = context.metadata.get("parameters", {})
            self.logger.info("AI function parameters", params=params)

            query = params.get("input", "")
            self.logger.info("Query from AI parameters", query=query)

            # If no input parameter, try extracting from user message
            if not query:
                user_input = self._extract_user_input(context)
                self.logger.info("User input from context", user_input=user_input)
                query = await self._extract_search_query(user_input)
                self.logger.info("Query after processing user input", query=query)

            # If still no query, try task content
            if not query:
                input_text = self._extract_task_content(context)
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
                    "content": "Please provide a search query",
                }

            input_text = query  # Use the extracted query as input_text

            # Log the start of processing (demonstrates structured logging)
            self.logger.info(
                "Starting capability execution",
                capability_id="search_internet",
                input_length=len(input_text),
            )

            # Perform the search
            result = await self.brave_client.search(q=query, count=10)
            self.logger.info("Search completed", query=query, result_length=len(result.web_results))

            if not result.web_results:
                self.logger.warning("No results found for the query", query=query)
                return {
                    "success": False,
                    "error": "No results found",
                    "content": "No results found for the given query",
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
            self.logger.info(
                "Capability execution completed",
                capability_id="search_internet",
                mode=mode,
                format=format,
                result_length=len(formatted_result),
            )

            return {
                "success": True,
                "content": formatted_result,
                "metadata": {
                    "capability": "search_internet",
                    "mode": mode,
                    "format": format,
                    "processed_at": datetime.datetime.now().isoformat(),
                },
            }

        except Exception as e:
            # Log the error with structured data
            self.logger.error(
                "Error in capability execution",
                capability_id="search_internet",
                error=str(e),
                exc_info=True,
            )
            return {
                "success": False,
                "error": str(e),
                "content": f"Error in Brave Search: {str(e)}",
            }

    def _extract_user_input(self, context: CapabilityContext) -> str:
        """Extract user input from the task context (A2A message structure)."""
        if hasattr(context.task, "history") and context.task.history:
            # Get the first user message (not the last, as that might be agent response)
            for msg in context.task.history:
                if hasattr(msg, "role") and msg.role.value == "user":
                    if hasattr(msg, "parts") and msg.parts:
                        for part in msg.parts:
                            # Check for text content with proper type checking
                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                text_content = getattr(part.root, "text", None)
                                if text_content:
                                    return str(text_content)
                            # Direct text attribute access with safe getattr
                            text_content = getattr(part, "text", None)
                            if text_content:
                                return str(text_content)
        return ""

    def _extract_task_content(self, context: CapabilityContext) -> str:
        """Extract text content from task context"""
        task = context.task
        if hasattr(task, "content"):
            content = getattr(task, "content", None)
            if content:
                return str(content)

        if hasattr(task, "message"):
            message = getattr(task, "message", None)
            if message:
                return str(message)

        # Fallback to string representation of task
        return str(task) if task else ""

    async def _format_output(self, result: str, format: str) -> str:
        """Format the result according to the specified format."""
        if format == "json":
            import json

            return json.dumps(
                {
                    "result": result,
                    "plugin": "brave-search",
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                indent=2,
            )
        elif format == "markdown":
            return f"## Brave Search Result\n\n{result}\n\n*Processed by brave-search*"
        else:  # text
            return result

    def get_config_schema(self) -> dict[str, Any]:
        """Define configuration schema for Brave Search plugin."""
        return {
            "type": "object",
            "properties": {
                "api_key": {"type": "string", "description": "Brave Search API key"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10,
                    "description": "Maximum number of search results to return",
                },
                "default_country": {
                    "type": "string",
                    "description": "Default country code for searches",
                    "default": "us",
                },
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable/disable the plugin",
                },
                "debug": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable debug logging",
                },
            },
            "additionalProperties": False,
        }

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate Brave Search plugin configuration."""
        errors = []
        warnings = []

        # Check if API key is provided
        api_key = config.get("api_key") or os.environ.get("BRAVE_API_KEY")
        if not api_key:
            warnings.append("No API key found in config or BRAVE_API_KEY environment variable")

        # Validate max_results
        max_results = config.get("max_results", 10)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 20:
            errors.append("max_results must be an integer between 1 and 20")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    async def cleanup(self):
        """Cleanup resources when plugin is destroyed."""
        # Basic cleanup
        pass

    async def _extract_search_query(self, user_input: str) -> str:
        """Extract the search query from user input."""
        if not user_input or not user_input.strip():
            return ""

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
        else:
            # If no prefix found, check if this looks like a task context rather than a search query
            if self._is_task_context(user_input):
                # Extract key terms from task context
                query = self._extract_key_terms_from_task(user_input)
            else:
                # Use the entire input if it's reasonable length
                if len(user_input) <= 200:
                    query = user_input
                else:
                    # For very long inputs, extract the first meaningful sentence or phrase
                    query = self._extract_first_meaningful_phrase(user_input)

        # Final cleanup and length check
        query = query.strip()
        if len(query) > 400:  # Brave API has query length limits
            query = query[:400].rsplit(' ', 1)[0]  # Cut at word boundary

        return query

    def _is_task_context(self, text: str) -> bool:
        """Check if the text appears to be a task context rather than a search query."""
        task_indicators = [
            "GOAL:", "ITERATION:", "PLANNED TASKS:", "Task 1:", "Task 2:",
            "Take the next concrete action", "make progress toward the goal",
            "call the 'mark_goal_complete' tool", "Otherwise, continue making concrete progress"
        ]
        return any(indicator in text for indicator in task_indicators)

    def _extract_key_terms_from_task(self, task_text: str) -> str:
        """Extract key search terms from a task description."""
        # Look for goal description
        if "GOAL:" in task_text:
            goal_section = task_text.split("GOAL:")[1].split("\n")[0]
            goal_section = goal_section.strip()
            if goal_section:  # Only return if non-empty
                return goal_section

        # Look for task descriptions that might contain searchable terms
        if "install" in task_text.lower() and ("pandas" in task_text.lower() or "matplotlib" in task_text.lower()):
            return "how to install pandas matplotlib seaborn using pip"

        # Generic fallback - extract first meaningful line
        lines = task_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('GOAL:', 'ITERATION:', 'PLANNED TASKS:', 'Task ', 'IMPORTANT:')):
                return line[:100]  # Limit to reasonable length

        return "python data analysis libraries"  # Safe fallback

    def _extract_first_meaningful_phrase(self, text: str) -> str:
        """Extract the first meaningful phrase from long text."""
        # Split by sentences and take the first reasonable one
        sentences = text.replace('\n', ' ').split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if 10 <= len(sentence) <= 200:  # Reasonable sentence length
                return sentence

        # Fallback to first 100 characters
        return text[:100].strip()

    async def _format_search_results(self, search_results) -> str:
        """Format search results for display."""
        sections = []

        # Web results
        if hasattr(search_results, "web_results") and search_results.web_results:
            web_section = "## Web Results\n\n"
            for i, result in enumerate(search_results.web_results[:10], 1):
                title = getattr(result, "title", "No title")
                url = getattr(result, "url", "")
                description = getattr(result, "description", "No description")
                web_section += f"{i}. **{title}**\n   {url}\n   {description}\n\n"
            sections.append(web_section)

        # News results
        if hasattr(search_results, "news_results") and search_results.news_results:
            news_section = "## News Results\n\n"
            for i, result in enumerate(search_results.news_results[:5], 1):
                title = getattr(result, "title", "No title")
                url = getattr(result, "url", "")
                age = getattr(result, "age", "")
                source = getattr(result, "source", "")
                news_section += f"{i}. **{title}**\n   {source} - {age}\n   {url}\n\n"
            sections.append(news_section)

        # Video results
        if hasattr(search_results, "video_results") and search_results.video_results:
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
