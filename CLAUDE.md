# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this AgentUp plugin.

## Plugin Overview

This is an AgentUp plugin that provides Brave Search functionality. It follows the modern AgentUp plugin architecture using the `@capability` decorator system.

## Plugin Structure

```
agentup-brave-search/
├── src/
│   └── agentup_brave/
│       ├── __init__.py
│       └── plugin.py           # Main plugin implementation
├── tests/
│   └── test_brave.py
├── static/
│   └── logo.png               # Plugin logo
├── dist/                      # Distribution files
├── pyproject.toml             # Package configuration with AgentUp entry point
├── README.md                  # Plugin documentation
├── example-agentup.yml        # Example configuration
├── uv.lock                    # UV lock file
└── CLAUDE.md                  # This file
```

## Core Plugin Architecture

### Modern Plugin System
The plugin inherits from `Plugin` base class and uses decorators:

- `@capability` - Decorator to define plugin capabilities
- `async def initialize()` - Initialize plugin with config and services
- `async def cleanup()` - Clean up resources when plugin is destroyed

### Entry Point
The plugin is registered via entry point in `pyproject.toml`:
```toml
[project.entry-points."agentup.plugins"]
agentup_brave = "agentup_brave.plugin:BraveSearchPlugin"
```

## Development Guidelines

### Code Style
- Follow PEP 8 and Python best practices
- Use type hints throughout the codebase
- Use async/await for I/O operations
- Handle errors gracefully with proper error responses

### Plugin Implementation Patterns

#### 1. Plugin Class Structure
```python
from agent.plugins.base import Plugin
from agent.plugins.decorators import capability
from agent.plugins.models import CapabilityContext

class BraveSearchPlugin(Plugin):
    def __init__(self):
        super().__init__()
        self.name = "brave_search"
        self.version = "1.0.0"
        self.api_key = None
        
    def configure(self, config: dict[str, Any]) -> None:
        """Configure the plugin with settings."""
        super().configure(config)
        self.config = config
        # Get API key from config or environment
        self.api_key = config.get("api_key") or os.environ.get("BRAVE_API_KEY")
        # Initialize Brave client if API key is available
        if self.api_key:
            self.brave_client = BraveClient(api_key=self.api_key, logger=self.logger)
```

#### 2. Capability Definition
```python
@capability(
    id="search_internet",
    name="Brave Search",
    description="Search the web using Brave Search API",
    scopes=["api:read"],
    ai_function=True,
    ai_parameters={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The search query"
            },
            "format": {
                "type": "string",
                "enum": ["text", "json", "markdown"],
                "description": "Output format",
                "default": "text"
            }
        },
        "required": ["input"]
    }
)
async def search_internet(self, context: CapabilityContext) -> dict[str, Any]:
    """Execute the search capability."""
    # Get config from context
    config = context.config or {}
    api_key = config.get("api_key") or os.environ.get("BRAVE_API_KEY")
    
    # Extract parameters
    params = context.metadata.get("parameters", {})
    query = params.get("input", "")
    
    # Perform search and return result
    return {
        "success": True,
        "content": formatted_results,
        "metadata": {"capability": "search_internet"}
    }
```

#### 3. Configuration Access
```python
async def search_internet(self, context: CapabilityContext) -> dict[str, Any]:
    # Access plugin configuration from context
    config = context.config or {}
    
    # Get API key with fallback to environment variable
    api_key = config.get("api_key") or os.environ.get("BRAVE_API_KEY")
    
    # Use other config values
    max_results = config.get("max_results", 10)
    default_country = config.get("default_country", "us")
```

#### 4. Error Handling
```python
async def search_internet(self, context: CapabilityContext) -> dict[str, Any]:
    try:
        # Your capability logic here
        result = await self.brave_client.search(query)
        return {
            "success": True,
            "content": result,
            "metadata": {"capability": "search_internet"}
        }
    except Exception as e:
        self.logger.error("Search failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "content": f"Error in Brave Search: {str(e)}"
        }
```

### Configuration

#### Plugin Configuration in agentup.yml
```yaml
plugins:
  - plugin_id: agentup_brave
    package: agentup-brave
    name: Brave Search Plugin
    enabled: true
    config:
      # Use environment variable substitution (recommended)
      api_key: ${BRAVE_API_KEY}
      # Or direct configuration (not recommended for production)
      # api_key: "your-api-key-here"
      max_results: 10
      default_country: us
      debug: false
    capabilities:
      - capability_id: search_internet
        required_scopes: ["api:read"]
        enabled: true
```

#### Environment Variable Substitution
AgentUp automatically expands `${VARIABLE_NAME}` syntax in configuration files using the `expand_env_vars` function.

### Testing
- Write comprehensive tests for all plugin functionality
- Test both success and error cases
- Mock external dependencies
- Use pytest and async test patterns

## Development Workflow

### Local Development
1. Install in development mode: `pip install -e .`
2. Create test agent: `agentup agent create test-agent --template minimal`
3. Configure plugin in agent's `agentup.yml`
4. Test with: `agentup agent serve`

### Testing
```bash
# Run tests
pytest tests/ -v

# Check plugin loading
agentup plugin list

# Validate plugin
agentup plugin validate agentup_brave
```

### External Dependencies
- Use AgentUp's service registry for HTTP clients, databases, etc.
- Declare all dependencies in pyproject.toml
- Use async libraries for better performance

## Plugin Capabilities

### AI Function Support
When `ai_function=True` is set in `@capability` decorator:
- The capability becomes callable by LLMs
- Parameters are defined in `ai_parameters`
- The main orchestrating LLM handles the interpretation

### Context Access
The `CapabilityContext` provides:
- `task`: The current task being executed
- `config`: Plugin-specific configuration from agentup.yml
- `services`: Service registry instance
- `state`: Execution state
- `metadata`: Context metadata including AI function parameters

## Best Practices

### Performance
- Use async/await for I/O operations
- Implement caching for expensive operations
- Use connection pooling for external APIs
- Initialize clients once in `initialize()` method

### Security
- Validate all inputs
- Sanitize outputs
- Use secure authentication methods
- Never log sensitive data
- Support both config and environment variables for secrets

### Maintainability
- Follow single responsibility principle
- Keep functions small and focused
- Use descriptive variable names
- Add docstrings to all public methods
- Use structured logging with structlog

## Common Patterns

### External API Integration
```python
class BraveClient:
    def __init__(self, api_key: str, logger=None):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.logger = logger or structlog.get_logger(__name__)
    
    async def search(self, q: str, count: int = 10) -> BraveSearchResponse:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        # Use httpx for async HTTP requests
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, headers=headers, params={"q": q})
            response.raise_for_status()
            return BraveSearchResponse(response.json())
```

### Configuration Schema
```python
async def get_config_schema(self) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string",
                "description": "Brave Search API key"
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10
            }
        },
        "additionalProperties": False
    }
```

### Logging
```python
import structlog

class BraveSearchPlugin(Plugin):
    def __init__(self):
        super().__init__()
        self.logger = structlog.get_logger(__name__)
    
    async def search_internet(self, context: CapabilityContext) -> dict[str, Any]:
        self.logger.info("Starting search", capability_id="search_internet")
        # ... implementation
```

## Debugging Tips

### Common Issues
- Plugin not loading: Check entry point in pyproject.toml
- API key not found: Check both config and environment variable
- Import errors: Ensure package structure matches entry point
- Capability not available: Verify `@capability` decorator parameters

### Testing Configuration
```python
# Test that plugin reads config correctly
def test_config_reading():
    plugin = BraveSearchPlugin()
    config = {"api_key": "test-key", "max_results": 5}
    plugin.configure(config)
    assert plugin.api_key == "test-key"
```

## Distribution

### Package Structure
- Follow Python package conventions
- Include comprehensive README.md
- Add LICENSE file
- Include example-agentup.yml for configuration reference

### Publishing
1. Test thoroughly with various agents
2. Update version in pyproject.toml
3. Build package: `python -m build`
4. Upload to PyPI: `python -m twine upload dist/*`

## Important Notes

### Modern Plugin System
- No longer uses pluggy hooks (`@hookimpl`)
- Uses `@capability` decorator for defining capabilities
- Inherits from `Plugin` base class
- Configuration handled via `configure()` method
- Context passed via `CapabilityContext`

### Framework Integration
- Leverage AgentUp's built-in features
- Use provided utilities and helpers
- Follow established patterns from other plugins
- Maintain compatibility with different agent templates

### Community Guidelines
- Write clear documentation
- Provide usage examples
- Follow semantic versioning
- Respond to issues and pull requests

## Resources

- [AgentUp Documentation](https://docs.agentup.dev)
- [Plugin Development Guide](https://docs.agentup.dev/plugins/development)
- [Testing Guide](https://docs.agentup.dev/plugins/testing)
- [AI Functions Guide](https://docs.agentup.dev/plugins/ai-functions)

---

Remember: This plugin is part of the AgentUp ecosystem. Always use the modern plugin architecture with `@capability` decorator and proper configuration handling through `CapabilityContext`.