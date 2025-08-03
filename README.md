# Brave Search Plugin for AgentUp

A plugin that provides Brave Search functionality to AgentUp agents, enabling web search, news search, and video search capabilities.

## Features

- **Web Search**: Search the internet for web pages, articles, and information
- **News Search**: Find recent news articles and current events
- **Video Search**: Discover video content from across the web
- **Configurable**: Customizable search result counts and API settings

## Installation

### For development:
```bash
cd brave
pip install -e .
```

### From AgentUp Registry or PyPi (when published):
```bash
pip install agentup-brave
```

## Configuration

### API Key Setup

You need a Brave Search API key to use this plugin. You can obtain one from [Brave Search API](https://brave.com/search/api/).

Set your API key using one of these methods:

1. **Environment Variable** (recommended for security):
   ```bash
   export BRAVE_API_KEY="your-api-key-here"
   ```

2. **Agent Configuration** (in `agentup.yml`):
   ```yaml
   plugins:
     - plugin_id: brave_search
       package: agentup-brave
       config:
         api_key: ${BRAVE_API_KEY}  # Use environment variable substitution
         default_count: 10  # Optional: number of results (1-20)
       capabilities:
         - capability_id: search_internet
           required_scopes: ["api:read"]
           enabled: true
   ```

### Full Configuration Example

```yaml
# agentup.yml
agent:
  name: Search Agent
  description: An agent with web search capabilities

plugins:
  - plugin_id: brave_search
    package: agentup-brave
    name: Brave Search Plugin
    enabled: true
    config:
      # Recommended: Use environment variable substitution
      api_key: ${BRAVE_API_KEY}
      
      # Alternative: Direct configuration (not recommended for production)
      # api_key: "your-brave-api-key-here"
      
      default_count: 10  # Optional: default number of results
    capabilities:
      - capability_id: search_internet
        required_scopes: ["api:read"]
        enabled: true
```

**Note:** AgentUp automatically expands `${VARIABLE_NAME}` syntax in the configuration file. The plugin will use the API key from the config, or fall back to the `BRAVE_API_KEY` environment variable if the config value is not set.

## Usage

### Natural Language Queries

The plugin responds to natural language queries containing search-related keywords:

- "Search for machine learning tutorials"
- "Find information about climate change"
- "What is quantum computing?"
- "Look up the latest news on AI"
- "Show me videos about cooking pasta"

### AI Function Calls

When used with LLMs, the plugin provides two functions:

1. **brave_web_search**
```json
{
  "query": "machine learning tutorials",
  "count": 10,
  "search_type": "web"
}
```

2. **brave_news_search**
```json
{
  "query": "artificial intelligence breakthroughs",
  "count": 5
}
```

## Capabilities

The plugin registers with the following capabilities:
- **ID**: `web_search`
- **Types**: `TEXT`, `AI_FUNCTION`
- **Tags**: `search`, `web`, `brave`, `news`, `videos`

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=brave
```

## API Response Format

### Web Results
```markdown
## Web Results

1. **Title of Result**
   https://example.com/page
   Description of the web page content
```

### News Results
```markdown
## News Results

1. **News Article Title**
   Source Name - 2 hours ago
   https://news-site.com/article
```

### Video Results
```markdown
## Video Results

1. **Video Title**
   By Creator Name - 10:30
   https://video-platform.com/watch
```

## License

Apache 2.0