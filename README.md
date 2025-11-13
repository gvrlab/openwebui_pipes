# OpenWebUI with Anthropic Claude Integration

This Docker Compose setup provides OpenWebUI with integrated support for all Claude models via the Anthropic API pipe.

## Features

- **Full Claude Model Support**: All Claude 3, 3.5, 3.7, and 4 models
- **Extended Thinking**: Claude 3.7 and 4 models with up to 32K thinking tokens
- **128K Output**: Support for 128K output tokens on capable models
- **Vision Support**: Image processing with auto-resizing
- **PDF Processing**: PDF document analysis
- **Streaming Responses**: Real-time response streaming
- **Prompt Caching**: Efficient API usage with server-side caching

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose installed
- Anthropic API key from [console.anthropic.com](https://console.anthropic.com/)

### 2. Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Anthropic API key:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
   ```

### 3. Start the Service

```bash
docker-compose up -d
```

This will:
- Pull the latest OpenWebUI image
- Start the service on http://localhost:3000
- Mount the Anthropic pipe automatically
- Load your API key from the environment

### 4. Access OpenWebUI

Open your browser and navigate to:
```
http://localhost:3000
```

### 5. Using Claude Models

1. Create an account or log in
2. Go to Settings → Models
3. You'll see all available Claude models:
   - Standard models (3, 3.5, 3.7, 4)
   - Extended Thinking variants (3.7, 4)
4. Select a model and start chatting!

## Available Models

### Standard Models
- Claude 3 (Opus, Sonnet, Haiku)
- Claude 3.5 (Sonnet, Haiku)
- Claude 3.7 Sonnet (Standard)
- Claude 4 (Sonnet, Opus) - Standard

### Extended Thinking Models
- Claude 3.7 Sonnet (Extended Thinking)
- Claude 4 Sonnet (Extended Thinking)
- Claude 4 Opus (Extended Thinking)

Extended Thinking models show their reasoning process in `<thinking>` tags.

## Configuration Options

### Pipe Settings

The Anthropic pipe can be configured via the OpenWebUI interface:

- **ENABLE_THINKING**: Show/hide thinking process (default: true)
- **MAX_OUTPUT_TOKENS**: Use maximum output tokens (default: true)
- **THINKING_BUDGET_TOKENS**: Thinking token budget (default: 16000, max: 32000)
- **AUTO_RESIZE_IMAGES**: Automatically resize large images (default: true)
- **REQUEST_TIMEOUT**: API request timeout in seconds (default: 300)
- **TRACK_TOKEN_USAGE**: Log token usage statistics (default: true)

### Docker Compose Settings

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - WEBUI_NAME=Your Custom Name
  - ENABLE_OLLAMA_API=false
  - ENABLE_OPENAI_API=true
```

## Management Commands

### View Logs
```bash
docker-compose logs -f openwebui
```

### Stop Service
```bash
docker-compose down
```

### Restart Service
```bash
docker-compose restart
```

### Update to Latest Version
```bash
docker-compose pull
docker-compose up -d
```

### Remove Everything (including data)
```bash
docker-compose down -v
```

## Data Persistence

Your conversations and settings are stored in a Docker volume named `openwebui-data`. This ensures your data persists across container restarts.

## Troubleshooting

### Pipe Not Loading
1. Check logs: `docker-compose logs -f openwebui`
2. Verify the pipe file exists: `ls -l Anthropic_pipe_V1.py`
3. Ensure the file is readable

### API Key Errors
1. Verify your API key in `.env`
2. Ensure the key starts with `sk-ant-`
3. Check API key validity at console.anthropic.com

### Port Already in Use
If port 3000 is already in use, edit `docker-compose.yml`:
```yaml
ports:
  - "8080:8080"  # Change 3000 to any available port
```

### Image Processing Errors
The pipe automatically resizes oversized images. If you encounter issues:
1. Check image size (max 5MB per image, 100MB total)
2. Ensure images are in supported formats (JPEG, PNG, GIF, WebP)

## Security Notes

- Never commit your `.env` file with real API keys
- Keep your API key secure and rotate it regularly
- The pipe validates API key format for additional security
- Images are auto-resized to prevent oversized uploads

## Support

- OpenWebUI: https://github.com/open-webui/open-webui
- Anthropic API: https://docs.anthropic.com/
- Pipe Documentation: See comments in `Anthropic_pipe_V1.py`

## License

This setup uses:
- OpenWebUI (MIT License)
- Anthropic Pipe v5.1 (MIT License)
