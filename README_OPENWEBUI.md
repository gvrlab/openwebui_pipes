# Anthropic Claude API Integration - Complete Model Support with Extended Thinking

A comprehensive OpenWebUI pipe providing full integration with all Anthropic Claude models, including extended thinking capabilities, vision support, PDF processing, and 128K output tokens.

## Features

### Model Support
- **All Claude 3 Models**: Opus, Sonnet, Haiku
- **Claude 3.5 Models**: Sonnet, Haiku
- **Claude 3.7 Sonnet**: Standard & Extended Thinking modes
- **Claude 4 Models**: Sonnet & Opus (Standard & Extended Thinking modes)

### Advanced Capabilities
- **Extended Thinking**: Claude 3.7 and 4 models with up to 32K thinking tokens
- **128K Output**: Support for 128,000 output tokens on capable models
- **Vision Processing**: Image analysis with automatic resizing for oversized images
- **PDF Analysis**: Document processing on compatible models
- **Streaming Responses**: Real-time streaming with thinking visualization
- **Prompt Caching**: Server-side caching for efficient API usage
- **Function Calling**: Full tool/function calling support

### Security & Reliability
- Secure API key handling via environment variables
- API key format validation
- Automatic image resizing with multi-stage compression
- Comprehensive error handling with request ID tracking
- Token usage tracking and monitoring
- Configurable request timeouts (60-600s)

## Requirements

```
pydantic>=2.0.0
requests>=2.0.0
aiohttp>=3.8.0
pillow>=9.0.0
```

## Installation

### Method 1: OpenWebUI Interface

1. Go to **Settings** → **Pipelines**
2. Click **Add Pipeline**
3. Upload `Anthropic_pipe_V1.py`
4. Add your Anthropic API key in the pipeline settings

### Method 2: Docker Compose (Recommended)

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-directory>
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Edit `.env` and add your API key:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

4. Start OpenWebUI:
```bash
docker-compose up -d
```

5. Access at `http://localhost:3000`

## Configuration

### Pipeline Settings (Valves)

Configure these settings in the OpenWebUI interface under the pipeline settings:

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required) | - | - |
| `ENABLE_THINKING` | Show thinking process in responses | `true` | true/false |
| `MAX_OUTPUT_TOKENS` | Use maximum available output tokens | `true` | true/false |
| `THINKING_BUDGET_TOKENS` | Token budget for extended thinking | `16000` | 0-32000 |
| `AUTO_RESIZE_IMAGES` | Automatically resize oversized images | `true` | true/false |
| `REQUEST_TIMEOUT` | API request timeout in seconds | `300` | 60-600 |
| `TRACK_TOKEN_USAGE` | Log token usage statistics | `true` | true/false |

### Extended Thinking Mode

Extended Thinking models display their reasoning process:
- **Claude 3.7 Sonnet (Extended Thinking)**: Up to 16K thinking tokens
- **Claude 4 Sonnet/Opus (Extended Thinking)**: Up to 32K thinking tokens

The thinking process appears in `<thinking>` tags before the final response.

### Image Processing

The pipe automatically handles images with intelligent processing:
- **Auto-Resizing**: Oversized images are automatically compressed
- **Multi-Stage Compression**: 7-stage fallback (2048px@85% → 512px@50%)
- **Format Conversion**: PNG with transparency converted to JPEG
- **Size Limits**: 5MB per image, 100MB total
- **Supported Formats**: JPEG, PNG, GIF, WebP

### PDF Processing

Available on compatible models (Claude 3.5+, 3.7, 4):
- Maximum size: 32MB per PDF
- Automatic format validation
- Efficient processing with prompt caching

## Usage Examples

### Standard Chat
Select any Claude model and start chatting normally.

### Image Analysis
1. Select a vision-capable model (e.g., Claude 3.5 Sonnet)
2. Upload or paste an image
3. Ask questions about the image

### Extended Thinking
1. Select a model with "(Extended Thinking)" in the name
2. Enable `ENABLE_THINKING` in settings
3. Ask complex reasoning questions
4. See the thinking process in `<thinking>` tags

### PDF Analysis
1. Select a PDF-compatible model (Claude 3.5+)
2. Upload a PDF document
3. Ask questions about the content

## API Key

Get your API key from [Anthropic Console](https://console.anthropic.com/):
1. Create an account or log in
2. Navigate to API Keys
3. Generate a new key (starts with `sk-ant-`)
4. Add to `.env` file or pipeline settings

## Troubleshooting

### API Key Errors
- Verify key starts with `sk-ant-`
- Check key validity at console.anthropic.com
- Ensure key is properly set in environment or settings

### Image Processing Issues
- Images are auto-resized if oversized
- Check format is supported (JPEG, PNG, GIF, WebP)
- Verify total size doesn't exceed 100MB

### Timeout Errors
- Increase `REQUEST_TIMEOUT` in settings
- Default is 300s, maximum is 600s
- Large images/PDFs may need more time

### Model Not Found
- Verify model name is correctly formatted
- Check model is available in your region
- Some models may require API access approval

## Token Usage Tracking

When `TRACK_TOKEN_USAGE` is enabled, the pipe logs:
- Input tokens
- Output tokens
- Cache read tokens
- Cache creation tokens

View logs with:
```bash
docker-compose logs -f openwebui
```

## Security Best Practices

- Never commit `.env` files with real API keys
- Use environment variables for API keys
- Rotate API keys regularly
- Keep your OpenWebUI instance updated
- Use HTTPS in production environments

## Version History

### v5.1 (Current)
- Fixed hardcoded API key security vulnerability
- Added intelligent image auto-resizing with 7-stage compression
- Enhanced model support with accurate vision detection
- Improved token usage tracking
- Added configurable request timeouts
- Better error handling with request ID tracking

### v5.0
- Claude 4 models support
- Extended thinking with configurable budgets
- 128K output tokens support
- PDF processing capabilities

## License

MIT License - See LICENSE file for details

## Support

- **OpenWebUI**: https://github.com/open-webui/open-webui
- **Anthropic API Documentation**: https://docs.anthropic.com/
- **Issues**: Report bugs via GitHub issues

## Credits

- **Author**: gvrlab
- **Based on work by**: Balaxxe and nbellochi
- **Version**: 5.1
