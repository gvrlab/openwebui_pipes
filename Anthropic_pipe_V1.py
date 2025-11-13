"""
title: Anthropic API Integration w/ Extended Thinking, 128K Output, & Complete Claude Model Support
author: gvrlab (based on work by Balaxxe and nbellochi)
version: 5.1
license: MIT
requirements: pydantic>=2.0.0, requests>=2.0.0, aiohttp>=3.8.0, pillow>=9.0.0
environment_variables:
    - ANTHROPIC_API_KEY (required)

Supports:
- All Claude 3, 3.5, 3.7, and 4 models (including latest versions)
- Extended thinking with configurable budget
- 128K output tokens for Claude 3.7 and 4 models
- Streaming responses with thinking visualization
- Image and PDF processing with auto-resizing
- Prompt caching (server-side)
- Function calling
- Cache Control
- Comprehensive error handling
- Token usage tracking

Updates in v5.1 (gvrlab):
Built upon v5.0 by adding the following security and functionality improvements:

Security Enhancements:
- Fixed critical hardcoded API key security vulnerability
- Added API key validation with format checking
- API key now properly loaded from environment variables

Image Processing Improvements:
- Implemented intelligent auto-resizing for oversized images
- Added PIL/Pillow integration with multi-stage compression
- Progressive quality reduction with 7-stage fallback (2048px@85% → 512px@50%)
- Automatic RGBA to RGB conversion for PNG transparency handling
- JPEG progressive encoding for optimized loading
- Per-image size validation (5MB limit)
- Cumulative image size validation (100MB total limit)
- Better error messages with specific image index and size information

Enhanced Model Support:
- Added VISION_SUPPORTED_MODELS set for accurate vision capability detection
- Proper handling of vision-only models (excluding claude-3-5-haiku-20241022)
- Fixed model validation in content processing

Robustness & Monitoring:
- Token usage tracking across requests (input, output, cache read, cache creation)
- Configurable request timeout via valves (60-600s range)
- Enhanced error context in all failure scenarios
- Request ID tracking in all error messages
- Better logging for debugging and monitoring

Code Quality:
- Added type hints for better code clarity
- Improved validator for API key in Valves
- Better separation of concerns in image processing
- More detailed inline documentation
- Fixed thinking budget calculation logic for Claude 4 models

Based on Original v5.0 (Balaxxe, nbellochi):
- Claude 4 models support (Sonnet 4, Opus 4)
- Extended thinking with configurable budgets
- 128K output tokens support
- PDF processing capabilities
- Comprehensive model support and metadata
"""

import os
import json
import time
import hashlib
import logging
import asyncio
import random
import base64
import io
from datetime import datetime
from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Dict,
    Optional,
    AsyncIterator,
    Tuple,
)
from pydantic import BaseModel, Field, validator
from open_webui.utils.misc import pop_system_message
import aiohttp

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Image resizing disabled.")


class Pipe:
    API_VERSION = "2023-06-01"
    MODEL_URL = "https://api.anthropic.com/v1/messages"
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]

    # Models with vision support
    VISION_SUPPORTED_MODELS = {
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        # claude-3-5-haiku does NOT support vision
        "claude-3-7-sonnet-20250219",
        "claude-3-7-sonnet-latest",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-sonnet-4-latest",
        "claude-opus-4-latest",
    }

    SUPPORTED_PDF_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet-latest",
        "claude-3-7-sonnet-20250219",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-sonnet-4-latest",
        "claude-opus-4-latest",
    ]

    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
    MAX_PDF_SIZE = 32 * 1024 * 1024  # 32MB per PDF
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB total
    MAX_IMAGE_DIMENSION = 2048  # Recommended max dimension for resize

    PDF_BETA_HEADER = "pdfs-2024-09-25"
    OUTPUT_128K_BETA = "output-128k-2025-02-19"
    BETA_HEADER = "prompt-caching-2024-07-31"

    # Model max tokens - comprehensive list
    MODEL_MAX_TOKENS = {
        # Claude 3 family
        "claude-3-opus-20240229": 4096,
        "claude-3-sonnet-20240229": 4096,
        "claude-3-haiku-20240307": 4096,
        # Claude 3.5 family
        "claude-3-5-sonnet-20240620": 8192,
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-5-haiku-20241022": 8192,
        # Claude 3.7 family
        "claude-3-7-sonnet-20250219": 16384,
        # Claude 4 family
        "claude-sonnet-4-20250514": 32000,
        "claude-opus-4-20250514": 32000,
        # Latest aliases
        "claude-3-opus-latest": 4096,
        "claude-3-sonnet-latest": 4096,
        "claude-3-haiku-latest": 4096,
        "claude-3-5-sonnet-latest": 8192,
        "claude-3-5-haiku-latest": 8192,
        "claude-3-7-sonnet-latest": 16384,
        "claude-sonnet-4-latest": 32000,
        "claude-opus-4-latest": 32000,
    }

    # Model context lengths
    MODEL_CONTEXT_LENGTH = {model: 200000 for model in MODEL_MAX_TOKENS.keys()}

    # Models that support extended thinking
    THINKING_SUPPORTED_MODELS = [
        "claude-3-7-sonnet-latest",
        "claude-3-7-sonnet-20250219",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-sonnet-4-latest",
        "claude-opus-4-latest",
    ]

    REQUEST_TIMEOUT = 300
    DEFAULT_THINKING_BUDGET = 16000
    CLAUDE_4_MAX_THINKING_BUDGET = 32000

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""  # FIXED: No default key for security
        ENABLE_THINKING: bool = True
        MAX_OUTPUT_TOKENS: bool = True
        ENABLE_TOOL_CHOICE: bool = True
        ENABLE_SYSTEM_PROMPT: bool = True
        THINKING_BUDGET_TOKENS: int = Field(default=16000, ge=0, le=32000)
        AUTO_RESIZE_IMAGES: bool = True  # New: Auto-resize oversized images
        REQUEST_TIMEOUT: int = Field(default=300, ge=60, le=600)  # Configurable timeout
        TRACK_TOKEN_USAGE: bool = True  # New: Track token usage

        @validator("ANTHROPIC_API_KEY")
        def validate_api_key(cls, v):
            if not v:
                raise ValueError("ANTHROPIC_API_KEY is required")
            if not v.startswith("sk-ant-"):
                logging.warning("API key doesn't match expected format")
            return v

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "anthropic"
        # Initialize valves with API key from environment if available
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.valves = self.Valves(ANTHROPIC_API_KEY=api_key) if api_key else self.Valves()
        self.request_id = None
        self.total_tokens_used = {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_creation": 0,
        }

    def get_anthropic_models(self) -> List[dict]:
        """Returns a list of all supported Anthropic Claude models with their capabilities."""

        standard_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-latest",
            "claude-3-sonnet-latest",
            "claude-3-haiku-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]

        hybrid_models = [
            "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-latest",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-sonnet-4-latest",
            "claude-opus-4-latest",
        ]

        models = []

        # Add standard models
        for name in standard_models:
            models.append(
                {
                    "id": f"anthropic/{name}",
                    "name": name,
                    "context_length": self.MODEL_CONTEXT_LENGTH.get(name, 200000),
                    "supports_vision": name in self.VISION_SUPPORTED_MODELS,
                    "supports_thinking": False,
                    "is_hybrid_model": False,
                    "max_output_tokens": self.MODEL_MAX_TOKENS.get(name, 4096),
                }
            )

        # Add hybrid models - both standard and thinking versions
        for name in hybrid_models:
            # Standard mode
            models.append(
                {
                    "id": f"anthropic/{name}",
                    "name": f"{name} (Standard)",
                    "context_length": self.MODEL_CONTEXT_LENGTH.get(name, 200000),
                    "supports_vision": name in self.VISION_SUPPORTED_MODELS,
                    "supports_thinking": False,
                    "is_hybrid_model": True,
                    "thinking_mode": "standard",
                    "max_output_tokens": self.MODEL_MAX_TOKENS.get(name, 16384),
                }
            )

            # Thinking mode
            models.append(
                {
                    "id": f"anthropic/{name}-thinking",
                    "name": f"{name} (Extended Thinking)",
                    "context_length": self.MODEL_CONTEXT_LENGTH.get(name, 200000),
                    "supports_vision": name in self.VISION_SUPPORTED_MODELS,
                    "supports_thinking": True,
                    "is_hybrid_model": True,
                    "thinking_mode": "extended",
                    "max_output_tokens": (
                        131072
                        if self.valves.MAX_OUTPUT_TOKENS
                        else self.MODEL_MAX_TOKENS.get(name, 16384)
                    ),
                }
            )

        return models

    def pipes(self) -> List[dict]:
        return self.get_anthropic_models()

    def resize_image_if_needed(
        self, base64_data: str, media_type: str
    ) -> Tuple[str, bool]:
        """
        Resize image if it exceeds size limits with aggressive multi-stage compression.
        Returns tuple of (base64_data, was_resized)
        """
        if not PIL_AVAILABLE or not self.valves.AUTO_RESIZE_IMAGES:
            return base64_data, False

        try:
            # Decode and check size
            img_bytes = base64.b64decode(base64_data)
            original_size = len(img_bytes)

            if original_size <= self.MAX_IMAGE_SIZE:
                return base64_data, False

            logging.info(
                f"Image size {original_size/1024/1024:.2f}MB exceeds {self.MAX_IMAGE_SIZE/1024/1024}MB limit. Starting resize..."
            )

            # Open image
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert RGBA to RGB if needed (for PNG with transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Multi-stage resize attempts with progressive compression
            resize_attempts = [
                # (max_dimension, quality, format_override)
                (2048, 85, None),      # Try original format first
                (2048, 75, 'JPEG'),    # Force JPEG with good quality
                (1600, 70, 'JPEG'),    # Reduce dimensions
                (1024, 65, 'JPEG'),    # Further reduction
                (800, 60, 'JPEG'),     # More aggressive
                (640, 55, 'JPEG'),     # Very aggressive
                (512, 50, 'JPEG'),     # Last resort
            ]

            resized_bytes = None
            successful_attempt = None

            for max_dim, quality, force_format in resize_attempts:
                try:
                    # Create a copy for this attempt
                    img_copy = img.copy()
                    
                    # Resize maintaining aspect ratio
                    img_copy.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                    
                    buffer = io.BytesIO()
                    
                    # Determine format
                    if force_format:
                        save_format = force_format
                        output_media_type = f"image/{force_format.lower()}"
                    else:
                        format_map = {
                            "image/jpeg": "JPEG",
                            "image/png": "JPEG",  # Convert PNG to JPEG for better compression
                            "image/gif": "JPEG",
                            "image/webp": "WEBP",
                        }
                        save_format = format_map.get(media_type, "JPEG")
                        output_media_type = f"image/{save_format.lower()}"
                    
                    # Save with optimization
                    save_kwargs = {
                        "format": save_format,
                        "quality": quality,
                        "optimize": True
                    }
                    
                    # Additional JPEG optimization
                    if save_format == "JPEG":
                        save_kwargs["progressive"] = True
                    
                    img_copy.save(buffer, **save_kwargs)
                    resized_bytes = buffer.getvalue()
                    
                    # Check if this attempt succeeded
                    if len(resized_bytes) <= self.MAX_IMAGE_SIZE:
                        successful_attempt = (max_dim, quality, save_format)
                        logging.info(
                            f"Successfully resized image: {original_size/1024/1024:.2f}MB -> {len(resized_bytes)/1024/1024:.2f}MB "
                            f"(dimensions: {max_dim}x{max_dim}, quality: {quality}, format: {save_format})"
                        )
                        break
                    else:
                        logging.debug(
                            f"Attempt failed (dim:{max_dim}, q:{quality}): {len(resized_bytes)/1024/1024:.2f}MB still too large"
                        )
                        
                except Exception as attempt_error:
                    logging.debug(f"Resize attempt failed: {str(attempt_error)}")
                    continue

            # Check if any attempt succeeded
            if resized_bytes and len(resized_bytes) <= self.MAX_IMAGE_SIZE:
                new_base64 = base64.b64encode(resized_bytes).decode("utf-8")
                return new_base64, True
            else:
                # All attempts failed
                final_size = len(resized_bytes) if resized_bytes else original_size
                logging.error(
                    f"Failed to resize image below {self.MAX_IMAGE_SIZE/1024/1024}MB limit. "
                    f"Final size: {final_size/1024/1024:.2f}MB"
                )
                return base64_data, False

        except Exception as e:
            logging.error(f"Failed to resize image: {str(e)}")
            return base64_data, False

    def process_image(self, image_data: dict, image_index: int = 0) -> dict:
        """Process image with better error handling and auto-resizing."""
        try:
            if image_data["image_url"]["url"].startswith("data:image"):
                header, base64_data = image_data["image_url"]["url"].split(",", 1)
                media_type = header.split(":")[1].split(";")[0]

                if media_type not in self.SUPPORTED_IMAGE_TYPES:
                    raise ValueError(
                        f"Image {image_index + 1}: Unsupported format {media_type}"
                    )

                # Check actual decoded size
                try:
                    decoded_size = len(base64.b64decode(base64_data))
                except Exception:
                    raise ValueError(
                        f"Image {image_index + 1}: Invalid base64 encoding"
                    )

                if decoded_size > self.MAX_IMAGE_SIZE:
                    if self.valves.AUTO_RESIZE_IMAGES:
                        base64_data, was_resized = self.resize_image_if_needed(
                            base64_data, media_type
                        )
                        if was_resized:
                            logging.info(
                                f"Image {image_index + 1} was automatically resized"
                            )
                        else:
                            raise ValueError(
                                f"Image {image_index + 1}: Size {decoded_size/1024/1024:.1f}MB exceeds limit"
                            )
                    else:
                        raise ValueError(
                            f"Image {image_index + 1}: Size {decoded_size/1024/1024:.1f}MB exceeds {self.MAX_IMAGE_SIZE/1024/1024}MB limit"
                        )

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
            else:
                return {
                    "type": "image",
                    "source": {"type": "url", "url": image_data["image_url"]["url"]},
                }
        except Exception as e:
            raise ValueError(f"Failed to process image {image_index + 1}: {str(e)}")

    def process_pdf(self, pdf_data: dict) -> dict:
        """Process PDF with size validation."""
        if (
            pdf_data.get("pdf_url", {})
            .get("url", "")
            .startswith("data:application/pdf")
        ):
            mime_type, base64_data = pdf_data["pdf_url"]["url"].split(",", 1)

            # Check actual decoded size
            try:
                pdf_size = len(base64.b64decode(base64_data))
            except Exception:
                raise ValueError("Invalid PDF base64 encoding")

            if pdf_size > self.MAX_PDF_SIZE:
                raise ValueError(
                    f"PDF size {pdf_size/1024/1024:.1f}MB exceeds {self.MAX_PDF_SIZE/1024/1024}MB limit"
                )

            document = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64_data,
                },
            }

            if pdf_data.get("cache_control"):
                document["cache_control"] = pdf_data["cache_control"]

            return document
        else:
            document = {
                "type": "document",
                "source": {"type": "url", "url": pdf_data["pdf_url"]["url"]},
            }

            if pdf_data.get("cache_control"):
                document["cache_control"] = pdf_data["cache_control"]

            return document

    def process_content(
        self, content: Union[str, List[dict]], model_name: str = ""
    ) -> List[dict]:
        """Process content with cumulative size validation."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        processed_content = []
        total_image_size = 0
        image_count = 0

        for item in content:
            if item["type"] == "text":
                processed_content.append({"type": "text", "text": item["text"]})

            elif item["type"] == "image_url":
                # Check vision support
                if model_name and model_name not in self.VISION_SUPPORTED_MODELS:
                    raise ValueError(f"Model {model_name} does not support images")

                processed_image = self.process_image(item, image_count)

                # Track cumulative size
                if processed_image["source"]["type"] == "base64":
                    image_size = len(
                        base64.b64decode(processed_image["source"]["data"])
                    )
                    total_image_size += image_size
                    if total_image_size > self.TOTAL_MAX_IMAGE_SIZE:
                        raise ValueError(
                            f"Total image size {total_image_size/1024/1024:.1f}MB exceeds {self.TOTAL_MAX_IMAGE_SIZE/1024/1024}MB limit"
                        )

                processed_content.append(processed_image)
                image_count += 1

            elif item["type"] == "pdf_url":
                if model_name and model_name not in self.SUPPORTED_PDF_MODELS:
                    raise ValueError(
                        f"PDF support not available for model: {model_name}"
                    )
                processed_content.append(self.process_pdf(item))

            elif item["type"] in ["tool_calls", "tool_results"]:
                processed_content.append(item)

        return processed_content

    def _process_messages(
        self, messages: List[dict], model_name: str = ""
    ) -> List[dict]:
        """Process messages for the Anthropic API format."""
        processed_messages = []

        for message in messages:
            processed_content = []
            for content in self.process_content(message["content"], model_name):
                if (
                    message.get("role") == "assistant"
                    and content.get("type") == "tool_calls"
                ):
                    content["cache_control"] = {"type": "ephemeral"}
                elif (
                    message.get("role") == "user"
                    and content.get("type") == "tool_results"
                ):
                    content["cache_control"] = {"type": "ephemeral"}
                processed_content.append(content)

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        return processed_messages

    def calculate_thinking_budget(self, model_name: str) -> int:
        """Calculate appropriate thinking budget for model."""
        is_claude_4 = any(
            claude4 in model_name for claude4 in ["claude-sonnet-4", "claude-opus-4"]
        )

        if is_claude_4:
            # Claude 4 models can use up to 32K thinking tokens
            return min(
                self.valves.THINKING_BUDGET_TOKENS, self.CLAUDE_4_MAX_THINKING_BUDGET
            )
        else:
            # Claude 3.7 models use standard budget
            return min(self.valves.THINKING_BUDGET_TOKENS, self.DEFAULT_THINKING_BUDGET)

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, AsyncIterator[str]]:
        """Process a request to the Anthropic API."""

        # Validate API key
        if not self.valves.ANTHROPIC_API_KEY:
            error_msg = "Error: ANTHROPIC_API_KEY is required"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg

        try:
            system_message, messages = pop_system_message(body["messages"])
            model_name = body["model"].split("/")[-1]

            # Handle thinking model variants
            is_thinking_variant = model_name.endswith("-thinking")
            actual_model_name = (
                model_name.replace("-thinking", "")
                if is_thinking_variant
                else model_name
            )

            if actual_model_name not in self.MODEL_MAX_TOKENS:
                logging.warning(
                    f"Unknown model: {actual_model_name}, using default token limit"
                )

            # Get max tokens for the model
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(actual_model_name, 4096)

            if self.valves.MAX_OUTPUT_TOKENS:
                max_tokens = max_tokens_limit
            else:
                max_tokens = min(
                    body.get("max_tokens", max_tokens_limit), max_tokens_limit
                )

            # Process messages with model validation
            try:
                processed_messages = self._process_messages(messages, actual_model_name)
            except ValueError as e:
                error_msg = str(e)
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": error_msg, "done": True},
                        }
                    )
                return error_msg

            payload = {
                "model": actual_model_name,
                "messages": processed_messages,
                "max_tokens": max_tokens,
                "temperature": (
                    float(body.get("temperature"))
                    if body.get("temperature") is not None
                    else None
                ),
                "top_k": (
                    int(body.get("top_k")) if body.get("top_k") is not None else None
                ),
                "top_p": (
                    float(body.get("top_p")) if body.get("top_p") is not None else None
                ),
                "stream": body.get("stream", False),
                "metadata": body.get("metadata", {}),
            }

            # Handle thinking mode
            should_enable_thinking = (
                is_thinking_variant
                and actual_model_name in self.THINKING_SUPPORTED_MODELS
            )

            if should_enable_thinking:
                thinking_budget = self.calculate_thinking_budget(actual_model_name)
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                logging.info(f"Thinking enabled with budget: {thinking_budget} tokens")

            payload = {k: v for k, v in payload.items() if v is not None}

            # Add system message if enabled
            if system_message and self.valves.ENABLE_SYSTEM_PROMPT:
                payload["system"] = str(system_message)

            # Add tools if enabled
            if "tools" in body and self.valves.ENABLE_TOOL_CHOICE:
                payload["tools"] = [
                    {"type": "function", "function": tool} for tool in body["tools"]
                ]
                payload["tool_choice"] = body.get("tool_choice")

            if "response_format" in body:
                payload["response_format"] = {
                    "type": body["response_format"].get("type")
                }

            # Prepare headers
            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            }

            beta_headers = []

            # Add beta headers as needed
            if any(
                isinstance(msg["content"], list)
                and any(item.get("type") == "document" for item in msg["content"])
                for msg in payload.get("messages", [])
            ):
                beta_headers.append(self.PDF_BETA_HEADER)

            if any(
                isinstance(msg["content"], list)
                and any(item.get("cache_control") for item in msg["content"])
                for msg in payload.get("messages", [])
            ):
                beta_headers.append(self.BETA_HEADER)

            if (
                actual_model_name in self.THINKING_SUPPORTED_MODELS
                and self.valves.MAX_OUTPUT_TOKENS
            ):
                beta_headers.append(self.OUTPUT_128K_BETA)

            if beta_headers:
                headers["anthropic-beta"] = ",".join(beta_headers)

            # Process request
            if payload["stream"]:
                return self._stream_with_ui(
                    self.MODEL_URL, headers, payload, body, __event_emitter__
                )

            response_data, cache_metrics = await self._send_request(
                self.MODEL_URL, headers, payload
            )

            # Track token usage
            if self.valves.TRACK_TOKEN_USAGE and cache_metrics:
                self.total_tokens_used["input"] += cache_metrics.get("input_tokens", 0)
                self.total_tokens_used["output"] += cache_metrics.get(
                    "output_tokens", 0
                )
                self.total_tokens_used["cache_read"] += cache_metrics.get(
                    "cache_read_input_tokens", 0
                )
                self.total_tokens_used["cache_creation"] += cache_metrics.get(
                    "cache_creation_input_tokens", 0
                )

                logging.info(
                    f"Token usage - Total: {json.dumps(self.total_tokens_used)}"
                )

            # Handle errors in response
            if (
                isinstance(response_data, dict)
                and response_data.get("format") == "text"
            ):
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": response_data["content"],
                                "done": True,
                            },
                        }
                    )
                return response_data["content"]

            # Handle tool calls
            if any(
                block.get("type") == "tool_use"
                for block in response_data.get("content", [])
            ):
                tool_blocks = [
                    block
                    for block in response_data.get("content", [])
                    if block.get("type") == "tool_use"
                ]
                tool_calls = []
                for block in tool_blocks:
                    tool_use = block["tool_use"]
                    tool_calls.append(
                        {
                            "id": tool_use["id"],
                            "type": "function",
                            "function": {
                                "name": tool_use["name"],
                                "arguments": tool_use["input"],
                            },
                        }
                    )

                if tool_calls:
                    return json.dumps({"type": "tool_calls", "tool_calls": tool_calls})

            # Extract thinking and text content
            thinking_content = None
            if should_enable_thinking:
                thinking_blocks = [
                    block
                    for block in response_data.get("content", [])
                    if block.get("type") == "thinking"
                ]
                if thinking_blocks:
                    thinking_content = thinking_blocks[0].get("thinking", "")

            text_blocks = [
                block
                for block in response_data.get("content", [])
                if block.get("type") == "text"
            ]
            response_text = text_blocks[0]["text"] if text_blocks else ""

            # Combine thinking and response
            if thinking_content:
                response_text = (
                    f"<thinking>{thinking_content}</thinking>\n\n{response_text}"
                )

            return response_text

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"

            logging.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg

    async def _stream_with_ui(
        self, url: str, headers: dict, payload: dict, body: dict, __event_emitter__=None
    ) -> AsyncIterator[str]:
        """Stream responses from the Anthropic API with UI event updates."""
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
                async with session.post(
                    url, headers=headers, json=payload, timeout=timeout
                ) as response:
                    self.request_id = response.headers.get("x-request-id")

                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"Error: HTTP {response.status}: {error_text}"
                        if self.request_id:
                            error_msg += f" (Request ID: {self.request_id})"
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": error_msg, "done": True},
                                }
                            )
                        yield error_msg
                        return

                    is_thinking = False
                    token_count = {"thinking": 0, "text": 0}

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                line_text = line[6:].decode("utf-8").strip()
                                if line_text == "[DONE]":
                                    if is_thinking:
                                        yield "</thinking>\n\n"
                                    break

                                data = json.loads(line_text)

                                if data["type"] == "content_block_start":
                                    block_type = data.get("content_block", {}).get(
                                        "type"
                                    )
                                    if block_type == "thinking":
                                        is_thinking = True
                                        if self.valves.ENABLE_THINKING:
                                            yield "<thinking>"
                                    elif block_type == "text":
                                        is_thinking = False

                                elif data["type"] == "content_block_delta":
                                    delta_type = data.get("delta", {}).get("type")
                                    if (
                                        is_thinking
                                        and delta_type == "thinking_delta"
                                        and self.valves.ENABLE_THINKING
                                    ):
                                        thinking_text = data["delta"].get(
                                            "thinking", ""
                                        )
                                        yield thinking_text
                                        token_count["thinking"] += len(
                                            thinking_text.split()
                                        )
                                    elif not is_thinking and delta_type == "text_delta":
                                        text = data["delta"].get("text", "")
                                        yield text
                                        token_count["text"] += len(text.split())

                                elif data["type"] == "content_block_stop":
                                    if is_thinking:
                                        yield "</thinking>\n\n"
                                        is_thinking = False

                                elif data["type"] == "message_stop":
                                    if self.valves.TRACK_TOKEN_USAGE:
                                        logging.info(
                                            f"Stream tokens - Thinking: {token_count['thinking']}, Text: {token_count['text']}"
                                        )
                                    break

                            except json.JSONDecodeError as e:
                                logging.error(
                                    f"Failed to parse streaming response: {e}"
                                )
                                continue

        except asyncio.TimeoutError:
            error_msg = f"Request timed out after {self.valves.REQUEST_TIMEOUT}s"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            yield error_msg

        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            yield error_msg

    async def _send_request(
        self, url: str, headers: dict, payload: dict
    ) -> Tuple[dict, Optional[dict]]:
        """Send a request to the Anthropic API with enhanced retry logic."""
        retry_count = 0
        base_delay = 1
        max_retries = 5
        retry_status_codes = [429, 500, 502, 503, 504]

        while retry_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
                    async with session.post(
                        url, headers=headers, json=payload, timeout=timeout
                    ) as response:
                        self.request_id = response.headers.get("x-request-id")
                        response_text = await response.text()

                        # Handle rate limiting and server errors
                        if response.status in retry_status_codes:
                            retry_after = int(
                                response.headers.get(
                                    "retry-after", base_delay * (2**retry_count)
                                )
                            )
                            jitter = random.uniform(0, 0.1 * retry_after)
                            retry_time = retry_after + jitter

                            logging.warning(
                                f"Request failed with status {response.status}. "
                                f"Retrying in {retry_time:.2f}s. "
                                f"Retry {retry_count + 1}/{max_retries}"
                            )
                            await asyncio.sleep(retry_time)
                            retry_count += 1
                            continue

                        if response.status != 200:
                            error_msg = f"Error: HTTP {response.status}"
                            try:
                                error_data = json.loads(response_text).get("error", {})
                                error_msg += (
                                    f": {error_data.get('message', response_text)}"
                                )
                                if error_data.get("type"):
                                    error_msg += f" (Type: {error_data.get('type')})"
                            except:
                                error_msg += f": {response_text}"

                            if self.request_id:
                                error_msg += f" (Request ID: {self.request_id})"

                            logging.error(error_msg)
                            return {"content": error_msg, "format": "text"}, None

                        result = json.loads(response_text)
                        usage = result.get("usage", {})
                        cache_metrics = {
                            "cache_creation_input_tokens": usage.get(
                                "cache_creation_input_tokens", 0
                            ),
                            "cache_read_input_tokens": usage.get(
                                "cache_read_input_tokens", 0
                            ),
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                        }

                        logging.info(
                            f"Request successful. Input: {usage.get('input_tokens', 0)}, "
                            f"Output: {usage.get('output_tokens', 0)}, "
                            f"Cache read: {usage.get('cache_read_input_tokens', 0)}"
                        )

                        return result, cache_metrics

            except aiohttp.ClientError as e:
                logging.error(f"Request failed: {str(e)}")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    retry_time = base_delay * (2**retry_count)
                    logging.info(
                        f"Retrying in {retry_time}s. Retry {retry_count}/{max_retries}"
                    )
                    await asyncio.sleep(retry_time)
                    continue
                raise

            except asyncio.TimeoutError:
                logging.error(f"Request timed out after {self.valves.REQUEST_TIMEOUT}s")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    retry_time = base_delay * (2**retry_count)
                    logging.info(
                        f"Retrying after timeout. Retry {retry_count}/{max_retries}"
                    )
                    await asyncio.sleep(retry_time)
                    continue
                raise

        logging.error(f"Max retries ({max_retries}) exceeded")
        return {
            "content": f"Max retries ({max_retries}) exceeded",
            "format": "text",
        }, None
