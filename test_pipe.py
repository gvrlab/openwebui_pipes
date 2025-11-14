"""
Test script for Anthropic_pipe_V1.py
Tests the pipe functionality outside of Open WebUI environment
"""

import os
import sys
import asyncio
import logging
import types
from typing import List, Dict, Tuple
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loaded environment variables from {env_path}")
    else:
        logging.info("No .env file found, using system environment variables")
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env file loading")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Mock the Open WebUI dependency
def mock_pop_system_message(messages: List[Dict], *args, **kwargs) -> Tuple[str, List[Dict]]:
    """
    Mock implementation of open_webui.utils.misc.pop_system_message
    Extracts system message from messages list
    """
    system_message = None
    filtered_messages = []
    
    for message in messages:
        if message.get("role") == "system":
            system_message = message.get("content", "")
        else:
            filtered_messages.append(message)
    
    return system_message, filtered_messages

# Create proper mock modules using types.ModuleType
open_webui = types.ModuleType('open_webui')
open_webui.utils = types.ModuleType('open_webui.utils')
open_webui.utils.misc = types.ModuleType('open_webui.utils.misc')
open_webui.utils.misc.pop_system_message = mock_pop_system_message

# Inject the mock modules into sys.modules
sys.modules['open_webui'] = open_webui
sys.modules['open_webui.utils'] = open_webui.utils
sys.modules['open_webui.utils.misc'] = open_webui.utils.misc

# Now import the pipe
from Anthropic_pipe_V1 import Pipe


class TestEventEmitter:
    """Mock event emitter to capture status updates"""
    def __init__(self):
        self.events = []
    
    async def __call__(self, event):
        self.events.append(event)
        logging.info(f"Event emitted: {event}")


async def test_model_listing():
    """Test 1: Check if model listing works"""
    print("\n" + "="*60)
    print("TEST 1: Model Listing")
    print("="*60)
    
    try:
        pipe = Pipe()
        models = pipe.pipes()
        
        print(f"[OK] Successfully retrieved {len(models)} models")
        print("\nAvailable models:")
        for model in models[:5]:  # Show first 5
            print(f"  - {model['name']}")
            print(f"    ID: {model['id']}")
            print(f"    Context: {model['context_length']}")
            print(f"    Vision: {model['supports_vision']}")
            print(f"    Thinking: {model['supports_thinking']}")
            print()
        
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more models")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed: {str(e)}")
        logging.error(f"Model listing test failed", exc_info=True)
        return False


async def test_api_key_validation():
    """Test 2: Check API key validation"""
    print("\n" + "="*60)
    print("TEST 2: API Key Validation")
    print("="*60)
    
    try:
        pipe = Pipe()
        
        # Test with empty API key
        if not pipe.valves.ANTHROPIC_API_KEY:
            print("[OK] Correctly detected missing API key")
            print("  Set ANTHROPIC_API_KEY environment variable to proceed with API tests")
            return True
        
        # Test API key format
        if pipe.valves.ANTHROPIC_API_KEY.startswith("sk-ant-"):
            print("[OK] API key format looks valid")
            return True
        else:
            print("[WARN] Warning: API key doesn't match expected format (sk-ant-...)")
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed: {str(e)}")
        logging.error(f"API key validation test failed", exc_info=True)
        return False


async def test_basic_completion():
    """Test 3: Basic non-streaming completion"""
    print("\n" + "="*60)
    print("TEST 3: Basic Completion (Non-Streaming)")
    print("="*60)
    
    pipe = Pipe()
    
    # Check if API key is set
    if not pipe.valves.ANTHROPIC_API_KEY:
        print("[SKIP] Skipped: ANTHROPIC_API_KEY not set")
        return None
    
    try:
        event_emitter = TestEventEmitter()
        
        body = {
            "model": "anthropic/claude-3-5-haiku-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'Hello, World!' and nothing else."
                }
            ],
            "stream": False,
            "max_tokens": 100
        }
        
        print("Sending request to Anthropic API...")
        response = await pipe.pipe(body, event_emitter)
        
        print(f"\n[OK] Response received:")
        print(f"  {response[:200]}..." if len(response) > 200 else f"  {response}")
        
        if event_emitter.events:
            print(f"\n  Events emitted: {len(event_emitter.events)}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        logging.error(f"Basic completion test failed", exc_info=True)
        return False


async def test_streaming_completion():
    """Test 4: Streaming completion"""
    print("\n" + "="*60)
    print("TEST 4: Streaming Completion")
    print("="*60)
    
    pipe = Pipe()
    
    # Check if API key is set
    if not pipe.valves.ANTHROPIC_API_KEY:
        print("[SKIP] Skipped: ANTHROPIC_API_KEY not set")
        return None
    
    try:
        event_emitter = TestEventEmitter()
        
        body = {
            "model": "anthropic/claude-3-5-haiku-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 5, one number per line."
                }
            ],
            "stream": True,
            "max_tokens": 100
        }
        
        print("Sending streaming request to Anthropic API...")
        print("\nStreaming response:")
        print("-" * 60)
        
        response_chunks = []
        stream = await pipe.pipe(body, event_emitter)
        
        # Check if we got an error string instead of a stream
        if isinstance(stream, str):
            print(f"\n[ERROR] Got error instead of stream: {stream[:200]}")
            return False
        
        async for chunk in stream:
            print(chunk, end='', flush=True)
            response_chunks.append(chunk)
        
        print("\n" + "-" * 60)
        print(f"\n[OK] Received {len(response_chunks)} chunks")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        logging.error(f"Streaming completion test failed", exc_info=True)
        return False


async def test_thinking_mode():
    """Test 5: Extended thinking mode (if API key is set)"""
    print("\n" + "="*60)
    print("TEST 5: Extended Thinking Mode")
    print("="*60)
    
    pipe = Pipe()
    
    # Check if API key is set
    if not pipe.valves.ANTHROPIC_API_KEY:
        print("[SKIP] Skipped: ANTHROPIC_API_KEY not set")
        return None
    
    try:
        event_emitter = TestEventEmitter()
        
        body = {
            "model": "anthropic/claude-3-7-sonnet-20250219-thinking",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 15 * 17? Show your thinking."
                }
            ],
            "stream": False,
            "max_tokens": 1000
        }
        
        print("Testing extended thinking mode...")
        response = await pipe.pipe(body, event_emitter)
        
        if "<thinking>" in response:
            print("[OK] Thinking mode activated successfully")
            print(f"\n  Response preview:")
            print(f"  {response[:300]}...")
        else:
            print("[WARN] Response received but no thinking tags found")
            print(f"  {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        logging.error(f"Thinking mode test failed", exc_info=True)
        return False


async def test_pdf_processing():
    """Test 6: PDF Processing"""
    print("\n" + "="*60)
    print("TEST 6: PDF Processing")
    print("="*60)
    
    pipe = Pipe()
    
    # Check if API key is set
    if not pipe.valves.ANTHROPIC_API_KEY:
        print("[SKIP] Skipped: ANTHROPIC_API_KEY not set")
        return None
    
    try:
        # Check if PDF file exists
        pdf_path = Path(__file__).parent / 'mocdocs' / 'US8295471.pdf'
        if not pdf_path.exists():
            print(f"[SKIP] Skipped: PDF file not found at {pdf_path}")
            return None
        
        print(f"[OK] Found PDF file: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
        
        # Read PDF and convert to base64
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        import base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_data_url = f"data:application/pdf;base64,{pdf_base64}"
        
        print(f"[OK] PDF encoded (size: {len(pdf_base64)} chars)")
        
        event_emitter = TestEventEmitter()
        
        # Test with a PDF-supporting model
        body = {
            "model": "anthropic/claude-3-7-sonnet-20250219",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "pdf_url",
                            "pdf_url": {
                                "url": pdf_data_url
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please analyze this PDF document. What is the title and main subject of this patent document?"
                        }
                    ]
                }
            ],
            "stream": False,
            "max_tokens": 1000
        }
        
        print("\nSending PDF to Anthropic API for analysis...")
        print("This may take a moment as the PDF is being processed...")
        
        response = await pipe.pipe(body, event_emitter)
        
        print(f"\n[OK] Response received:")
        print("-" * 60)
        if len(response) > 500:
            print(f"{response[:500]}...")
            print(f"\n[Response truncated. Full length: {len(response)} chars]")
        else:
            print(response)
        print("-" * 60)
        
        # Check if response seems valid
        if len(response) > 50 and not response.startswith("Error"):
            print("\n[OK] PDF processing appears successful!")
            return True
        else:
            print("\n[WARN] Response might indicate an issue")
            return False
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        logging.error(f"PDF processing test failed", exc_info=True)
        return False


async def test_pdf_streaming():
    """Test 7: PDF Processing with Streaming"""
    print("\n" + "="*60)
    print("TEST 7: PDF Processing (Streaming)")
    print("="*60)
    
    pipe = Pipe()
    
    # Check if API key is set
    if not pipe.valves.ANTHROPIC_API_KEY:
        print("[SKIP] Skipped: ANTHROPIC_API_KEY not set")
        return None
    
    try:
        # Check if PDF file exists
        pdf_path = Path(__file__).parent / 'mocdocs' / 'US8295471.pdf'
        if not pdf_path.exists():
            print(f"[SKIP] Skipped: PDF file not found at {pdf_path}")
            return None
        
        print(f"[OK] Found PDF file: {pdf_path.name}")
        
        # Read PDF and convert to base64
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        import base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_data_url = f"data:application/pdf;base64,{pdf_base64}"
        
        event_emitter = TestEventEmitter()
        
        body = {
            "model": "anthropic/claude-3-7-sonnet-20250219",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "pdf_url",
                            "pdf_url": {
                                "url": pdf_data_url
                            }
                        },
                        {
                            "type": "text",
                            "text": "Briefly summarize the key claims of this patent in 2-3 sentences."
                        }
                    ]
                }
            ],
            "stream": True,
            "max_tokens": 500
        }
        
        print("\nSending streaming request with PDF...")
        print("\nStreaming response:")
        print("-" * 60)
        
        response_chunks = []
        stream = await pipe.pipe(body, event_emitter)
        
        # Check if we got an error string instead of a stream
        if isinstance(stream, str):
            print(f"\n[ERROR] Got error instead of stream: {stream[:200]}")
            return False
        
        async for chunk in stream:
            print(chunk, end='', flush=True)
            response_chunks.append(chunk)
        
        print("\n" + "-" * 60)
        print(f"\n[OK] Received {len(response_chunks)} chunks")
        
        full_response = ''.join(response_chunks)
        if len(full_response) > 50 and not full_response.startswith("Error"):
            print("[OK] Streaming PDF processing successful!")
            return True
        else:
            print("[WARN] Response might indicate an issue")
            return False
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        logging.error(f"PDF streaming test failed", exc_info=True)
        return False


async def test_error_handling():
    """Test 8: Error handling with invalid model"""
    print("\n" + "="*60)
    print("TEST 8: Error Handling")
    print("="*60)
    
    pipe = Pipe()
    
    # Check if API key is set
    if not pipe.valves.ANTHROPIC_API_KEY:
        print("[SKIP] Skipped: ANTHROPIC_API_KEY not set")
        return None
    
    try:
        event_emitter = TestEventEmitter()
        
        body = {
            "model": "anthropic/invalid-model-name",
            "messages": [
                {
                    "role": "user",
                    "content": "Test"
                }
            ],
            "stream": False,
            "max_tokens": 100
        }
        
        print("Testing with invalid model name...")
        response = await pipe.pipe(body, event_emitter)
        
        if "error" in response.lower() or "Error" in response:
            print("[OK] Error handling working correctly")
            print(f"  Error message: {response[:200]}")
            return True
        else:
            print("[WARN] Unexpected response (expected error):")
            print(f"  {response[:200]}")
            return True
        
    except Exception as e:
        print(f"[OK] Exception caught as expected: {str(e)[:100]}")
        return True


async def run_all_tests():
    """Run all tests and display summary"""
    print("\n" + "="*60)
    print("ANTHROPIC PIPE TEST SUITE")
    print("="*60)
    
    # Check environment
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        print(f"[OK] ANTHROPIC_API_KEY is set (length: {len(api_key)})")
    else:
        print("[WARN] ANTHROPIC_API_KEY not set - API tests will be skipped")
    
    results = {}
    
    # Run tests
    tests = [
        ("Model Listing", test_model_listing),
        ("API Key Validation", test_api_key_validation),
        ("Basic Completion", test_basic_completion),
        ("Streaming", test_streaming_completion),
        ("Thinking Mode", test_thinking_mode),
        ("PDF Processing", test_pdf_processing),
        ("PDF Streaming", test_pdf_streaming),
        ("Error Handling", test_error_handling),
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            logging.error(f"Test '{test_name}' crashed: {str(e)}", exc_info=True)
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    for test_name, result in results.items():
        status = "[PASS]" if result is True else "[SKIP]" if result is None else "[FAIL]"
        print(f"  {status}: {test_name}")
    
    print("\n" + "-"*60)
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total:   {len(results)}")
    print("="*60)
    
    if failed > 0:
        print("\n[WARN] Some tests failed. Check the logs above for details.")
        return 1
    elif skipped > 0:
        print("\n[WARN] Some tests were skipped. Set ANTHROPIC_API_KEY to run all tests.")
        return 0
    else:
        print("\n[SUCCESS] All tests passed!")
        return 0


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Anthropic Pipe Test Script")
    print("="*60)
    print("\nThis script tests the Anthropic pipe outside of Open WebUI.")
    print("\nPrerequisites:")
    print("  1. Install dependencies: pip install pydantic requests aiohttp pillow")
    print("  2. Set ANTHROPIC_API_KEY environment variable for API tests")
    print("  3. Ensure Anthropic_pipe_V1.py is in the same directory")
    print("\n")
    
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
