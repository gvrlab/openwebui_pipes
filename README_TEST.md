# Testing the Anthropic Pipe Outside Open WebUI

This guide explains how to test the `Anthropic_pipe_V1.py` pipe function independently of Open WebUI to identify any errors or issues.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Required Python packages:**
   ```bash
   pip install pydantic>=2.0.0 requests>=2.0.0 aiohttp>=3.8.0 pillow>=9.0.0
   ```

3. **Anthropic API Key** (optional for full testing):
   - Get your API key from https://console.anthropic.com/
   - Set it as an environment variable

## Files in This Directory

- `Anthropic_pipe_V1.py` - The main pipe function for Open WebUI
- `test_pipe.py` - Standalone test script
- `README_TEST.md` - This file

## Quick Start

### Basic Test (No API Key Required)

Test the pipe structure and model listing without making actual API calls:

```bash
python test_pipe.py
```

This will run basic tests including:
- ✓ Model listing verification
- ✓ API key validation
- ⊘ API-dependent tests (skipped without key)

### Full Test (With API Key)

To test actual API functionality, set your Anthropic API key first:

**On Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-api-key-here"
python test_pipe.py
```

**On Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-your-api-key-here
python test_pipe.py
```

**On Linux/Mac:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-api-key-here"
python test_pipe.py
```

## What Gets Tested

The test suite includes 6 comprehensive tests:

1. **Model Listing** - Verifies all Claude models are properly registered
2. **API Key Validation** - Checks API key format and presence
3. **Basic Completion** - Tests non-streaming text generation
4. **Streaming Completion** - Tests real-time streaming responses
5. **Extended Thinking Mode** - Tests Claude 3.7/4 thinking capabilities
6. **Error Handling** - Verifies graceful error handling

## Expected Output

### Without API Key:
```
============================================================
ANTHROPIC PIPE TEST SUITE
============================================================
⚠ ANTHROPIC_API_KEY not set - API tests will be skipped

============================================================
TEST 1: Model Listing
============================================================
✓ Successfully retrieved 30 models

============================================================
TEST SUMMARY
============================================================
  ✓ PASS: Model Listing
  ✓ PASS: API Key Validation
  ⊘ SKIP: Basic Completion
  ⊘ SKIP: Streaming
  ⊘ SKIP: Thinking Mode
  ⊘ SKIP: Error Handling

------------------------------------------------------------
  Passed:  2
  Failed:  0
  Skipped: 4
  Total:   6
============================================================
```

### With API Key:
All tests will run and you'll see actual API responses, including:
- Claude's text completions
- Streaming output in real-time
- Extended thinking visualization (for Claude 3.7/4 models)
- Proper error messages for invalid requests

## Troubleshooting

### Import Error: "No module named 'pydantic'"
Install the required dependencies:
```bash
pip install pydantic requests aiohttp pillow
```

### Import Error: "No module named 'Anthropic_pipe_V1'"
Make sure `test_pipe.py` is in the same directory as `Anthropic_pipe_V1.py`:
```bash
ls -l  # Linux/Mac
dir    # Windows
```

### API Error: "Invalid API Key"
Verify your API key:
- Should start with `sk-ant-`
- Check for typos or extra spaces
- Ensure it's active in the Anthropic Console

### Timeout Errors
The default timeout is 300 seconds. If tests timeout:
- Check your internet connection
- Try again (might be temporary API issues)
- Increase timeout in the pipe's Valves configuration

### Type Checking Warnings (Pylance)
You may see type checking warnings in VSCode/Pylance. These are expected and don't affect functionality:
- The mock module creation triggers type warnings
- Tests will run correctly despite these warnings
- These warnings don't occur in the actual Open WebUI environment

## Understanding Test Results

- **✓ PASS** - Test completed successfully
- **✗ FAIL** - Test failed (check error messages in output)
- **⊘ SKIP** - Test skipped (usually due to missing API key)

## Integration with Open WebUI

Once tests pass, the pipe is ready for Open WebUI:

1. Copy `Anthropic_pipe_V1.py` to your Open WebUI pipes directory
2. Set `ANTHROPIC_API_KEY` as an environment variable in your Open WebUI deployment
3. Restart Open WebUI
4. The Anthropic models should appear in your model selector

## Customizing Tests

You can modify `test_pipe.py` to add custom tests:

```python
async def test_custom_scenario():
    """Your custom test"""
    pipe = Pipe()
    
    # Your test code here
    body = {
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "messages": [...],
        "stream": False
    }
    
    response = await pipe.pipe(body)
    print(response)
```

Add your test to the `tests` list in `run_all_tests()` function.

## Additional Resources

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Open WebUI Documentation](https://docs.openwebui.com/)
- [Claude Model Comparison](https://www.anthropic.com/claude)

## Support

If you encounter issues:
1. Check the error messages in the test output
2. Review the logs (set `logging.DEBUG` for more detail)
3. Verify all prerequisites are met
4. Test with a simple API call using curl to rule out API issues

## Notes

- The test script mocks Open WebUI dependencies, so it won't be 100% identical to the actual Open WebUI environment
- Some advanced features (like PDF processing) require additional setup
- Extended thinking mode is only available for Claude 3.7 and 4 models
- API costs apply when running tests with actual API calls
