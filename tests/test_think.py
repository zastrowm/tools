"""
Tests for the think tool using the Agent interface.
"""

from unittest.mock import patch

from strands_tools import think
from strands_tools.think import ThoughtProcessor


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_think_tool_direct():
    """Test direct invocation of the think tool."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "name": "think",
        "input": {
            "thought": "What are the implications of quantum computing on cryptography?",
            "cycle_count": 2,
            "system_prompt": "You are an expert analytical thinker.",
        },
    }

    # Mock use_llm function since we don't want to actually call the LLM
    with patch("strands_tools.think.use_llm") as mock_use_llm:
        # Setup mock response
        mock_use_llm.return_value = {
            "status": "success",
            "content": [{"text": "This is a mock analysis of quantum computing."}],
        }

        # Call the think function directly
        tool_input = tool_use.get("input", {})
        result = think.think(
            thought=tool_input.get("thought"),
            cycle_count=tool_input.get("cycle_count"),
            system_prompt=tool_input.get("system_prompt"),
        )

        # Verify the result has the expected structure
        assert result["status"] == "success"
        assert "Cycle 1/2" in result["content"][0]["text"]
        assert "Cycle 2/2" in result["content"][0]["text"]

        # Verify use_llm was called twice (once for each cycle)
        assert mock_use_llm.call_count == 2


def test_think_one_cycle():
    """Test think tool with a single cycle."""
    tool_use = {
        "toolUseId": "test-one-cycle",
        "name": "think",
        "input": {
            "thought": "Simple thought for one cycle",
            "cycle_count": 1,
            "system_prompt": "You are an expert analytical thinker.",
        },
    }

    with patch("strands_tools.think.use_llm") as mock_use_llm:
        mock_use_llm.return_value = {
            "status": "success",
            "content": [{"text": "Analysis for single cycle."}],
        }

        tool_input = tool_use.get("input", {})
        result = think.think(
            thought=tool_input.get("thought"),
            cycle_count=tool_input.get("cycle_count"),
            system_prompt=tool_input.get("system_prompt"),
        )

        assert result["status"] == "success"
        assert "Cycle 1/1" in result["content"][0]["text"]
        assert mock_use_llm.call_count == 1


def test_think_error_handling():
    """Test error handling in the think tool."""
    tool_use = {
        "toolUseId": "test-error-case",
        "name": "think",
        "input": {
            "thought": "Thought that will cause an error",
            "cycle_count": 2,
            "system_prompt": "You are an expert analytical thinker.",
        },
    }

    with patch("strands_tools.think.use_llm") as mock_use_llm:
        # Make use_llm raise an exception
        mock_use_llm.side_effect = Exception("Test error")

        tool_input = tool_use.get("input", {})
        result = think.think(
            thought=tool_input.get("thought"),
            cycle_count=tool_input.get("cycle_count"),
            system_prompt=tool_input.get("system_prompt"),
        )

        assert result["status"] == "error"
        assert "Error in think tool" in result["content"][0]["text"]


def test_thought_processor():
    """Test the ThoughtProcessor class."""
    processor = ThoughtProcessor({"system_prompt": "System prompt", "messages": []})

    # Test creating thinking prompt
    prompt = processor.create_thinking_prompt("Test thought", 1, 3)
    assert "Test thought" in prompt
    assert "Current Cycle: 1/3" in prompt
    assert "DO NOT call the think tool again" in prompt
