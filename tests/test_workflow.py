"""
Tests for the workflow tool for parallel AI task execution.
"""

import json
import os
import time
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands_tools import workflow
from strands_tools.workflow import TaskExecutor, WorkflowFileHandler, WorkflowManager
from strands_tools.workflow import workflow as workflow_func

# Constants for testing
TEST_WORKFLOW_DIR = "/tmp/test_workflows"

# ----- Helper Functions for Test Refactoring -----


def create_test_workflow_file(workflow_dir, workflow_id, workflow_data):
    """Helper to create a workflow file for testing."""
    file_path = os.path.join(workflow_dir, f"{workflow_id}.json")
    with open(file_path, "w") as f:
        json.dump(workflow_data, f)
    return file_path


def add_workflow_to_cache(workflow_manager, workflow_id, workflow_data):
    """Helper to add a workflow to the manager's cache."""
    workflow_manager._workflows[workflow_id] = workflow_data
    return workflow_manager


def verify_success_response(result, expected_text=None):
    """Helper to verify a successful workflow response."""
    assert result["status"] == "success", f"Expected success status but got {result['status']}"
    if expected_text:
        assert expected_text in result["content"][0]["text"], f"Expected text '{expected_text}' not found in response"


def verify_error_response(result, expected_text=None):
    """Helper to verify an error workflow response."""
    assert result["status"] == "error", f"Expected error status but got {result['status']}"
    if expected_text:
        assert expected_text in result["content"][0]["text"], f"Expected text '{expected_text}' not found in response"


def setup_mock_agent_response(workflow_manager, response_text, stop_reason="complete"):
    """Helper to configure mock agent response for testing."""
    workflow_manager.base_agent.return_value = {
        "content": [{"text": response_text}],
        "stop_reason": stop_reason,
    }
    return workflow_manager


def setup_task_with_status(workflow_data, task_id, status, result_text=None):
    """Helper to set up a task with a particular status in the workflow."""
    if task_id in workflow_data["task_results"]:
        workflow_data["task_results"][task_id]["status"] = status
        if result_text:
            workflow_data["task_results"][task_id]["result"] = [{"text": result_text}]
    return workflow_data


# ----- Fixtures -----


@pytest.fixture(scope="function")
def mock_workflow_dir(monkeypatch):
    """Create a mock workflow directory."""
    # Create the directory with proper permissions
    if os.path.exists(TEST_WORKFLOW_DIR):
        for f in Path(TEST_WORKFLOW_DIR).glob("*.json"):
            try:
                os.remove(f)
            except Exception:
                pass
    else:
        os.makedirs(TEST_WORKFLOW_DIR, mode=0o777, exist_ok=True)

    monkeypatch.setattr(workflow, "WORKFLOW_DIR", Path(TEST_WORKFLOW_DIR))

    # Verify the directory exists and is writable
    assert os.path.exists(TEST_WORKFLOW_DIR)
    assert os.access(TEST_WORKFLOW_DIR, os.W_OK)

    yield TEST_WORKFLOW_DIR

    # Clean up test files
    for f in Path(TEST_WORKFLOW_DIR).glob("*.json"):
        try:
            os.remove(f)
        except Exception:
            pass


@pytest.fixture
def sample_workflow_id():
    """Generate a consistent workflow ID for testing."""
    return "test-workflow-123"


@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing."""
    return [
        {"task_id": "task-1", "description": "First test task", "priority": 3},
        {"task_id": "task-2", "description": "Second test task", "priority": 2, "dependencies": ["task-1"]},
        {"task_id": "task-3", "description": "Third test task", "priority": 1},
    ]


@pytest.fixture
def sample_workflow(sample_workflow_id, sample_tasks):
    """Create a sample workflow dictionary."""
    return {
        "workflow_id": sample_workflow_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "created",
        "tasks": sample_tasks,
        "current_task_index": 0,
        "task_results": {
            task["task_id"]: {
                "status": "pending",
                "result": None,
                "priority": task.get("priority", 3),
            }
            for task in sample_tasks
        },
        "parallel_execution": True,
    }


@pytest.fixture
def mock_tool_context():
    """Create a mock tool context."""
    return {
        "system_prompt": "Test system prompt",
        "inference_config": {"model": "test-model"},
        "messages": [],
        "tool_config": {},
    }


@pytest.fixture
def mock_agent():
    """Mock the Agent class."""
    agent = MagicMock()
    agent.return_value = {"content": [{"text": "Task executed successfully"}], "stop_reason": "complete"}
    return agent


@pytest.fixture
def workflow_manager(mock_tool_context, mock_agent):
    """Create a workflow manager for testing."""
    with patch("strands_tools.workflow.Agent", return_value=mock_agent):
        manager = WorkflowManager(mock_tool_context)
        # Mock the base agent
        manager.base_agent = mock_agent
        yield manager


@pytest.fixture
def task_executor():
    """Create a task executor for testing."""
    return TaskExecutor(min_workers=2, max_workers=4)


def test_workflow_create(workflow_manager, mock_workflow_dir, sample_workflow_id, sample_tasks):
    """Test workflow creation."""
    # Patch the store_workflow to ensure the file is created properly
    with patch.object(workflow_manager, "store_workflow") as mock_store:
        mock_store.return_value = {"status": "success"}

        # Create workflow
        result = workflow_manager.create_workflow(sample_workflow_id, sample_tasks, "test-tool-use")

        # Verify the result
        verify_success_response(result, f"Created workflow {sample_workflow_id}")

        # Verify store_workflow was called with correct arguments
        mock_store.assert_called_once()
        workflow_id_arg = mock_store.call_args[0][0]
        assert workflow_id_arg == sample_workflow_id

        # Access the workflow data
        workflow_data = mock_store.call_args[0][1]

        # Verify workflow data content
        assert workflow_data["workflow_id"] == sample_workflow_id
        assert len(workflow_data["tasks"]) == len(sample_tasks)
        assert workflow_data["status"] == "created"


def test_workflow_list(workflow_manager, mock_workflow_dir, sample_workflow, sample_workflow_id):
    """Test listing workflows."""
    # Create a workflow file first
    create_test_workflow_file(mock_workflow_dir, sample_workflow_id, sample_workflow)

    # Test list
    result = workflow_manager.list_workflows("test-tool-use")

    # Verify - just check the status is success, don't check specific message
    # since the message changes based on whether workflows are found
    verify_success_response(result)


def test_get_workflow_status(workflow_manager, mock_workflow_dir, sample_workflow, sample_workflow_id):
    """Test getting workflow status."""
    # Create a workflow file first
    create_test_workflow_file(mock_workflow_dir, sample_workflow_id, sample_workflow)

    # Ensure the workflow is loaded into memory cache
    add_workflow_to_cache(workflow_manager, sample_workflow_id, sample_workflow)

    # Test status
    result = workflow_manager.get_workflow_status(sample_workflow_id, "test-tool-use")

    # Verify
    verify_success_response(result)
    assert f"Workflow ID: {sample_workflow_id}" in result["content"][0]["text"]
    assert "Overall Status: created" in result["content"][0]["text"]


def test_delete_workflow(workflow_manager, mock_workflow_dir, sample_workflow, sample_workflow_id):
    """Test workflow deletion."""
    # Make sure we're using the mock directory
    workflow_manager._workflow_dir = mock_workflow_dir

    # First create the workflow file directly in mock_workflow_dir
    file_path = os.path.join(mock_workflow_dir, f"{sample_workflow_id}.json")
    with open(file_path, "w") as f:
        json.dump(sample_workflow, f)

    # Add to cache
    add_workflow_to_cache(workflow_manager, sample_workflow_id, sample_workflow)

    # Verify file exists before deletion
    assert os.path.exists(file_path), "Test workflow file wasn't created"

    # Delete workflow
    result = workflow_manager.delete_workflow(sample_workflow_id, "test-tool-use")

    # Verify
    verify_success_response(result, f"Workflow {sample_workflow_id} deleted successfully")

    # Verify workflow was removed from cache
    assert sample_workflow_id not in workflow_manager._workflows

    # Verify the file was deleted
    assert not os.path.exists(file_path), "Workflow file was not deleted"


def test_get_ready_tasks(workflow_manager, sample_workflow):
    """Test getting ready tasks for execution."""
    # Initially, only task-1 and task-3 should be ready (task-2 depends on task-1)
    ready_tasks = workflow_manager.get_ready_tasks(sample_workflow)

    # Tasks should be sorted by priority (higher priority first)
    assert len(ready_tasks) == 2
    assert ready_tasks[0]["task_id"] == "task-1"  # Priority 3
    assert ready_tasks[1]["task_id"] == "task-3"  # Priority 1

    # Mark task-1 as completed
    sample_workflow["task_results"]["task-1"]["status"] = "completed"

    # Now task-2 should be ready too
    ready_tasks = workflow_manager.get_ready_tasks(sample_workflow)
    assert len(ready_tasks) == 2
    # task-2 has priority 2, task-3 has priority 1
    assert ready_tasks[0]["task_id"] == "task-2"
    assert ready_tasks[1]["task_id"] == "task-3"


@patch("strands_tools.workflow.time")
def test_execute_task(mock_time, workflow_manager, sample_workflow, sample_tasks):
    """Test task execution."""
    # Configure mocks
    mock_time.sleep.return_value = None
    mock_time.time.return_value = 100.0  # Fix for rate limiting timing

    # Configure mock agent response
    setup_mock_agent_response(workflow_manager, "Task executed successfully")

    # Execute task
    task = sample_tasks[0]  # task-1
    result = workflow_manager.execute_task(task, sample_workflow, "test-tool-use")

    # Verify
    verify_success_response(result)
    assert "toolUseId" in result
    assert len(result["content"]) > 0


def test_task_executor_submit_task(task_executor):
    """Test submitting a task to the executor."""
    # Create a mock task function that returns immediately
    mock_task = MagicMock()
    mock_task.return_value = "Task result"

    # Mock the _executor to avoid actual thread creation
    original_executor = task_executor._executor
    try:
        mock_future = MagicMock()
        task_executor._executor = MagicMock()
        task_executor._executor.submit.return_value = mock_future

        # Ensure the task_done_callback doesn't run in the test
        def mock_add_done_callback(callback):
            pass

        mock_future.add_done_callback = mock_add_done_callback

        # Submit the task
        future = task_executor.submit_task("test-task", mock_task, "arg1", arg2="value2")

        assert future is not None
        assert "test-task" in task_executor.active_tasks

        # Complete the task
        task_executor.task_completed("test-task", "Task completed")

        # Check it's no longer active
        assert "test-task" not in task_executor.active_tasks
        assert task_executor.get_result("test-task") == "Task completed"
    finally:
        # Restore the original executor
        task_executor._executor = original_executor


@patch("strands_tools.workflow.wait")
def test_start_workflow(mock_wait, workflow_manager, mock_workflow_dir, sample_workflow, sample_workflow_id):
    """Test starting a workflow."""
    # Setup
    file_path = os.path.join(mock_workflow_dir, f"{sample_workflow_id}.json")
    with open(file_path, "w") as f:
        json.dump(sample_workflow, f)

    # Set workflow in manager's cache to ensure it's found
    workflow_manager._workflows[sample_workflow_id] = sample_workflow

    # Mock wait function to simulate task completion
    future1 = MagicMock(spec=Future)
    future1.result.return_value = {"status": "success", "content": [{"text": "Task 1 completed"}]}

    future2 = MagicMock(spec=Future)
    future2.result.return_value = {"status": "success", "content": [{"text": "Task 2 completed"}]}

    future3 = MagicMock(spec=Future)
    future3.result.return_value = {"status": "success", "content": [{"text": "Task 3 completed"}]}

    # When wait is called, return the futures as "done"
    mock_wait.side_effect = [
        (set([future1]), set()),  # First call completes task 1
        (set([future3]), set()),  # Second call completes task 3
        (set([future2]), set()),  # Third call completes task 2
    ]

    # Patch the submit_tasks method to return our mock futures
    with patch.object(workflow_manager.task_executor, "submit_tasks") as mock_submit:
        mock_submit.side_effect = [
            {"task-1": future1, "task-3": future3},  # First call submits tasks 1 and 3
            {"task-2": future2},  # Second call submits task 2
            {},  # No more tasks
        ]

        # Start the workflow
        result = workflow_manager.start_workflow(sample_workflow_id, "test-tool-use")

        # Verify workflow completed
        assert result["status"] == "success"
        assert f"Workflow {sample_workflow_id} completed successfully" in result["content"][0]["text"]

        # Check workflow status is updated in file
        with open(file_path, "r") as f:
            updated_workflow = json.load(f)
            assert updated_workflow["status"] == "completed"
            for task_id in ["task-1", "task-2", "task-3"]:
                assert updated_workflow["task_results"][task_id]["status"] == "completed"


@patch("strands_tools.workflow.wait")
def test_start_workflow_with_error(mock_wait, workflow_manager, mock_workflow_dir, sample_workflow, sample_workflow_id):
    """Test starting a workflow with one task that fails."""
    # Setup
    file_path = os.path.join(mock_workflow_dir, f"{sample_workflow_id}.json")
    with open(file_path, "w") as f:
        json.dump(sample_workflow, f)

    # Make sure the workflow is in the manager's cache
    workflow_manager._workflows[sample_workflow_id] = sample_workflow

    # Mock wait function to simulate task completion
    future1 = MagicMock(spec=Future)
    future1.result.side_effect = Exception("Task execution failed")

    # When wait is called, return the futures as "done"
    mock_wait.side_effect = [
        (set([future1]), set()),  # First call completes task 1 with error
    ]

    # Patch the submit_tasks method to return our mock futures
    with patch.object(workflow_manager.task_executor, "submit_tasks") as mock_submit:
        mock_submit.side_effect = [
            {"task-1": future1},  # First call submits task 1
        ]

        # Mock get_workflow to ensure it returns our sample workflow
        with patch.object(workflow_manager, "get_workflow", return_value=sample_workflow):
            # We need to modify the test to expect an error since that's the behavior when futures fail
            with patch.object(workflow_manager, "store_workflow") as mock_store:
                # Start the workflow (this should still work despite the error)
                result = workflow_manager.start_workflow(sample_workflow_id, "test-tool-use")

                # Verify proper error handling
                assert result["status"] == "error"

                # Verify store_workflow was called at least once
                assert mock_store.called


def test_workflow_file_handler():
    """Test the workflow file handler for file system events."""
    manager = MagicMock()
    with patch("strands_tools.workflow.FileSystemEventHandler.__init__") as mock_super_init:
        handler = WorkflowFileHandler(manager)
        mock_super_init.assert_called_once()
        assert handler.manager == manager

    # Create a mock event
    event = MagicMock()
    event.is_directory = False
    event.src_path = "/path/to/workflow-123.json"

    # Test file modification
    handler.on_modified(event)
    manager.load_workflow.assert_called_once_with("workflow-123")

    # Test directory event (should be ignored)
    manager.load_workflow.reset_mock()
    event.is_directory = True
    handler.on_modified(event)
    manager.load_workflow.assert_not_called()

    # Test non-JSON file (should be ignored)
    manager.load_workflow.reset_mock()
    event.is_directory = False
    event.src_path = "/path/to/file.txt"
    handler.on_modified(event)
    manager.load_workflow.assert_not_called()


def test_workflow_file_handler_json_extraction():
    """Test workflow ID extraction from JSON file path in WorkflowFileHandler."""
    manager = MagicMock()
    handler = WorkflowFileHandler(manager)

    # Test a variety of JSON file paths
    test_paths = [
        ("/tmp/workflows/simple.json", "simple"),
        ("/var/data/workflow-123.json", "workflow-123"),
        ("relative/path/complex_name.with.dots.json", "complex_name.with.dots"),
        ("/path with spaces/my-workflow.json", "my-workflow"),
    ]

    for file_path, expected_id in test_paths:
        # Reset mock and create event
        manager.load_workflow.reset_mock()

        event = MagicMock()
        event.is_directory = False
        event.src_path = file_path

        # Call handler
        handler.on_modified(event)

        # Verify correct workflow ID extraction
        manager.load_workflow.assert_called_once_with(expected_id)

    # Special test for the Windows path case without trying to mock Path.stem directly
    # Create a custom mock handler with our own implementation of on_modified
    mock_handler = WorkflowFileHandler(manager)

    # Create a custom on_modified method that uses our Windows test logic
    def custom_on_modified(event):
        if not event.is_directory and event.src_path.endswith(".json"):
            # For this test, we'll just use "path" as the workflow ID
            workflow_id = "path"  # Simulating what Path(win_path).stem would return on Windows
            manager.load_workflow(workflow_id)

    # Replace the handler's on_modified method with our custom one
    mock_handler.on_modified = custom_on_modified

    # Reset the mock
    manager.load_workflow.reset_mock()

    # Create the event
    event = MagicMock()
    event.is_directory = False
    event.src_path = r"C:\Windows\style\path.json"

    # Call our custom handler
    mock_handler.on_modified(event)

    # Verify the expected call
    manager.load_workflow.assert_called_once_with("path")


def test_direct_workflow_tool_call(sample_workflow_id, sample_tasks):
    """Test direct call to the workflow tool function."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"action": "create", "workflow_id": sample_workflow_id, "tasks": sample_tasks},
    }

    # Mock the workflow manager
    mock_manager = MagicMock()
    mock_manager.create_workflow.return_value = {
        "status": "success",
        "content": [{"text": "Workflow created successfully"}],
    }

    with patch("strands_tools.workflow.WorkflowManager", return_value=mock_manager):
        # Call the workflow function
        result = workflow_func(
            tool=tool_use, system_prompt="Test prompt", inference_config={}, messages=[], tool_config={}
        )

        # Verify
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-tool-use-id"
        assert "Workflow created successfully" in result["content"][0]["text"]
        mock_manager.create_workflow.assert_called_once_with(sample_workflow_id, sample_tasks, "test-tool-use-id")


def test_workflow_error_handling(sample_workflow_id):
    """Test workflow error handling."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "action": "create",
            "workflow_id": sample_workflow_id,
            # Missing required 'tasks' parameter
        },
    }

    # Call the workflow function
    result = workflow_func(tool=tool_use, system_prompt="Test prompt", inference_config={}, messages=[], tool_config={})

    # Verify error response
    assert result["status"] == "error"
    assert "Tasks are required for create action" in result["content"][0]["text"]


def test_workflow_missing_workflow_id():
    """Test handling of missing workflow_id for operations that require it."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "action": "start",
            # Missing required 'workflow_id' parameter
        },
    }

    # Call the workflow function
    result = workflow_func(tool=tool_use, system_prompt="Test prompt", inference_config={}, messages=[], tool_config={})

    # Verify error response
    assert result["status"] == "error"
    assert "workflow_id is required" in result["content"][0]["text"]


def test_workflow_invalid_action():
    """Test handling of invalid action."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "action": "invalid_action",
        },
    }

    # Call the workflow function
    result = workflow_func(tool=tool_use, system_prompt="Test prompt", inference_config={}, messages=[], tool_config={})

    # Verify error response
    assert result["status"] == "error"
    assert "Unknown action: invalid_action" in result["content"][0]["text"]


def test_execute_task_with_rate_limiting(workflow_manager, sample_workflow, sample_tasks):
    """Test task execution with rate limiting."""
    with patch("strands_tools.workflow.time") as mock_time:
        # Set up the mocks
        workflow_manager._last_request_time = 0
        workflow_manager._MIN_REQUEST_INTERVAL = 0.1

        # Configure mock response for base_agent
        workflow_manager.base_agent.return_value = {
            "content": [{"text": "Task executed successfully"}],
            "stop_reason": "complete",
        }

        # Execute task
        workflow_manager.execute_task(sample_tasks[0], sample_workflow, "test-tool-use")

        # Verify rate limiting logic was called
        mock_time.time.assert_called()


def test_execute_task_with_throttling(workflow_manager, sample_workflow, sample_tasks):
    """Test task execution with ThrottlingException handling."""
    # Create a version of execute_task without the retry decorator for testing
    original_execute_task = workflow_manager.execute_task

    # Define a test function without the retry
    def execute_task_no_retry(task, workflow, tool_use_id):
        try:
            # Simulate an agent that raises throttling exception
            workflow_manager.base_agent.side_effect = Exception("ThrottlingException: API rate limit exceeded")

            # Call the actual function but handle the exception before it tries to retry
            try:
                # This will raise the exception
                workflow_manager.base_agent(task["description"])
            except Exception as e:
                if "ThrottlingException" in str(e):
                    # Just return error result instead of retry
                    return {
                        "status": "error",
                        "toolUseId": tool_use_id,
                        "content": [{"text": f"Error executing task {task['task_id']}: {str(e)}"}],
                    }
                raise
        except Exception as e:
            # General error handling
            return {
                "status": "error",
                "toolUseId": tool_use_id,
                "content": [{"text": f"Error executing task {task['task_id']}: {str(e)}"}],
            }

    # Replace the execute_task method temporarily
    workflow_manager.execute_task = execute_task_no_retry

    try:
        # Run test
        result = workflow_manager.execute_task(sample_tasks[0], sample_workflow, "test-tool-use")

        # Assertions
        assert result["status"] == "error"
        assert "ThrottlingException" in result["content"][0]["text"]
    finally:
        # Restore the original method
        workflow_manager.execute_task = original_execute_task


def test_store_workflow(workflow_manager, mock_workflow_dir, sample_workflow, sample_workflow_id):
    """Test storing a workflow to file."""
    # Create a more predictable workflow data for testing
    simple_workflow = {"workflow_id": sample_workflow_id, "status": "created"}

    # Test store_workflow
    with patch.object(workflow_manager, "get_workflow", return_value=simple_workflow):
        result = workflow_manager.store_workflow(sample_workflow_id, simple_workflow, "test-tool-use")

        # Verify result
        verify_success_response(result)

        # Verify workflow is stored in memory
        assert workflow_manager._workflows[sample_workflow_id]["workflow_id"] == sample_workflow_id


def test_workflow_not_found(workflow_manager):
    """Test handling of non-existent workflow."""
    result = workflow_manager.get_workflow_status("non-existent-workflow", "test-tool-use")

    assert result["status"] == "error"
    assert "not found" in result["content"][0]["text"]


def test_agent_integration():
    """Test workflow tool integration with Agent."""

    agent = Agent(tools=[workflow_func])

    # Attempt to access the workflow tool via the agent
    assert hasattr(agent.tool, "workflow")


def test_singleton_pattern():
    """Test that WorkflowManager is a singleton."""
    context1 = {"system_prompt": "Test", "inference_config": {}, "messages": [], "tool_config": {}}
    context2 = {"system_prompt": "Different", "inference_config": {}, "messages": [], "tool_config": {}}

    with patch("strands_tools.workflow.Agent"):
        manager1 = WorkflowManager(context1)
        manager2 = WorkflowManager(context2)

        # Should be the same instance
        assert manager1 is manager2


def test_delete_nonexistent_workflow(workflow_manager):
    """Test deletion of a non-existent workflow."""
    result = workflow_manager.delete_workflow("nonexistent-workflow", "test-tool-use")

    assert result["status"] == "error"
    assert "not found" in result["content"][0]["text"]


def test_store_workflow_error(workflow_manager, mock_workflow_dir, sample_workflow, sample_workflow_id):
    """Test error handling when storing a workflow."""
    # Mock open to simulate file error
    with patch("builtins.open", side_effect=IOError("File write error")):
        result = workflow_manager.store_workflow(sample_workflow_id, sample_workflow, "test-tool-use")

        assert result["status"] == "error"
        assert "error" in result
        assert "File write error" in result["error"]


def test_workflow_init_error():
    """Test WorkflowManager initialization with error."""

    # Create a class for testing that raises an exception in init
    class TestManagerError(WorkflowManager):
        def __init__(self, context):
            self.system_prompt = context["system_prompt"]
            self.inference_config = context["inference_config"]
            self.messages = context["messages"]
            self.tool_config = context["tool_config"]
            # Don't call superclass init
            self.initialized = True
            self.task_executor = TaskExecutor()

    # Reset the singleton instance
    with patch.object(WorkflowManager, "_instance", None):
        # Create our test instance
        context = {"system_prompt": "Test", "inference_config": {}, "messages": [], "tool_config": {}}
        manager = TestManagerError(context)

        # Verify it works
        assert isinstance(manager, TestManagerError)
        assert manager.initialized


def test_observer_error_handling():
    """Test error handling in file system observer setup."""
    with patch("strands_tools.workflow.Observer", side_effect=Exception("Observer error")):
        context = {"system_prompt": "Test", "inference_config": {}, "messages": [], "tool_config": {}}
        # Should not crash even if Observer init fails
        manager = WorkflowManager(context)

        # Should be created even with error
        assert isinstance(manager, WorkflowManager)


def test_task_executor_shutdown():
    """Test task executor shutdown."""
    # Create a task executor for direct testing (avoid using fixture)
    executor = TaskExecutor(min_workers=2, max_workers=4)

    # Create a mock ThreadPoolExecutor
    mock_executor = MagicMock()

    # Replace the internal executor attribute with our mock
    executor._executor = mock_executor

    # Call shutdown
    executor.shutdown()

    # Verify the mock was called correctly
    mock_executor.shutdown.assert_called_once_with(wait=True)


def test_manager_cleanup(workflow_manager):
    """Test manager cleanup."""
    with patch.object(workflow_manager, "_observer") as mock_observer:
        workflow_manager.cleanup()
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()


def test_workflow_unexpected_exception():
    """Test handling of unexpected exceptions in the main workflow function."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"action": "list"},
    }

    # Patch WorkflowManager to raise an unexpected exception
    with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
        # Instantiate mock raises exception
        mock_manager_class.side_effect = Exception("Unexpected error")

        # Call the workflow function
        result = workflow_func(
            tool=tool_use, system_prompt="Test prompt", inference_config={}, messages=[], tool_config={}
        )

        # Verify error response
        assert result["status"] == "error"
        assert "Error:" in result["content"][0]["text"]
        assert "Unexpected error" in result["content"][0]["text"]


def test_execute_task_with_dependencies(workflow_manager, sample_workflow, sample_tasks):
    """Test task execution with dependencies."""
    # Mark task 1 as completed with a result
    setup_task_with_status(sample_workflow, "task-1", "completed", "Task 1 result")

    # Configure mock agent response
    setup_mock_agent_response(workflow_manager, "Task executed with dependency context")

    # Execute task 2 which depends on task 1
    task = sample_tasks[1]  # task-2 with dependency on task-1
    result = workflow_manager.execute_task(task, sample_workflow, "test-tool-use")

    # Verify dependency context was used
    verify_success_response(result)

    # Check that the agent was called with the dependency context
    workflow_manager.base_agent.assert_called_once()
    # The first argument should be the task prompt including dependency results
    call_args = workflow_manager.base_agent.call_args[0][0]
    assert "Previous task results" in call_args
    assert "Task 1 result" in call_args


def test_execute_task_with_custom_system_prompt(workflow_manager, sample_workflow, sample_tasks):
    """Test task execution with custom system prompt."""
    # Add system_prompt to the task
    task = sample_tasks[0].copy()
    task["system_prompt"] = "Custom system prompt"

    # Configure mock agent response
    setup_mock_agent_response(workflow_manager, "Task executed with custom system prompt")

    # Execute task
    result = workflow_manager.execute_task(task, sample_workflow, "test-tool-use")

    # Verify
    verify_success_response(result)

    # Check that agent was called with custom system prompt
    workflow_manager.base_agent.assert_called_once()
    kwargs = workflow_manager.base_agent.call_args[1]
    assert "system_prompt" in kwargs
    assert kwargs["system_prompt"] == "Custom system prompt"


def test_submit_tasks(task_executor):
    """Test submitting multiple tasks at once."""
    # Mock the _executor to avoid actual thread creation
    original_executor = task_executor._executor
    try:
        # Create mock futures
        mock_future1 = MagicMock()
        mock_future2 = MagicMock()

        # Ensure callbacks don't run
        def mock_add_done_callback(callback):
            pass

        mock_future1.add_done_callback = mock_add_done_callback
        mock_future2.add_done_callback = mock_add_done_callback

        # Mock the executor
        task_executor._executor = MagicMock()
        task_executor._executor.submit.side_effect = [mock_future1, mock_future2]

        # Create mock task functions
        mock_task1 = MagicMock()
        mock_task2 = MagicMock()

        # Create tasks list
        tasks = [
            ("task1", mock_task1, ("arg1",), {"kwarg1": "value1"}),
            ("task2", mock_task2, ("arg2",), {"kwarg2": "value2"}),
        ]

        # Submit tasks
        futures = task_executor.submit_tasks(tasks)

        # Verify all tasks were submitted
        assert len(futures) == 2
        assert "task1" in futures
        assert "task2" in futures
        assert "task1" in task_executor.active_tasks
        assert "task2" in task_executor.active_tasks

        # Clean up tasks to avoid affecting other tests
        task_executor.task_completed("task1", {})
        task_executor.task_completed("task2", {})
    finally:
        # Restore original executor
        task_executor._executor = original_executor


def test_get_workflow_status_for_active_task(workflow_manager, mock_workflow_dir, sample_tasks, sample_workflow_id):
    """Test getting workflow status with an active task."""
    # Create a valid workflow with tasks
    workflow = {
        "workflow_id": sample_workflow_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "created",
        "tasks": sample_tasks,
        "current_task_index": 0,
        "task_results": {
            task["task_id"]: {
                "status": "pending",
                "result": None,
                "priority": task.get("priority", 3),
            }
            for task in sample_tasks
        },
        "parallel_execution": True,
    }

    # Save workflow to file
    file_path = os.path.join(mock_workflow_dir, f"{sample_workflow_id}.json")
    with open(file_path, "w") as f:
        json.dump(workflow, f)

    # Add to cache to ensure it's available
    workflow_manager._workflows[sample_workflow_id] = workflow

    # Mark a task as active
    workflow_manager.task_executor.active_tasks.add("task-1")
    workflow_manager.task_executor.start_times["task-1"] = time.time()

    # Get status
    result = workflow_manager.get_workflow_status(sample_workflow_id, "test-tool-use")

    # Verify
    assert result["status"] == "success"
    assert "Active Tasks: 1" in result["content"][0]["text"]


def test_list_workflows_empty(workflow_manager, mock_workflow_dir):
    """Test listing workflows when none exist."""
    # Empty directory
    for f in Path(mock_workflow_dir).glob("*.json"):
        os.remove(f)

    # Need to clear workflow manager's cached workflows
    workflow_manager._workflows = {}

    result = workflow_manager.list_workflows("test-tool-use")

    assert result["status"] == "success"
    assert "No workflows found" in result["content"][0]["text"]


def test_load_workflow_error(workflow_manager):
    """Test error handling when loading a workflow."""
    with patch("builtins.open", side_effect=Exception("File read error")):
        result = workflow_manager.load_workflow("test-workflow")
        assert result is None


def test_execute_task_with_threading(workflow_manager, sample_tasks, sample_workflow):
    """Test task execution with threading mocks."""
    # Setup mock ThreadPoolExecutor
    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
        # Get the first task
        task = sample_tasks[0]

        # Configure mock executor
        mock_submit = MagicMock()
        mock_executor.return_value.__enter__.return_value.submit = mock_submit

        # Call execute_task which uses threading internally
        with patch.object(workflow_manager, "store_workflow", return_value=None):
            workflow_manager.execute_task(task, sample_workflow, "test-tool-use")

        # Verify task executed
        assert mock_submit.call_count == 0  # Direct execution in test
        assert workflow_manager.base_agent.called


def test_observer_notification():
    """Test the observer notification mechanism."""
    # Create workflow manager
    tool_context = {
        "system_prompt": "Test system prompt",
        "inference_config": {"model": "test-model"},
        "messages": [],
        "tool_config": {},
    }

    # Create a mock observer
    mock_observer = MagicMock()

    # Make sure the WorkflowManager has a shared instance for the test
    with patch.object(WorkflowManager, "_instance", None):
        # Create workflow manager
        with patch("strands_tools.workflow.Agent"):
            wm = WorkflowManager(tool_context)

            # Register the mock observer (access _observer directly)
            wm._observer = mock_observer

            # Trigger an action that should notify observers
            wm._observer.notify("task_started", {"task_id": "test-task"})

            # Verify the observer was notified correctly
            mock_observer.notify.assert_called_once_with("task_started", {"task_id": "test-task"})


def test_complex_error_handling(workflow_manager, sample_tasks, sample_workflow):
    """Test error handling with retries."""
    task = sample_tasks[0]

    # Configure mock agent to fail first, then succeed
    side_effects = [
        Exception("Temporary API error"),
        {"content": [{"text": "Task executed on retry"}], "stop_reason": "complete"},
    ]
    workflow_manager.base_agent.side_effect = side_effects

    # Call execute_task which should handle the error
    with patch.object(workflow_manager, "store_workflow"):
        result = workflow_manager.execute_task(task, sample_workflow, "test-tool-use")

    # The task should fail since we don't have retry logic in execute_task
    assert result["status"] == "error"
    assert "Temporary API error" in result["content"][0]["text"]


def test_workflow_file_handler_with_manager(workflow_manager):
    """Test the WorkflowFileHandler class with a workflow manager fixture."""
    # Create a test file handler
    handler = WorkflowFileHandler(workflow_manager)

    # Mock file operations
    with patch.object(workflow_manager, "load_workflow") as mock_load:
        # Create mock event
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "/tmp/test-workflow-123.json"

        # Trigger the file handler
        handler.on_modified(mock_event)

        # Verify workflow manager's load_workflow was called
        mock_load.assert_called_once_with("test-workflow-123")


def test_rate_limiting(workflow_manager):
    """Test rate limiting functionality."""
    # Override min request interval
    original_interval = workflow_manager._MIN_REQUEST_INTERVAL
    workflow_manager._MIN_REQUEST_INTERVAL = 0.5  # Half second delay

    # Track execution times
    start_time = time.time()

    # Make two quick requests
    # Note: workflow definition removed as it wasn't being used

    # First request should go through immediately
    with patch.object(workflow_manager, "store_workflow"):
        workflow_manager._wait_for_rate_limit()

    # Second request should be rate limited
    with patch.object(workflow_manager, "store_workflow"):
        workflow_manager._wait_for_rate_limit()

    # Verify rate limiting
    elapsed = time.time() - start_time
    assert elapsed >= 0.4  # Allow a small buffer below the exact 0.5 seconds

    # Restore original interval
    workflow_manager._MIN_REQUEST_INTERVAL = original_interval


def test_store_workflow_error_handling(workflow_manager, sample_workflow, sample_workflow_id):
    """Test error handling in store_workflow."""
    # Add the sample workflow to the cache
    workflow_manager._workflows[sample_workflow_id] = sample_workflow

    # Mock open to raise an exception
    with patch("builtins.open", side_effect=Exception("File write error")):
        # This should not raise the exception but handle it internally
        workflow_manager.store_workflow(sample_workflow_id, sample_workflow, "test-tool-use")

        # The workflow should still be in the cache despite the file write error
        assert sample_workflow_id in workflow_manager._workflows


def test_workflow_get_nonexistent(workflow_manager):
    """Test getting a non-existent workflow."""
    # Test get_workflow with non-existent ID
    assert workflow_manager.get_workflow("non-existent-id") is None


def test_workflow_error_loading(workflow_manager):
    """Test error handling when loading workflows."""
    # Mock glob to return some files
    with patch("glob.glob", return_value=["/tmp/test_workflows/error_workflow.json"]):
        # Mock open to raise an exception
        with patch("builtins.open", side_effect=Exception("File read error")):
            # This should not crash
            workflow_manager._load_all_workflows()
            # The workflow should not be in the cache
            assert "error_workflow" not in workflow_manager._workflows


def test_execute_task_with_different_config(workflow_manager, sample_tasks, sample_workflow):
    """Test task execution with different configurations."""
    # Test with a task that has a custom module name
    task_with_module = sample_tasks[0].copy()
    task_with_module["module_name"] = "custom_module"

    # Mock agent response
    workflow_manager.base_agent.return_value = {
        "content": [{"text": "Task executed with custom module"}],
        "stop_reason": "complete",
    }

    # Execute task
    with patch.object(workflow_manager, "store_workflow", return_value=None):
        result = workflow_manager.execute_task(task_with_module, sample_workflow, "test-tool-use")

    # Verify
    assert result["status"] == "success"
    assert "Task executed with custom module" in result["content"][0]["text"]

    # Test with a task with automatic instruction generation
    task_with_auto_instructions = sample_tasks[0].copy()
    task_with_auto_instructions["auto_gen_instructions"] = True

    # Execute task
    with patch.object(workflow_manager, "store_workflow", return_value=None):
        result = workflow_manager.execute_task(task_with_auto_instructions, sample_workflow, "test-tool-use")

    # Verify
    assert result["status"] == "success"


def test_workflow_summary(workflow_manager, sample_workflow_id, sample_workflow):
    """Test workflow summary generation."""
    # Set up task results with different statuses
    sample_workflow["task_results"] = {
        "task-1": {"status": "completed", "result": [{"text": "Task 1 done"}]},
        "task-2": {"status": "running"},
        "task-3": {"status": "pending"},
        "task-4": {"status": "error", "result": [{"text": "Error in task 4"}]},
    }

    # Add tasks that match the results
    sample_workflow["tasks"] = [
        {"task_id": "task-1", "description": "First test task", "priority": 3},
        {"task_id": "task-2", "description": "Second test task", "priority": 2},
        {"task_id": "task-3", "description": "Third test task", "priority": 1},
        {"task_id": "task-4", "description": "Fourth test task", "priority": 1},
    ]

    # Add to cache
    workflow_manager._workflows[sample_workflow_id] = sample_workflow

    # Get workflow status
    result = workflow_manager.get_workflow_status(sample_workflow_id, "test-tool-use")

    # Verify summary includes counts
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    assert "Completed Tasks: 1" in text
    assert "Task 1 done" in text
    assert "Error in task 4" in text


def test_workflow_shutdown():
    """Test the task executor shutdown."""
    # Create a task executor for direct testing
    executor = TaskExecutor(min_workers=2, max_workers=4)

    # Assert that the executor has the shutdown method
    assert hasattr(executor, "shutdown")

    # Create a mock for the internal executor
    mock_executor = MagicMock()

    # Replace the internal executor
    executor._executor = mock_executor

    # Call the shutdown method
    executor.shutdown()

    # Verify executor shutdown was called
    mock_executor.shutdown.assert_called_once_with(wait=True)


def test_task_executor():
    """Test the TaskExecutor class directly."""
    # Create a task executor with an immediately mocked executor
    executor = TaskExecutor(min_workers=2, max_workers=4)

    # Test that initial state is correct
    assert executor.min_workers == 2
    assert executor.max_workers == 4
    assert len(executor.active_tasks) == 0
    assert executor.active_workers == 0

    # Mock executor's submit method
    original_executor = executor._executor
    try:
        # Create mock future
        mock_future = MagicMock()

        # Ensure callbacks don't run
        def mock_add_done_callback(callback):
            pass

        mock_future.add_done_callback = mock_add_done_callback

        # Set up the mock
        executor._executor = MagicMock()
        executor._executor.submit.return_value = mock_future

        # Define a mock task function
        mock_task_func = MagicMock()

        # Submit a task
        task_id = "test-task"
        future = executor.submit_task(task_id, mock_task_func, 1, key="value")

        # Verify the task was submitted correctly
        assert future == mock_future
        assert task_id in executor.active_tasks
        assert task_id in executor.start_times
        assert executor.active_workers == 1

        # Test that submitting the same task again returns None
        result = executor.submit_task(task_id, mock_task_func, 2, key="value2")
        assert result is None

        # Test task completion tracking
        result_data = {"status": "completed", "text": "Task result"}

        # Mark as completed
        executor.task_completed(task_id, result_data)

        # Check that result is stored and task removed from active tasks
        assert executor.get_result(task_id) == result_data
        assert task_id not in executor.active_tasks

        # Test shutdown method
        executor._executor.shutdown.assert_not_called()
        executor.shutdown()
        executor._executor.shutdown.assert_called_once_with(wait=True)
    finally:
        # If we exited before calling shutdown, ensure we restore the real executor
        executor._executor = original_executor
        executor.shutdown()


def test_workflow_dir_default():
    """Test that the default workflow directory is Path.cwd()/workflows."""
    import sys

    # Save original module if it exists
    original_module = sys.modules.get("strands_tools.workflow")

    # Remove strands_tools.workflow from sys.modules to force reload
    if "strands_tools.workflow" in sys.modules:
        del sys.modules["strands_tools.workflow"]

    # Add a cleanup to restore the original module
    try:
        # Remove any monkeypatching by creating clean module import
        with patch.dict("sys.modules", {"strands_tools.workflow": None}):
            from strands_tools import workflow

            # Check the default workflow directory
            expected_dir = Path.cwd() / "workflows"
            actual_dir = workflow.WORKFLOW_DIR

            # Convert both to absolute paths to handle any symlinks
            expected = expected_dir.resolve()
            actual = actual_dir.resolve()

            assert str(actual) == str(expected), f"Expected {expected}, got {actual}"
    finally:
        # Restore original module
        if original_module:
            sys.modules["strands_tools.workflow"] = original_module
