import asyncio
import requests
from fastapi.testclient import TestClient
from main import app

# Create a test client
client = TestClient(app)

def test_create_task():
    """Test creating a task that will ask for human input"""
    # Create a task that will ask for human input
    task_data = {
        "task_description": "Login to a website that requires a verification code",
        "config": {
            "headless": False,  # Show the browser for testing
            "browser": "chrome"
        }
    }
    
    # Create the task
    response = client.post("/tasks", json=task_data)
    assert response.status_code == 200
    
    # Get the task ID
    task_id = response.json()["task_id"]
    print(f"Created task with ID: {task_id}")
    
    return task_id

def test_provide_input(task_id: str):
    """Test providing input to a task"""
    # Provide input to the task
    input_data = {
        "input": "123456"  # Example verification code
    }
    
    # Send the input
    response = client.post(f"/tasks/{task_id}/input", json=input_data)
    assert response.status_code == 200
    
    print(f"Provided input to task: {response.json()}")

def main():
    # Create a task
    task_id = test_create_task()
    
    # Wait for the task to ask for input
    print("Waiting for the task to ask for human input...")
    print("(You may need to interact with the browser to trigger the verification code request)")
    
    # Wait for user to manually trigger the input request
    input("Press Enter when the task is waiting for input...")
    
    # Provide input to the task
    test_provide_input(task_id)
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
