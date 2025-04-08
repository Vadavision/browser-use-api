import requests
import time
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def create_task():
    """Create a test task that will use the custom system prompt"""
    # Task data with a simple task that might require verification
    task_data = {
        "task_description": "Go to example.com and check if there's a login form. If there is, try to login with test@example.com and password123.",
        "config": {
            "headless": False,  # Show the browser for testing
            "browser": "chrome"
        }
    }
    
    # Create the task
    response = requests.post(f"{BASE_URL}/api/browser-use/tasks/", json=task_data)
    
    if response.status_code == 200:
        task_id = response.json().get("task_id")
        print(f"✅ Created task with ID: {task_id}")
        return task_id
    else:
        print(f"❌ Failed to create task: {response.status_code} - {response.text}")
        return None

def main():
    # Create a task
    task_id = create_task()
    
    if not task_id:
        return
    
    print("\n📋 Task created successfully. Check the server logs for:")
    print("  - 🔍 HumanHelpSystemPrompt initialized")
    print("  - 🔍 HumanHelpSystemPrompt.important_rules called")
    print("  - 🔍 HumanHelpSystemPrompt.system_prompt called")
    print("\nThese log messages indicate that your custom system prompt is being used.")
    print("\nIf you see a message about the agent asking for human input (🙋),")
    print("that means your custom controller is working correctly.")
    
    print("\n⏳ Waiting for the task to complete or request input...")
    print("Press Ctrl+C to exit")
    
    # Keep checking the task status
    try:
        while True:
            time.sleep(2)
            response = requests.get(f"{BASE_URL}/api/browser-use/tasks/{task_id}/status")
            if response.status_code == 200:
                status = response.json()
                if status.get("waiting_for_input"):
                    input_requirements = status.get("input_requirements", {})
                    print(f"\n🎯 Task is waiting for input: {json.dumps(input_requirements, indent=2)}")
                    
                    # Provide input
                    user_input = input("\nEnter your input: ")
                    input_data = {"input_text": user_input}
                    
                    # Send the input
                    input_response = requests.post(
                        f"{BASE_URL}/api/browser-use/tasks/{task_id}/input", 
                        json=input_data
                    )
                    
                    if input_response.status_code == 200:
                        print("✅ Input provided successfully")
                    else:
                        print(f"❌ Failed to provide input: {input_response.status_code} - {input_response.text}")
                
                if status.get("status") == "completed":
                    print("\n✅ Task completed successfully")
                    break
                elif status.get("status") == "failed":
                    print(f"\n❌ Task failed: {status.get('error')}")
                    break
            else:
                print(f"❌ Failed to get task status: {response.status_code} - {response.text}")
                break
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")

if __name__ == "__main__":
    main()
