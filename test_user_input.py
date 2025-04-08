import asyncio
import json
from datetime import datetime

# For testing in local environment
import aiohttp

async def test_user_input_flow():
    """Test the user input flow for browser-use tasks"""
    base_url = "http://localhost:8000"  # Adjust if your API runs on a different port
    
    # Step 1: Create a task that will require user input (like OTP verification)
    print("Creating a task that will require user input...")
    async with aiohttp.ClientSession() as session:
        # Create a task that will likely require OTP (login to a service)
        task_data = {
            "task": "Log in to my email account",
            "url": "https://mail.google.com",  # This will likely trigger OTP verification
            "options": {
                "headless": False  # Set to False to see the browser
            }
        }
        
        async with session.post(f"{base_url}/api/browser-use/tasks", json=task_data) as response:
            if response.status != 200:
                print(f"Error creating task: {await response.text()}")
                return
            
            task_info = await response.json()
            task_id = task_info['task_id']
            print(f"Created task with ID: {task_id}")
        
        # Step 2: Start the task and listen for state updates
        print("Starting task and listening for state updates...")
        async with session.put(f"{base_url}/api/browser-use/tasks/{task_id}/run") as response:
            if response.status != 200:
                print(f"Error starting task: {await response.text()}")
                return
            
            # Process the SSE stream
            requires_input = False
            input_requirements = None
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if not line or line.startswith(':'):
                    continue
                
                if line.startswith('event:'):
                    event_type = line[6:].strip()
                    continue
                
                if line.startswith('data:'):
                    data = json.loads(line[5:].strip())
                    
                    # Check if this is a state update
                    if 'event' in data and data['event'] == 'state':
                        state_data = json.loads(data['data'])
                        print(f"State update: {state_data.get('message', 'No message')}")
                        
                        # Check if input is required
                        if 'requires_input' in state_data and state_data['requires_input']:
                            requires_input = True
                            input_requirements = state_data.get('input_requirements', {})
                            print("\n" + "=" * 50)
                            print(f"Task requires input: {input_requirements.get('prompt', 'Please provide input')}")
                            print(f"Input type: {input_requirements.get('input_type', 'text')}")
                            
                            if input_requirements.get('options'):
                                print("Options:")
                                for i, option in enumerate(input_requirements['options']):
                                    print(f"  {i+1}. {option}")
                            
                            print("=" * 50)
                            
                            # Break the loop to provide input
                            break
                    
                    # Check if task is complete
                    if 'event' in data and data['event'] == 'complete':
                        print("Task completed!")
                        complete_data = json.loads(data['data'])
                        print(f"Result: {complete_data}")
                        return
        
        # Step 3: If input is required, provide it
        if requires_input:
            print("\nProviding user input...")
            
            # Get input from user
            if input_requirements and input_requirements.get('input_type') == 'select' and input_requirements.get('options'):
                # For selection, show options and get user choice
                user_choice = input("Enter the number of your selection: ")
                try:
                    choice_idx = int(user_choice) - 1
                    if 0 <= choice_idx < len(input_requirements['options']):
                        selected_option = input_requirements['options'][choice_idx]
                        user_input = {"selected_option": selected_option}
                    else:
                        print("Invalid selection, using first option")
                        user_input = {"selected_option": input_requirements['options'][0]}
                except ValueError:
                    print("Invalid input, using first option")
                    user_input = {"selected_option": input_requirements['options'][0]}
            else:
                # For text/OTP input
                text_input = input("Enter the required text/OTP: ")
                user_input = {"input_text": text_input}
            
            # Send the input back to the API
            async with session.put(f"{base_url}/api/browser-use/tasks/{task_id}/run", json=user_input) as response:
                if response.status != 200:
                    print(f"Error providing input: {await response.text()}")
                    return
                
                print("Input provided successfully, continuing to monitor task...")
                
                # Continue processing the SSE stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if not line or line.startswith(':'):
                        continue
                    
                    if line.startswith('event:'):
                        event_type = line[6:].strip()
                        continue
                    
                    if line.startswith('data:'):
                        data = json.loads(line[5:].strip())
                        
                        # Check if this is a state update
                        if 'event' in data and data['event'] == 'state':
                            state_data = json.loads(data['data'])
                            print(f"State update: {state_data.get('message', 'No message')}")
                        
                        # Check if task is complete
                        if 'event' in data and data['event'] == 'complete':
                            print("Task completed!")
                            complete_data = json.loads(data['data'])
                            print(f"Result: {complete_data}")
                            return

async def main():
    await test_user_input_flow()

if __name__ == "__main__":
    asyncio.run(main())
