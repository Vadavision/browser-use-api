from browser_use import Controller, ActionResult, Browser
from pydantic import BaseModel
import asyncio
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger('browser-use-api')

# Define a model for the ask_human action parameters
class AskHumanParams(BaseModel):
    message: str
    input_type: str = 'text'
    options: Optional[List[str]] = None

# Create a custom controller by extending the base Controller
class HumanHelpController(Controller):
    """
    Custom controller that adds the ask_human action to the browser-use agent.
    This allows the agent to request human input when needed.
    """
    
    def __init__(self):
        super().__init__()
        # Register our custom action using the registry
        self.register_ask_human()
    
    def register_ask_human(self):
        """Register the ask_human action with the controller"""
        # Register the action with the registry
        # The function name becomes the action name, so we name it 'ask_human'
        @self.registry.action(
            'Request input from a human user, such as verification codes or choices',
            param_model=AskHumanParams  # Use our parameter model
        )
        async def ask_human(params: AskHumanParams, browser: Browser = None) -> ActionResult:
            """
            Request input from a human user
            
            Args:
                params: The parameters for the ask_human action
                browser: The browser instance
                
            Returns:
                ActionResult with success status and human input when available
            """
            # Log that we're waiting for human input
            logger.info(f"🙋 Agent is asking for human input: {params.message}")
            
            # Set a flag in the task to indicate we're waiting for input
            # This will be handled by the API endpoint
            if hasattr(browser, 'task'):
                browser.task.waiting_for_input = True
                browser.task.input_requirements = {
                    'message': params.message,
                    'input_type': params.input_type,
                    'options': params.options or []
                }
                
                # Create an event that will be set when input is received
                if not hasattr(browser.task, 'input_event'):
                    browser.task.input_event = asyncio.Event()
                else:
                    browser.task.input_event.clear()
                    
                # Wait for the input event to be set
                await browser.task.input_event.wait()
                
                # Get the input that was provided
                user_input = getattr(browser.task, 'latest_user_input', None)
                
                # Return the input as the result
                return ActionResult(
                    success=True,
                    message=f"Received human input: {user_input}",
                    data=user_input
                )
            
            # If we can't set the waiting_for_input flag, return an error
            return ActionResult(
                success=False,
                message="Could not request human input - browser task not available"
            )
