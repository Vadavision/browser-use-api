from browser_use import Controller, ActionResult, Browser
from pydantic import BaseModel
from typing import Optional, List
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
        @self.registry.action(
            'Request input from user if you\'re unsure about some input like one time password (otp)/verification code, 2fa, sensitive information like passwords or multiple choices, also ask user if you get stuck',
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
            logger.info(f"Agent is asking for human input: {params.message}")
            
            # Simply return success - the main.py will handle the pause/resume
            # and adding the input to the agent's state
            return ActionResult(
                success=True,
                message=f"Requested human input: {params.message}"
            )
 
