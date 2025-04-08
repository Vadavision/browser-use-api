from browser_use import SystemPrompt
import logging

logger = logging.getLogger('browser-use-api')

class HumanHelpSystemPrompt(SystemPrompt):
    """
    Custom system prompt that enables the browser-use agent to ask for human help
    when needed, such as for OTP verification or other user input requirements.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("🔍 HumanHelpSystemPrompt initialized - Custom prompt with ask_human capability is active")
    
    def important_rules(self) -> str:
        # Get existing rules from parent class
        existing_rules = super().important_rules()
        
        # Add custom rules for asking human help
        new_rules = """
9. HUMAN HELP RULE:
- When you encounter situations requiring human input, 
  use the ask_human action to request assistance.
- Format your request clearly explaining what input is needed and why.
- You can specify the input_type (text, otp, select) and provide options if applicable.
- Example: { "type": "ask_human", "message": "Please provide the 6-digit verification code sent to your phone", "input_type": "otp" }
- Example: { "type": "ask_human", "message": "Please select which account to use", "input_type": "select", "options": ["Personal", "Work"] }
- Wait for human input before proceeding with the task.
"""
        
        # Log that we're adding custom rules
        logger.info("🔍 HumanHelpSystemPrompt.important_rules called - Adding custom human help rules")
        
        # Make sure to use this pattern to preserve existing rules
        return f'{existing_rules}\n{new_rules}'
    
    def system_prompt(self) -> str:
        prompt = super().system_prompt()
        logger.info("🔍 HumanHelpSystemPrompt.system_prompt called - Custom system prompt generated")
        return prompt
