import asyncio
import json
import logging
import os
import uuid
from queue import Queue
from threading import Lock
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Import S3 uploader
from s3_utils import s3_uploader

from browser_use import Agent, Browser
from custom_controller import HumanHelpController
from browser_use import BrowserConfig, BrowserContextConfig
# Use patchright's BrowserContext when available, otherwise fallback to browser_use's
try:
    from patchright.async_api import BrowserContext
    PATCHRIGHT_AVAILABLE = True
    print("Using patchright.async_api.BrowserContext for enhanced stealth capabilities")
except ImportError:
    from browser_use.browser.context import BrowserContext
    PATCHRIGHT_AVAILABLE = False
    print("Falling back to browser_use.browser.context.BrowserContext")

# IPRoyal proxy settings
PROXY_ENABLED = os.environ.get("USE_PROXY", "false").lower() == "true"
PROXY_SERVER = os.environ.get("PROXY_SERVER")
PROXY_USERNAME = os.environ.get("PROXY_USERNAME")
PROXY_PASSWORD = os.environ.get("PROXY_PASSWORD")

# Log proxy status
if PROXY_ENABLED:
    print(f"Proxy enabled: {PROXY_SERVER}")
else:
    print("Proxy disabled. Set USE_PROXY=true to enable proxy.")

# Patch Playwright connection to handle KeyError: 'error'
try:
    from playwright._impl._connection import Connection
    
    # Store the original dispatch method
    original_dispatch = Connection.dispatch
    
    # Create a patched version that handles the KeyError
    def patched_dispatch(self, msg):
        try:
            return original_dispatch(self, msg)
        except KeyError as e:
            # Handle the specific KeyError: 'error'
            if str(e) == "'error'":
                # Just log it and continue
                logging.getLogger("playwright").warning(f"Ignoring KeyError in Playwright dispatch: {e}")
                return None
            # Re-raise other KeyErrors
            raise
    
    # Apply the monkey patch
    Connection.dispatch = patched_dispatch
    print("Patched Playwright connection to handle KeyError: 'error'")
except Exception as e:
    print(f"Failed to patch Playwright connection: {e}")

# Add patchright import - will be used if available
try:
    import patchright
    # Standard patchright import without monkey patching
    PATCHRIGHT_AVAILABLE = True
    print("Patchright is available and will be used for enhanced stealth mode")
except ImportError:
    PATCHRIGHT_AVAILABLE = False
    print("Patchright is not available. Install with: pip install patchright")

# Optional Redis imports - will be used if available
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('browser-use-api')

# Create a queue for log messages
log_queue = Queue()
log_lock = Lock()


class LogHandler(logging.Handler):
	def emit(self, record):
		log_entry = self.format(record)
		with log_lock:
			log_queue.put(log_entry)


# Add custom handler to the root logger
root_logger = logging.getLogger('browser-use-api')
root_logger.addHandler(LogHandler())

app = FastAPI()

# Enable CORS
app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],  # Allows all origins
	allow_credentials=True,
	allow_methods=['*'],  # Allows all methods
	allow_headers=['*'],  # Allows all headers
)

# Create static directory if it doesn't exist
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Serve static files
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


class TaskRequest(BaseModel):
	task: str
	task_id: Optional[str] = None  # Optional task ID, will be generated if not provided
	headless: bool = True  # Run browser in headless mode by default
	redis_url: Optional[str] = None  # Optional Redis URL for state persistence (legacy format)
	redis_host: Optional[str] = None  # Optional Redis host (new format)
	redis_port: Optional[int] = None  # Optional Redis port (new format)
	model: str = 'gpt-4o-mini'  # LLM model to use


class ResumeTaskRequest(BaseModel):
	"""Request model for resuming a paused task with user input"""
	input_text: Optional[str] = None  # Free-form text input from user
	selected_option: Optional[str] = None  # Selected option from a set of choices
	other_data: Optional[Dict[str, Any]] = None  # Any other data required


class AgentTask:
	"""Represents a single agent task with its associated resources"""
	def __init__(self, task_id: str, task_description: str, model: str = 'gpt-4o-mini', headless: bool = True):
		self.task_id = task_id
		self.task_description = task_description
		self.agent: Optional[Agent] = None
		self._running = False
		self.browser_debug_url: Optional[str] = None
		self.remote_debugging_port = self._get_available_port()  # Dynamic port allocation
		self.state_queue = asyncio.Queue()  # Queue for state updates
		self.screenshot_queue = asyncio.Queue()  # Queue for screenshots
		self.task_completed = False
		self.model = model
		self.headless = headless
		self.creation_time = asyncio.get_event_loop().time()
		self.last_activity_time = self.creation_time
		self._paused_for_input = False  # Flag to track if task is paused for human input
		
	@staticmethod
	def _get_available_port(start_port=9222, max_attempts=100):
		"""Find an available port for browser debugging"""
		import socket
		for port in range(start_port, start_port + max_attempts):
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				if s.connect_ex(('localhost', port)) != 0:
					return port
		return start_port  # Fallback to default if no ports available
	
	async def initialize(self, controller=None):
		"""Initialize the agent with browser and optional controller"""
		try:
			# Configure extra browser arguments for stealth mode (used for local browser)
			extra_args = [
				f'--remote-debugging-port={self.remote_debugging_port}',
				'--remote-debugging-address=0.0.0.0',  # Allow external connections
				'--disable-blink-features=AutomationControlled',  # Hide automation flags
				'--disable-infobars',
				'--disable-dev-shm-usage',  # Overcome limited /dev/shm size in containers
				'--no-sandbox',  # Required for running in containers
				'--disable-setuid-sandbox',  # Additional sandbox disabling for containers
			]
			
			# Log whether patchright is being used
			if PATCHRIGHT_AVAILABLE:
				logger.info("Using patchright for enhanced stealth capabilities")
				# Add additional stealth-focused arguments when patchright is available
				extra_args.extend([
					'--disable-features=IsolateOrigins,site-per-process',  # Disable site isolation for better stealth
					'--disable-web-security',  # Disable web security for better functionality
					'--disable-site-isolation-trials'  # Disable site isolation trials
				])
			else:
				logger.warning("Patchright not available - stealth capabilities will be limited")
			
			browser = None
			
			# Configure browser with proxy if enabled
			if PROXY_ENABLED and PROXY_SERVER and PROXY_USERNAME and PROXY_PASSWORD:
				logger.info(f"Configuring browser with proxy: {PROXY_SERVER}")
				
				# Configure proxy settings
				proxy_settings = {
					"server": f"http://{PROXY_SERVER}",
					"username": PROXY_USERNAME,
					"password": PROXY_PASSWORD
				}
				
				# Create browser config with proxy settings and enhanced stealth
				browser_config = BrowserConfig(
					headless=self.headless,
					channel="chrome",
					connection_timeout=60000,  # 60 seconds timeout for more reliable connection
					proxy=proxy_settings,
					extra_chromium_args=extra_args,
					# Enhanced stealth settings
					bypass_csp=True,  # Bypass Content Security Policy for better functionality
					ignore_https_errors=True,  # Ignore HTTPS errors
					has_touch=True,  # Simulate touch capability
					is_mobile=False,  # Don't simulate mobile device
					locale="en-US",  # Set locale to US English
					timezone_id="America/New_York",  # Set timezone to US East Coast
					user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"  # Use a common user agent
				)
				
				# Create browser with proxy config
				browser = Browser(config=browser_config)
				
				# Create the browser context
				context = await browser.new_context()
				
				# Set the debug URL for local browser
				self.browser_debug_url = f'http://localhost:{self.remote_debugging_port}'
			else:
				# Use standard local browser without proxy
				logger.info("Using standard browser configuration without proxy")
				
				# Create local browser config with standard browser-use options and enhanced stealth
				browser_config = BrowserConfig(
					headless=self.headless,
					channel="chrome",  # Use Chrome instead of Chromium for better stealth
					extra_chromium_args=extra_args,
					# Enhanced stealth settings
					bypass_csp=True,  # Bypass Content Security Policy for better functionality
					ignore_https_errors=True,  # Ignore HTTPS errors
					has_touch=True,  # Simulate touch capability
					is_mobile=False,  # Don't simulate mobile device
					locale="en-US",  # Set locale to US English
					timezone_id="America/New_York",  # Set timezone to US East Coast
					user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"  # Use a common user agent
				)
				
				# Create browser with local config
				browser = Browser(config=browser_config)
				
				# Create the browser context
				context = await browser.new_context()
				
				# Set the debug URL for local browser
				self.browser_debug_url = f'http://localhost:{self.remote_debugging_port}'
			
			# Create custom controller if not provided
			if controller is None:
				controller = HumanHelpController(
					task_id=self.task_id,
					state_queue=self.state_queue,
					screenshot_queue=self.screenshot_queue
				)
			
			# Initialize LLM
			llm = ChatOpenAI(model=self.model)
			
			# Create agent with the configured browser and controller
			self.agent = Agent(
				task=self.task_description, 
				llm=llm, 
				browser_context=context,
				controller=controller
			)
			
			# Store browser for cleanup
			self._browser = browser
			
			# Register callbacks
			self.agent.register_new_step_callback = self.step_callback
			self.agent.register_done_callback = self.done_callback
			
			self._running = False
			
			return self
		except Exception as e:
			logger.error(f"Error initializing agent: {e}")
			raise
	
	async def step_callback(self, browser_state, agent_output, step_number):
		"""Callback for each step of the agent"""
		# Update last activity time
		self.last_activity_time = asyncio.get_event_loop().time()
		
		# Extract the relevant data from browser_state
		state_data = {
			# Basic page info
			"url": browser_state.url,
			"title": browser_state.title,
			"step_number": step_number,
			"task_id": self.task_id,
			
			# Agent output information
			"action": self._filter_active_actions([a.model_dump() for a in agent_output.action]) if agent_output and hasattr(agent_output, 'action') else None,
			"thought": agent_output.thought if agent_output and hasattr(agent_output, 'thought') else None,
			"current_state": agent_output.current_state.model_dump() if agent_output and hasattr(agent_output, 'current_state') else None
		}
		
		# Check if this is a requires_input state (ask_human action)
		requires_input = False
		# agent_output.action is a list of ActionModel objects and each ActionModel has action_name
		
		# Optional debug logging for action structure (uncomment if needed)
		# if agent_output and hasattr(agent_output, 'action') and agent_output.action:
		# 	logger.info(f"DEBUG - Agent actions found: {len(agent_output.action)}")
		# 	for action in agent_output.action:
		# 		if hasattr(action, 'model_dump'):
		# 			logger.info(f"DEBUG - Action model_dump: {action.model_dump()}")
		
		if agent_output and hasattr(agent_output, 'action') and agent_output.action:
			# Process all actions to get their active type
			filtered_actions = self._filter_active_actions([a.model_dump() for a in agent_output.action])
			
			# Look for ask_human actions in the filtered actions
			ask_human_actions = [a for a in filtered_actions if "ask_human" in a] if filtered_actions else []
			
			if ask_human_actions:
				# Get the first ask_human action
				ask_human_action = ask_human_actions[0]
				ask_human_params = ask_human_action.get('ask_human', {})
				
				# Pause the agent before setting anything in state data
				self.agent.pause()
				
				# Set flag for requiring input
				requires_input = True
				state_data["requires_input"] = True
				
				# Extract input requirements directly from the action parameters
				state_data["input_requirements"] = {
					"input_type": ask_human_params.get("input_type", "text"),
					"prompt": ask_human_params.get("message", "Please provide input"),
					"options": ask_human_params.get("options", [])
				}
		
		# Save the screenshot (base64 encoded) for separate screenshot stream
		if hasattr(browser_state, 'screenshot') and browser_state.screenshot:
			# Upload screenshot to S3 if configured
			screenshot_url = None
			try:
				screenshot_url = await s3_uploader.upload_screenshot(
					base64_screenshot=browser_state.screenshot,
					task_id=self.task_id,
					step_number=step_number
				)
				if screenshot_url:
					logger.info(f"Uploaded screenshot to S3: {screenshot_url}")
			except Exception as e:
				logger.error(f"Error uploading screenshot to S3: {e}")
				# Continue with local screenshot if S3 upload fails
			
			# Store screenshot with step info for the screenshot stream
			screenshot_data = {
				"task_id": self.task_id,
				"step_number": step_number,
				"timestamp": asyncio.get_event_loop().time()
			}
			
			# Add screenshot URL if available, otherwise include base64 data
			if screenshot_url:
				screenshot_data["screenshot_url"] = screenshot_url
			else:
				screenshot_data["screenshot"] = browser_state.screenshot
				
			# Add to separate screenshot queue
			await self.screenshot_queue.put(json.loads(json.dumps({
				"type": "screenshot",
				"data": screenshot_data
			}, default=str)))
		
		# Convert to proper JSON format (ensure None becomes null)
		# Put the state data in the queue - ensure it's properly JSON serialized
		await self.state_queue.put(json.loads(json.dumps(state_data, default=str)))
	
		# If this requires input, auto-pause the agent
		if requires_input:
			logger.info(f"Task {self.task_id} paused waiting for human input")
			self._paused_for_input = True
			await self.pause()
	
	async def done_callback(self, agent_history):
		"""Callback when the agent is done"""
		self.task_completed = True
		
		# Get the final result
		final_result = None
		if agent_history:
			# Use the correct method to access history data
			# AgentHistoryList in newer versions doesn't have 'steps' attribute
			try:
				# Try to get the final result using the final_result method
				final_result = agent_history.final_result()
			except (AttributeError, TypeError):
				# Fallback if final_result method is not available
				try:
					# Try to access action_results if available
					action_results = agent_history.action_results()
					if action_results and len(action_results) > 0:
						final_result = action_results[-1]
				except (AttributeError, TypeError):
					# Final fallback - try to extract content from the history
					try:
						extracted = agent_history.extracted_content()
						if extracted:
							final_result = extracted
					except (AttributeError, TypeError):
						logger.warning("Could not extract final result from agent history")
		
		# Put the final result in the queue
		await self.state_queue.put({
			"type": "result",
			"data": final_result or 'Task completed',
			"task_id": self.task_id
		})
	
	async def run(self):
		"""Run the agent task"""
		self._running = True
		try:
			await self.agent.run()
			# When run() completes, mark the task as completed if not already done
			if not self.task_completed:
				self.task_completed = True
				await self.state_queue.put({
					"type": "result",
					"data": "Task completed without explicit result",
					"task_id": self.task_id
				})
		finally:
			self._running = False
	
	async def stop(self):
		"""Stop the agent task"""
		if self.agent:
			self.agent.stop()
			self._running = False
			self.task_completed = True
			# Add a stopped message to the queue
			await self.state_queue.put({
				"type": "result",
				"data": "Task stopped by user",
				"task_id": self.task_id
			})
	
	async def pause(self):
		"""Pause the agent task"""
		if self.agent:
			self.agent.pause()
	
	async def resume(self, user_input=None):
		"""Resume the agent task, optionally with user input"""
		if self.agent:
			# If we have user input and the task was paused waiting for input
			if user_input and self._paused_for_input:
				logger.info(f"Resuming task {self.task_id} with user input: {user_input}")
				self._paused_for_input = False
				self.agent.add_new_task(f'Continue the task with this user input: {user_input}')
			
			# Resume the agent
			self.agent.resume()
	
	@property
	def is_running(self):
		return self._running
	
	@staticmethod
	def _filter_active_actions(actions):
		"""Filter out null actions"""
		if not actions:
			return None
		
		filtered_actions = []
		for action in actions:
			# Find the active action (the one with a non-None value)
			active_action = {}
			for key, value in action.items():
				if value is not None:
					active_action[key] = value
			
			# Only add actions that have at least one non-None property
			if active_action:
				filtered_actions.append(active_action)
		
		return filtered_actions if filtered_actions else None


class MultiAgentManager:
	"""Manages multiple agent tasks"""
	def __init__(self, redis_url: Optional[str] = None):
		self.tasks: Dict[str, AgentTask] = {}
		self.redis_client = None
		self.use_redis = False
		# Create a shared controller for all agents
		self.shared_controller = HumanHelpController()
		
		# Initialize Redis if URL is provided and Redis is available
		if redis_url and REDIS_AVAILABLE:
			try:
				self.redis_client = aioredis.from_url(redis_url)
				self.use_redis = True
				logger.info(f"Redis connection established at {redis_url}")
			except Exception as e:
				logger.error(f"Failed to connect to Redis: {e}")
				self.use_redis = False
		else:
			logger.info("Using in-memory storage for task state")
		
		# We'll initialize the cleanup task when the event loop is running
		self._cleanup_task = None
	
	async def _cleanup_old_tasks(self, max_age_hours: int = 24, check_interval: int = 3600):
		"""Periodically clean up old completed tasks"""
		while True:
			try:
				current_time = asyncio.get_event_loop().time()
				tasks_to_remove = []
				
				for task_id, task in self.tasks.items():
					# Remove tasks that are completed and older than max_age_hours
					if task.task_completed and (current_time - task.last_activity_time) > (max_age_hours * 3600):
						tasks_to_remove.append(task_id)
						
				# Remove the tasks
				for task_id in tasks_to_remove:
					await self.remove_task(task_id)
					logger.info(f"Cleaned up old task {task_id}")
					
				# Also clean up Redis keys if using Redis
				if self.use_redis and self.redis_client:
					# Find all task keys older than max_age_hours
					pattern = "browser_task:*"
					async for key in self.redis_client.scan_iter(match=pattern):
						try:
							# Get the last_activity_time field from the hash instead of getting the entire JSON
							last_activity_time = await self.redis_client.hget(key, "last_activity_time")
							if last_activity_time:
								# Convert string to float
								last_activity_time = float(last_activity_time)
								if (current_time - last_activity_time) > (max_age_hours * 3600):
									await self.redis_client.delete(key)
									logger.info(f"Cleaned up Redis key {key}")
						except Exception as e:
							logger.error(f"Error cleaning up Redis key: {e}")
			except Exception as e:
				logger.error(f"Error in cleanup task: {e}")
				
			# Sleep for the check interval
			await asyncio.sleep(check_interval)
	
	async def create_task(self, task_description: str, task_id: Optional[str] = None, model: str = 'gpt-4o-mini', headless: bool = True) -> str:
		"""Create a new agent task"""
		# Generate a task ID if not provided
		if not task_id:
			task_id = str(uuid.uuid4())
			
		# Create and initialize the task with shared controller
		task = AgentTask(task_id, task_description, model, headless)
		await task.initialize(controller=self.shared_controller)
		
		# Store the task
		self.tasks[task_id] = task
		
		# Store task info in Redis if available
		if self.use_redis and self.redis_client:
			task_info = {
				"task_id": task_id,
				"task_description": task_description,
				"model": model,
				"headless": headless,
				"creation_time": task.creation_time,
				"last_activity_time": task.last_activity_time,
				"status": "created"
			}
			# Convert task_info to individual hash fields instead of using SET with a JSON string
			for key, value in task_info.items():
				await self.redis_client.hset(f"browser_task:{task_id}", key, str(value))
			
		return task_id
	
	async def get_task(self, task_id: str) -> Optional[AgentTask]:
		"""Get a task by ID"""
		return self.tasks.get(task_id)
	
	async def run_task(self, task_id: str) -> bool:
		"""Run a task by ID"""
		task = await self.get_task(task_id)
		if task:
			# Start the task in a background task
			asyncio.create_task(task.run())
			
			# Update Redis if available
			if self.use_redis and self.redis_client:
				await self.redis_client.hset(f"browser_task:{task_id}", "status", "running")
				await self.redis_client.hset(f"browser_task:{task_id}", "last_activity_time", asyncio.get_event_loop().time())
			
			return True
		return False
	
	async def stop_task(self, task_id: str) -> bool:
		"""Stop a task by ID"""
		task = await self.get_task(task_id)
		if task:
			await task.stop()
			
			# Update Redis if available
			if self.use_redis and self.redis_client:
				await self.redis_client.hset(f"browser_task:{task_id}", "status", "stopped")
				await self.redis_client.hset(f"browser_task:{task_id}", "last_activity_time", asyncio.get_event_loop().time())
			
			return True
		return False
	
	async def pause_task(self, task_id: str) -> bool:
		"""Pause a task by ID"""
		task = await self.get_task(task_id)
		if task:
			await task.pause()
			
			# Update Redis if available
			if self.use_redis and self.redis_client:
				await self.redis_client.hset(f"browser_task:{task_id}", "status", "paused")
				await self.redis_client.hset(f"browser_task:{task_id}", "last_activity_time", asyncio.get_event_loop().time())
			
			return True
		return False
	
	async def resume_task(self, task_id: str, user_input=None) -> bool:
		"""Resume a task by ID, optionally with user input"""
		task = await self.get_task(task_id)
		if task:
			# Resume with user input if provided
			await task.resume(user_input)
			
			# Update Redis if available
			if self.use_redis and self.redis_client:
				await self.redis_client.hset(f"browser_task:{task_id}", "status", "running")
				await self.redis_client.hset(f"browser_task:{task_id}", "last_activity_time", asyncio.get_event_loop().time())
			
			return True
		return False
	
	async def remove_task(self, task_id: str) -> bool:
		"""Remove a task by ID"""
		task = await self.get_task(task_id)
		if task:
			# Stop the task if it's running
			if task.is_running:
				await task.stop()
				
			# Remove from memory
			del self.tasks[task_id]
			
			# Remove from Redis if available
			if self.use_redis and self.redis_client:
				await self.redis_client.delete(f"browser_task:{task_id}")
				
			return True
		return False
	
	async def get_all_tasks(self) -> List[Dict[str, Any]]:
		"""Get info about all tasks"""
		tasks_info = []
		
		for task_id, task in self.tasks.items():
			tasks_info.append({
				"task_id": task_id,
				"task_description": task.task_description,
				"status": "running" if task.is_running else ("completed" if task.task_completed else "ready"),
				"creation_time": task.creation_time,
				"last_activity_time": task.last_activity_time,
				"browser_debug_url": task.browser_debug_url
			})
			
		return tasks_info
	
	async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
		"""Get status of a specific task"""
		task = await self.get_task(task_id)
		if task:
			return {
				"task_id": task_id,
				"task_description": task.task_description,
				"status": "running" if task.is_running else ("completed" if task.task_completed else "ready"),
				"creation_time": task.creation_time,
				"last_activity_time": task.last_activity_time,
				"browser_debug_url": task.browser_debug_url
			}
		return None


# Get Redis configuration from environment variables
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')

# Construct Redis URL if host is provided
redis_url = None
if REDIS_HOST:
    redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}"
    logger.info(f"Initializing Redis connection to {REDIS_HOST}:{REDIS_PORT}")

# Create a singleton instance of the multi-agent manager
agent_manager = MultiAgentManager(redis_url)


@app.post('/api/browser-use/tasks/')
async def create_task(request: TaskRequest):
	"""Create a new browser task without starting it"""
	global agent_manager
	try:
		# Create a new task with the provided task_id or generate one
		task_id = request.task_id or str(uuid.uuid4())
		
		# Handle Redis connection based on provided parameters
		if REDIS_AVAILABLE and not agent_manager.use_redis:
			# Priority 1: Use redis_url if provided directly in the request
			if request.redis_url:
				agent_manager = MultiAgentManager(request.redis_url)
				logger.info(f"Using Redis URL from request: {request.redis_url}")
			# Priority 2: Use redis_host and redis_port if provided
			elif request.redis_host:
				redis_port = request.redis_port or 6379
				redis_url = f"redis://{request.redis_host}:{redis_port}"
				agent_manager = MultiAgentManager(redis_url)
				logger.info(f"Using Redis from request: {request.redis_host}:{redis_port}")
		
		# Start the cleanup task if not already started
		if agent_manager._cleanup_task is None:
			agent_manager._cleanup_task = asyncio.create_task(agent_manager._cleanup_old_tasks())
			logger.info("Started background cleanup task")
		
		# Create the task
		await agent_manager.create_task(request.task, task_id, request.model, request.headless)
		
		# Get the task
		task = await agent_manager.get_task(task_id)
		if not task:
			raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
		
		# Return task info with stream URLs for compatibility with TypeScript client
		return {
			"task_id": task_id,
			"status": "created",
			"streams": {
				"state": f"/api/browser-use/tasks/{task_id}/run",
				"screenshots": f"/api/browser-use/tasks/{task_id}/screenshots"
			},
			"browser_debug_url": task.browser_debug_url
		}
	except Exception as e:
		logger.error(f"Error creating task: {e}")
		raise HTTPException(status_code=400, detail=str(e))


@app.put('/api/browser-use/tasks/{task_id}/run')
async def run_and_stream(task_id: str, resume_request: ResumeTaskRequest = None):
	"""Run and stream a browser task, or resume it with user input"""
	global agent_manager
	try:
		# Get the task
		task = await agent_manager.get_task(task_id)
		if not task:
			raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
		
		# Check if task is already running
		if task.is_running and not resume_request:
			logger.info(f"Task {task_id} is already running")
		# Otherwise start or resume the task
		else:
			if resume_request and resume_request.input_text:
				# Resume with user input
				await agent_manager.resume_task(task_id, resume_request)
				logger.info(f"Resumed task {task_id} with user input")
			else:
				# Start the task
				await agent_manager.run_task(task_id)
				logger.info(f"Started task {task_id}")

		# Stream state updates
		async def generate():
			while True:
				# Process state updates
				try:
					# Use wait_for to avoid blocking indefinitely
					state_data = await asyncio.wait_for(task.state_queue.get(), 0.5)
					
					if 'type' in state_data and state_data['type'] == 'result':
						# This is the final result
						logger.info(f'📄 Result for task {task_id}: {state_data["data"]}')
						# Ensure proper JSON serialization for completion event
						yield {"event": "complete", "data": json.dumps(state_data, default=str)}
						break
					elif 'requires_input' in state_data and state_data['requires_input']:
						# This state requires human input
						logger.info(f'👤 Task {task_id} requires human input')
						yield {"event": "requires_input", "data": json.dumps(state_data, default=str)}
					else:
						# Regular state update
						yield {"event": "state", "data": json.dumps(state_data, default=str)}
				except asyncio.TimeoutError:
					# No updates in the timeout period
					pass
				except asyncio.CancelledError:
					# Stream was cancelled
					break
				except Exception as e:
					logger.error(f"Error in state stream for task {task_id}: {e}")
					yield {"event": "error", "data": json.dumps({"error": str(e)}, default=str)}
					break
				
				# Check if task is complete
				if task.task_completed and task.state_queue.empty():
					yield {"event": "complete", "data": json.dumps({"message": "Task completed", "task_id": task_id}, default=str)}
					break

		return EventSourceResponse(generate())
	except Exception as e:
		logger.error(f"Error in run_and_stream: {e}")
		raise HTTPException(status_code=400, detail=str(e))


@app.get('/api/browser-use/tasks/{task_id}/screenshots')
async def stream_screenshots(task_id: str):
	"""Stream screenshots for a specific browser task"""
	try:
		# Get the task
		task = await agent_manager.get_task(task_id)
		if not task:
			raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
		
		# Stream screenshots
		async def generate():
			while True:
				try:
					# Use wait_for to avoid blocking indefinitely
					screenshot_data = await asyncio.wait_for(task.screenshot_queue.get(), 0.5)
					
					# Check if we have a screenshot_url or screenshot data
					screenshot_event_data = screenshot_data['data']
					
					# Ensure compatibility with TypeScript client - data must be a JSON string
					yield {"event": "screenshot", "data": json.dumps(screenshot_event_data, default=str)}
				except asyncio.TimeoutError:
					# No screenshots in the timeout period
					pass
				except asyncio.CancelledError:
					# Stream was cancelled
					break
				except Exception as e:
					logger.error(f"Error in screenshot stream for task {task_id}: {e}")
					yield {"event": "error", "data": json.dumps({"error": str(e)}, default=str)}
					break
				
				# Check if task is complete and no more screenshots
				if task.task_completed and task.screenshot_queue.empty():
					# Just continue; don't break the stream as more screenshots might come
					pass

		return EventSourceResponse(generate())
	except Exception as e:
		logger.error(f"Error in stream_screenshots: {e}")
		raise HTTPException(status_code=400, detail=str(e))


@app.get('/health')
async def health_check():
	return {
		'status': 'healthy',
		'version': '1.0.0',
		'service': 'browser-use',
		'redis_enabled': agent_manager.use_redis
	}


@app.get('/api/browser-use/tasks')
async def list_tasks():
	"""List all browser tasks"""
	tasks = await agent_manager.get_all_tasks()
	return {
		'tasks': tasks,
		'count': len(tasks),
		'using_redis': agent_manager.use_redis
	}

@app.post('/api/browser-use/tasks/{task_id}/stop')
async def stop_task(task_id: str):
	"""Stop a specific browser task"""
	try:
		success = await agent_manager.stop_task(task_id)
		if not success:
			raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
		return {'status': 'stopped', 'task_id': task_id}
	except Exception as e:
		logger.error(f"Error stopping task: {e}")
		raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/browser-use/tasks/{task_id}/pause')
async def pause_task(task_id: str):
	"""Pause a specific browser task"""
	try:
		success = await agent_manager.pause_task(task_id)
		if not success:
			raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
		return {'status': 'paused', 'task_id': task_id}
	except Exception as e:
		logger.error(f"Error pausing task: {e}")
		raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/browser-use/tasks/{task_id}/resume')
async def resume_task(task_id: str, resume_request: ResumeTaskRequest = None):
	"""Resume a specific browser task, optionally with user input"""
	try:
		# Process user input if provided
		user_input = None
		if resume_request:
			user_input = resume_request
			
		success = await agent_manager.resume_task(task_id, user_input)
		if not success:
			raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
		return {'status': 'resumed', 'task_id': task_id}
	except Exception as e:
		logger.error(f"Error resuming task: {e}")
		raise HTTPException(status_code=400, detail=str(e))


@app.delete('/api/browser-use/tasks/{task_id}')
async def delete_browser_task(task_id: str):
	"""Delete a specific browser task"""
	try:
		success = await agent_manager.remove_task(task_id)
		if not success:
			raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
		return {'status': 'deleted', 'task_id': task_id}
	except Exception as e:
		logger.error(f"Error deleting task: {e}")
		raise HTTPException(status_code=400, detail=str(e))


@app.get('/api/browser-use/tasks/{task_id}')
async def get_task_info(task_id: str):
	"""Get information about a specific browser task"""
	task_status = await agent_manager.get_task_status(task_id)
	if not task_status:
		raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
	return task_status


@app.get('/api/browser-use/health')
async def health_check():
	"""Health check endpoint for Kubernetes"""
	try:
		# Check if agent_manager is initialized
		if agent_manager is None:
			return {"status": "error", "message": "Agent manager not initialized"}
		
		# Return basic health information
		return {
			"status": "ok",
			"version": "1.0.0",
			"service": "browser-use-api",
			"redis_connected": agent_manager.use_redis
		}
	except Exception as e:
		logger.error(f"Health check failed: {e}")
		return {"status": "error", "message": str(e)}

@app.get('/browser/debug-url')
async def get_browser_debug_url(task_id: str):
	"""Return the browser's remote debugging URL for a specific task"""
	task = await agent_manager.get_task(task_id)
	if not task:
		raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
	
	# The debug URL is set when the agent is created with remote debugging enabled
	if task.browser_debug_url:
		# For a better user experience, provide different URLs for different purposes
		devtools_url = f"{task.browser_debug_url}/devtools/devtools_app.html"
		json_version_url = f"{task.browser_debug_url}/json/version"
		json_list_url = f"{task.browser_debug_url}/json/list"
		
		return {
			'status': 'available',
			'task_id': task_id,
			'debug_urls': {
				'devtools': devtools_url,  # Chrome DevTools UI
				'json_version': json_version_url,  # Browser version info
				'json_list': json_list_url,  # List of debuggable targets
				'base': task.browser_debug_url  # Base debugging URL
			},
			'instructions': {
				'devtools': 'Open this URL in a browser to use Chrome DevTools UI',
				'websocket': 'To connect via WebSocket, get the webSocketDebuggerUrl from the json_list endpoint',
				'playwright': 'Use browser.connect_over_cdp(endpoint_url=base_url) to connect with Playwright'
			}
		}
	else:
		return {
			'status': 'unavailable',
			'task_id': task_id,
			'message': 'Browser debugging not enabled or browser not yet initialized'
		}

if __name__ == '__main__':
	import uvicorn

	uvicorn.run(app, host='0.0.0.0', port=8000)