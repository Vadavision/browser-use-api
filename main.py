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

from browser_use import Agent

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
		self.task_completed = False
		self.model = model
		self.headless = headless
		self.creation_time = asyncio.get_event_loop().time()
		self.last_activity_time = self.creation_time
		
	@staticmethod
	def _get_available_port(start_port=9222, max_attempts=100):
		"""Find an available port for browser debugging"""
		import socket
		for port in range(start_port, start_port + max_attempts):
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				if s.connect_ex(('localhost', port)) != 0:
					return port
		return start_port  # Fallback to default if no ports available
	
	async def initialize(self):
		"""Initialize the agent with browser"""
		llm = ChatOpenAI(model=self.model)
		
		# Create a Browser instance with remote debugging enabled
		from browser_use import Browser, BrowserConfig
		
		# Configure browser with remote debugging enabled
		extra_args = [
			f'--remote-debugging-port={self.remote_debugging_port}',
			'--remote-debugging-address=0.0.0.0'  # Allow external connections
		]
		
		browser_config = BrowserConfig(
			headless=self.headless,
			extra_chromium_args=extra_args
		)
		
		browser = Browser(config=browser_config)
		
		# Create agent with the configured browser
		self.agent = Agent(task=self.task_description, llm=llm, browser=browser)
		self._running = False
		
		# Set the debug URL (will be available after the browser is launched)
		self.browser_debug_url = f'http://localhost:{self.remote_debugging_port}'
		
		# Register callbacks
		self.agent.register_new_step_callback = self.step_callback
		self.agent.register_done_callback = self.done_callback
		
		return self
	
	async def step_callback(self, browser_state, agent_output, step_number):
		"""Callback for each step of the agent"""
		# Update last activity time
		self.last_activity_time = asyncio.get_event_loop().time()
		
		# Extract the relevant data from browser_state
		state_data = {
			# Basic page info
			"url": browser_state.url,
			"title": browser_state.title,
			# "screenshot": browser_state.screenshot,  # Include screenshot by default
			"step_number": step_number,
			"task_id": self.task_id,
			
			# Agent output information
			"action": self._filter_active_actions([a.model_dump() for a in agent_output.action]) if agent_output and hasattr(agent_output, 'action') else None,
			"thought": agent_output.thought if agent_output and hasattr(agent_output, 'thought') else None,
			"current_state": agent_output.current_state.model_dump() if agent_output and hasattr(agent_output, 'current_state') else None
		}
		
		# Convert to proper JSON format (ensure None becomes null)
		import json
		
		# Put the state data in the queue - ensure it's properly JSON serialized
		await self.state_queue.put(json.loads(json.dumps(state_data, default=str)))
	
	async def done_callback(self, agent_history):
		"""Callback when the agent is done"""
		self.task_completed = True
		
		# Get the final result
		final_result = None
		if agent_history and len(agent_history.steps) > 0:
			last_step = agent_history.steps[-1]
			if last_step and last_step.results and len(last_step.results) > 0:
				last_result = last_step.results[-1]
				if hasattr(last_result, 'extracted_content'):
					final_result = last_result.extracted_content
		
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
	
	async def resume(self):
		"""Resume the agent task"""
		if self.agent:
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
		
		# Start background task for cleanup
		asyncio.create_task(self._cleanup_old_tasks())
	
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
							task_data = await self.redis_client.get(key)
							if task_data:
								task_info = json.loads(task_data)
								if "last_activity_time" in task_info and (current_time - task_info["last_activity_time"]) > (max_age_hours * 3600):
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
			
		# Create and initialize the task
		task = AgentTask(task_id, task_description, model, headless)
		await task.initialize()
		
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
			await self.redis_client.set(f"browser_task:{task_id}", json.dumps(task_info))
			
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
	
	async def resume_task(self, task_id: str) -> bool:
		"""Resume a task by ID"""
		task = await self.get_task(task_id)
		if task:
			await task.resume()
			
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


@app.post('/api/browser-use/tasks/start')
async def run_and_stream(request: TaskRequest):
	"""Create, start, and stream a browser task in one operation"""
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
			# Priority 2: Construct URL from redis_host and redis_port if provided in the request
			elif request.redis_host:
				port = request.redis_port or 6379
				constructed_url = f"redis://{request.redis_host}:{port}"
				agent_manager = MultiAgentManager(constructed_url)
				logger.info(f"Using Redis connection from request parameters: {request.redis_host}:{port}")
		
		# Create the task
		task_id = await agent_manager.create_task(
			task_description=request.task,
			task_id=task_id,
			model=request.model,
			headless=request.headless
		)
		
		# Get the task
		task = await agent_manager.get_task(task_id)
		if not task:
			raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
		
		# Start the task in the background
		asyncio.create_task(task.run())

		# Only stream state updates (disable log streaming)
		async def generate():
			while True:
				# Process log entries - just consume them but don't yield
				if not log_queue.empty():
					with log_lock:
						while not log_queue.empty():
							# Just get the logs to clear the queue but don't yield them
							log_queue.get()
				
				# Process state updates (including screenshots)
				try:
					# Use get_nowait to avoid blocking
					state_data = await asyncio.wait_for(task.state_queue.get(), 0.01)
					
					if 'type' in state_data and state_data['type'] == 'result':
						# This is the final result
						logger.info(f'📄 Result for task {task_id}: {state_data["data"]}')
						# Ensure proper JSON serialization for completion event
						import json
						yield {"event": "complete", "data": json.dumps(state_data, default=str)}
						break
					else:
						# This is a state update with screenshot
						# Ensure proper JSON serialization using json.dumps
						import json
						yield {"event": "state", "data": json.dumps(state_data, default=str)}
				except (asyncio.TimeoutError, asyncio.QueueEmpty):
					pass  # No state updates available, continue
				
				# Check if task is complete
				if task.task_completed and task.state_queue.empty():
					yield {"event": "complete", "data": json.dumps({"message": "Task completed", "task_id": task_id}, default=str)}
					break
				
				await asyncio.sleep(0.1)

		return EventSourceResponse(generate())
	except Exception as e:
		logger.error(f"Error in run_and_stream: {e}")
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
async def resume_task(task_id: str):
	"""Resume a specific browser task"""
	try:
		success = await agent_manager.resume_task(task_id)
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
