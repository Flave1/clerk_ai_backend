"""
Browser Bot Orchestrator Service

Handles launching and managing browser bot containers/processes.
Supports multiple deployment methods:
- Docker (local development)
- AWS ECS Fargate (production)
- Subprocess (fallback/development)
"""
import asyncio
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BrowserBotOrchestrator:
    """Orchestrator for managing browser bot containers/processes."""
    
    def __init__(self):
        """Initialize the browser bot orchestrator."""
        self.docker_client = None
        self.ecs_client = None
        self.active_bots: Dict[str, Dict[str, Any]] = {}
        
        # Configuration from settings
        self.browser_bot_enabled = getattr(settings, 'browser_bot_enabled', True)
        self.bot_image = getattr(settings, 'bot_image', 'bot_staging')
        self.bot_container_cpu = getattr(settings, 'bot_container_cpu', '1024')
        self.bot_container_memory = getattr(settings, 'bot_container_memory', '2048')
        self.bot_join_timeout_sec = getattr(settings, 'bot_join_timeout_sec', 60)
        self.max_concurrent_bots = getattr(settings, 'max_concurrent_bots', 5)
        self.bot_deployment_method = getattr(settings, 'bot_deployment_method', 'auto')  # 'docker', 'ecs', 'subprocess', 'auto'
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize Docker and AWS clients based on environment."""
        # Initialize Docker client
        if self.bot_deployment_method in ('auto', 'docker'):
            try:
                import docker
                self.docker_client = docker.from_env()
                logger.info("✅ Docker client initialized")
            except ImportError:
                logger.debug("Docker SDK not available")
                self.docker_client = None
            except Exception as e:
                logger.warning(f"Failed to initialize Docker client: {e}")
                self.docker_client = None
        
        # Initialize AWS ECS client
        if self.bot_deployment_method in ('auto', 'ecs'):
            try:
                self.ecs_client = boto3.client('ecs', region_name=getattr(settings, 'aws_region', 'us-east-1'))
                logger.info("✅ AWS ECS client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AWS ECS client: {e}")
                self.ecs_client = None
    
    async def launch_bot(
        self,
        meeting_id: str,
        platform: str,
        meeting_url: str,
        bot_name: str = None,
        session_id: str = None,
        additional_env_vars: Dict[str, str] = None
    ) -> bool:
        """
        Launch a browser bot for a meeting.
        
        Args:
            meeting_id: Unique meeting identifier
            platform: Meeting platform ('google_meet', 'zoom', 'teams')
            meeting_url: URL to join the meeting
            bot_name: Name of the bot (defaults to settings.ai_email)
            session_id: Optional session ID (will generate if not provided)
            additional_env_vars: Additional environment variables to pass to bot
        
        Returns:
            True if bot launched successfully, False otherwise
        """
        try:
            logger.info(f"🚀 Launching browser bot for meeting: {meeting_id}")
            
            if not self.browser_bot_enabled:
                logger.warning("Browser bot is disabled in settings")
                return False
            
            # Check if we're at capacity
            # if len(self.active_bots) >= self.max_concurrent_bots:
            #     logger.warning(f"Maximum concurrent bots ({self.max_concurrent_bots}) reached")
            #     return False
            
            # Check if bot already exists for this meeting
            if meeting_id in self.active_bots:
                logger.warning(f"Bot already running for meeting: {meeting_id}")
                return True
            
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid4())
            
            # Prepare environment variables
            env_vars = self._prepare_env_vars(
                meeting_id=meeting_id,
                platform=platform,
                meeting_url=meeting_url,
                bot_name=bot_name,
                session_id=session_id,
                additional_env_vars=additional_env_vars or {}
            )
            
            # Try to launch based on deployment method
            success = False
            
            if self.bot_deployment_method == 'docker' or (self.bot_deployment_method == 'auto' and self.docker_client):
                success = await self._launch_docker_bot(meeting_id, session_id, env_vars)
                if success:
                    return True
            
            if self.bot_deployment_method == 'ecs' or (self.bot_deployment_method == 'auto' and self.ecs_client and not success):
                success = await self._launch_ecs_bot(meeting_id, session_id, env_vars)
                if success:
                    return True
            
            if self.bot_deployment_method == 'subprocess' or (self.bot_deployment_method == 'auto' and not success):
                success = await self._launch_subprocess_bot(meeting_id, session_id, env_vars)
                if success:
                    return True
            
            logger.error("❌ Failed to launch browser bot with any available method")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error launching browser bot: {e}", exc_info=True)
            return False
    
    def _prepare_env_vars(
        self,
        meeting_id: str,
        platform: str,
        meeting_url: str,
        bot_name: str = None,
        session_id: str = None,
        additional_env_vars: Dict[str, str] = None
    ) -> Dict[str, str]:
        """Prepare environment variables for the bot."""
        # Normalize platform name for browser_bot (it expects 'teams', not 'microsoft_teams')
        platform_for_bot = platform.lower()
        if platform_for_bot == 'microsoft_teams':
            platform_for_bot = 'teams'
        
        # Use host.docker.internal for macOS/Windows Docker Desktop
        # On Linux with proper host networking, localhost works
        import platform as platform_module
        rt_gateway_url = (
            getattr(settings, 'rt_gateway_base_url', None)
            or getattr(settings, 'rt_gateway_url', None)
        )
        api_base_url = getattr(settings, 'api_base_url', None)

        if not rt_gateway_url:
            host = getattr(settings, 'rt_gateway_host', '127.0.0.1') or '127.0.0.1'
            if host == '0.0.0.0':
                host = '127.0.0.1'
            port = getattr(settings, 'rt_gateway_port', 8001) or 8001
            rt_gateway_url = f"http://{host}:{port}"

        if not api_base_url:
            host = getattr(settings, 'api_host', '127.0.0.1') or '127.0.0.1'
            if host == '0.0.0.0':
                host = '127.0.0.1'
            port = getattr(settings, 'api_port', 8000) or 8000
            api_base_url = f"http://{host}:{port}{settings.api_prefix.rstrip('/') if getattr(settings, 'api_prefix', None) else ''}"
        
        # On macOS/Windows, Docker Desktop uses host.docker.internal
        if platform_module.system() in ['Darwin', 'Windows']:
            if rt_gateway_url:
                rt_gateway_url = rt_gateway_url.replace('localhost', 'host.docker.internal')
            if api_base_url:
                api_base_url = api_base_url.replace('localhost', 'host.docker.internal')
        
        env_vars = {
            'MEETING_URL': meeting_url,
            'BOT_NAME': bot_name or getattr(settings, 'ai_email', 'Clerk AI Bot'),
            'PLATFORM': platform_for_bot,
            'MEETING_ID': meeting_id,
            'SESSION_ID': session_id or str(uuid4()),
            'RT_GATEWAY_URL': rt_gateway_url or '',
            'API_BASE_URL': api_base_url or '',
            'JOIN_TIMEOUT_SEC': str(self.bot_join_timeout_sec),
            'NAVIGATION_TIMEOUT_MS': '45000',
            'AUDIO_SAMPLE_RATE': '16000',
            'AUDIO_CHANNELS': '1',
            'ENABLE_AUDIO_CAPTURE': 'true',
            'ENABLE_TTS_PLAYBACK': 'true',
            'HEADLESS': 'true',  # Must be true for Docker containers (no X server)
            'BROWSER_LOCALE': 'en-US',
            'LOG_LEVEL': 'info',
        }
        
        # Add any additional environment variables
        if additional_env_vars:
            env_vars.update(additional_env_vars)
        
        return env_vars
    
    async def _launch_docker_bot(
        self,
        meeting_id: str,
        session_id: str,
        env_vars: Dict[str, str]
    ) -> bool:
        """Launch browser bot using Docker."""
        try:
            if not self.docker_client:
                logger.debug("Docker client not available")
                return False
            
            logger.info(f"🐳 Launching Docker bot for meeting: {meeting_id}")
            
            # Create container name
            container_name = f"clerk-bot-{meeting_id}-{session_id[:8]}"
            
            # Run container
            container = self.docker_client.containers.run(
                image=self.bot_image,
                name=container_name,
                environment=env_vars,
                detach=True,
                remove=True,
                network_mode='host',
                shm_size='2g',
                mem_limit=f'{self.bot_container_memory}m',
                cpu_quota=int(self.bot_container_cpu) * 1000,  # Docker uses microseconds
                security_opt=['seccomp:unconfined'],
                cap_add=['SYS_ADMIN'],
                devices=['/dev/snd:/dev/snd'] if os.path.exists('/dev/snd') else None,
                volumes={
                    '/dev/shm': {'bind': '/dev/shm', 'mode': 'rw'},
                    '/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'rw'} if os.path.exists('/tmp/.X11-unix') else None,
                } if os.path.exists('/dev/shm') else None,
            )
            
            # Store container info
            self.active_bots[meeting_id] = {
                'deployment_method': 'docker',
                'container_id': container.id,
                'container_name': container_name,
                'session_id': session_id,
                'started_at': datetime.utcnow(),
                'platform': env_vars['PLATFORM'],
                'meeting_url': env_vars['MEETING_URL']
            }
            
            logger.info(f"✅ Docker bot launched successfully: {container_name} (ID: {container.id[:12]})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error launching Docker bot: {e}", exc_info=True)
            return False
    
    async def _launch_ecs_bot(
        self,
        meeting_id: str,
        session_id: str,
        env_vars: Dict[str, str]
    ) -> bool:
        """Launch browser bot using AWS ECS Fargate."""
        try:
            if not self.ecs_client:
                logger.debug("ECS client not available")
                return False
            
            logger.info(f"☁️  Launching ECS bot for meeting: {meeting_id}")
            
            # Prepare task definition overrides
            overrides = {
                'containerOverrides': [
                    {
                        'name': 'browser-bot',
                        'environment': [
                            {'name': k, 'value': v} for k, v in env_vars.items()
                        ]
                    }
                ]
            }
            
            # Get ECS configuration from settings
            cluster_name = getattr(settings, 'ecs_cluster_name', 'clerk-cluster')
            task_definition = getattr(settings, 'ecs_task_definition', 'clerk-browser-bot')
            subnet_ids = getattr(settings, 'ecs_subnet_ids', [])
            security_group_ids = getattr(settings, 'ecs_security_group_ids', [])
            
            # Convert comma-separated strings to lists if needed
            if isinstance(subnet_ids, str):
                subnet_ids = [s.strip() for s in subnet_ids.split(',') if s.strip()]
            if isinstance(security_group_ids, str):
                security_group_ids = [s.strip() for s in security_group_ids.split(',') if s.strip()]
            
            if not subnet_ids:
                logger.error("ECS subnet IDs not configured")
                return False
            
            # Run ECS task
            response = self.ecs_client.run_task(
                cluster=cluster_name,
                taskDefinition=task_definition,
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': subnet_ids,
                        'securityGroups': security_group_ids,
                        'assignPublicIp': 'ENABLED'  # Allow outbound connections
                    }
                },
                overrides=overrides,
                tags=[
                    {'key': 'MeetingId', 'value': meeting_id},
                    {'key': 'SessionId', 'value': session_id},
                    {'key': 'Platform', 'value': env_vars['PLATFORM']},
                    {'key': 'ManagedBy', 'value': 'clerk-orchestrator'}
                ]
            )
            
            if response.get('tasks') and len(response['tasks']) > 0:
                task = response['tasks'][0]
                task_arn = task['taskArn']
                
                # Store task info
                self.active_bots[meeting_id] = {
                    'deployment_method': 'ecs',
                    'task_arn': task_arn,
                    'task_id': task_arn.split('/')[-1],
                    'cluster': cluster_name,
                    'session_id': session_id,
                    'started_at': datetime.utcnow(),
                    'platform': env_vars['PLATFORM'],
                    'meeting_url': env_vars['MEETING_URL']
                }
                
                logger.info(f"✅ ECS bot launched successfully: {task_arn}")
                return True
            else:
                failures = response.get('failures', [])
                if failures:
                    error_msg = '; '.join([f.get('reason', 'Unknown error') for f in failures])
                    logger.error(f"❌ ECS task launch failed: {error_msg}")
                else:
                    logger.error("❌ No tasks created in ECS response")
                return False
                
        except ClientError as e:
            logger.error(f"❌ AWS ECS error: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"❌ Error launching ECS bot: {e}", exc_info=True)
            return False
    
    async def _launch_subprocess_bot(
        self,
        meeting_id: str,
        session_id: str,
        env_vars: Dict[str, str]
    ) -> bool:
        """
        Launch browser bot using subprocess (fallback method).
        
        This is the original method from webhooks.py, kept as a fallback option.
        """
        try:
            logger.info(f"📝 Launching subprocess bot for meeting: {meeting_id}")
            
            # Find bot entry script
            # __file__ would be: clerk_backend/services/meeting_agent/bot_orchestrator.py
            # We need to go up to root, then into browser_bot/
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            bot_path = os.path.join(base_dir, "browser_bot", "bot_entry.js")
            
            logger.info(f"Bot entry path: {bot_path}")
            
            if not os.path.exists(bot_path):
                logger.error(f"❌ Bot entry file not found at: {bot_path}")
                return False
            
            # Create log file path
            log_dir = os.path.join(base_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'bot_{meeting_id}_{session_id[:8]}.log')
            
            # Write startup info to log
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Bot process starting at {datetime.now().isoformat()}\n")
                f.write(f"Deployment method: subprocess\n")
                f.write(f"Meeting ID: {meeting_id}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Environment: {', '.join([f'{k}={v}' for k, v in env_vars.items()])}\n")
                f.write(f"{'='*80}\n")
            
            # Open log file for subprocess
            log_fh = open(log_file, 'a', buffering=1)
            
            try:
                process = subprocess.Popen(
                    ['node', bot_path],
                    env={**os.environ, **env_vars},
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(bot_path),
                    start_new_session=True,
                    close_fds=False,
                )
                
                # Write PID to log
                log_fh.write(f"Bot process PID: {process.pid}\n")
                log_fh.flush()
                
                logger.info(f"✅ Subprocess bot launched with PID: {process.pid}, log: {log_file}")
                
                # Store process info
                self.active_bots[meeting_id] = {
                    'deployment_method': 'subprocess',
                    'process_id': process.pid,
                    'log_file': log_file,
                    'session_id': session_id,
                    'started_at': datetime.utcnow(),
                    'platform': env_vars['PLATFORM'],
                    'meeting_url': env_vars['MEETING_URL'],
                    'log_handle': log_fh  # Keep reference to prevent premature closing
                }
                
                return True
                
            except Exception as e:
                log_fh.close()
                raise
                
        except Exception as e:
            logger.error(f"❌ Error launching subprocess bot: {e}", exc_info=True)
            return False
    
    async def stop_bot(self, meeting_id: str) -> bool:
        """Stop a browser bot."""
        try:
            logger.info(f"🛑 Stopping browser bot for meeting: {meeting_id}")
            
            if meeting_id not in self.active_bots:
                logger.warning(f"No active bot found for meeting: {meeting_id}")
                return False
            
            bot_info = self.active_bots[meeting_id]
            deployment_method = bot_info.get('deployment_method')
            success = False
            
            if deployment_method == 'docker' and 'container_id' in bot_info:
                success = await self._stop_docker_bot(bot_info)
            
            elif deployment_method == 'ecs' and 'task_arn' in bot_info:
                success = await self._stop_ecs_bot(bot_info)
            
            elif deployment_method == 'subprocess' and 'process_id' in bot_info:
                success = await self._stop_subprocess_bot(bot_info)
            
            if success:
                # Remove from active bots
                del self.active_bots[meeting_id]
                logger.info(f"✅ Bot stopped successfully for meeting: {meeting_id}")
            else:
                logger.warning(f"⚠️  Failed to stop bot for meeting: {meeting_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error stopping browser bot: {e}", exc_info=True)
            return False
    
    async def _stop_docker_bot(self, bot_info: Dict[str, Any]) -> bool:
        """Stop a Docker bot container."""
        try:
            if not self.docker_client:
                return False
            
            container = self.docker_client.containers.get(bot_info['container_id'])
            container.stop(timeout=10)
            logger.info(f"✅ Docker bot stopped: {bot_info.get('container_name', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping Docker bot: {e}")
            return False
    
    async def _stop_ecs_bot(self, bot_info: Dict[str, Any]) -> bool:
        """Stop an ECS bot task."""
        try:
            if not self.ecs_client:
                return False
            
            self.ecs_client.stop_task(
                cluster=bot_info.get('cluster', 'clerk-cluster'),
                task=bot_info['task_arn'],
                reason='Meeting ended - stopped by orchestrator'
            )
            logger.info(f"✅ ECS bot stopped: {bot_info.get('task_arn', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping ECS bot: {e}")
            return False
    
    async def _stop_subprocess_bot(self, bot_info: Dict[str, Any]) -> bool:
        """Stop a subprocess bot."""
        try:
            import psutil
            process = psutil.Process(bot_info['process_id'])
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                process.kill()
            
            # Close log file handle if it exists
            log_handle = bot_info.get('log_handle')
            if log_handle and not log_handle.closed:
                log_handle.close()
            
            logger.info(f"✅ Subprocess bot stopped: PID {bot_info['process_id']}")
            return True
            
        except psutil.NoSuchProcess:
            logger.warning(f"Process {bot_info['process_id']} already terminated")
            return True
        except ImportError:
            logger.warning("psutil not available, using os.kill")
            import signal
            import os
            try:
                os.kill(bot_info['process_id'], signal.SIGTERM)
                return True
            except ProcessLookupError:
                return True
        except Exception as e:
            logger.error(f"❌ Error stopping subprocess bot: {e}")
            return False
    
    async def get_active_bots(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active bots."""
        return self.active_bots.copy()
    
    async def get_bot_status(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific bot."""
        return self.active_bots.get(meeting_id)
    
    async def cleanup_all_bots(self) -> None:
        """Cleanup all active bots."""
        logger.info(f"🧹 Cleaning up {len(self.active_bots)} active bots...")
        
        meeting_ids = list(self.active_bots.keys())
        for meeting_id in meeting_ids:
            await self.stop_bot(meeting_id)
        
        logger.info("✅ Bot cleanup completed")


# Singleton instance
_orchestrator_instance: Optional[BrowserBotOrchestrator] = None


def get_bot_orchestrator() -> BrowserBotOrchestrator:
    """Get or create the bot orchestrator singleton instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = BrowserBotOrchestrator()
    return _orchestrator_instance
