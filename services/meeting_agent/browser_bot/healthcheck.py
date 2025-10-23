#!/usr/bin/env python3
"""
Health check script for the browser bot
"""

import os
import sys
import asyncio
import aiohttp
from datetime import datetime

async def health_check():
    """Perform health check by pinging the backend API"""
    try:
        api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
        api_url = f"{api_base_url}/api/v1/meetings/bot-status"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print(f"[{datetime.now()}] Health check passed")
                    return True
                else:
                    print(f"[{datetime.now()}] Health check failed: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"[{datetime.now()}] Health check error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(health_check())
    sys.exit(0 if success else 1)
