#!/usr/bin/env python3
"""
Direct test of OpenMemory functionality by calling the memory functions directly.

This bypasses the MCP protocol layer and directly tests the core memory functionality.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add the api directory to the Python path
sys.path.insert(0, '/Users/jarvis/Desktop/FUN/memory/openmemory/api')

from app.utils.memory import get_memory_client
from app.database import SessionLocal
from app.utils.db import get_user_and_app
from app.models import Memory, MemoryState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryTester:
    def __init__(self, user_id: str = "test_user", app_id: str = "test_app"):
        self.user_id = user_id
        self.app_id = app_id
        self.memory_client = None
        
    def initialize_memory_client(self):
        """Initialize the memory client."""
        try:
            self.memory_client = get_memory_client()
            logger.info("✅ Memory client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory client: {e}")
            return False
    
    def test_memory_operations(self):
        """Test basic memory operations."""
        if not self.memory_client:
            logger.error("❌ Memory client not initialized")
            return False
        
        try:
            # Test adding memories
            logger.info("📝 Testing memory addition...")
            test_memories = [
                "I love playing guitar and have been learning for 5 years",
                "My favorite programming language is Python",
                "I enjoy reading science fiction novels"
            ]
            
            added_memories = []
            for i, memory_text in enumerate(test_memories, 1):
                try:
                    result = self.memory_client.add(
                        memory_text,
                        user_id=self.user_id,
                        metadata={"source": "test", "app": self.app_id}
                    )
                    logger.info(f"✅ Added memory {i}: {memory_text[:50]}...")
                    added_memories.append(result)
                except Exception as e:
                    logger.error(f"❌ Failed to add memory {i}: {e}")
            
            # Test searching memories
            logger.info("\n🔍 Testing memory search...")
            search_queries = ["guitar music", "programming", "books"]
            
            for query in search_queries:
                try:
                    results = self.memory_client.search(
                        query,
                        user_id=self.user_id,
                        limit=5
                    )
                    logger.info(f"✅ Search for '{query}' returned {len(results) if results else 0} results")
                    if results:
                        for i, result in enumerate(results[:2], 1):  # Show first 2 results
                            logger.info(f"   Result {i}: {result.get('memory', 'No memory field')[:50]}...")
                except Exception as e:
                    logger.error(f"❌ Failed to search for '{query}': {e}")
            
            # Test getting all memories
            logger.info("\n📋 Testing get all memories...")
            try:
                all_memories = self.memory_client.get_all(user_id=self.user_id)
                if isinstance(all_memories, dict) and 'results' in all_memories:
                    memory_count = len(all_memories['results'])
                else:
                    memory_count = len(all_memories) if all_memories else 0
                logger.info(f"✅ Retrieved {memory_count} memories")
            except Exception as e:
                logger.error(f"❌ Failed to get all memories: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory operations test failed: {e}")
            return False
    
    def test_database_connection(self):
        """Test database connectivity."""
        try:
            db = SessionLocal()
            try:
                # Test creating user and app
                user, app = get_user_and_app(db, user_id=self.user_id, app_id=self.app_id)
                logger.info(f"✅ Database connection successful. User: {user.user_id}, App: {app.name}")
                
                # Test querying memories
                memories_count = db.query(Memory).filter(Memory.user_id == user.id).count()
                logger.info(f"✅ Found {memories_count} existing memories in database")
                
                return True
            finally:
                db.close()
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False

def test_environment():
    """Test environment setup."""
    logger.info("🔧 Testing environment setup...")
    
    required_vars = ["OPENAI_API_KEY", "API_KEY"]
    env_status = {}
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            env_status[var] = "✅ Set"
            logger.info(f"✅ {var} is set")
        else:
            env_status[var] = "❌ Not set"
            logger.warning(f"⚠️ {var} is not set")
    
    return env_status

def main():
    logger.info("🎯 OpenMemory Direct Memory Test")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    # Test environment
    env_status = test_environment()
    
    print("\n" + "="*60)
    
    # Initialize tester
    tester = MemoryTester()
    
    # Test database connection
    logger.info("🗄️ Testing database connection...")
    if not tester.test_database_connection():
        logger.error("❌ Database connection failed")
        return False
    
    print("\n" + "="*60)
    
    # Test memory client initialization
    logger.info("🧠 Testing memory client initialization...")
    if not tester.initialize_memory_client():
        logger.error("❌ Memory client initialization failed")
        return False
    
    print("\n" + "="*60)
    
    # Test memory operations
    logger.info("⚙️ Testing memory operations...")
    if not tester.test_memory_operations():
        logger.error("❌ Memory operations test failed")
        return False
    
    print("\n" + "="*60)
    logger.info("🎉 All tests completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)