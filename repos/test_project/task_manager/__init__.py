"""
Task Manager Package
Provides task management functionality
"""

__version__ = "0.1.0"
__author__ = "Test Project"

from .task_handler import TaskHandler
from .storage import TaskStorage

__all__ = ['TaskHandler', 'TaskStorage']
