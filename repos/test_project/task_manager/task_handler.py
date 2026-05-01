"""
Task Handler Module
Handles all task-related operations (add, update, delete, complete)
"""

from typing import List, Dict, Optional
from datetime import datetime
from .storage import TaskStorage
from utils.validators import validate_task_title


class Task:
    """Represents a single task"""

    def __init__(self, task_id: int, title: str, priority: str = 'medium'):
        self.id = task_id
        self.title = title
        self.priority = priority
        self.completed = False
        self.created_at = datetime.now()
        self.completed_at = None

    def to_dict(self) -> Dict:
        """Convert task to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'priority': self.priority,
            'completed': self.completed,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

    def mark_complete(self):
        """Mark task as completed"""
        self.completed = True
        self.completed_at = datetime.now()


class TaskHandler:
    """Handles task operations"""

    def __init__(self, storage: TaskStorage):
        self.storage = storage
        # Calculate next ID based on existing tasks
        existing_tasks = self.storage.load_all()
        if existing_tasks:
            self._next_id = max(task['id'] for task in existing_tasks) + 1
        else:
            self._next_id = 1

    def add_task(self, title: str, priority: str = 'medium') -> int:
        """Add a new task"""
        if not validate_task_title(title):
            raise ValueError("Invalid task title")

        task = Task(self._next_id, title, priority)
        self.storage.save(task)
        self._next_id += 1
        return task.id

    def list_tasks(self, include_completed: bool = True) -> List[Dict]:
        """List all tasks"""
        tasks = self.storage.load_all()

        if not include_completed:
            tasks = [t for t in tasks if not t['completed']]

        return sorted(tasks, key=lambda x: x['id'])

    def complete_task(self, task_id: int) -> bool:
        """Mark a task as completed"""
        task = self.storage.load_by_id(task_id)
        if task:
            task.mark_complete()
            self.storage.update(task)
            return True
        return False

    def delete_task(self, task_id: int) -> bool:
        """Delete a task"""
        return self.storage.delete(task_id)

    def get_task_by_priority(self, priority: str) -> List[Dict]:
        """Get tasks filtered by priority"""
        all_tasks = self.list_tasks()
        return [t for t in all_tasks if t['priority'] == priority]

    # DEAD CODE - Never called
    def archive_completed_tasks(self):
        """
        This function was meant to archive old completed tasks
        but the feature was never finished
        """
        completed = [t for t in self.list_tasks() if t['completed']]
        # TODO: Implement archival logic
        pass
