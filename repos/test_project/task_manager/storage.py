"""
Storage Module
Handles data persistence for tasks
"""

import os
import pickle
from typing import List, Dict, Optional
from pathlib import Path


class TaskStorage:
    """Handles task storage and retrieval"""

    def __init__(self, storage_path: str = ".tasks.pkl"):
        self.storage_path = Path(storage_path)
        self.tasks = self._load_from_disk()

    def save(self, task) -> None:
        """Save a task"""
        self.tasks[task.id] = task
        self._persist_to_disk()

    def load_all(self) -> List[Dict]:
        """Load all tasks"""
        return [task.to_dict() for task in self.tasks.values()]

    def load_by_id(self, task_id: int):
        """Load a specific task by ID"""
        return self.tasks.get(task_id)

    def update(self, task) -> None:
        """Update an existing task"""
        if task.id in self.tasks:
            self.tasks[task.id] = task
            self._persist_to_disk()

    def delete(self, task_id: int) -> bool:
        """Delete a task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._persist_to_disk()
            return True
        return False

    def _load_from_disk(self) -> Dict:
        """Load tasks from disk"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading tasks: {e}")
                return {}
        return {}

    def _persist_to_disk(self) -> None:
        """Save tasks to disk"""
        try:
            with open(self.storage_path, 'wb') as f:
                pickle.dump(self.tasks, f)
        except Exception as e:
            print(f"Error saving tasks: {e}")

    def clear_all(self) -> None:
        """Clear all tasks (for testing)"""
        self.tasks = {}
        self._persist_to_disk()


# DEAD CODE - This class is defined but never instantiated
class DatabaseStorage:
    """
    Alternative storage backend using SQLite
    This was planned but never implemented
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def connect(self):
        """Connect to database"""
        pass

    def execute_query(self, query: str):
        """Execute SQL query"""
        pass
