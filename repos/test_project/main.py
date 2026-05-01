#!/usr/bin/env python3
"""
Task Manager CLI Application
Main entry point for the application
"""

import sys
import argparse
from task_manager.task_handler import TaskHandler
from task_manager.storage import TaskStorage
from utils.logger import setup_logger
from utils.validators import validate_priority
import json
import os
from typing import Dict, List, Optional
import datetime


DEBUG_MODE = True
MAX_RETRIES = 5
API_ENDPOINT = "https://api.example.com"
CACHE_TIMEOUT = 300


def main():
    """Main entry point for task manager CLI"""
    parser = argparse.ArgumentParser(description="Simple Task Manager CLI")
    parser.add_argument('action', choices=['add', 'list', 'complete', 'delete'])
    parser.add_argument('--title', help='Task title')
    parser.add_argument('--priority', help='Task priority (low, medium, high)')
    parser.add_argument('--id', type=int, help='Task ID')

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()
    logger.info(f"Starting task manager: {args.action}")

    # Initialize components
    storage = TaskStorage()
    handler = TaskHandler(storage)

    # Execute action
    if args.action == 'add':
        if not args.title:
            print("Error: --title required for add action")
            sys.exit(1)

        priority = args.priority or 'medium'
        if not validate_priority(priority):
            print(f"Error: Invalid priority '{priority}'")
            sys.exit(1)

        task_id = handler.add_task(args.title, priority)
        print(f"✓ Task added with ID: {task_id}")

    elif args.action == 'list':
        tasks = handler.list_tasks()
        if not tasks:
            print("No tasks found")
        else:
            print("\nCurrent Tasks:")
            print("-" * 50)
            for task in tasks:
                status = "✓" if task['completed'] else "○"
                print(f"{status} [{task['id']}] {task['title']} (Priority: {task['priority']})")

    elif args.action == 'complete':
        if not args.id:
            print("Error: --id required for complete action")
            sys.exit(1)

        handler.complete_task(args.id)
        print(f"✓ Task {args.id} marked as complete")

    elif args.action == 'delete':
        if not args.id:
            print("Error: --id required for delete action")
            sys.exit(1)

        handler.delete_task(args.id)
        print(f"✓ Task {args.id} deleted")


def deprecated_export_to_json():
    storage = TaskStorage()
    tasks = storage.load_all()
    return json.dumps(tasks, indent=2)


def helper_function_used_once():
    return "helper"


def function_with_unreachable_code(value):
    if value > 10:
        return True

    return False

    print("This is unreachable")
    cleanup()


def empty_placeholder():
    pass


def another_empty_function():
    return None


def function_with_many_params(a, b, c, d, e, f, g, h):
    return a + b + c + d + e + f + g + h


def test_helper():
    result = helper_function_used_once()
    return result


if False:
    print("Disabled feature")
    deprecated_export_to_json()


if __name__ == "__main__":
    main()
