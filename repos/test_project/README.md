# Task Manager CLI

A simple command-line task management application built with Python.

## Features

- Add tasks with priority levels
- List all tasks
- Mark tasks as complete
- Delete tasks

## Usage

```bash
# Add a task
python main.py add --title "Buy groceries" --priority high

# List all tasks
python main.py list

# Complete a task
python main.py complete --id 1

# Delete a task
python main.py delete --id 1
```

## Project Structure

```
test_project/
├── main.py                      # Entry point
├── task_manager/
│   ├── __init__.py
│   ├── task_handler.py         # Task operations
│   └── storage.py              # Data persistence
└── utils/
    ├── __init__.py
    ├── logger.py               # Logging utilities
    └── validators.py           # Input validation
```

## Testing the AST Parser

This project is designed to test various AST parsing features:

1. **Multiple modules** - 6 Python files across 2 packages
2. **Functions and classes** - Various function signatures and class methods
3. **Imports** - Mix of used and unused imports
4. **Dead code** - Intentionally unused functions and classes
5. **Decorators** - None in this simple version, but structure supports it
6. **Entry points** - `main.py` has `if __name__ == "__main__"`

## Known Dead Code (for testing)

- `main.py::deprecated_export_to_json()` - Never called
- `task_handler.py::TaskHandler.archive_completed_tasks()` - Never called
- `storage.py::DatabaseStorage` - Entire class never used
- `logger.py::log_to_file()` - Never called
- `validators.py::validate_phone_number()` - Never called
- `validators.py::sanitize_input()` - Never called
- `main.py` - Unused import: `json`
