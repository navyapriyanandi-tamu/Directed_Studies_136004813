"""
Validators Module
Input validation functions
"""

import re
from typing import Optional


def validate_priority(priority: str) -> bool:
    """
    Validate task priority

    Args:
        priority: Priority string to validate

    Returns:
        True if valid, False otherwise
    """
    valid_priorities = ['low', 'medium', 'high']
    return priority.lower() in valid_priorities


def validate_task_title(title: str) -> bool:
    """
    Validate task title

    Args:
        title: Task title to validate

    Returns:
        True if valid, False otherwise
    """
    if not title or len(title.strip()) == 0:
        return False

    if len(title) > 200:
        return False

    return True


def validate_email(email: str) -> bool:
    """
    Validate email format

    Args:
        email: Email address to validate

    Returns:
        True if valid email format, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


# DEAD CODE - These functions are defined but never called
def validate_phone_number(phone: str) -> bool:
    """
    Validate phone number (US format)
    Was created for future user profile feature
    """
    pattern = r'^\+?1?\d{10}$'
    return bool(re.match(pattern, phone.replace('-', '').replace(' ', '')))


def sanitize_input(text: str) -> str:
    """
    Sanitize user input
    Planned for security but never integrated
    """
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', ';', '&', '|']
    for char in dangerous_chars:
        text = text.replace(char, '')
    return text.strip()
