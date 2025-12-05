"""
TreePO algorithm: Tree-based Policy Optimization.
"""

import re
from collections import deque

from .sample_schema import TreePOSampleSchema

__all__ = [
    "TreePOSampleSchema",
    "TreeNode",
    "get_ancestors",
    "backpropagation",
    "extract_boxed_answer",
]


def extract_boxed_answer(solution_text):
    """
    Extract the answer from \\boxed{...} pattern in solution text.
    """
    match = re.search(r"\\boxed\{(.*?)\}", solution_text)
    if match:
        return match.group(1)
    return None


class TreeNode:
    """Node in the TreePO search tree."""

    def __init__(self, item, budget, parent_depth=-1):
        self.budget = budget
        self.endurance = 3
        self.item = item
        self.children = []
        self.children_rewards = []
        self.parent = None
        self.birth_order = 0
        self.child_count = 0
        self.depth = parent_depth + 1

    def add_child(self, child_node):
        self.budget -= 1
        child_node.birth_order = self.child_count
        self.child_count += 1
        self.children.append(child_node)


def get_ancestors(node):
    """Get all ancestor nodes of a node."""
    ancestors = []
    current = node
    while hasattr(current, "parent") and current.parent is not None:
        current = current.parent
        ancestors.append(current)
    return ancestors


def backpropagation(node):
    """Backpropagate reward through the tree."""
    current = node
    reward = node.item["reward"]["total"]
    while hasattr(current, "parent") and current.parent is not None:
        current = current.parent
        current.children_rewards.append(reward)
    return

