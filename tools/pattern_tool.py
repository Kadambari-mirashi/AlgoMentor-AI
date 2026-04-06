"""
pattern_tool.py
Tool: get_pattern_hint

Returns common algorithmic problem-solving patterns for a given topic
or list of tags.  Used by the Mentor Agent to give pattern-level guidance.
"""

TOOL_METADATA = {
    "type": "function",
    "function": {
        "name": "get_pattern_hint",
        "description": (
            "Return common DSA problem-solving patterns for a topic or tag list. "
            "Helps the student connect the current problem to well-known strategies."
        ),
        "parameters": {
            "type": "object",
            "required": ["topic_or_tags"],
            "properties": {
                "topic_or_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of topic names or problem tags.",
                },
            },
        },
    },
}

_PATTERN_MAP: dict[str, list[str]] = {
    "arrays": [
        "Two Pointer technique",
        "Sliding Window",
        "Prefix Sum / Running total",
        "Sort-then-scan",
    ],
    "hashing": [
        "Hash Map for O(1) lookup",
        "Frequency counting",
        "Two-pass hash approach",
    ],
    "binary search": [
        "Standard binary search on sorted data",
        "Binary search on the answer space",
        "Bisect left / bisect right variants",
    ],
    "trees": [
        "DFS — preorder, inorder, postorder",
        "BFS / Level-order traversal",
        "Recursion with return values",
    ],
    "graphs": [
        "BFS for shortest path in unweighted graphs",
        "DFS for connected-component detection",
        "Union-Find / Disjoint Set Union",
        "Topological Sort for DAGs",
    ],
    "dynamic programming": [
        "Top-down memoization",
        "Bottom-up tabulation",
        "State reduction to lower dimensions",
    ],
    "linked lists": [
        "Fast / Slow pointer (Floyd's cycle)",
        "Dummy head node for simpler edge cases",
        "In-place reversal",
    ],
    "sorting": [
        "Merge Sort (stable, O(n log n))",
        "Quick Sort (in-place, average O(n log n))",
        "Counting / Bucket Sort for bounded ranges",
    ],
    "two pointer": [
        "Opposite-end pointers on sorted data",
        "Same-direction fast/slow pointers",
    ],
    "sliding window": [
        "Fixed-size window with running aggregate",
        "Variable-size window with expand/contract",
    ],
    "recursion": [
        "Divide and conquer",
        "Backtracking with pruning",
    ],
    "design": [
        "Combine data structures (hash map + linked list)",
        "Amortized analysis for operations",
    ],
    "greedy": [
        "Local optimum leads to global optimum",
        "Sorting + greedy selection",
    ],
}


def get_pattern_hint(topic_or_tags) -> dict:
    """Return pattern hints relevant to the given topics or tags.

    Parameters
    ----------
    topic_or_tags : str or list[str]
        A single topic string or a list of tags / topic names.

    Returns
    -------
    dict  with keys ``query`` and ``patterns``.
    """
    if isinstance(topic_or_tags, str):
        keys = [topic_or_tags]
    else:
        keys = list(topic_or_tags)

    normalized = [k.lower().strip() for k in keys]

    patterns: list[str] = []
    for token in normalized:
        for map_key, map_patterns in _PATTERN_MAP.items():
            if token in map_key or map_key in token:
                patterns.extend(map_patterns)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if not unique:
        unique = [
            "No specific patterns matched. "
            "Try breaking the problem into smaller sub-problems."
        ]

    return {"query": topic_or_tags, "patterns": unique}
