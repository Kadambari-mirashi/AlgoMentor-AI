"""
testcase_tool.py
Tool: generate_test_cases

Produces 3-5 sample and edge-case test descriptions for a given DSA problem.
Used by the Evaluator Agent to remind the student of cases they should handle.
"""

TOOL_METADATA = {
    "type": "function",
    "function": {
        "name": "generate_test_cases",
        "description": (
            "Generate 3-5 sample and edge test-case descriptions for a "
            "DSA problem, based on its topic, title, and statement."
        ),
        "parameters": {
            "type": "object",
            "required": ["problem"],
            "properties": {
                "problem": {
                    "type": "object",
                    "description": "A problem dict with at least 'id', 'title', and 'topic'.",
                },
            },
        },
    },
}

_CASES_BY_ID: dict[str, list[str]] = {
    "two-sum": [
        "nums = [2, 7, 11, 15], target = 9  →  [0, 1]",
        "nums = [3, 2, 4], target = 6  →  [1, 2]",
        "nums = [-1, -2, -3, -4, -5], target = -8  →  [2, 4]",
        "Edge: single pair  nums = [1, 2], target = 3  →  [0, 1]",
        "Edge: negative + positive  nums = [-3, 4, 3, 90], target = 0  →  [0, 2]",
    ],
    "valid-anagram": [
        's = "anagram", t = "nagaram"  →  True',
        's = "rat", t = "car"  →  False',
        's = "", t = ""  →  True  (empty strings)',
        's = "a", t = "a"  →  True  (single char)',
        'Edge: different lengths  s = "ab", t = "abc"  →  False',
    ],
    "binary-search": [
        "nums = [-1, 0, 3, 5, 9, 12], target = 9  →  4",
        "nums = [-1, 0, 3, 5, 9, 12], target = 2  →  -1",
        "nums = [5], target = 5  →  0  (single element found)",
        "Edge: empty array  nums = [], target = 1  →  -1",
        "Edge: target at boundaries  nums = [1, 2, 3], target = 1 or 3",
    ],
    "best-time-to-buy-sell-stock": [
        "prices = [7, 1, 5, 3, 6, 4]  →  5  (buy@1, sell@6)",
        "prices = [7, 6, 4, 3, 1]  →  0  (no profit possible)",
        "prices = [1, 2]  →  1",
        "Edge: single day  prices = [5]  →  0",
        "Edge: constant prices  prices = [3, 3, 3]  →  0",
    ],
    "merge-two-sorted-lists": [
        "list1 = [1,2,4], list2 = [1,3,4]  →  [1,1,2,3,4,4]",
        "list1 = [], list2 = []  →  []",
        "list1 = [], list2 = [0]  →  [0]",
        "Edge: one list much longer than the other",
        "Edge: overlapping duplicate values",
    ],
    "max-depth-binary-tree": [
        "root = [3,9,20,null,null,15,7]  →  3",
        "root = [1,null,2]  →  2",
        "Edge: empty tree  root = []  →  0",
        "Edge: single node  root = [1]  →  1",
        "Edge: completely left-skewed tree (like a linked list)",
    ],
    "number-of-islands": [
        'grid = [["1","1","0"],["1","1","0"],["0","0","1"]]  →  2',
        'grid = [["1","0","1"],["0","0","0"],["1","0","1"]]  →  4',
        "Edge: entire grid is water  →  0",
        "Edge: entire grid is land  →  1",
        'Edge: single cell  grid = [["1"]]  →  1',
    ],
    "three-sum": [
        "nums = [-1, 0, 1, 2, -1, -4]  →  [[-1,-1,2],[-1,0,1]]",
        "nums = [0, 1, 1]  →  []",
        "nums = [0, 0, 0]  →  [[0,0,0]]",
        "Edge: array with fewer than 3 elements  →  []",
        "Edge: many duplicates require proper skipping",
    ],
    "coin-change": [
        "coins = [1, 5, 10, 25], amount = 30  →  2  (25+5)",
        "coins = [2], amount = 3  →  -1  (impossible)",
        "coins = [1], amount = 0  →  0",
        "Edge: amount = 0  →  0 regardless of coins",
        "Edge: single large coin exceeding amount",
    ],
    "lru-cache": [
        "capacity=2: put(1,1), put(2,2), get(1)→1, put(3,3), get(2)→-1",
        "capacity=1: put(1,1), put(2,2), get(1)→-1, get(2)→2",
        "Edge: update existing key value",
        "Edge: get on empty cache  →  -1",
        "Edge: capacity = 0 (if allowed by constraints)",
    ],
}

_GENERIC_CASES: list[str] = [
    "Normal / happy-path input",
    "Empty or minimal input",
    "Large input to stress-test performance",
    "Input with duplicates",
    "Negative numbers or boundary values",
]


def generate_test_cases(problem: dict) -> dict:
    """Return sample and edge test cases for a problem.

    Parameters
    ----------
    problem : dict
        A problem entry from the knowledge base (must contain ``id``).

    Returns
    -------
    dict  with keys ``problem_id``, ``title``, and ``test_cases``.
    """
    pid = problem.get("id", "")
    cases = _CASES_BY_ID.get(pid, _GENERIC_CASES)

    return {
        "problem_id": pid,
        "title": problem.get("title", "Unknown"),
        "test_cases": cases,
    }
