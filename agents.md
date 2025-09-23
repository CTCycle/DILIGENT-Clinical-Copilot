# INSTRUCTIONS

You will assist me in modifying my code according to the given task.

# RULES

Always follow **PEP 8** guidelines and check online to stay up-to-date.

Apply the following rules when modifying code:

## Type Hinting

1. Use modern type hints (e.g., `list`, `dict`, `tuple`) instead of `List`, `Dict`, `Tuple`.
2. Replace deprecated constructs like `Optional` and `Union` with the `|` operator (e.g., `str | None`).
3. Always import `Callable` from `collections.abc`.

## Comments and Docstrings

1. Add comments that are concise, clear, and minimal.
2. Use separator comments (e.g., `# ----`) consistent with the project’s layout.
3. Do not write docstrings unless explicitly requested.  
   When required, follow this structure:
   - Brief and clear description  
   - Keyword arguments  
   - Return value  

## Code Structure

1. Maintain the overall project structure by reusing existing classes and workflows whenever possible.
2. Wrap loose functions inside classes when reasonable.
3. Do not define functions inside other functions.
4. Avoid unnecessary fragmentation — do not split logic into overly small functions.
5. Always place imports at the top of the module, never inside a function or class.