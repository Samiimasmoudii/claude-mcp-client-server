"""
Tool execution permission system.
Prompts the user before running any tool that hasn't been pre-approved.
"""
import json
from core import ui


class PermissionManager:
    def __init__(self, auto_approve: bool = False):
        self._always: set[str] = set()
        self.auto_approve = auto_approve  # --yes flag bypasses all prompts

    def request(self, tool_name: str, tool_input: dict) -> bool:
        """Return True if execution should proceed, False to deny."""
        if self.auto_approve or tool_name in self._always:
            return True

        ui.print_tool_call(tool_name, tool_input)
        while True:
            try:
                raw = input("  Allow? [y] once  [a] always  [n] deny: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False

            if raw in ("y", "yes", ""):
                return True
            if raw == "a":
                self._always.add(tool_name)
                ui.print_info(f"'{tool_name}' added to always-allow list for this session.")
                return True
            if raw in ("n", "no"):
                return False

            ui.print_warning("Enter y, a, or n.")
