from __future__ import annotations

from pathlib import Path


class PromptStore:
    def __init__(self, prompt_paths: dict[str, str], root: str | Path = ".") -> None:
        self.root = Path(root)
        self.prompt_paths = prompt_paths
        self._cache: dict[str, str] = {}

    def render(self, name: str, **values: str) -> str:
        template = self.get(name)
        # Prompt files contain JSON examples with literal braces, so replace only
        # known placeholders instead of calling str.format on the full template.
        for key, value in values.items():
            template = template.replace("{" + key + "}", value or "")
        return template

    def get(self, name: str) -> str:
        if name not in self._cache:
            path = self.root / self.prompt_paths[name]
            self._cache[name] = path.read_text(encoding="utf-8")
        return self._cache[name]
