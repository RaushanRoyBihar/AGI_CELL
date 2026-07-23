"""Adapter configuration loader. No PyYAML dependency in this environment,
so this is a small parser for the flat/one-level-nested subset of YAML
that config/adapters.yaml actually uses (scalars + one level of nested
mappings, no lists, `#` comments). If PyYAML is available in a real
deployment, swap `_parse_yaml_subset` for `yaml.safe_load` — the rest of
this module (the `AdapterConfig` shape and `load_adapter_config`) does not
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _parse_yaml_subset(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        value = value.strip().strip('"')
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            new_map: dict = {}
            parent[key] = new_map
            stack.append((indent, new_map))
        else:
            parent[key] = value
    return root


@dataclass
class AdapterConfig:
    associative: str = "local"
    graph: str = "sqlite"
    telemetry: str = "sqlite"
    artifacts: str = "local_file"
    fleet_sync: str = "local_sqlite"
    raw: dict = field(default_factory=dict)

    def is_minimal_deployment(self) -> bool:
        return (self.associative == "local" and self.graph == "sqlite"
                and self.telemetry == "sqlite" and self.artifacts == "local_file"
                and self.fleet_sync == "local_sqlite")


def load_adapter_config(path: str | None = None) -> AdapterConfig:
    if path is None:
        return AdapterConfig()
    with open(path, "r") as fh:
        parsed = _parse_yaml_subset(fh.read())
    return AdapterConfig(
        associative=parsed.get("associative", "local"),
        graph=parsed.get("graph", "sqlite"),
        telemetry=parsed.get("telemetry", "sqlite"),
        artifacts=parsed.get("artifacts", "local_file"),
        fleet_sync=parsed.get("fleet_sync", "local_sqlite"),
        raw=parsed,
    )
