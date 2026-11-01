"""Validate the published example-admission metadata, not live environment values."""

import math
import re


def validate_contract(contract):
    def require(condition):
        if not condition:
            raise ValueError("invalid environment contract")

    require(isinstance(contract, dict) and set(contract) == {"schema", "settings"})
    require(contract["schema"] == "bridge-env-example-contract/v1")
    settings = contract["settings"]
    require(isinstance(settings, dict) and bool(settings))
    types = {"credential_path", "string", "secret", "boolean", "integer", "number",
             "enum", "csv", "json", "url", "path", "budget"}
    for name, spec in settings.items():
        require(isinstance(name, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is not None)
        require(isinstance(spec, dict) and spec.get("type") in types)
        kind = spec["type"]
        allowed = {"type"}
        if kind in {"integer", "number"}:
            allowed |= {"min", "max", "minimum_key"}
            for key in ("min", "max"):
                if key in spec:
                    value = spec[key]
                    require(type(value) in (int, float) and math.isfinite(value))
                    if kind == "integer":
                        require(type(value) is int)
            if "min" in spec and "max" in spec:
                require(spec["min"] <= spec["max"])
            if "minimum_key" in spec:
                ref = spec["minimum_key"]
                require(isinstance(ref, str) and ref != name and ref in settings)
                require(isinstance(settings[ref], dict) and settings[ref].get("type") == kind)
        if kind == "enum":
            allowed |= {"values", "case_insensitive"}
            values = spec.get("values")
            require(isinstance(values, list) and bool(values))
            require(all(isinstance(v, str) and v for v in values))
            require(len(values) == len(set(values)))
            if "case_insensitive" in spec:
                require(type(spec["case_insensitive"]) is bool)
                if spec["case_insensitive"]:
                    require(len({v.casefold() for v in values}) == len(values))
        if kind == "csv":
            allowed.add("max_items")
            if "max_items" in spec:
                require(type(spec["max_items"]) is int and spec["max_items"] > 0)
        require(set(spec) <= allowed)
