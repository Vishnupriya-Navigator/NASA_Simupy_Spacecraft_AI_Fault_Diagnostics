# scripts/list_simupy_cases.py
"""
Probe your installed simupy_flight for likely NESC/example case builder functions.
This helps us pick a real case and wire the adapter without guesswork.
"""
import pkgutil, inspect, importlib, sys


def iter_modules(prefix):
    try:
        mod = importlib.import_module(prefix)
    except Exception as e:
        print(f"[skip] cannot import {prefix}: {e}")
        return
    if hasattr(mod, "__path__"):
        for m in pkgutil.walk_packages(mod.__path__, mod.__name__ + "."):
            yield m.name


def looks_like_builder(name, obj):
    # Heuristic: functions that return a model or build a 'case'
    if not inspect.isfunction(obj):
        return False
    lname = name.lower()
    return any(k in lname for k in ("build", "case", "example"))


def main():
    print("=== Scanning simupy_flight modules for case builders ===")
    found = []
    for modname in iter_modules("simupy_flight"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod):
            if looks_like_builder(name, obj):
                sig = None
                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    sig = "(signature unknown)"
                path = f"{modname}.{name}"
                print(f"[builder] {path} {sig}")
                found.append(path)

    if not found:
        print("No obvious builders found. We can inspect specific submodules next.")
    else:
        print("\nPick one builder path from above and we will wire the adapter to it.")
    print(
        "\nTip: If you recognize a known NESC case module, tell me its path directly."
    )


if __name__ == "__main__":
    main()
