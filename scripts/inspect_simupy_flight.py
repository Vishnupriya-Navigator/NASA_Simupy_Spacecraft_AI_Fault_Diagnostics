# scripts/inspect_simupy_flight.py
import pkgutil, inspect, importlib, sys, re

TARGET_PACKAGE = "simupy_flight"

KEY_FN_TOKENS = ("build", "case", "example", "model")
KEY_SIG_TOKENS = ("p", "q", "r", "q0", "q1", "q2", "q3", "att", "quat", "rates")
MAX_DOC_LEN = 160


def walk_modules(prefix):
    try:
        mod = importlib.import_module(prefix)
    except Exception as e:
        print(f"[skip] cannot import {prefix}: {e}")
        return
    yield prefix
    if hasattr(mod, "__path__"):
        for m in pkgutil.walk_packages(mod.__path__, mod.__name__ + "."):
            yield m.name


def shortlist_functions(mod):
    out = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        lname = name.lower()
        if any(tok in lname for tok in KEY_FN_TOKENS):
            out.append((name, obj))
    return out


def shortlist_classes(mod):
    out = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        lname = name.lower()
        if any(tok in lname for tok in KEY_FN_TOKENS):
            out.append((name, obj))
    return out


def safe_signature(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(signature unavailable)"


def trim(s, n=MAX_DOC_LEN):
    if not s:
        return ""
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    print("=== Inspecting installed simupy_flight package ===")
    found_any = False
    for modname in walk_modules(TARGET_PACKAGE):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        fns = shortlist_functions(mod)
        clss = shortlist_classes(mod)

        if fns or clss:
            print(f"\n[mod] {modname}")
            for name, fn in fns:
                doc = trim(inspect.getdoc(fn))
                sig = safe_signature(fn)
                print(f"  [fn] {modname}.{name}{sig}")
                if doc:
                    print(f"       doc: {doc}")
                found_any = True
            for name, cls in clss:
                doc = trim(inspect.getdoc(cls))
                print(f"  [cls] {modname}.{name}")
                if doc:
                    print(f"       doc: {doc}")
                found_any = True

    if not found_any:
        print("No obvious builders/classes with keywords found.")
        print("Let's also list top-level attributes for hints.\n")

        try:
            top = importlib.import_module(TARGET_PACKAGE)
            print("Top-level dir(simupy_flight):")
            print(sorted(dir(top)))
        except Exception as e:
            print(f"Could not import {TARGET_PACKAGE}: {e}")


if __name__ == "__main__":
    main()
