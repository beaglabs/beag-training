"""Catalog CLI — build, inspect, and search the framework catalog.

Usage:
  python -m frameworks.catalog_cli build         Build catalog.json + embeddings.json
  python -m frameworks.catalog_cli stats          Print framework statistics
  python -m frameworks.catalog_cli search <text>  Full-text search
  python -m frameworks.catalog_cli crosswalk <id> Show cross-framework mappings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cmd_build() -> None:
    from frameworks.catalog import load_catalog
    from frameworks.embeddings import EmbeddingStore

    print("Building catalog ...")
    cat = load_catalog(force_rebuild=True)
    stats = cat.stats()
    for fw, count in stats.items():
        print(f"  {fw}: {count} entries")

    print("Building embeddings ...")
    store = EmbeddingStore(cat)
    store.build()
    store.save()
    print(f"  embeddings saved ({store._matrix.shape if store._matrix is not None else 'N/A'})")
    print("Done.")


def cmd_stats() -> None:
    from frameworks.catalog import Catalog

    cat = Catalog.load()
    for fw, entries in cat.frameworks.items():
        families = len(cat.list_by_framework(fw, "family"))
        controls = len(cat.list_by_framework(fw, "control"))
        enhancements = len(cat.list_by_framework(fw, "enhancement"))
        print(f"{fw.value}: {len(entries)} total | {families} families | {controls} controls | {enhancements} enhancements")


def cmd_search(query: str) -> None:
    from frameworks.catalog import Catalog
    from frameworks.embeddings import EmbeddingStore

    cat = Catalog.load()
    store = EmbeddingStore.load(cat)

    print(f"Keyword matches for '{query}':")
    for entry in cat.search_text(query):
        print(f"  [{entry.framework.value}] {entry.id}: {entry.title}")
    print()

    if store._matrix is not None:
        print(f"Semantic matches for '{query}':")
        for entry, score in store.search(query, top_k=5):
            print(f"  [{entry.framework.value}] {entry.id} (score={score:.3f}): {entry.title}")


def cmd_crosswalk(control_id: str) -> None:
    from frameworks.catalog import Catalog, Framework

    cat = Catalog.load()
    found = False
    for fw in Framework:
        entry = cat.get(fw, control_id)
        if entry:
            found = True
            print(f"{fw.value} :: {entry.id} — {entry.title}")
            print(f"  Granularity: {entry.granularity.value}")
            if entry.parent_id:
                parent = cat.get(fw, entry.parent_id)
                print(f"  Parent: {parent.title if parent else entry.parent_id}")
            if entry.children:
                print(f"  Children: {', '.join(entry.children)}")
            if entry.crosswalk:
                print("  Cross-framework mappings:")
                for cw in entry.crosswalk:
                    target = cat.get(cw.framework, cw.control_id)
                    label = target.title if target else cw.control_id
                    print(f"    → [{cw.framework.value}] {cw.control_id}: {label}")
            else:
                print("  (no cross-framework mappings)")
            print()
    if not found:
        print(f"No entry found for '{control_id}' in any framework.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIST Framework Catalog CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("build", help="Build catalog.json and embeddings.json")
    sub.add_parser("stats", help="Print framework statistics")

    search_p = sub.add_parser("search", help="Search the catalog")
    search_p.add_argument("query", help="Search query text")

    cw_p = sub.add_parser("crosswalk", help="Show cross-framework mappings")
    cw_p.add_argument("control_id", help="Control or subcategory ID")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build()
    elif args.command == "stats":
        cmd_stats()
    elif args.command == "search":
        cmd_search(args.query)
    elif args.command == "crosswalk":
        cmd_crosswalk(args.control_id)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
