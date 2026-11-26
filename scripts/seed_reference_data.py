"""
Load the country reference data (data protection laws and digital ID systems)
into MongoDB. This is what the frontend's maps and Sources page display.

By default, collections that already contain data are left untouched, so the
script is safe to run more than once.

Usage (from the repository root):
    uv run python scripts/seed_reference_data.py                       # first-time load
    uv run python scripts/seed_reference_data.py --update --dry-run    # preview changes
    uv run python scripts/seed_reference_data.py --update --dry-run -v # preview, field by field
    uv run python scripts/seed_reference_data.py --update              # apply changes
    uv run python scripts/seed_reference_data.py --export exported     # save the database's records

--update matches records by isoAlpha2 and replaces the stored record with the one
in the JSON file. Edits made to those records through the API are overwritten, so
run --dry-run first. If the database holds newer data than the JSON files, use
--export and merge the exported files into app/data/ before updating.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db.base_db import mongo_client  # noqa: E402

SEED_FILES = {
    "policy_law_docs": ROOT / "app/data/countries_law_data.json",
    "digital_ids_docs": ROOT / "app/data/accurate-digital-id-data.json",
}


def _short(value, width=70):
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= width else text[: width - 3] + "..."


def describe_diff(existing, record):
    """List how a stored record differs from the one in the JSON file."""
    lines = []
    for field in sorted(set(existing) | set(record)):
        db_value, file_value = existing.get(field), record.get(field)
        if db_value == file_value:
            continue
        if field == "links":
            db_urls = {l.get("url") for l in db_value or []}
            file_urls = {l.get("url") for l in file_value or []}
            for url in sorted(file_urls - db_urls):
                lines.append(f"links: file adds   {_short(url, 90)}")
            for url in sorted(db_urls - file_urls):
                lines.append(f"links: db only     {_short(url, 90)}")
            if db_urls == file_urls:
                lines.append("links: same URLs, different titles or types")
        else:
            lines.append(f"{field}: db={_short(db_value)!r}  file={_short(file_value)!r}")
    return lines


def update_collection(collection, records, dry_run):
    added, changed, unchanged, details = [], [], 0, {}
    for record in records:
        iso = record["isoAlpha2"]
        existing = collection.find_one({"isoAlpha2": iso}, {"_id": 0})
        if existing is None:
            added.append(record["country"])
        elif existing != record:
            changed.append(record["country"])
            details[record["country"]] = describe_diff(existing, record)
        else:
            unchanged += 1
            continue
        if not dry_run:
            collection.replace_one({"isoAlpha2": iso}, record, upsert=True)
    return added, changed, unchanged, details


def export_collection(collection, out_path):
    records = sorted(collection.find({}, {"_id": 0}), key=lambda r: r.get("country", ""))
    out_path.write_text(json.dumps(records, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="update existing databases from the JSON files")
    parser.add_argument("--dry-run", action="store_true", help="with --update: show changes without writing")
    parser.add_argument("-v", "--verbose", action="store_true", help="with --update: show each differing field")
    parser.add_argument("--export", metavar="DIR", help="write the database's records to DIR as JSON files")
    args = parser.parse_args()
    if (args.dry_run or args.verbose) and not args.update:
        parser.error("--dry-run and --verbose only make sense together with --update")

    if not mongo_client.is_connected():
        sys.exit("Could not connect to MongoDB. Check MONGODB_CONNECTION_STRING in your .env file.")

    if args.export:
        out_dir = Path(args.export)
        out_dir.mkdir(parents=True, exist_ok=True)
        for collection_name, path in SEED_FILES.items():
            count = export_collection(mongo_client.get_collection(collection_name), out_dir / path.name)
            print(f"Exported {count} records from {collection_name} to {out_dir / path.name}")
        return

    for collection_name, path in SEED_FILES.items():
        collection = mongo_client.get_collection(collection_name)
        records = json.loads(path.read_text(encoding="utf-8"))

        if args.update:
            added, changed, unchanged, details = update_collection(collection, records, args.dry_run)
            verb = "Would change" if args.dry_run else "Updated"
            print(f"{verb} {collection_name}: {len(added)} added, {len(changed)} changed, {unchanged} unchanged")
            for name in added:
                print(f"  + {name}")
            for name in changed:
                print(f"  ~ {name}")
                if args.verbose:
                    for line in details[name]:
                        print(f"      {line}")
            continue

        existing = collection.count_documents({})
        if existing:
            print(f"Skipped {collection_name}: already has {existing} records (use --update to refresh it)")
            continue

        collection.insert_many(records)
        print(f"Loaded {collection_name}: {len(records)} records from {path.name}")


if __name__ == "__main__":
    main()
