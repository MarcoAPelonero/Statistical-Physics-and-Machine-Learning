import json
import os
import re
import sys

CANDIDATES = ["Dataset.jsonl", "dataset.jsonl", "dataset_backup.jsonl", "Dataset_backup.jsonl"]

def find_file():
    for p in CANDIDATES:
        if os.path.exists(p):
            return p
    return None

UI_RE = re.compile(r"^D0*\d+$")

def main(path=None):
    if path is None:
        path = find_file()
    if path is None:
        print("No dataset file found. Tried:", CANDIDATES)
        sys.exit(2)

    total = 0
    missing_pub_year = []
    bad_ui = []
    ui_missing_field = []

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Line {i}: JSON parse error: {e}")
                continue
            total += 1
            pmid = obj.get('pmid')
            if 'pub_year' not in obj or obj.get('pub_year') is None:
                missing_pub_year.append({'line': i, 'pmid': pmid})

            mesh = obj.get('mesh', [])
            if not isinstance(mesh, list):
                ui_missing_field.append({'line': i, 'pmid': pmid, 'reason': 'mesh not list'})
                continue
            for entry in mesh:
                ui = entry.get('ui') if isinstance(entry, dict) else None
                if ui is None:
                    ui_missing_field.append({'line': i, 'pmid': pmid, 'entry': entry})
                else:
                    if not UI_RE.match(str(ui)):
                        bad_ui.append({'line': i, 'pmid': pmid, 'ui': ui})

    print("\nCHECK SUMMARY")
    print("File:", path)
    print(f"Total records scanned: {total}")
    print(f"Missing or null pub_year: {len(missing_pub_year)}")
    if missing_pub_year:
        print("Examples (up to 10):")
        for ex in missing_pub_year[:10]:
            print(f"  line {ex['line']} pmid={ex['pmid']}")

    print(f"Mesh entries with missing 'ui' field: {len(ui_missing_field)}")
    if ui_missing_field:
        for ex in ui_missing_field[:10]:
            print(f"  line {ex['line']} pmid={ex['pmid']} reason={ex.get('reason')}")

    print(f"Mesh 'ui' values not matching ^D0*\\d+$: {len(bad_ui)}")
    if bad_ui:
        print("Examples (up to 10):")
        for ex in bad_ui[:10]:
            print(f"  line {ex['line']} pmid={ex['pmid']} ui={ex['ui']}")

    if len(missing_pub_year) == 0 and len(ui_missing_field) == 0 and len(bad_ui) == 0:
        print("\nAll records have pub_year and mesh.ui values match the expected pattern.")
        return 0
    else:
        print("\nIssues detected. See counts and examples above.")
        return 1

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    rc = main(path)
    sys.exit(rc)
