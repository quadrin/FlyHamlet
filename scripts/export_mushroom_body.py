#!/usr/bin/env python3
"""Export mushroom-body cell annotations for the browser model as dense indices.

Reads the public FlyWire v783 neurons and classification tables (verified against the
SHA-256 values recorded in site/model/manifest.json), maps root IDs to the manifest's
dense ascending-root-ID indices, and writes site/model/mushroom-body.json with Kenyon
cells, mushroom-body output neurons (MBONs) and uniglomerular antennal-lobe projection
neurons. No pandas; only the standard library.
"""
import csv, gzip, hashlib, json, struct, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "site/model/manifest.json"
OUTPUT = ROOT / "site/model/mushroom-body.json"


def fetch(source: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(source["url"]).name
    if not path.exists():
        print(f"downloading {source['url']}", file=sys.stderr)
        with urllib.request.urlopen(source["url"]) as response, open(path, "wb") as handle:
            handle.write(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"] or path.stat().st_size != source["bytes"]:
        raise RuntimeError(f"{path.name}: SHA-256 or size does not match the manifest provenance")
    return path


def main(cache="data/cache"):
    manifest = json.loads(MANIFEST.read_text())
    sources = {item["table"]: item for item in manifest["provenance"]["sources"]}
    neurons = fetch(sources["neurons"], ROOT / cache)
    classification = fetch(sources["classification"], ROOT / cache)
    with gzip.open(neurons, "rt") as handle:
        root_ids = sorted(int(row["root_id"]) for row in csv.DictReader(handle))
    if len(root_ids) != manifest["n"]:
        raise RuntimeError("neuron count differs from the manifest")
    ordering = hashlib.sha256(b"".join(struct.pack("<q", value) for value in root_ids)).hexdigest()
    if ordering != manifest["provenance"]["rootIdsSha256"]:
        raise RuntimeError("root ID ordering differs from the manifest")
    index = {root_id: position for position, root_id in enumerate(root_ids)}
    groups = {"kenyonCells": [], "mbons": [], "uniglomerularPNs": []}
    subtypes = {}
    with gzip.open(classification, "rt") as handle:
        for row in csv.DictReader(handle):
            root_id = int(row["root_id"])
            if root_id not in index:
                continue
            cell_type = row.get("cell_type") or ""
            hemibrain = row.get("hemibrain_type") or ""
            if row.get("class") == "Kenyon_Cell":
                groups["kenyonCells"].append(index[root_id])
                subtypes[str(index[root_id])] = cell_type or hemibrain or "KC"
            elif cell_type.startswith("MBON") or hemibrain.startswith("MBON"):
                groups["mbons"].append(index[root_id])
                subtypes[str(index[root_id])] = hemibrain or cell_type
            elif row.get("class") == "ALPN" and row.get("sub_class") == "uniglomerular":
                groups["uniglomerularPNs"].append(index[root_id])
                subtypes[str(index[root_id])] = cell_type or hemibrain or "uPN"
    for name in groups:
        groups[name].sort()
    payload = {
        "format": "flyhamlet-mushroom-body-v1",
        "n": manifest["n"],
        "criteria": {
            "kenyonCells": "classification class == 'Kenyon_Cell'",
            "mbons": "cell_type or hemibrain_type starts with 'MBON'",
            "uniglomerularPNs": "classification class == 'ALPN' and sub_class == 'uniglomerular'",
        },
        "counts": {name: len(values) for name, values in groups.items()},
        "provenance": {
            "neurons": sources["neurons"],
            "classification": sources["classification"],
            "rootIdsSha256": ordering,
            "ordering": manifest["provenance"]["ordering"],
        },
        **groups,
        "types": subtypes,
    }
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {payload['counts']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
