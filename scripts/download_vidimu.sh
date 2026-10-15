#!/bin/bash
# Download VIDIMU v2.0.0 (Zenodo record 15075076, CC BY 4.0) to VIDIMU_DIR (default data/vidimu), verify every
# file's md5, and unzip.
#
#   bash scripts/download_vidimu.sh          # "small": everything except the 41 GB full-size videos (about 1.5 GB)
#   bash scripts/download_vidimu.sh all      # also the full-size videos
#
# Safe to rerun: verified files are skipped, and interrupted downloads resume. Data stays outside the repository.
# Credit: Martinez-Zarzuela et al. (2023), Scientific Data 10, 648, https://doi.org/10.1038/s41597-023-02554-9
set -euo pipefail
RECORD=15075076
MODE="${1:-small}"
DEST="${VIDIMU_DIR:-data/vidimu}"
mkdir -p "$DEST/zips"

# 1) The file list, sizes and md5 checksums, from Zenodo's API (or a saved copy, via VIDIMU_API_JSON).
python3 - "$RECORD" "${VIDIMU_API_JSON:-}" > "$DEST/files.tsv" <<'PY'
import json, sys, urllib.request
record, saved = sys.argv[1], sys.argv[2]
if saved:
    meta = json.load(open(saved))
else:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as r:
        meta = json.load(r)
md = meta.get("metadata", {})
lic = md.get("license", {})
print(f"# {md.get('title', '?')} | version {md.get('version', '?')} | licence {lic.get('id', lic) if isinstance(lic, dict) else lic}",
      file=sys.stderr)
files = meta.get("files", [])
if isinstance(files, dict):            # newer API shape: {"entries": [...]}
    files = files.get("entries", [])
if not files:
    sys.exit("No files found in the record's metadata; check https://zenodo.org/records/" + record)
for f in files:
    name = f.get("key") or f.get("filename")
    md5 = (f.get("checksum") or "").split(":", 1)[-1]
    links = f.get("links", {})
    url = links.get("self") or links.get("content") or links.get("download")
    if not (name and md5 and url):
        sys.exit(f"Unexpected file entry: {f}")
    print(name, f.get("size", 0), md5, url, sep="\t")
PY

# 2) Download, resuming after interruptions, and verify each file against its md5.
md5_of() { md5sum "$1" | cut -d' ' -f1; }
while IFS=$'\t' read -r name size md5 url; do
  if [[ "$MODE" != "all" && "$name" == videosfullsize* ]]; then
    echo "skip   $name ($((size / 1024 / 1024 / 1024)) GiB; rerun with 'all')"; continue
  fi
  out="$DEST/zips/$name"
  if [[ -f "$out" && "$(stat -c %s "$out")" == "$size" && "$(md5_of "$out")" == "$md5" ]]; then
    echo "ok     $name (already verified)"; continue
  fi
  echo "fetch  $name ($((size / 1024 / 1024)) MiB)"
  for attempt in $(seq 1 20); do
    if curl -L --fail --silent --show-error -C - -o "$out" "$url"; then break; fi
    [[ -f "$out" && "$(stat -c %s "$out")" -ge "$size" ]] && break     # already complete
    echo "       attempt $attempt failed; retrying in 30 s" >&2; sleep 30
  done
  got="$(md5_of "$out")"
  if [[ "$got" != "$md5" ]]; then
    echo "md5 mismatch for $name: got $got, expected $md5. Delete $out and rerun." >&2; exit 1
  fi
  echo "ok     $name (verified)"
done < "$DEST/files.tsv"

# 3) Unzip each verified archive into its own folder (via a temporary name, so a failed unzip leaves no half folder).
for z in "$DEST"/zips/*.zip; do
  [[ -e "$z" ]] || continue
  base="$(basename "$z" .zip)"
  if [[ -d "$DEST/$base" ]]; then echo "ok     $base/ (already unzipped)"; continue; fi
  rm -rf "$DEST/$base.tmp" && unzip -q "$z" -d "$DEST/$base.tmp" && mv "$DEST/$base.tmp" "$DEST/$base"
  echo "unzip  $base/"
done

# 4) Provenance
{
  echo "VIDIMU, Zenodo record $RECORD, downloaded $(date -Iseconds) by scripts/download_vidimu.sh ($MODE)"
  echo "Licence: CC BY 4.0. Credit: Martinez-Zarzuela et al. (2023), https://doi.org/10.1038/s41597-023-02554-9"
  cut -f1,2,3 "$DEST/files.tsv"
} > "$DEST/PROVENANCE.txt"
echo "Done: $DEST (see PROVENANCE.txt)"
