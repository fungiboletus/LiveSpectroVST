#!/usr/bin/env bash
set -euo pipefail

platform="${1:?usage: package-release.sh <platform>}"
bundle_dir="target/bundled"
dist_dir="target/dist"
archive="${dist_dir}/LiveSpectroVST-${platform}.zip"

mkdir -p "${dist_dir}"
rm -f "${archive}"

if [[ "${platform}" == macos-* ]]; then
    stage="${dist_dir}/LiveSpectroVST-${platform}"
    rm -rf "${stage}"
    mkdir -p "${stage}"
    ditto "${bundle_dir}/Live Spectro.vst3" "${stage}/Live Spectro.vst3"
    ditto "${bundle_dir}/Live Spectro.clap" "${stage}/Live Spectro.clap"
    ditto -c -k --sequesterRsrc --keepParent "${stage}" "${archive}"
    rm -rf "${stage}"
else
    python - "${bundle_dir}" "${archive}" <<'PY'
import pathlib
import sys
import zipfile

bundle_dir = pathlib.Path(sys.argv[1])
archive = pathlib.Path(sys.argv[2])
paths = [bundle_dir / "Live Spectro.vst3"]
clap = bundle_dir / "Live Spectro.clap"
if clap.exists():
    paths.append(clap)

with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
    for path in paths:
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    output.write(child, child.relative_to(bundle_dir))
        else:
            output.write(path, path.relative_to(bundle_dir))
PY
fi

echo "Created ${archive}"
