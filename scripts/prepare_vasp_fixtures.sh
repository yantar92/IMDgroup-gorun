#!/usr/bin/env bash

# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of IMDgroup-gorun package
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# prepare_vasp_fixtures.sh - regenerate the converged VASP fixture
#
# Copies a real, converged VASP run into tests/fixtures/vasp_converged/,
# storing large output files gzip-compressed and excluding files that
# cannot be redistributed (POTCAR, vdw_kernel.bindat) under the VASP
# license.
#
# Usage:
#   scripts/prepare_vasp_fixtures.sh <source-vasp-directory>
#
# Example:
#   scripts/prepare_vasp_fixtures.sh \
    #     ~/helios/2026.CNT.pore-filling/00.Li/..../SCF

set -euo pipefail

SRC="${1:?usage: $0 <source-vasp-directory>}"
if [[ ! -d "$SRC" ]]; then
    echo "error: not a directory: $SRC" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/tests/fixtures/vasp_converged"
mkdir -p "$DEST"

# Small text files stored uncompressed (readable in the repo).
for f in INCAR POSCAR KPOINTS CONTCAR OSZICAR; do
    if [[ ! -f "$SRC/$f" ]]; then
        echo "error: missing required file: $SRC/$f" >&2
        exit 1
    fi
    cp "$SRC/$f" "$DEST/$f"
done

# Large output files stored gzip-compressed.  ``-n`` omits the
# filename/timestamp so regenerating from identical sources is
# byte-for-byte reproducible.
for f in OUTCAR vasprun.xml; do
    if [[ ! -f "$SRC/$f" ]]; then
        echo "error: missing required file: $SRC/$f" >&2
        exit 1
    fi
    gzip -n -c "$SRC/$f" > "$DEST/$f.gz"
done

cat <<EOF
Wrote fixture to $DEST

Intentionally excluded (VASP license, must not be committed):
  POTCAR            (pseudopotentials)
  vdw_kernel.bindat (distributed VASP kernel file)

Commit the resulting .gz files:
  git add tests/fixtures/vasp_converged
EOF
