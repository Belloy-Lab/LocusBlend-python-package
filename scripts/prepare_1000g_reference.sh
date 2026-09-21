#!/usr/bin/env bash
#
# prepare_1000g_reference.sh - build a LocusBlend reference layout from a
# user-prepared 1000 Genomes GRCh38 PLINK2 PGEN dataset.
#
# LocusBlend does not download, host or redistribute reference data. You supply
# the PGEN dataset, the per-ancestry keep lists and (optionally) the
# related-sample removal list; this script only runs PLINK2 locally.
# See docs/reference_preparation.md for the documented workflow and citation.
#
# Usage:
#   bash scripts/prepare_1000g_reference.sh <PGEN_PREFIX> <KEEP_DIR> <OUTPUT_ROOT> [REMOVE_FILE]
#   bash scripts/prepare_1000g_reference.sh --print-template
#
# Arguments (or the equivalently named environment variables):
#   PGEN_PREFIX   PLINK2 dataset prefix (<prefix>.pgen/.pvar/.psam)
#   KEEP_DIR      directory containing AFR.txt AMR.txt EAS.txt EUR.txt SAS.txt
#   OUTPUT_ROOT   reference root; files go to <OUTPUT_ROOT>/1000g/<ANCESTRY>/
#   REMOVE_FILE   optional related-sample removal list (used historically)
#
# The output file names are derived from src/locusblend/reference.py
# (ReferenceManager.bfile_prefix_template) so they always match what the package
# expects. Chromosome X is written as chX, never the historical ch23.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REFERENCE_PY="${REFERENCE_PY:-${REPO_ROOT}/src/locusblend/reference.py}"

ANCESTRIES=(AFR AMR EAS EUR SAS)
CHROMOSOMES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X)

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/prepare_1000g_reference.sh <PGEN_PREFIX> <KEEP_DIR> <OUTPUT_ROOT> [REMOVE_FILE]
  bash scripts/prepare_1000g_reference.sh --print-template

  PGEN_PREFIX   PLINK2 dataset prefix (<prefix>.pgen/.pvar/.psam)
  KEEP_DIR      directory containing AFR.txt AMR.txt EAS.txt EUR.txt SAS.txt
  OUTPUT_ROOT   reference root; files go to <OUTPUT_ROOT>/1000g/<ANCESTRY>/
  REMOVE_FILE   optional related-sample removal list (used historically)

Requires plink2 on PATH. LocusBlend never downloads reference data; see
docs/reference_preparation.md.
USAGE
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
info() { printf '[locusblend] %s\n' "$*"; }

extract_bfile_prefix_template() {
  [[ -f "$REFERENCE_PY" ]] || die "cannot find ${REFERENCE_PY} (set REFERENCE_PY to src/locusblend/reference.py)"
  local raw template
  raw="$(awk '
    /bfile_prefix_template[[:space:]]*=/ {
      if ($0 ~ /=[[:space:]]*"[^"]*"/) { print $0; exit }
      inside = 1
      next
    }
    inside {
      if ($0 ~ /^[[:space:]]*\)/) { exit }
      print
    }
  ' "$REFERENCE_PY")"
  template="$(printf '%s\n' "$raw" | grep -o '"[^"]*"' | tr -d '"' | tr -d '\n')"
  [[ -n "$template" ]] || die "could not parse bfile_prefix_template from ${REFERENCE_PY}"
  [[ "$template" == *'{ancestry}'* && "$template" == *'{chrom}'* ]] || die "unexpected bfile_prefix_template in ${REFERENCE_PY}: ${template}"
  printf '%s' "$template"
}

print_template() {
  local template example_14 example_x
  template="$(extract_bfile_prefix_template)"
  example_14="${template//\{ancestry\}/EUR}"; example_14="${example_14//\{chrom\}/14}"
  example_x="${template//\{ancestry\}/EUR}"; example_x="${example_x//\{chrom\}/X}"
  printf 'bfile_prefix_template: %s\n' "$template"
  printf 'example (EUR, chr14): %s.bed/.bim/.fam\n' "$example_14"
  printf 'example (EUR, chrX):  %s.bed/.bim/.fam\n' "$example_x"
}

main() {
  case "${1:-}" in
    "") usage >&2; exit 2 ;;
    --help | -h) usage; exit 0 ;;
    --print-template) print_template; exit 0 ;;
  esac

  local pgen_prefix="${1:-${PGEN_PREFIX:-}}"
  local keep_dir="${2:-${KEEP_DIR:-}}"
  local output_root="${3:-${OUTPUT_ROOT:-}}"
  local remove_file="${4:-${REMOVE_FILE:-}}"

  [[ -n "$pgen_prefix" ]] || { usage >&2; die "PGEN_PREFIX is required"; }
  [[ -n "$keep_dir" ]] || { usage >&2; die "KEEP_DIR is required"; }
  [[ -n "$output_root" ]] || { usage >&2; die "OUTPUT_ROOT is required"; }

  command -v plink2 >/dev/null 2>&1 || die "plink2 not found on PATH (this script requires PLINK2)"

  local suffix
  for suffix in .pgen .pvar .psam; do
    [[ -f "${pgen_prefix}${suffix}" ]] || die "missing input file: ${pgen_prefix}${suffix}"
  done
  [[ -d "$keep_dir" ]] || die "KEEP_DIR is not a directory: ${keep_dir}"

  local anc missing=()
  for anc in "${ANCESTRIES[@]}"; do
    [[ -f "${keep_dir}/${anc}.txt" ]] || missing+=("${keep_dir}/${anc}.txt")
  done
  if [[ "${#missing[@]}" -gt 0 ]]; then
    die "missing keep list(s): ${missing[*]}"
  fi
  if [[ -n "$remove_file" ]]; then
    [[ -f "$remove_file" ]] || die "REMOVE_FILE not found: ${remove_file}"
  fi

  local template
  template="$(extract_bfile_prefix_template)"
  info "plink2: $(command -v plink2)"
  info "PGEN prefix: ${pgen_prefix}"
  info "output root: ${output_root}"
  info "output name template: ${template}"

  local chr out_dir out_name out_prefix keep_file args
  for anc in "${ANCESTRIES[@]}"; do
    keep_file="${keep_dir}/${anc}.txt"
    out_dir="${output_root}/1000g/${anc}"
    mkdir -p "$out_dir"
    for chr in "${CHROMOSOMES[@]}"; do
      out_name="${template//\{ancestry\}/$anc}"
      out_name="${out_name//\{chrom\}/$chr}"
      out_prefix="${out_dir}/${out_name}"
      args=(--pfile "$pgen_prefix" vzs --keep "$keep_file")
      if [[ -n "$remove_file" ]]; then
        args+=(--remove "$remove_file")
      fi
      args+=(--chr "$chr" --set-missing-var-ids '@:#$r:$a' --new-id-max-allele-len 1000 --make-bed --out "$out_prefix")
      info "${anc} chr${chr} -> ${out_prefix}.bed/.bim/.fam"
      plink2 "${args[@]}"
    done
  done

  info "done: ${output_root}/1000g/<ancestry> now holds the chr1-22 and chrX PLINK files"
}

main "$@"