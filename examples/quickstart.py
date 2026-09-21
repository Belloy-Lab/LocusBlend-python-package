"""Low-level LocusBlend quickstart (component level).

This example points LocusBlend at an external reference directory, validates it
without reading any data, and describes a locus configuration. For the
high-level pipeline use ``locusblend.plot`` instead (see
``examples/python_api.py``).

Run it with::

    python examples/quickstart.py
"""

from locusblend import LocusBlendConfig, ReferenceManager
from locusblend.config import parse_highlighted_genes

# Reference data live outside the Git repository (never commit them).
REFERENCE_DIR = "D:/locusblend_reference"


def main():
    ref = ReferenceManager(reference_dir=REFERENCE_DIR, ancestry="EUR")
    print(ref.describe())

    # Cheap preflight: reports missing configuration/files without reading data.
    report = ref.validate_for_locus("14")
    print(report.describe())
    if report.ok:
        print(f"chr14 PLINK bfile prefix: {ref.get_bfile_prefix('14')}")
        print(f"GENCODE annotation: {ref.get_gtf_path('14')}")
    else:
        print("Reference data are incomplete; see the errors above.")

    # The recombination BigWig is optional (warning only when missing).
    print(f"Recombination track: {ref.get_recombination_bw_path()}")

    config = LocusBlendConfig(
        mode="Three-index LocusBlend",
        ancestry="EUR",
        chromosome="14",
        center_bp=73238768,
        window_kb=500,
        index_variants=("rs11159021", "rs3742825", "rs74986264"),
        highlighted_genes="PSEN1, PAPLN, HEATR4",
        title_top="Alzheimer's disease Female GWAS",
        title_bottom="Monocyte PSEN1 eQTL",
    )

    print()
    print(f"mode: {config.mode}")
    print(f"requires {config.required_n_indices} index variant(s)")
    print(f"locus: chr{config.chromosome}:{config.center_bp} +/- {config.window_kb} kb")
    print(f"window_bp: {config.window_bp}")
    print(f"highlighted genes: {sorted(parse_highlighted_genes(config.highlighted_genes))}")


if __name__ == "__main__":
    main()
