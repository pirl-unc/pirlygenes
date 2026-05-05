import json

import pandas as pd

from pirlygenes.rna_qc import (
    collect_rna_quant_qc,
    rna_quant_qc_markdown,
    rna_quant_qc_summary_line,
)


def test_collect_salmon_qc_from_output_dir(tmp_path):
    salmon = tmp_path / "salmon-output"
    aux = salmon / "aux_info"
    aux.mkdir(parents=True)
    (aux / "meta_info.json").write_text(
        json.dumps(
            {
                "salmon_version": "1.10.1",
                "library_types": ["ISR"],
                "num_processed": 59_900_883,
                "num_mapped": 20_054_496,
                "percent_mapped": 33.479,
                "num_decoy_fragments": 1_669_268,
                "num_alignments_below_threshold_for_mapped_fragments_vm": 119_412_458,
                "num_fragments_filtered_vm": 7_871_607,
                "frag_length_mean": 213.1,
                "frag_length_sd": 84.0,
            }
        )
    )
    (salmon / "lib_format_counts.json").write_text(
        json.dumps({"expected_format": "ISR", "strand_mapping_bias": 0.00004})
    )
    pd.DataFrame({"Name": ["g1", "g2", "g3"], "TPM": [999_990.0, 5.0, 0.0]}).to_csv(
        salmon / "quant.genes.sf",
        sep="\t",
        index=False,
    )
    pd.DataFrame({"Name": ["t1", "t2"], "TPM": [999_999.0, 1.0]}).to_csv(
        salmon / "quant.sf",
        sep="\t",
        index=False,
    )

    qc = collect_rna_quant_qc(salmon / "quant.gene_tpm.csv")

    assert qc["available"]
    assert qc["source"] == "salmon"
    assert qc["gene_detection"]["detected_ge1"] == 2
    assert qc["transcript_detection"]["detected_ge1"] == 2
    assert any("mapping rate is low" in warning for warning in qc["warnings"])
    assert "Salmon mapping 33.5%" in rna_quant_qc_summary_line(qc)
    md = rna_quant_qc_markdown(qc, heading="## RNA QC")
    assert "Mapping rate" in md
    assert "Gene TPM sum" in md
