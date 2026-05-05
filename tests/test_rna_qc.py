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


def test_collect_alignment_idxstats_qc_flags_rdna_repeat_density(tmp_path):
    idxstats = tmp_path / "alignment.idxstats"
    idxstats.write_text(
        "\n".join(
            [
                "chr1\t100000000\t1000\t0",
                "chr21\t40000000\t9000\t0",
                "chr22_KI270733v1_random\t180000\t5400\t0",
                "chrUn_GL000220v1\t160000\t4400\t0",
                "*\t0\t0\t881",
            ]
        )
    )

    qc = collect_rna_quant_qc(
        tmp_path / "quant.gene_tpm.csv",
        alignment_qc_path=idxstats,
    )

    assert qc["available"]
    assert qc["source"] == "alignment_qc"
    assert qc["alignment_qc"]["rdna_mapped"] == 9800
    assert qc["alignment_qc"]["rdna_density_over_chr1"] > 1000
    assert any("rDNA-like contigs" in warning for warning in qc["warnings"])
    assert any("chr21 is the top primary chromosome" in warning for warning in qc["warnings"])
    md = rna_quant_qc_markdown(qc, heading="## RNA QC")
    assert "rDNA-like contig burden" in md
    assert "Top rDNA-like contigs" in md
