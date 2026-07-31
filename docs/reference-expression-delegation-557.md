# cancer_reference_expression delegation parity (#557)

Compares pirlygenes' delegated source-union compatibility rows against oncoref's canonical selected/QC-aware artifact view, per `(cancer_code, Ensembl_Gene_ID)` at `tpm_clean`. Relative deltas are on the median `expression`, over genes with pg TPM >= 1.0.

This intentionally compares two different oncoref products: the all-sample source-union compatibility rows and the default pass-QC selected artifact. Large outliers therefore identify QC/source-product differences, not adapter distortion; exact adapter-to-source parity for TCGA, heme, microarray, subtype, and computed-union cases is gated in `tests/test_reference_expression_delegation.py`.

- cancer_codes compared: **130**
- served by both sides: **130**
- n_samples agreement: **74/130** codes match exactly
- median relative delta (across codes): **0.1176%**
- worst-code p95 relative delta: **3462.0134%**

## Per-code detail

| cancer_code | qc | n_samp pg/on | genes shared (pg-only/on-only) | rel median | rel p95 | divergent |
| --- | --- | --- | --- | --- | --- | --- |
| MBL_WNT | artifact | 17/4 warn | 30930 (1679/1946) | 1450.2955% | 3462.0134% | 7562 |
| MBL_SHH | artifact | 25/7 warn | 30930 (1679/1946) | 1615.4739% | 3166.1213% | 7180 |
| MBL | artifact | 125/58 warn | 32651 (1686/225) | 420.5962% | 1025.8829% | 11303 |
| MBL_G4 | artifact | 39/18 warn | 30930 (1679/1946) | 367.1439% | 854.4698% | 10831 |
| HL | artifact | 5/2 warn | 55986 (2681/1814) | 45.4203% | 283.2629% | 6549 |
| MM | artifact | 764/47 warn | 57233 (2616/426) | 133.0175% | 196.8736% | 11088 |
| SARC_LGFMS | artifact | 2/2 | 18778 (94/14098) | 49.2665% | 180.1510% | 6292 |
| MBL_G3 | artifact | 44/29 warn | 30930 (1679/1946) | 39.2941% | 148.9891% | 4755 |
| ACINIC | artifact | 3/2 warn | 56357 (2594/0) | 18.3425% | 128.5182% | 2480 |
| SARC_CCS | artifact | 5/2 warn | 16462 (257/16) | 22.1006% | 103.3026% | 2419 |
| NPC | artifact | 4/3 warn | 32651 (1686/225) | 18.4273% | 77.0931% | 2082 |
| NET_PANCREAS | artifact | 113/30 warn | 55986 (2681/1814) | 20.6177% | 66.3002% | 2128 |
| SARC_CHON | artifact | 54/36 warn | 59413 (2620/463) | 13.6496% | 51.0011% | 770 |
| SARC_IFS | artifact | 2/2 | 32651 (1686/225) | 4.7730% | 50.4475% | 735 |
| MCL | artifact | 51/10 warn | 25466 (122/143) | 8.5260% | 45.5507% | 439 |
| SCLC_YAP1 | artifact | 2/2 | 16805 (82/5) | 5.4194% | 44.4777% | 429 |
| T_ALL | artifact | 264/176 warn | 57233 (2616/426) | 16.4180% | 44.3316% | 480 |
| SARC_RMS_SSRMS | artifact | 8/7 warn | 32651 (1686/225) | 7.3257% | 38.1285% | 316 |
| READ_MSI | artifact | 2/2 | 32651 (1686/225) | 1.5066% | 28.6634% | 182 |
| NET_LUNG | artifact | 118/64 warn | 56143 (2616/415) | 5.5422% | 26.5121% | 259 |
| MPN | artifact | 45/29 warn | 35635 (2181/256) | 4.2010% | 26.3788% | 225 |
| FL | artifact | 6/5 warn | 48572 (2568/235) | 4.3683% | 26.3754% | 169 |
| ATRT | artifact | 4/4 | 32651 (1686/225) | 0.9110% | 21.0053% | 107 |
| NEC_MERKEL | artifact | 91/76 warn | 23098 (111/72) | 4.6530% | 17.7220% | 45 |
| ACC | artifact | 77/68 warn | 32651 (1686/225) | 3.2773% | 15.0294% | 28 |
| NEC_LUNG_LARGECELL | artifact | 69/58 warn | 56143 (2616/415) | 2.2081% | 14.4122% | 104 |
| RT | artifact | 63/58 warn | 57233 (2616/426) | 2.5748% | 13.2745% | 48 |
| KICH | artifact | 66/55 warn | 32651 (1686/225) | 2.4737% | 12.4236% | 16 |
| SARC_PLEOLPS | artifact | 4/4 | 18790 (82/35) | 0.5214% | 12.3698% | 42 |
| LAML_APL | artifact | 20/19 warn | 57233 (2616/426) | 1.9536% | 11.3345% | 23 |
| B_ALL | artifact | 154/141 warn | 57233 (2616/426) | 1.5962% | 8.6496% | 27 |
| NET_MIDGUT | artifact | 81/77 warn | 55986 (2681/1814) | 1.2556% | 7.9543% | 48 |
| SARC_IMT | artifact | 4/4 | 32651 (1686/225) | 0.2353% | 6.6002% | 86 |
| SARC_RMS_PRMS | artifact | 6/6 | 32651 (1686/225) | 0.2857% | 6.2263% | 21 |
| UCEC_CNH | artifact | 85/83 warn | 32871 (1700/5) | 0.9275% | 6.0678% | 0 |
| SARC_RMS_ARMS | artifact | 73/71 warn | 32651 (1686/225) | 0.1455% | 5.4407% | 7 |
| SARC_SYN | artifact | 50/49 warn | 32651 (1686/225) | 0.8915% | 5.2792% | 11 |
| SCLC_NEUROD1 | artifact | 8/8 | 16805 (82/5) | 0.1791% | 5.1845% | 3 |
| ADCC | artifact | 57/56 warn | 56357 (2594/0) | 0.9052% | 4.8991% | 0 |
| UVM | artifact | 79/78 warn | 32651 (1686/225) | 0.5816% | 3.3643% | 7 |
| LAML_ELNint | artifact | 154/152 warn | 57233 (2616/426) | 0.5823% | 3.0882% | 22 |
| SARC_RMS_ERMS | artifact | 95/94 warn | 32651 (1686/225) | 0.5168% | 2.9550% | 6 |
| UCEC | artifact | 181/179 warn | 32651 (1686/225) | 0.4141% | 2.8520% | 7 |
| SARC_EWS | artifact | 101/100 warn | 32651 (1686/225) | 0.4224% | 2.8252% | 6 |
| MDS | artifact | 82/81 warn | 55986 (2681/1814) | 0.4866% | 2.7298% | 46 |
| PCPG | artifact | 182/179 warn | 32651 (1686/225) | 0.4323% | 2.5203% | 7 |
| KIRP | artifact | 289/282 warn | 32651 (1686/225) | 0.5079% | 2.4604% | 7 |
| PAAD | artifact | 179/177 warn | 32651 (1686/225) | 0.4220% | 2.4375% | 6 |
| LAML_ELNfav | artifact | 238/234 warn | 57233 (2616/426) | 0.3475% | 2.3031% | 22 |
| NBL_MYCNnonamp | artifact | 122/121 warn | 57233 (2616/426) | 0.3891% | 2.2672% | 21 |
| SARC_KS | artifact | 10/10 | 55340 (2611/416) | 0.1310% | 2.1417% | 13 |
| SCLC_POU2F3 | artifact | 10/10 | 16805 (82/5) | 0.1271% | 2.1257% | 0 |
| LUAD_STK11 | artifact | 142/141 warn | 32651 (1686/225) | 0.3314% | 1.8482% | 5 |
| TGCT | artifact | 154/153 warn | 32651 (1686/225) | 0.3199% | 1.8172% | 7 |
| ESCA | artifact | 182/181 warn | 32651 (1686/225) | 0.2885% | 1.7402% | 6 |
| LAML_ELNadv | artifact | 189/188 warn | 57233 (2616/426) | 0.2954% | 1.6577% | 22 |
| LUAD_KRAS | artifact | 153/152 warn | 32651 (1686/225) | 0.3000% | 1.6454% | 5 |
| BLCA | artifact | 407/404 warn | 32651 (1686/225) | 0.2192% | 1.3120% | 6 |
| SKCM | artifact | 469/466 warn | 32651 (1686/225) | 0.2453% | 1.2880% | 7 |
| UCEC_POLE | artifact | 16/16 | 32871 (1700/5) | 0.0879% | 1.1807% | 0 |
| COAD_MSS | artifact | 226/225 warn | 32651 (1686/225) | 0.1836% | 0.9903% | 5 |
| STAD | artifact | 414/412 warn | 32651 (1686/225) | 0.1468% | 0.8888% | 6 |
| HEPB | artifact | 20/20 | 32651 (1686/225) | 0.0749% | 0.8794% | 7 |
| CLL | artifact | 708/703 warn | 49221 (2575/277) | 0.1488% | 0.8329% | 12 |
| THCA | artifact | 512/508 warn | 32651 (1686/225) | 0.1667% | 0.8279% | 9 |
| COAD | artifact | 290/289 warn | 32651 (1686/225) | 0.1498% | 0.8077% | 5 |
| NET_RECTAL | artifact | 18/18 | 55986 (2681/1814) | 0.0670% | 0.7346% | 51 |
| SARC_CHOR | artifact | 20/20 | 59413 (2620/463) | 0.0577% | 0.7244% | 14 |
| KIRC | artifact | 531/528 warn | 32651 (1686/225) | 0.1332% | 0.7004% | 5 |
| SARC_ANGIO | artifact | 20/20 | 32651 (1686/225) | 0.0684% | 0.5160% | 11 |
| LUAD | artifact | 515/514 warn | 32651 (1686/225) | 0.1080% | 0.5103% | 5 |
| UCEC_CNL | artifact | 30/30 | 32871 (1700/5) | 0.0599% | 0.3645% | 0 |
| STAD_EBV | artifact | 30/30 | 32871 (1700/5) | 0.0571% | 0.2720% | 0 |
| NUTM | artifact | 3/3 | 19773 (172/147) | 0.1064% | 0.2712% | 25 |
| BRCA | artifact | 1099/1098 warn | 32651 (1686/225) | 0.0713% | 0.2624% | 6 |
| CHOL | artifact | 36/36 | 32651 (1686/225) | 0.0551% | 0.2365% | 6 |
| SARC_DDLPS | artifact | 48/48 | 32651 (1686/225) | 0.0507% | 0.1857% | 6 |
| SARC_PEC | artifact | 60/60 | 21351 (95/0) | 0.0501% | 0.1798% | 0 |
| STAD_GS | artifact | 50/50 | 32871 (1700/5) | 0.0502% | 0.1758% | 0 |
| COAD_MSI | artifact | 50/50 | 32651 (1686/225) | 0.0497% | 0.1711% | 6 |
| SARC_MYXLPS | artifact | 28/28 | 13520 (112/25) | 0.0528% | 0.1623% | 12 |
| SCLC_ASCL1 | artifact | 61/61 | 16805 (82/5) | 0.0476% | 0.1562% | 0 |
| HNSC_HPVpos | artifact | 72/72 | 32651 (1686/225) | 0.0472% | 0.1562% | 6 |
| CTCL | artifact | 7/7 | 16900 (102/10) | 0.0496% | 0.1533% | 1 |
| SCLC | artifact | 81/81 | 16805 (82/5) | 0.0474% | 0.1515% | 0 |
| LAML | artifact | 173/173 | 32651 (1686/225) | 0.0465% | 0.1500% | 11 |
| SARC_LPS_UNSPEC | artifact | 92/92 | 32651 (1686/225) | 0.0471% | 0.1492% | 7 |
| LGG | artifact | 523/523 | 32651 (1686/225) | 0.0466% | 0.1465% | 7 |
| SARC_GIST | artifact | 19/19 | 32651 (1686/225) | 0.0465% | 0.1464% | 4 |
| SARC_UPS | artifact | 110/110 | 32651 (1686/225) | 0.0474% | 0.1464% | 6 |
| READ_MSS | artifact | 83/83 | 32651 (1686/225) | 0.0458% | 0.1445% | 5 |
| BRCA_LumA | artifact | 501/501 | 32651 (1686/225) | 0.0457% | 0.1436% | 7 |
| SARC_ASPS | artifact | 3/3 | 32651 (1686/225) | 0.0447% | 0.1435% | 6 |
| PRAD | artifact | 496/496 | 32651 (1686/225) | 0.0455% | 0.1433% | 6 |
| GBM | artifact | 166/166 | 32651 (1686/225) | 0.0456% | 0.1431% | 6 |
| STAD_MSI | artifact | 73/73 | 32871 (1700/5) | 0.0453% | 0.1426% | 0 |
| SARC_DSRCT | artifact | 9/9 | 32651 (1686/225) | 0.0458% | 0.1424% | 6 |
| BRCA_LumB | artifact | 199/199 | 32651 (1686/225) | 0.0462% | 0.1423% | 5 |
| READ | artifact | 93/93 | 32651 (1686/225) | 0.0466% | 0.1422% | 5 |
| SARC_EHE | artifact | 1/1 | 32651 (1686/225) | 0.0436% | 0.1421% | 5 |
| CML | artifact | 5/5 | 19192 (103/11) | 0.0458% | 0.1414% | 0 |
| THYM | artifact | 119/119 | 32651 (1686/225) | 0.0445% | 0.1410% | 7 |
| SARC_MYXFIB | artifact | 41/41 | 32651 (1686/225) | 0.0462% | 0.1397% | 6 |
| DLBC | artifact | 47/47 | 32651 (1686/225) | 0.0429% | 0.1391% | 5 |
| SARC_OS | artifact | 262/262 | 32651 (1686/225) | 0.0466% | 0.1390% | 6 |
| BRCA_Basal | artifact | 172/172 | 32651 (1686/225) | 0.0451% | 0.1389% | 6 |
| SARC_LMS | artifact | 151/151 | 32651 (1686/225) | 0.0458% | 0.1384% | 6 |
| UCEC_MSI | artifact | 41/41 | 32871 (1700/5) | 0.0439% | 0.1381% | 0 |
| CESC | artifact | 306/306 | 32651 (1686/225) | 0.0453% | 0.1374% | 5 |
| WILMS | artifact | 125/125 | 57233 (2616/426) | 0.0410% | 0.1366% | 22 |
| STAD_CIN | artifact | 221/221 | 32871 (1700/5) | 0.0452% | 0.1365% | 0 |
| UCS | artifact | 57/57 | 32651 (1686/225) | 0.0437% | 0.1365% | 6 |
| BRCA_Normal | artifact | 35/35 | 32651 (1686/225) | 0.0459% | 0.1362% | 6 |
| HNSC_HPVneg | artifact | 415/415 | 32651 (1686/225) | 0.0437% | 0.1357% | 6 |
| LUSC | artifact | 498/498 | 32651 (1686/225) | 0.0441% | 0.1353% | 5 |
| MESO | artifact | 87/87 | 32651 (1686/225) | 0.0437% | 0.1353% | 8 |
| SARC_MPNST | artifact | 13/13 | 32651 (1686/225) | 0.0447% | 0.1352% | 5 |
| BRCA_HER2 | artifact | 77/77 | 32651 (1686/225) | 0.0447% | 0.1352% | 5 |
| SARC_EPITH | artifact | 5/5 | 32651 (1686/225) | 0.0447% | 0.1347% | 5 |
| SARC_WDLPS | artifact | 5/5 | 32651 (1686/225) | 0.0440% | 0.1342% | 7 |
| OV | artifact | 426/426 | 32651 (1686/225) | 0.0443% | 0.1332% | 5 |
| LUAD_EGFR | artifact | 67/67 | 32651 (1686/225) | 0.0453% | 0.1329% | 6 |
| NBL | artifact | 155/155 | 57233 (2616/426) | 0.0415% | 0.1296% | 21 |
| HNSC | artifact | 520/520 | 32651 (1686/225) | 0.0438% | 0.1296% | 6 |
| MTC | artifact | 52/52 | 18844 (122/24) | 0.0533% | 0.1287% | 1 |
| NBL_MYCNamp | artifact | 33/33 | 57233 (2616/426) | 0.0408% | 0.1261% | 18 |
| LIHC | artifact | 368/368 | 32651 (1686/225) | 0.0410% | 0.1239% | 5 |
| BL | artifact | 175/175 | 57659 (2620/0) | 0.0399% | 0.1085% | 0 |
| SARC_MMNST | artifact | 3/3 | 41855 (2591/0) | 0.0390% | 0.1036% | 0 |
| RB | artifact | 15/15 | 32651 (1686/225) | 0.0357% | 0.0955% | 7 |
