# cancer_reference_expression delegation parity (#557)

Compares pirlygenes' delegated source-union compatibility rows against oncoref's canonical selected/QC-aware artifact view, per `(cancer_code, Ensembl_Gene_ID)` at `tpm_clean`. Relative deltas are on the median `expression`, over genes with pg TPM >= 1.0.

This intentionally compares two different oncoref products: the all-sample source-union compatibility rows and the default pass-QC selected artifact. Large outliers therefore identify QC/source-product differences, not adapter distortion; exact adapter-to-source parity for TCGA, heme, microarray, subtype, and computed-union cases is gated in `tests/test_reference_expression_delegation.py`.

- cancer_codes compared: **139**
- served by both sides: **139**
- n_samples agreement: **124/139** codes match exactly
- median relative delta (across codes): **0.0463%**
- worst-code p95 relative delta: **237.9842%**

## Per-code detail

| cancer_code | qc | n_samp pg/on | genes shared (pg-only/on-only) | rel median | rel p95 | divergent |
| --- | --- | --- | --- | --- | --- | --- |
| HL | artifact | 5/3 warn | 52687 (7777/0) | 10.2348% | 237.9842% | 4032 |
| SARC_LGFMS | artifact | 2/2 | 18812 (93/10824) | 49.2361% | 180.5148% | 6286 |
| MM | artifact | 764/81 warn | 52545 (7734/0) | 97.1611% | 139.1152% | 10692 |
| ACINIC | artifact | 3/2 warn | 51380 (7571/0) | 18.2981% | 128.4601% | 2473 |
| NET_PANCREAS | artifact | 113/30 warn | 52687 (7777/0) | 20.0932% | 68.2410% | 2158 |
| SARC_IFS | artifact | 2/2 | 29636 (4935/0) | 4.7388% | 50.0245% | 716 |
| SCLC_YAP1 | artifact | 2/2 | 16810 (79/0) | 5.4194% | 44.4872% | 430 |
| READ_MSI | artifact | 2/2 | 29636 (4935/0) | 1.5070% | 28.0949% | 169 |
| NPC | artifact | 4/4 | 29636 (4935/0) | 1.5300% | 26.6206% | 123 |
| SARC_CHON | artifact | 54/47 warn | 54762 (7735/0) | 2.9210% | 24.0884% | 214 |
| ATRT | artifact | 4/4 | 29636 (4935/0) | 0.9108% | 20.6634% | 96 |
| SARC_PLEOLPS | artifact | 4/4 | 18825 (80/0) | 0.5212% | 12.3692% | 42 |
| SARC_ESS_HG | artifact | 4/4 | 19941 (164/0) | 0.3174% | 12.1989% | 74 |
| SARC_PEC | artifact | 69/66 warn | 21351 (95/0) | 1.9047% | 11.2162% | 3 |
| LAML_APL | artifact | 20/19 warn | 52545 (7734/0) | 1.9161% | 10.8540% | 7 |
| SARC_IMT | artifact | 4/4 | 29636 (4935/0) | 0.2360% | 6.3114% | 72 |
| SARC_RMS_PRMS | artifact | 6/6 | 29636 (4935/0) | 0.2862% | 6.1338% | 13 |
| NET_MIDGUT | artifact | 81/78 warn | 52687 (7777/0) | 1.2162% | 6.0757% | 2 |
| SARC_RMS_SSRMS | artifact | 8/8 | 29636 (4935/0) | 0.2477% | 5.8354% | 8 |
| SCLC_NEUROD1 | artifact | 8/8 | 16810 (79/0) | 0.1791% | 5.1834% | 3 |
| MDS | artifact | 82/81 warn | 52687 (7777/0) | 0.4813% | 2.6351% | 0 |
| FL | artifact | 6/6 | 44399 (6983/0) | 0.1122% | 2.4759% | 9 |
| GBC | artifact | 10/10 | 28043 (2665/0) | 0.1133% | 2.4029% | 0 |
| B_ALL | artifact | 154/153 warn | 52545 (7734/0) | 0.4446% | 2.3678% | 0 |
| BL | artifact | 184/182 warn | 52545 (7734/0) | 0.3527% | 2.1968% | 0 |
| SARC_KS | artifact | 10/10 | 50643 (7732/0) | 0.1431% | 2.1892% | 3 |
| SCLC_POU2F3 | artifact | 10/10 | 16810 (79/0) | 0.1271% | 2.1309% | 0 |
| cSCC | artifact | 10/10 | 15107 (156/0) | 0.1041% | 1.5116% | 0 |
| PAAD | artifact | 179/178 warn | 29636 (4935/0) | 0.2255% | 1.2063% | 0 |
| UCEC_POLE | artifact | 16/16 | 29636 (4935/0) | 0.0872% | 1.1596% | 0 |
| HEPB | artifact | 20/20 | 29636 (4935/0) | 0.0750% | 0.8706% | 0 |
| SARC_CHOR | artifact | 20/20 | 54762 (7735/0) | 0.0637% | 0.7593% | 2 |
| STAD | artifact | 414/413 warn | 29636 (4935/0) | 0.1337% | 0.6639% | 0 |
| NET_RECTAL | artifact | 18/18 | 52687 (7777/0) | 0.0669% | 0.6554% | 0 |
| SARC_ANGIO | artifact | 20/20 | 29636 (4935/0) | 0.0675% | 0.5003% | 0 |
| LGG | artifact | 523/522 warn | 29636 (4935/0) | 0.0947% | 0.4291% | 0 |
| UCEC_CNL | artifact | 30/30 | 29636 (4935/0) | 0.0612% | 0.3716% | 0 |
| STAD_EBV | artifact | 30/30 | 29636 (4935/0) | 0.0570% | 0.2654% | 0 |
| BRCA | artifact | 1099/1098 warn | 29636 (4935/0) | 0.0726% | 0.2616% | 0 |
| CHOL | artifact | 36/36 | 29636 (4935/0) | 0.0554% | 0.2380% | 0 |
| MBL_G3 | artifact | 44/44 | 29636 (4935/0) | 0.0564% | 0.2334% | 0 |
| SARC_DDLPS | artifact | 48/48 | 29636 (4935/0) | 0.0504% | 0.1849% | 0 |
| CMN | artifact | 12/12 | 13214 (179/0) | 0.0631% | 0.1819% | 0 |
| SARC_SYN | artifact | 50/50 | 29636 (4935/0) | 0.0532% | 0.1816% | 0 |
| STAD_GS | artifact | 50/50 | 29636 (4935/0) | 0.0498% | 0.1754% | 0 |
| NUTM | artifact | 3/3 | 19920 (111/0) | 0.0567% | 0.1688% | 0 |
| COAD_MSI | artifact | 50/50 | 29636 (4935/0) | 0.0508% | 0.1681% | 0 |
| SARC_MYXLPS | artifact | 28/28 | 13471 (182/0) | 0.0534% | 0.1634% | 0 |
| KICH | artifact | 66/66 | 29636 (4935/0) | 0.0483% | 0.1584% | 0 |
| SCLC_ASCL1 | artifact | 61/61 | 16810 (79/0) | 0.0476% | 0.1562% | 0 |
| HCL | artifact | 5/5 | 14704 (94/0) | 0.0508% | 0.1558% | 0 |
| BCC | artifact | 25/25 | 15107 (156/0) | 0.0483% | 0.1552% | 0 |
| HNSC_HPVpos | artifact | 72/72 | 29636 (4935/0) | 0.0480% | 0.1535% | 0 |
| CTCL | artifact | 7/7 | 16905 (99/0) | 0.0494% | 0.1528% | 0 |
| LAML | artifact | 173/173 | 29636 (4935/0) | 0.0475% | 0.1523% | 0 |
| SCLC | artifact | 81/81 | 16810 (79/0) | 0.0474% | 0.1515% | 0 |
| SARC_ESS_LG | artifact | 9/9 | 19941 (164/0) | 0.0445% | 0.1502% | 0 |
| SARC_UPS | artifact | 110/110 | 29636 (4935/0) | 0.0476% | 0.1483% | 0 |
| MBL_WNT | artifact | 17/17 | 29636 (4935/0) | 0.0497% | 0.1478% | 0 |
| SARC_LPS_UNSPEC | artifact | 92/92 | 29636 (4935/0) | 0.0474% | 0.1470% | 0 |
| SARC_CCS | artifact | 5/5 | 16310 (415/0) | 0.0470% | 0.1469% | 0 |
| CML | artifact | 5/5 | 19168 (129/0) | 0.0459% | 0.1468% | 0 |
| SARC_GIST | artifact | 19/19 | 29636 (4935/0) | 0.0463% | 0.1463% | 0 |
| UVM | artifact | 79/79 | 29636 (4935/0) | 0.0438% | 0.1459% | 0 |
| MBL_G4 | artifact | 39/39 | 29636 (4935/0) | 0.0468% | 0.1457% | 0 |
| SARC_MYXFIB | artifact | 41/41 | 29636 (4935/0) | 0.0454% | 0.1443% | 0 |
| MBL | artifact | 125/125 | 29636 (4935/0) | 0.0467% | 0.1439% | 0 |
| STAD_MSI | artifact | 73/73 | 29636 (4935/0) | 0.0458% | 0.1439% | 0 |
| READ | artifact | 93/93 | 29636 (4935/0) | 0.0461% | 0.1436% | 0 |
| SARC_DSRCT | artifact | 9/9 | 29636 (4935/0) | 0.0467% | 0.1434% | 0 |
| MBL_SHH | artifact | 25/25 | 29636 (4935/0) | 0.0470% | 0.1428% | 0 |
| SARC_EWS | artifact | 101/101 | 29636 (4935/0) | 0.0465% | 0.1427% | 0 |
| RB | artifact | 15/15 | 29636 (4935/0) | 0.0446% | 0.1425% | 0 |
| READ_MSS | artifact | 83/83 | 29636 (4935/0) | 0.0462% | 0.1423% | 0 |
| PRAD | artifact | 496/496 | 29636 (4935/0) | 0.0452% | 0.1422% | 0 |
| CLL | artifact | 708/708 | 44997 (7081/0) | 0.0444% | 0.1420% | 0 |
| EPN | artifact | 11/11 | 21374 (490/0) | 0.0469% | 0.1419% | 0 |
| SARC_ASPS | artifact | 3/3 | 29636 (4935/0) | 0.0449% | 0.1419% | 0 |
| SARC_RMS_ARMS | artifact | 73/73 | 29636 (4935/0) | 0.0462% | 0.1418% | 0 |
| SKCM | artifact | 469/469 | 29636 (4935/0) | 0.0450% | 0.1418% | 0 |
| COAD | artifact | 290/290 | 29636 (4935/0) | 0.0463% | 0.1416% | 0 |
| BRCA_LumA | artifact | 501/501 | 29636 (4935/0) | 0.0464% | 0.1411% | 0 |
| GBM | artifact | 166/166 | 29636 (4935/0) | 0.0459% | 0.1409% | 0 |
| TGCT | artifact | 154/154 | 29636 (4935/0) | 0.0445% | 0.1408% | 0 |
| PCPG | artifact | 182/182 | 29636 (4935/0) | 0.0450% | 0.1408% | 0 |
| ESCA | artifact | 182/182 | 29636 (4935/0) | 0.0458% | 0.1407% | 0 |
| DLBC | artifact | 47/47 | 29636 (4935/0) | 0.0437% | 0.1407% | 0 |
| KIRP | artifact | 289/289 | 29636 (4935/0) | 0.0455% | 0.1407% | 0 |
| BRCA_Basal | artifact | 172/172 | 29636 (4935/0) | 0.0451% | 0.1404% | 0 |
| SARC_EHE | artifact | 1/1 | 29636 (4935/0) | 0.0445% | 0.1402% | 0 |
| THCA | artifact | 512/512 | 29636 (4935/0) | 0.0459% | 0.1401% | 0 |
| KIRC | artifact | 531/531 | 29636 (4935/0) | 0.0459% | 0.1401% | 0 |
| UCEC | artifact | 181/181 | 29636 (4935/0) | 0.0444% | 0.1397% | 0 |
| COAD_MSS | artifact | 226/226 | 29636 (4935/0) | 0.0454% | 0.1395% | 0 |
| MCL | artifact | 51/51 | 25122 (605/0) | 0.0450% | 0.1394% | 0 |
| BRCA_LumB | artifact | 199/199 | 29636 (4935/0) | 0.0464% | 0.1391% | 0 |
| SARC_RMS_ERMS | artifact | 95/95 | 29636 (4935/0) | 0.0456% | 0.1388% | 0 |
| THYM | artifact | 119/119 | 29636 (4935/0) | 0.0447% | 0.1385% | 0 |
| OV | artifact | 426/426 | 29636 (4935/0) | 0.0448% | 0.1377% | 0 |
| BLCA | artifact | 407/407 | 29636 (4935/0) | 0.0443% | 0.1372% | 0 |
| LUAD_STK11 | artifact | 142/142 | 29636 (4935/0) | 0.0453% | 0.1371% | 0 |
| STAD_CIN | artifact | 221/221 | 29636 (4935/0) | 0.0460% | 0.1370% | 0 |
| UCEC_MSI | artifact | 41/41 | 29636 (4935/0) | 0.0440% | 0.1369% | 0 |
| WILMS | artifact | 125/125 | 52545 (7734/0) | 0.0411% | 0.1368% | 0 |
| CESC | artifact | 306/306 | 29636 (4935/0) | 0.0442% | 0.1366% | 0 |
| SARC_OS | artifact | 262/262 | 29636 (4935/0) | 0.0450% | 0.1363% | 0 |
| UCEC_CNH | artifact | 85/85 | 29636 (4935/0) | 0.0441% | 0.1362% | 0 |
| HNSC | artifact | 520/520 | 29636 (4935/0) | 0.0441% | 0.1357% | 0 |
| NEC_MERKEL | artifact | 91/91 | 23169 (96/0) | 0.0450% | 0.1357% | 0 |
| BRCA_HER2 | artifact | 77/77 | 29636 (4935/0) | 0.0456% | 0.1352% | 0 |
| ACC | artifact | 77/77 | 29636 (4935/0) | 0.0442% | 0.1349% | 0 |
| UCS | artifact | 57/57 | 29636 (4935/0) | 0.0435% | 0.1345% | 0 |
| SARC_LMS | artifact | 151/151 | 29636 (4935/0) | 0.0448% | 0.1343% | 0 |
| SARC_MPNST | artifact | 13/13 | 29636 (4935/0) | 0.0454% | 0.1342% | 0 |
| NBL_MYCNnonamp | artifact | 122/122 | 52545 (7734/0) | 0.0427% | 0.1341% | 0 |
| SARC_WDLPS | artifact | 5/5 | 29636 (4935/0) | 0.0441% | 0.1338% | 0 |
| LUSC | artifact | 498/498 | 29636 (4935/0) | 0.0445% | 0.1331% | 0 |
| SARC_EPITH | artifact | 5/5 | 29636 (4935/0) | 0.0444% | 0.1329% | 0 |
| BRCA_Normal | artifact | 35/35 | 29636 (4935/0) | 0.0455% | 0.1327% | 0 |
| LUAD_EGFR | artifact | 67/67 | 29636 (4935/0) | 0.0443% | 0.1325% | 0 |
| LUAD_KRAS | artifact | 153/153 | 29636 (4935/0) | 0.0442% | 0.1324% | 0 |
| MPN | artifact | 45/45 | 32578 (5499/0) | 0.0402% | 0.1324% | 0 |
| LAML_ELNint | artifact | 154/154 | 52545 (7734/0) | 0.0411% | 0.1323% | 0 |
| MESO | artifact | 87/87 | 29636 (4935/0) | 0.0435% | 0.1320% | 0 |
| LUAD | artifact | 515/515 | 29636 (4935/0) | 0.0442% | 0.1316% | 0 |
| HNSC_HPVneg | artifact | 415/415 | 29636 (4935/0) | 0.0448% | 0.1304% | 0 |
| NBL_MYCNamp | artifact | 33/33 | 52545 (7734/0) | 0.0416% | 0.1297% | 0 |
| LAML_ELNadv | artifact | 189/189 | 52545 (7734/0) | 0.0404% | 0.1281% | 0 |
| LAML_ELNfav | artifact | 238/238 | 52545 (7734/0) | 0.0406% | 0.1266% | 0 |
| ADCC | artifact | 57/57 | 51380 (7571/0) | 0.0422% | 0.1262% | 0 |
| NBL | artifact | 155/155 | 52545 (7734/0) | 0.0422% | 0.1258% | 0 |
| MTC | artifact | 52/52 | 18862 (105/0) | 0.0536% | 0.1257% | 0 |
| RT | artifact | 63/63 | 52545 (7734/0) | 0.0402% | 0.1251% | 0 |
| CRANIO | artifact | 29/29 | 48965 (7732/0) | 0.0425% | 0.1231% | 0 |
| NET_LUNG | artifact | 118/118 | 51445 (7733/0) | 0.0400% | 0.1213% | 0 |
| LIHC | artifact | 368/368 | 29636 (4935/0) | 0.0410% | 0.1178% | 0 |
| T_ALL | artifact | 264/264 | 52545 (7734/0) | 0.0401% | 0.1129% | 0 |
| NEC_LUNG_LARGECELL | artifact | 69/69 | 51445 (7733/0) | 0.0386% | 0.1082% | 0 |
| SARC_MMNST | artifact | 3/3 | 38064 (6382/0) | 0.0391% | 0.1030% | 0 |
