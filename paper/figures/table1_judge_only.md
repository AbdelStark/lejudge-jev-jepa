| judge | source | description | n_items | canonical_precision | canonical_recall | canonical_f1 | canonical_accuracy | canonical_auroc | canonical_ece | paraphrase_precision | paraphrase_recall | paraphrase_f1 | paraphrase_accuracy | paraphrase_auroc | paraphrase_ece | negative_fpr | latency_ms_per_1000 | tokens_in_per_1000 | failed_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | executed | gt-words | 64 | 0.786 | 0.924 | 0.849 | 0.883 | 0.955 | 0.113 | 0.799 | 0.807 | 0.803 | 0.858 | 0.933 | 0.088 | 0.535 | 2458.217 | 139883.984 | 0.000 |
| jev | executed | probe-words | 64 | 0.777 | 0.862 | 0.817 | 0.862 | 0.939 | 0.083 | 0.808 | 0.773 | 0.790 | 0.853 | 0.920 | 0.077 | 0.529 | 2458.217 | 139883.984 | 0.000 |
| jev | imagined | probe-words | 128 | 0.752 | 0.813 | 0.781 | 0.881 | 0.937 | 0.107 | 0.765 | 0.731 | 0.748 | 0.871 | 0.922 | 0.076 | 0.383 | 2561.187 | 139829.395 | 0.000 |
| keyword | executed | gt-words | 64 | 0.883 | 0.825 | 0.853 | 0.898 | 0.918 | 0.075 | nan | 0.000 | nan | 0.642 | 0.500 | 0.358 | 0.025 | 0.000 | 0.000 | 0.000 |
| keyword | executed | probe-words | 64 | 0.869 | 0.796 | 0.831 | 0.884 | 0.901 | 0.086 | nan | 0.000 | nan | 0.642 | 0.500 | 0.358 | 0.024 | 0.000 | 0.000 | 0.000 |
| keyword | imagined | probe-words | 128 | 0.783 | 0.823 | 0.802 | 0.894 | 0.895 | 0.086 | nan | 0.000 | nan | 0.738 | 0.500 | 0.262 | 0.020 | 0.000 | 0.000 | 0.000 |
| llm | executed | gt-words | 8 | 0.474 | 0.514 | 0.493 | 0.615 | 0.611 | 0.332 | nan | nan | nan | 0.000 | nan | nan | nan | 565954.823 | 196402.344 | 0.667 |
| llm | executed | probe-words | 8 | 0.447 | 0.486 | 0.466 | 0.594 | 0.588 | 0.356 | nan | nan | nan | 0.000 | nan | nan | nan | 565954.823 | 196402.344 | 0.667 |
| oracle | executed | gt-words | 64 | 1.000 | 0.920 | 0.958 | 0.971 | 1.000 | 0.013 | 1.000 | 0.920 | 0.958 | 0.971 | 1.000 | 0.013 | nan | 0.000 | 0.000 | 0.000 |
| oracle | executed | probe-words | 64 | 1.000 | 0.920 | 0.958 | 0.971 | 1.000 | 0.013 | 1.000 | 0.920 | 0.958 | 0.971 | 1.000 | 0.013 | nan | 0.000 | 0.000 | 0.000 |
| oracle | imagined | probe-words | 128 | 1.000 | 0.945 | 0.972 | 0.986 | 1.000 | 0.003 | 1.000 | 0.945 | 0.972 | 0.986 | 1.000 | 0.003 | nan | 0.000 | 0.000 | 0.000 |