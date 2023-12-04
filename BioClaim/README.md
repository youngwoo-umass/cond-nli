# BioClaim
The BioClaim dataset is a token-level annotated dataset on conditionally-compatible sentence pairs. 
There are two token level labels "neutral tokens" and "contradictory tokens".
* "Neutral tokens" are the tokens that represents the different conditions of the two claims. 
* "Contradiction tokens" are the tokens that indicate opposite outcomes of the two claims.

# Files

"val.csv" and "test.csv" contain all the data for validation split and test split.

# Example

Each row of the  csv file represents the following information.

| sent1            | sent1_contradiction | sent1_neutral | sent2            | sent2_contradiction | sent2_neutral |
|------------------|---------------------|---------------|------------------|---------------------|---------------|
| We               | 0                   | 0             | Oral             | 0                   | 1             |
| conclude         | 0                   | 0             | L-arginine       | 0                   | 0             |
| that             | 0                   | 0             | supplementation  | 0                   | 0             |
| in               | 0                   | 0             | did              | 1                   | 0             |
| women            | 0                   | 0             | not              | 1                   | 0             |
| with             | 0                   | 0             | reduce           | 1                   | 0             |
| preeclampsia,    | 0                   | 0             | mean             | 0                   | 1             |
| prolonged        | 0                   | 1             | diastolic        | 0                   | 1             |
| dietary          | 0                   | 1             | blood            | 0                   | 0             |
| supplementation  | 0                   | 0             | pressure         | 0                   | 0             |
| with             | 0                   | 0             | after            | 0                   | 1             |
| l-arginine       | 0                   | 0             | 2                | 0                   | 1             |
| significantly    | 0                   | 0             | days             | 0                   | 1             |
| decreased        | 1                   | 0             | of               | 0                   | 1             |
| blood            | 0                   | 0             | treatment        | 0                   | 1             |
| pressure         | 0                   | 0             | compared          | 0                   | 1             |
| through          | 0                   | 0             | with             | 0                   | 1             |
| increased        | 0                   | 0             | placebo          | 0                   | 1             |
| endothelial      | 0                   | 0             | in               | 0                   | 0             |
| synthesis        | 0                   | 0             | pre-eclamptic    | 0                   | 0             |
| and/or           | 0                   | 0             | patients         | 0                   | 1             |
| bioavailability  | 0                   | 0             | with             | 0                   | 1             |
| of               | 0                   | 0             | gestational      | 0                   | 1             |
| NO.              | 0                   | 0             | length           | 0                   | 1             |
|                 |                     |               | varying          | 0                   | 1             |
|                 |                     |               | from             | 0                   | 1             |
|                 |                     |               | 28               | 0                   | 1             |
|                 |                     |               | to               | 0                   | 1             |
|                 |                     |               | 36               | 0                   | 1             |
|                 |                     |               | weeks            | 0                   | 1             |



# Source dataset

The BioClaim dataset is an extension of an existing dataset. 

*  Alamri, Abdulaziz, and Mark Stevenson. "A corpus of potentially contradictory research claims from cardiovascular research abstracts." Journal of biomedical semantics 7.1 (2016): 1-9.
* https://staffwww.dcs.shef.ac.uk/people/m.stevenson/resources/bio_contradictions/

# To cite this dataset

Please cite our EMNLP 2023 paper titled "Conditional Natural Langauge Inference"

# License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg