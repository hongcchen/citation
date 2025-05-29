# The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research

This repository contains the code and resources for our ACL 2025 paper: **[The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research](https://github.com/hongcchen/citation_fidelity/edit/main/README.md)**. 

In this study, we introduce a computational pipeline to quantify citation fidelity at scale. Using full texts of papers, the pipeline identifies citations in citing papers and the corresponding claims in cited papers, and applies supervised models to measure fidelity at the sentence level. Analyzing a large-scale multi-disciplinary dataset of approximately 13 million citation sentence pairs, we find that citation fidelity is higher when authors cite papers that are 1) more recent and intellectually close, 2) more accessible, and 3) the first author has a lower H-index and the author team is medium-sized.Using a quasi-experiment, we establish the "telephone effect" -- when citing papers have low fidelity to the original claim, future papers that cite the citing paper and the original have lower fidelity to the original. 

## Project Structure

- `data/`: data sample  
- `notebooks/`: Jupyter notebooks for exploratory analysis  
- `src/`: code scripts  
- `README.md`: Project description and usage
