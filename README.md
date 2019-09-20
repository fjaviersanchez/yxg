# YxG

This repository contains the analysis pipeline used in https://arxiv.org/abs/1909.09102. Please contact the authors if you have any question.

## Running the pipeline
1. Unpack and download all necessary data by running the bash script `dwl_data.sh`. This will unpack `data.tar.gz`, download all necessary Planck maps and compute the SZ masks using `mk_mask_sz.py`.
2. Compute all power spectra and covariance matrices running `python pipeline.py params.yaml`.
3. Run the likelihood sampler by running `python mcmc.py params.yaml`.
3. Most of the paper plots can be generated by running the `plot_***.py` scripts. Some hard-coded paths may need to be modified.

## Source code
- `dwl_data.sh` downloads all the data needed for the analysis.
- All the data analysis modules can be found in `analysis`.
- The theory prediction modules can be found in `model`.
- The likelihood modules are in `likelihood`.
- `pipeline.py` contains the power spectrum measurement pipeline.
- `mcmc.py` contains the likelihood pipeline.
- `plot_stuff.py` contains a few plotting routines.

## Theory notes
Theory notes can be found in `notes`

## Credit
We ask that you cite https://arxiv.org/abs/1909.09102 if you use this pipeline for any of your work.
