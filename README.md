# Steric Exclusion Model (SEM) v2  

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
*Predicting ionic currents through nanopores with ion concentration effects*  

---

## Table of Contents  
- [Overview](#overview)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Basic SEM (Original)](#1-running-sem-original-version)  
  - [SEM2 (With Concentration)](#2-running-sem2-with-ion-concentration)  
- [Citation](#citation)  

---

## Overview  
The **Steric Exclusion Model (SEM)** predicts ionic current through nanopores. The first version was published in [*ACS Sensors* (2018)](https://pubs.acs.org/doi/10.1021/acssensors.8b01375).  

**SEM2** extends the model by incorporating **ion concentration effects** for realistic electrolyte conditions.  

---

## Installation  
1. Clone this repository:  
     
   git clone https://github.com/your_username/SEM2.git  
   cd SEM2  

2. Set up the Conda environment 

    conda env create -f environment.yml
    conda activate sem2_env
---

## Usage

1. Running SEM (Original Version)

To predict ionic current without concentration effects, use:

python3 run_sem.py example.psf example.dcd \
    --sigma 11.2           # Conductivity of 1M KCl (S/m) \
    --volt 0.2           # Applied voltage (V) \
    --xymargin 10.0     # XY margin (Å) \
    --zmargin 5.0          # Z margin (Å) \
    --o currents.dat        # Output current file (nA)

2. Running SEM2 (With UnEven Distribution of Ion Concentration Considered)

To include ion concentration effects, add --consider_conc and specify bulk concentration (e.g., 1.0 M):

python3 run_sem.py example.psf example.dcd \
    --sigma 11.2 \
    --volt 0.2 \
    --xymargin 10.0 \
    --zmargin 5.0 \
    --consider_conc \       # Enable concentration effects
    --bulk_conc 1.0 \       # Bulk ion concentration (M)
    --o currents.dat


---
## Citation
If you use SEM/SEM2 in your research, please cite the original SEM paper:

@article{wilson2019rapid,
  title={Rapid and accurate determination of nanopore ionic current using a steric exclusion model},
  author={Wilson, James and Sarthak, Kumar and Si, Wei and Gao, Luyu and Aksimentiev, Aleksei},
  journal={Acs Sensors},
  volume={4},
  number={3},
  pages={634--644},
  year={2019},
  publisher={ACS Publications}
}


