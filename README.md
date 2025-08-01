

# IMDgroup-gorun

This is a set of scripts for supercomputer job submission for VASP and
ATAT via [slurm](https://slurm.schedmd.com/), that are tailored to research that is performed in
[Inverse Materials Design group](https://www.oimalyi.org/).


# Installation

    # Download the package and source dependencies
    git clone https://git.sr.ht/~yantar92/IMDgroup-gorun
    
    # Activate virtual environment
    pip -m venv .venv
    . .venv/bin/activate
    
    # Install into the environment
    pip install IMDgroup-gorun


# Features

-   Submit slurm jobs in a single unified command, consistent across
    different supercomputers (specific per-cluster configuration is
    configured separately)
-   Automatically detect the most suitable queue (experimental)
-   Make sure that the VASP input to be submitted is sane
    -   Check for any issues in ICNAR/POSCAR and automatically resolve (when safe)
    -   Check if VASP calcualtion is already converged
-   If an existing VASP output is present, automatically back it up
    and restart the VASP, using the results of the previous calculation
-   Automatically generate POTCAR file
-   Automatically setup vdW kernel
-   Submit ATAT jobs in a single unified command
    -   Customize individual VASP runs invoked by ATAT


# TODO Usage


# TODO Citing


# Contributing

We welcome contributions in all forms. If you want to contribute,
please fork this repository, make changes and send us a pull request!


# Acknowledgements

We acknowledge financial support from the National Centre for Research
and Development (NCBR) under project
WPC3/2022/50/KEYTECH/2024. Computational resources were provided by
the Polish high-performance computing infrastructure PLGrid, including
access to the LUMI supercomputer—owned by the EuroHPC Joint
Undertaking and hosted by CSC in Finland together with the LUMI
Consortium—through allocation PLL/2024/07/017633, as well as
additional resources at the PLGrid HPC centres ACK Cyfronet AGH and
WCSS under allocation PLG/2024/017498.

