# Cell Deconvolution

This package's primary intentions are to provide the user with the following:

1. Easy to Use
2. Easy to Deploy any current models from the literature
3. Contains a Gene Expression Signature Builder (gene-average, gene-sum, gene-average-of-non-zero-entries)
4. Contains all the modules and routines to reproduce benchmarking results
5. ⚠️ Has Good Documentation ⚠️

### Package Motivation

Cellular deconvolution (also referred to as cell type composition or cell proportion estimation) refers to computational techniques aiming at estimating the proportions of different cell types in samples collected from a tissue. Over the past few years many methods have been implemented using a wide spread of machine learning methods which have been considered the "State of the Art". However based on the paper, [Clustering-independent estimation of cell abundances in bulk tissues using single-cell RNA-seq data](https://www.biorxiv.org/content/10.1101/2023.02.06.527318v1.full.pdf), we were able to learn that a lot of deconvolutions methods accuracy is highly driven on the gene expression signature which are typically required as input to estimate the cell proportions. Since then there have been more methods developed that don't require a gene signature set but require some form of single cell reference to infer the cellular proportions. Therefore this notebook takes a look at widely different methods found across the literature and provides an easy to use interface for the user. Part of the challenge when working with open source codebases is that reproducability becomes a lot of work because there may be missing files, classified datasets involving real patients, etc. Therefore this notebook's emphasis is really just to provide the user with everything they will need to be able to perform a proper benchmark of different deconvolution methods for themselves. In this repository you will find two datasets in the `\data` folder, with the required paired bulk samples along with an othogonal flow cytometry matrix ("ground truth) to benchmark on PBMC related datasets (`GSE107572, GSE1479433`). If you wou;d like to retrain these models from scratch, you will need to provide a training set with a gene signature set coming from the same tissue. For validation purposes some form of orthogonal qunaitifcation is required. If you are interested in including your own models, you are also free to do so by following the pipeline demonstarted of the three different methods seen in this notebook: **cellanneal (annealing/rank coefficients minimization function), Kassandra (Ligh Gradient Boosting Decision Tree Model), & SVR (support vector regression)**. 

### Package Organization

```\data``` - this folder contains some starter benchmarking files. 

```\figures```

```\src```

```\test```

### Installing Environments & Requirements

```environment.yml ``` - file is included in the repository. Takes dependencies from ```requirements.txt```

To build an environment run the following command:

```conda env create -f environment.yml --name [environment_name] ```

```conda activate [environment_name]```

### What the Package does for You

### User requirements

### Gene Expression Signature Builder

### How to Run Package

### Authors



