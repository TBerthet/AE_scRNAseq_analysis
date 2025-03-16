# Project Scivar : Deep learning algorithms for single-cell RNA-seq data clustering

## Introduction
The project **Scivar** contains :
- implementation of scDeepCluster and Contrastive-sc
- reproduction of the results of those methods (and the PCA) using the PBMC 4k dataset
- study of hyperparameters for those methods using another dataset (Baron)

The notebooks at the root of the project are cleaned version of the implementation and illustrate the followed approached.

## Articles folder
Contains useful articles papers included scDeepCluster, Contrastive-sc and related works

## Data folder
This is the folder where results are saved into. So it contains different experiment results (pdf plots and csv files)

## Dataset folder
Contains the PBMC 10X 4k and the Baron dataset

## env_scivar folder
Used to create the virtual environnment with the necessary modules

## logs folder
Contains tensorboard statistics (from previous experiments)

## model folder
folder where are saved model weights

## other_notebooks folder
Contains bunch of notebooks not really clean use for experiments or draft

## scDeepCluster folder
Folder created to run the original implementation of scDeepCluster


