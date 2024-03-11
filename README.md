# FlowForecaster
The objective of this project is to leverage SVR models to predict traffic flow based on time. The models include a linear baseline, and Gaussian, RBF, and Polynomial kernels, all subject to hyperparameter optimization in Matlab

This project applies Support Vector Regression (SVR) with a focus on optimizing kernel functions for traffic flow prediction. We compare the performance of each Kernel. Additional implementations include hyperparameter optimization, randomsearch and grid-search. 

## Data Preprocessing
Dataaccess.m is the file for splitting into train and test

## Model Training
SVR models are trained with different kernels. Hyperparameter optimization is performed using Bayesian optimization to avoid extensive grid searches.
We have four scripts for each, Gaussian, RBF, Linear, Polynomial

## Usage
Instructions on how to set up, train, and test the models, with code snippets and examples.

## Experiment

Data obtained from 
https://hub.arcgis.com/datasets/9cb86b342f2d4f228067a7437a7f7313/about

![evaluate](/images/eva.png)

## Reference

	@article{SAEs,  
	  title={Traffic Flow Prediction With Big Data: A Support Vector Machine Approach},  
	  author={Manto, J.M},
	  journal={still drafting}
	}


## Copyright
See [LICENSE](LICENSE) for details.
