# Kolmogorov-Arnold Networks (KANs) and B-Splines Implementation

An academic exploration and implementation of Kolmogorov-Arnold Networks (KANs) as an alternative to traditional Multi-Layer Perceptrons (MLPs). This project demonstrates how replacing linear weights and static activation functions with learnable B-splines on the network edges can improve model accuracy and interpretability.

## Project Overview
Traditional neural networks (MLPs) struggle with the "curse of dimensionality" and often lack interpretability. Based on the Kolmogorov-Arnold representation theorem, this project builds a custom neural network architecture in R where non-linear computations are handled on the edges using splines, while nodes perform simple summations. 

The project evaluates this architecture's performance through two primary applications:
1.  **Symbolic Regression:** Testing the model's ability to discover underlying mathematical functions from simulated data.
2.  **Comparative Analysis:** Benchmarking the KAN model against a classic MLP using real-world housing market data.

## The Data
*   **Simulated Data:** A generated dataset of 1,000 samples used to test symbolic regression, following the hidden function $y = \sin(\pi x_1) + x_2^2$.
*   **Real Data (`Housing.csv`):** A housing dataset comprising 13 variables (1 dependent variable for price, 12 independent variables including area, bedrooms, and categorical features). The data is preprocessed and min-max scaled to the $[-1, 1]$ range for optimal neural network training.

## Methodology
*   **B-Splines Generation:** Implemented the recursive Cox-de Boor algorithm from scratch to calculate the B-spline basis functions of any degree.
*   **Custom KAN Layer:** Developed a dynamic network layer utilizing the `torch` package in R. The layer employs the SiLU (Sigmoid Linear Unit) activation function to stabilize early training before the B-splines fully adapt.
*   **Optimization:** Both the KAN and the baseline MLP were trained using the Adam optimizer (`optim_adam`) and evaluated using Mean Squared Error (MSE).

## Prerequisites
To run the scripts in this repository, you will need **R** and the following library installed:
*   `torch`

## Running the Code
1. Ensure the `Housing.csv` dataset is located in your working directory.
2. Open and run the `Αριθμητικες_μεθοδοι_απαλλακτικη.R` script.
3. The script will automatically:
   * Generate the simulated data and train the KAN model.
   * Output two plots demonstrating the network's symbolic regression capabilities (comparing the learned splines to the true mathematical functions).
   * Preprocess the housing dataset, train both the KAN and a standard ReLU-based MLP, and output a final plot comparing their training loss (MSE) over 300 epochs.

## Full Academic Report
For a deep dive into the mathematical proofs, the architectural differences between KANs and MLPs, and a detailed discussion of the scaling laws and local support features of splines, please refer to the complete report: `Αριθμητικές_Μέθοδοι_Απαλλακτική_Εργασία.pdf`.

## Key Findings
*   **High Interpretability:** The KAN architecture successfully isolated and reconstructed the exact mathematical shapes of the sine and quadratic functions during the symbolic regression test.
*   **Superior Accuracy:** Despite a slower per-epoch training time due to complex non-linear basis calculations, the KAN model converged to a lower Mean Squared Error (MSE) on the housing dataset compared to a standard MLP with an identical node structure.
