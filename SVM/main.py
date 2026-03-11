# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: ML_assignment_SVM
@File Name: main.py
@Software: Python
@Time: 17/Feb/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.1.2
@Description: Implementing the SVM method.
"""


import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class SupportVectorMachine:

    def __init__(self, kernel=None, C=1.0):
        """
        Initialize the SVM with a kernel function and regularization parameter.

        :param kernel: Kernel function (linear, polynomial, or RBF)
        :param C: Regularization parameter (controls soft margin)
        """
        self.kernel = kernel if kernel else self.linear_kernel  # Default to linear kernel
        self.C = C
        self.alpha = None
        self.b = None
        self.support_vectors = None
        self.sv_targets = None
        self.sv_alpha = None
        self.inputs = None
        self.targets = None
        self.P = None  # Precomputed P matrix for optimization

    @staticmethod
    def linear_kernel(x, y):
        """Compute the linear kernel K(x, y) = x.T @ y."""
        return np.dot(x, y)

    @staticmethod
    def polynomial_kernel(x, y, p=3):
        """Compute the polynomial kernel K(x, y) = (x.T @ y + 1)^p."""
        return (np.dot(x, y) + 1) ** p

    @staticmethod
    def rbf_kernel(x, y, sigma=0.5):
        """Compute the RBF kernel K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))."""
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

    def generate_data(self, seed=100, classA_centers=[[1.5, 0.5], [-1.5, 0.5]], classB_center=[0.0, -0.5], noise=0.2):
        """
        Generate test data with customizable cluster positions and noise.

        :param seed: Random seed
        :param classA_centers: List of cluster centers for Class A
        :param classB_center: Cluster center for Class B
        :param noise: Standard deviation of noise
        """
        np.random.seed(seed)

        # Generate Class A clusters
        classA = []
        for center in classA_centers:
            cluster = np.random.randn(10, 2) * noise + center
            classA.append(cluster)
        classA = np.concatenate(classA)

        # Generate Class B cluster
        classB = np.random.randn(20, 2) * noise + classB_center

        # Merge and shuffle data
        inputs = np.concatenate((classA, classB))
        targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
        self.inputs, self.targets = inputs[indices], targets[indices]

    def compute_P_matrix(self):
        """Compute the P matrix used in the quadratic optimization problem."""
        N = len(self.inputs)
        self.P = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                self.P[i, j] = self.targets[i] * self.targets[j] * self.kernel(self.inputs[i], self.inputs[j])

    def objective(self, alpha):
        """Objective function to be minimized."""
        return 0.5 * np.dot(alpha, np.dot(self.P, alpha)) - np.sum(alpha)

    def zerofun(self, alpha):
        """Equality constraint: sum(alpha_i * target_i) = 0."""
        return np.dot(alpha, self.targets)

    def train(self):
        """Train the SVM using quadratic optimization."""
        N = len(self.inputs)
        self.compute_P_matrix()

        # Define constraints and bounds
        start = np.zeros(N)
        bounds = [(0, self.C) for _ in range(N)]
        constraints = {'type': 'eq', 'fun': self.zerofun}

        # Perform optimization
        result = minimize(self.objective, start, bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError("Optimization failed!")

        self.alpha = result['x']

        # Extract support vectors
        sv_indices = self.alpha > 1e-5
        self.sv_alpha = self.alpha[sv_indices]
        self.support_vectors = self.inputs[sv_indices]
        self.sv_targets = self.targets[sv_indices]

        # Compute b value
        self.b = self.compute_b()

    def compute_b(self):
        """Calculate the bias term b using support vectors."""
        sv_indices = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)
        sv = self.inputs[sv_indices]
        sv_t = self.targets[sv_indices]
        sv_alpha = self.alpha[sv_indices]

        if len(sv_alpha) == 0:
            return 0  # Default to 0 if no valid support vectors found

        b_sum = 0
        for i in range(len(sv_alpha)):
            b_sum += sv_t[i] - np.sum(self.alpha * self.targets * np.array(
                [self.kernel(self.inputs[j], sv[i]) for j in range(len(self.alpha))]))
        return b_sum / len(sv_alpha)

    def indicator(self, x, y):
        """Decision function to classify a new point (x, y)."""
        s = np.array([x, y])
        result = np.sum(self.sv_alpha * self.sv_targets * np.array(
            [self.kernel(sv, s) for sv in self.support_vectors]))
        return result - self.b

    def plot_decision_boundary(self, title='Decision Boundary'):
        """Enhanced plotting with title support"""
        plt.plot([p[0] for p, t in zip(self.inputs, self.targets) if t == 1],
                 [p[1] for p, t in zip(self.inputs, self.targets) if t == 1], 'b.', label='Class A')
        plt.plot([p[0] for p, t in zip(self.inputs, self.targets) if t == -1],
                 [p[1] for p, t in zip(self.inputs, self.targets) if t == -1], 'r.', label='Class B')
        plt.plot(self.support_vectors[:, 0], self.support_vectors[:, 1], 'go',
                 markersize=10, fillstyle='none', label='Support Vectors')

        xgrid = np.linspace(-5, 5, 100)
        ygrid = np.linspace(-4, 4, 100)
        grid = np.array([[self.indicator(x, y) for x in xgrid] for y in ygrid])

        plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                    colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
        plt.title(title)
        plt.axis('equal')
        plt.legend()

    def compare_cluster_configurations(self):
        plt.figure(figsize=(12, 6))

        # Configuration 1: Well-separated clusters
        svm1 = SupportVectorMachine(kernel=SupportVectorMachine.linear_kernel, C=1.0)
        svm1.generate_data(classA_centers=[[2.0, 1.0], [-2.0, 1.0]], classB_center=[0.0, -1.0])
        svm1.train()

        plt.subplot(121)
        svm1.plot_decision_boundary(title="Well-Separated Clusters (Linear Kernel)")

        # Configuration 2: Overlapping clusters
        svm2 = SupportVectorMachine(kernel=SupportVectorMachine.rbf_kernel, C=1.0)
        svm2.generate_data(classA_centers=[[0.5, 0.5], [-0.5, 0.5]], noise=0.5)
        svm2.train()

        plt.subplot(122)
        svm2.plot_decision_boundary(title="Overlapping Clusters (RBF Kernel, σ=0.5)")

        plt.tight_layout()
        plt.show()


    def compare_kernel_parameters(self):
        plt.figure(figsize=(12, 12))

        # Polynomial Kernel Comparison
        plt.subplot(221)
        svm_p2 = SupportVectorMachine(kernel=lambda x, y: SupportVectorMachine.polynomial_kernel(x, y, p=2), C=1.0)
        svm_p2.generate_data()
        svm_p2.train()
        svm_p2.plot_decision_boundary(title="Polynomial Kernel (p=2)")

        plt.subplot(222)
        svm_p4 = SupportVectorMachine(kernel=lambda x, y: SupportVectorMachine.polynomial_kernel(x, y, p=4), C=1.0)
        svm_p4.generate_data()
        svm_p4.train()
        svm_p4.plot_decision_boundary(title="Polynomial Kernel (p=4)")

        # RBF Kernel Comparison
        plt.subplot(223)
        svm_sigma05 = SupportVectorMachine(kernel=lambda x, y: SupportVectorMachine.rbf_kernel(x, y, sigma=0.5), C=1.0)
        svm_sigma05.generate_data()
        svm_sigma05.train()
        svm_sigma05.plot_decision_boundary(title="RBF Kernel (σ=0.5)")

        plt.subplot(224)
        svm_sigma01 = SupportVectorMachine(kernel=lambda x, y: SupportVectorMachine.rbf_kernel(x, y, sigma=0.1), C=1.0)
        svm_sigma01.generate_data()
        svm_sigma01.train()
        svm_sigma01.plot_decision_boundary(title="RBF Kernel (σ=0.1)")

        plt.tight_layout()
        plt.show()


# Main execution
if __name__ == "__main__":
    # Initialize SVM with an RBF kernel
    svm = SupportVectorMachine(kernel=SupportVectorMachine.rbf_kernel, C=1.0)

    # Generate dataset
    svm.generate_data()

    # Train the SVM
    svm.train()

    svm.compare_cluster_configurations()  # For Problem 1
    svm.compare_kernel_parameters()       # For Problem 3

    # Plot decision boundary
    svm.plot_decision_boundary()
