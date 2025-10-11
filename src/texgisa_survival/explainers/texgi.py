# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
TEXGI (Time-dependent EXtreme Gradient Integration) Explainer

This module implements the TEXGI method for computing time-dependent feature
importance in survival analysis models using Expected Gradients.

Based on the ICDM paper: "TexGISa: Interpretable and Interactive Deep Survival
Analysis with Time-dependent Extreme Gradient Integration"
"""

import functools
import operator
import torch
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, List

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gather_nd(params, indices):
    """
    Gather elements from params using indices.

    Args:
        params: Tensor to index
        indices: k-dimension tensor of integers

    Returns:
        output: 1-dimensional tensor of elements
    """
    max_value = functools.reduce(operator.mul, list(params.size())) - 1
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    idx[idx < 0] = 0
    idx[idx > max_value] = 0
    return torch.take(params, idx)


class SimpleDataset(Dataset):
    """Simple dataset wrapper for reference samples."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx],)


class TEXGIExplainer:
    """
    TEXGI Explainer using Expected Gradients for survival analysis.

    This explainer computes time-dependent feature importance by:
    1. Creating interpolated paths between input and reference samples
    2. Computing gradients along these paths
    3. Integrating gradients to obtain feature attributions

    Parameters
    ----------
    background_data : torch.Tensor
        Reference/background samples for computing baselines
    batch_size : int, default 32
        Batch size for processing
    random_alpha : bool, default True
        Whether to use random interpolation coefficients
    k : int, default 10
        Number of reference samples to use per input
    scale_by_inputs : bool, default True
        Whether to scale gradients by input differences
    device : str, optional
        Device for computation ('cpu', 'cuda', 'mps')
    """

    def __init__(self,
                 background_data: torch.Tensor,
                 batch_size: int = 32,
                 random_alpha: bool = True,
                 k: int = 10,
                 scale_by_inputs: bool = True,
                 device: Optional[str] = None):

        self.random_alpha = random_alpha
        self.k = k
        self.scale_by_inputs = scale_by_inputs
        self.batch_size = batch_size

        # Set device
        if device is None:
            self.device = DEFAULT_DEVICE
        else:
            self.device = torch.device(device)

        # Create dataset and dataloader for reference samples
        self.ref_set = SimpleDataset(background_data.to(self.device))
        self.ref_sampler = DataLoader(
            dataset=self.ref_set,
            batch_size=batch_size * k,
            shuffle=True,
            drop_last=True
        )

    def _get_ref_batch(self):
        """Get a batch of reference samples."""
        return next(iter(self.ref_sampler))[0].float()

    def _get_samples_input(self, input_tensor, reference_tensor):
        """
        Create interpolated samples between input and reference.

        Args:
            input_tensor: Input samples [batch_size, n_features]
            reference_tensor: Reference samples [batch_size, k, n_features]

        Returns:
            samples_input: Interpolated samples [batch_size, k, n_features]
        """
        input_dims = list(input_tensor.size())[1:]
        num_input_dims = len(input_dims)

        batch_size = reference_tensor.size()[0]
        k_ = reference_tensor.size()[1]

        # Generate interpolation coefficients
        if self.random_alpha:
            t_tensor = torch.FloatTensor(batch_size, k_).uniform_(0, 1).to(self.device)
        else:
            if k_ == 1:
                t_tensor = torch.cat([torch.Tensor([1.0]) for _ in range(batch_size)]).to(self.device)
            else:
                t_tensor = torch.cat([torch.linspace(0, 1, k_) for _ in range(batch_size)]).to(self.device)

        shape = [batch_size, k_] + [1] * num_input_dims
        interp_coef = t_tensor.view(*shape)

        # Compute interpolated samples
        end_point_ref = (1.0 - interp_coef) * reference_tensor
        input_expand_mult = input_tensor.unsqueeze(1)
        end_point_input = interp_coef * input_expand_mult

        samples_input = end_point_input + end_point_ref
        return samples_input

    def _get_samples_delta(self, input_tensor, reference_tensor):
        """Compute difference between input and reference."""
        input_expand_mult = input_tensor.unsqueeze(1)
        sd = input_expand_mult - reference_tensor
        return sd

    def _get_grads(self, samples_input, model, sparse_labels=None):
        """
        Compute gradients for interpolated samples.

        Args:
            samples_input: Interpolated samples [batch_size, k, n_features]
            model: Survival model
            sparse_labels: Optional labels for specific outputs

        Returns:
            grad_tensors: List of gradient tensors for each task/time bin
        """
        samples_input = samples_input.detach().clone().requires_grad_(True)

        # Get the number of tasks (time bins) from model output
        temp_output = model(samples_input[:, 0])
        grad_tensors = [
            torch.zeros(samples_input.shape).float().to(self.device)
            for _ in range(len(temp_output))
        ]

        # Compute gradients for each interpolation step
        for i in range(self.k):
            particular_slice = samples_input[:, i]
            batch_outputs = model(particular_slice)

            for idx in range(len(batch_outputs)):
                batch_output = batch_outputs[idx]

                if batch_output.size(1) > 1:
                    sample_indices = torch.arange(0, batch_output.size(0)).to(self.device)
                    indices_tensor = torch.cat([
                        sample_indices.unsqueeze(1),
                        sparse_labels.unsqueeze(1)
                    ], dim=1)
                    batch_output = gather_nd(batch_output, indices_tensor)

                model_grads = grad(
                    outputs=batch_output,
                    inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(self.device),
                    create_graph=True
                )
                grad_tensors[idx][:, i, :] = model_grads[0]

        return grad_tensors

    def compute_attributions(self, model, input_tensor, sparse_labels=None):
        """
        Calculate TEXGI attributions (Expected Gradients).

        Args:
            model: Trained survival model (must output list of tensors for each time bin)
            input_tensor: Input samples to explain [batch_size, n_features]
            sparse_labels: Optional sparse labels

        Returns:
            expected_grads_list: List of attribution tensors for each time bin
        """
        # Get reference batch
        reference_tensor = self._get_ref_batch()
        shape = reference_tensor.shape
        reference_tensor = reference_tensor.view(
            self.batch_size,
            self.k,
            *(shape[1:])
        ).to(self.device)

        # Create interpolated samples
        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        samples_delta = self._get_samples_delta(input_tensor, reference_tensor)

        # Compute gradients
        grad_tensor = self._get_grads(samples_input, model, sparse_labels)

        # Scale by input differences if requested
        mult_grads_list = [
            samples_delta * grad_tensor[i] if self.scale_by_inputs else grad_tensor[i]
            for i in range(len(grad_tensor))
        ]

        # Average over interpolation steps
        expected_grads_list = [mult_grads_list[i].mean(1) for i in range(len(mult_grads_list))]

        return expected_grads_list

    def compute_global_importance(self, model, input_tensor, sparse_labels=None):
        """
        Compute aggregated global feature importance.

        Args:
            model: Trained survival model
            input_tensor: Input samples [batch_size, n_features]
            sparse_labels: Optional sparse labels

        Returns:
            importance: Global feature importance [n_features]
        """
        # Get attributions for all time bins
        attributions_list = self.compute_attributions(model, input_tensor, sparse_labels)

        # Average across time bins
        avg_attributions = sum(attributions_list) / len(attributions_list)

        # Average across samples and take absolute value
        global_importance = torch.abs(avg_attributions).mean(dim=0)

        return global_importance
