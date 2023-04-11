from typing import Callable, Dict, Tuple
import torch
from torch import nn
import e3nn

import schnetpack.properties as structure
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus

import schnetpack.nn as snn

__all__ = ["E3SchNet", "E3SchNetInteraction"]


@e3nn.util.jit.compile_mode("script")
class E3SchNetInteraction(nn.Module):
    r"""E(3)-equivariant SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        max_ell: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(E3SchNetInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters
        self.max_ell = max_ell

        input_irreps = e3nn.o3.Irreps(
            (self.n_atom_basis, (ir.l, ir.p))
            for _, ir in e3nn.o3.Irreps.spherical_harmonics(self.max_ell)
        )

        irreps_after_in2f = self.n_filters * e3nn.o3.Irreps.spherical_harmonics(
            self.max_ell
        )
        self.in2f = e3nn.o3.Linear(irreps_in=input_irreps, irreps_out=irreps_after_in2f)

        irreps_after_mul_to_axis = e3nn.o3.Irreps.spherical_harmonics(self.max_ell)

        Yr_irreps = e3nn.o3.Irreps.spherical_harmonics(self.max_ell)
        self.tensor_product_x_Yr = e3nn.o3.FullTensorProduct(
            irreps_after_mul_to_axis, Yr_irreps
        )
        irreps_after_tensor_product_x_Yr = self.tensor_product_x_Yr.irreps_out

        irreps_after_axis_to_mul = self.n_filters * irreps_after_tensor_product_x_Yr

        self.W_irreps = e3nn.o3.Irreps(f"{irreps_after_axis_to_mul.num_irreps}x0e")
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation),
            Dense(n_filters, self.W_irreps.dim),
        )

        self.continuous_filter_convolution = e3nn.o3.ElementwiseTensorProduct(
            irreps_after_axis_to_mul, self.W_irreps
        )
        irreps_after_continuous_filter_convolution = (
            self.continuous_filter_convolution.irreps_out
        )

        output_irreps = input_irreps
        self.f2out_1 = e3nn.o3.Linear(
            irreps_in=irreps_after_continuous_filter_convolution,
            irreps_out=output_irreps,
        )
        self.f2out_act = e3nn.nn.Activation(
            irreps_in=output_irreps,
            acts=[activation if ir.l == 0 else None for _, ir in output_irreps],
        )
        self.f2out_2 = e3nn.o3.Linear(irreps_in=output_irreps, irreps_out=output_irreps)

    def forward(
        self,
        x: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        f_ij: torch.Tensor,
        rcut_ij: torch.Tensor,
        Yr_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            idx_i: index of center atom i
            idx_j: index of neighbors j
            f_ij: d_ij passed through the embedding function
            rcut_ij: d_ij passed through the cutoff function
            r_ij: relative position of neighbor j to atom i
            Yr_ij: spherical harmonics of r_ij
        Returns:
            atom features after interaction
        """
        # print("x", x, self.input_irreps)
        # Embed the inputs.
        x = self.in2f(x)
        # print("x, irreps_after_in2f", x, irreps_after_in2f)

        # Previously x_j.shape == (num_edges, n_filters * x_irreps.dim)
        # We want x_j.shape == (num_edges, n_filters, x_irreps.dim)
        x_j = x[idx_j]
        # print("x_j, irreps_after_in2f", x_j[:1], irreps_after_in2f)

        x_j = x_j.reshape((x_j.shape[0], self.n_filters, -1))
        # print("x_j, irreps_after_mul_to_axis", x_j[:1], irreps_after_mul_to_axis)

        # Apply e3nn.o3.FullTensorProduct to get new x_j of shape (num_edges, n_filters, new_x_irreps).
        x_j = self.tensor_product_x_Yr(x_j, Yr_ij)
        # print("x_j, irreps_after_tensor_product_x_Yr", x_j[:1], irreps_after_tensor_product_x_Yr)

        # Reshape x_j back to (num_edges, n_filters * x_irreps.dim).
        x_j = x_j.reshape((x_j.shape[0], -1))
        # print("x_j, irreps_after_axis_to_mul", x_j[:1], irreps_after_axis_to_mul)

        # Compute filter.
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # Continuous-filter convolution.
        x_ij = self.continuous_filter_convolution(x_j, Wij)
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        # Apply final linear and activation layer.
        x = self.f2out_1(x)
        x = self.f2out_act(x)
        x = self.f2out_2(x)
        return x


@e3nn.util.jit.compile_mode("script")
class E3SchNet(nn.Module):
    """E(3)-equivariant SchNet architecture for learning representations of atomistic systems

    Reduces to standard SchNet when max_ell = 0.

    References:

    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        max_ell: int,
        n_filters: int = None,
        shared_interactions: bool = False,
        max_z: int = 100,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.max_ell = max_ell

        # Create n_atom_basis copies of each irreps.
        spherical_harmonics_irreps = e3nn.o3.Irreps.spherical_harmonics(self.max_ell)
        latent_irreps = e3nn.o3.Irreps(
            (self.n_atom_basis, (ir.l, ir.p)) for _, ir in spherical_harmonics_irreps
        )

        # Layers
        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)
        self.post_embedding = e3nn.o3.Linear(
            irreps_in=f"{self.n_atom_basis}x0e", irreps_out=latent_irreps
        )
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            spherical_harmonics_irreps, normalization="component", normalize=True
        )
        self.interactions = snn.replicate_module(
            lambda: E3SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
                max_ell=self.max_ell,
            ),
            n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]

        # Compute atom embeddings.
        # Initially, the atom embeddings are just scalars.
        x = self.embedding(atomic_numbers)
        x = self.post_embedding(x)

        # Compute radial basis functions to cut off interactions
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)
        r_ij = r_ij * rcut_ij[:, None]

        # Compute the spherical harmonics of relative positions.
        # r_ij: (n_edges, 3)
        # Yr_ij: (n_edges, (max_ell + 1) ** 2)
        # Reshape Yr_ij to (num_edges, 1, (max_ell + 1) ** 2).
        Yr_ij = self.spherical_harmonics(r_ij)
        Yr_ij = Yr_ij.reshape((Yr_ij.shape[0], 1, Yr_ij.shape[1]))

        # Compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, idx_i, idx_j, f_ij, rcut_ij, Yr_ij)
            x = x + v

        # Extract only the scalars.
        inputs["scalar_representation"] = x[:, : self.n_atom_basis]
        return inputs
