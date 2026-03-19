"""Vertex Block Descent (VBD) solver for volumetric muscle simulation.

Reference: Chen et al., "Vertex Block Descent" (SIGGRAPH 2024)

Minimizes the variational implicit Euler objective:
  G(x) = 1/(2h^2) ||x - y||^2_M + E(x)
where y = x^t + h*v^t + h^2*g is the inertial prediction.
"""

import numpy as np
from .fem import (
    compute_rest_data,
    compute_deformation_gradient,
    get_b_vec,
    vertex_gradient_from_pk1,
    compute_lumped_masses,
)
from .constitutive import total_pk1, vertex_hessian_fd, project_spd
from .coloring import build_vertex_adjacency, greedy_color


class VBDSolver:
    """VBD solver for transversely isotropic hyperelastic muscle tissue."""

    def __init__(self, nodes, tets, fiber_directions, *,
                 mu=5000.0, kappa=500000.0, sigma0=300000.0,
                 density=1060.0, damping=0.01, dt=0.001,
                 n_iterations=20, gravity=None,
                 scale=1.0, kPE=4.0, e0=0.6):
        """
        Args:
            nodes: Rest vertex positions, shape (n_vertices, 3).
            tets: Tet connectivity, shape (n_elements, 4).
            fiber_directions: Per-element fiber direction d0, shape (n_elements, 3).
            mu: Shear modulus (Pa).
            kappa: Bulk modulus (Pa).
            sigma0: Peak isometric stress (Pa).
            density: Material density (kg/m^3).
            damping: Rayleigh damping coefficient k_d.
            dt: Time step (s).
            n_iterations: VBD iterations per time step.
            gravity: Gravity vector, shape (3,). Default [0,0,-9.81].
            scale: Active force-length width scale.
            kPE, e0: Passive force-length parameters.
        """
        self.nodes_rest = np.array(nodes, dtype=float)
        self.x = self.nodes_rest.copy()
        self.v = np.zeros_like(self.x)
        self.tets = np.array(tets, dtype=int)
        self.d0 = np.array(fiber_directions, dtype=float)

        # Material parameters
        self.mu = mu
        self.kappa = kappa
        self.sigma0 = sigma0
        self.damping = damping
        self.dt = dt
        self.n_iterations = n_iterations
        self.gravity = (np.array(gravity, dtype=float)
                        if gravity is not None
                        else np.array([0.0, 0.0, -9.81]))
        self.scale = scale
        self.kPE = kPE
        self.e0 = e0

        # Precompute rest-state data
        n_verts = len(nodes)
        self.Dm_inv, self.volumes = compute_rest_data(self.nodes_rest, self.tets)
        self.masses = compute_lumped_masses(
            self.tets, self.volumes, density, n_verts)

        # Vertex-element incidence: for each vertex, list of (elem_idx, local_idx)
        self.vertex_elements = [[] for _ in range(n_verts)]
        for e in range(len(self.tets)):
            for k in range(4):
                self.vertex_elements[self.tets[e, k]].append((e, k))

        # Graph coloring
        adj = build_vertex_adjacency(self.tets, n_verts)
        self.colors, self.n_colors, self.color_groups = greedy_color(adj, n_verts)

        # Boundary conditions
        self.fixed_vertices = set()
        self.prescribed_positions = {}
        # Per-DOF constraints: vertex_id -> set of fixed DOF indices {0,1,2}
        self.fixed_dofs = {}

        # Last activation (for reaction force queries)
        self._last_activation = np.zeros(len(self.tets))

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def set_fixed_vertices(self, indices):
        """Fix vertices at their current positions (all 3 DOFs)."""
        self.fixed_vertices = set(indices)
        for i in indices:
            self.prescribed_positions[i] = self.x[i].copy()
            self.fixed_dofs[i] = {0, 1, 2}

    def set_prescribed_positions(self, indices, positions):
        """Prescribe Dirichlet positions for specified vertices (all 3 DOFs)."""
        for i, pos in zip(indices, positions):
            self.fixed_vertices.add(i)
            self.prescribed_positions[i] = np.array(pos, dtype=float)
            self.fixed_dofs[i] = {0, 1, 2}

    def set_fixed_dof(self, vertex_indices, dofs, values=None):
        """Fix specific DOFs for given vertices.

        Args:
            vertex_indices: list of vertex indices.
            dofs: set or list of DOF indices to fix (0=x, 1=y, 2=z).
            values: if provided, shape (len(vertex_indices), len(dofs)),
                    prescribed DOF values. Otherwise uses current positions.
        """
        dof_set = set(dofs)
        for k, vi in enumerate(vertex_indices):
            if vi not in self.fixed_dofs:
                self.fixed_dofs[vi] = set()
            self.fixed_dofs[vi] |= dof_set
            if vi not in self.prescribed_positions:
                self.prescribed_positions[vi] = self.x[vi].copy()
            if values is not None:
                for j, d in enumerate(sorted(dofs)):
                    self.prescribed_positions[vi][d] = values[k][j] if hasattr(values[k], '__len__') else values[k]
            # Mark fully fixed
            if self.fixed_dofs[vi] == {0, 1, 2}:
                self.fixed_vertices.add(vi)

    def clear_boundary_conditions(self):
        """Remove all boundary conditions."""
        self.fixed_vertices.clear()
        self.prescribed_positions.clear()
        self.fixed_dofs.clear()

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self, activation=0.0):
        """Advance one time step.

        Args:
            activation: Muscle activation, scalar or per-element array in [0, 1].
        Returns:
            Current vertex positions, shape (n_vertices, 3).
        """
        h = self.dt
        n_verts = len(self.x)

        # Store activation for reaction force queries
        if np.isscalar(activation):
            act = np.full(len(self.tets), float(activation))
        else:
            act = np.asarray(activation, dtype=float)
        self._last_activation = act

        # Apply prescribed positions before prediction
        for vi, pos in self.prescribed_positions.items():
            self.x[vi] = pos
            self.v[vi] = 0.0

        # Inertial prediction: y = x + h*v + h^2*g
        y = self.x + h * self.v + h ** 2 * self.gravity[np.newaxis, :]

        # Fix prediction for prescribed vertices
        for vi, pos in self.prescribed_positions.items():
            y[vi] = pos

        # Initialize guess
        x_new = y.copy()
        for vi, pos in self.prescribed_positions.items():
            x_new[vi] = pos
        x_old = self.x.copy()

        # VBD Gauss-Seidel iterations
        for _iter in range(self.n_iterations):
            for color_idx in range(self.n_colors):
                for vi in self.color_groups[color_idx]:
                    if vi in self.fixed_vertices:
                        continue

                    mi = self.masses[vi]

                    # Inertial gradient and Hessian
                    grad_i = (mi / h ** 2) * (x_new[vi] - y[vi])
                    hess_i = (mi / h ** 2) * np.eye(3)

                    # Accumulate elastic contributions
                    hess_elastic = np.zeros((3, 3))

                    for elem_idx, local_idx in self.vertex_elements[vi]:
                        tet = self.tets[elem_idx]
                        x_elem = x_new[tet]

                        F = compute_deformation_gradient(
                            x_elem, self.Dm_inv[elem_idx])
                        P = total_pk1(
                            F, self.d0[elem_idx],
                            self.mu, self.kappa, self.sigma0,
                            act[elem_idx], self.scale, self.kPE, self.e0)

                        b_vec = get_b_vec(self.Dm_inv[elem_idx], local_idx)
                        g_e = vertex_gradient_from_pk1(
                            P, self.volumes[elem_idx], b_vec)
                        grad_i += g_e

                        H_e = vertex_hessian_fd(
                            x_elem, self.Dm_inv[elem_idx],
                            self.volumes[elem_idx], local_idx,
                            self.d0[elem_idx],
                            self.mu, self.kappa, self.sigma0,
                            act[elem_idx], self.scale, self.kPE, self.e0)
                        hess_elastic += H_e

                    # Rayleigh damping
                    disp = x_new[vi] - x_old[vi]
                    grad_i += (self.damping / h) * (hess_elastic @ disp)
                    hess_i += hess_elastic + (self.damping / h) * hess_elastic

                    # SPD projection + regularization
                    if not np.all(np.isfinite(grad_i)):
                        continue
                    if not np.all(np.isfinite(hess_i)):
                        hess_i = np.eye(3) * max(1e-6, abs(grad_i).max())
                    hess_i = project_spd(hess_i)
                    hess_i += 1e-8 * np.eye(3)

                    # Newton step
                    try:
                        dx = np.linalg.solve(hess_i, -grad_i)
                    except np.linalg.LinAlgError:
                        dx = -grad_i * 1e-6
                    if np.all(np.isfinite(dx)):
                        x_new[vi] += dx

        # Update velocity and position
        self.v = (x_new - self.x) / h
        self.x = x_new.copy()

        for vi in self.fixed_vertices:
            self.v[vi] = 0.0

        return self.x.copy()

    # ------------------------------------------------------------------
    # Quasi-static solve
    # ------------------------------------------------------------------

    def solve_static(self, activation=0.0, n_iterations=None, tol=1e-8,
                     max_step=None):
        """Run VBD iterations until convergence (quasi-static).

        Args:
            activation: scalar or per-element array.
            n_iterations: max iterations (default: 10 * self.n_iterations).
            tol: Convergence tolerance (max vertex displacement per iteration).
            max_step: Maximum displacement per vertex per iteration (m).
                If None, estimated from mesh size.
        Returns:
            converged: bool.
            n_iters_used: number of iterations.
        """
        if n_iterations is None:
            n_iterations = 10 * self.n_iterations

        if np.isscalar(activation):
            act = np.full(len(self.tets), float(activation))
        else:
            act = np.asarray(activation, dtype=float)
        self._last_activation = act

        # Estimate max step from element size if not provided
        if max_step is None:
            avg_vol = self.volumes.mean()
            char_length = avg_vol ** (1.0 / 3.0)
            max_step = 0.1 * char_length  # 10% of element size

        # Apply prescribed DOF values
        for vi, pos in self.prescribed_positions.items():
            for d in self.fixed_dofs.get(vi, set()):
                self.x[vi, d] = pos[d]

        for iteration in range(n_iterations):
            max_dx = 0.0

            for color_idx in range(self.n_colors):
                for vi in self.color_groups[color_idx]:
                    if vi in self.fixed_vertices:
                        continue

                    grad_i = np.zeros(3)
                    hess_i = np.zeros((3, 3))

                    for elem_idx, local_idx in self.vertex_elements[vi]:
                        tet = self.tets[elem_idx]
                        x_elem = self.x[tet]

                        F = compute_deformation_gradient(
                            x_elem, self.Dm_inv[elem_idx])
                        P = total_pk1(
                            F, self.d0[elem_idx],
                            self.mu, self.kappa, self.sigma0,
                            act[elem_idx], self.scale, self.kPE, self.e0)

                        b_vec = get_b_vec(self.Dm_inv[elem_idx], local_idx)
                        grad_i += vertex_gradient_from_pk1(
                            P, self.volumes[elem_idx], b_vec)

                        H_e = vertex_hessian_fd(
                            x_elem, self.Dm_inv[elem_idx],
                            self.volumes[elem_idx], local_idx,
                            self.d0[elem_idx],
                            self.mu, self.kappa, self.sigma0,
                            act[elem_idx], self.scale, self.kPE, self.e0)
                        hess_i += H_e

                    if not np.all(np.isfinite(grad_i)):
                        continue
                    if not np.all(np.isfinite(hess_i)):
                        hess_i = np.eye(3) * max(1e-6, abs(grad_i).max())
                    hess_i = project_spd(hess_i)
                    hess_i += 1e-8 * np.eye(3)

                    try:
                        dx = np.linalg.solve(hess_i, -grad_i)
                    except np.linalg.LinAlgError:
                        dx = -grad_i * 1e-6

                    if not np.all(np.isfinite(dx)):
                        continue

                    # Zero out fixed DOFs
                    for d in self.fixed_dofs.get(vi, set()):
                        dx[d] = 0.0

                    # Step size limiting
                    dx_norm = np.linalg.norm(dx)
                    if dx_norm > max_step:
                        dx = dx * (max_step / dx_norm)
                        dx_norm = max_step

                    self.x[vi] += dx
                    max_dx = max(max_dx, dx_norm)

            if max_dx < tol:
                return True, iteration + 1

        return False, n_iterations

    # ------------------------------------------------------------------
    # Reaction forces
    # ------------------------------------------------------------------

    def compute_reaction_forces(self, vertex_indices, activation=None):
        """Compute elastic reaction forces at specified vertices.

        The reaction force at a fixed vertex equals the elastic gradient
        (force needed to maintain the constraint).

        Args:
            vertex_indices: list/array of vertex indices.
            activation: if None, uses last activation from step/solve_static.
        Returns:
            dict: {vertex_idx: force_vector (3,)}.
        """
        if activation is None:
            act = self._last_activation
        elif np.isscalar(activation):
            act = np.full(len(self.tets), float(activation))
        else:
            act = np.asarray(activation, dtype=float)

        forces = {}
        for vi in vertex_indices:
            grad = np.zeros(3)
            for elem_idx, local_idx in self.vertex_elements[vi]:
                tet = self.tets[elem_idx]
                x_elem = self.x[tet]
                F = compute_deformation_gradient(
                    x_elem, self.Dm_inv[elem_idx])
                P = total_pk1(
                    F, self.d0[elem_idx],
                    self.mu, self.kappa, self.sigma0,
                    act[elem_idx], self.scale, self.kPE, self.e0)
                b_vec = get_b_vec(self.Dm_inv[elem_idx], local_idx)
                grad += vertex_gradient_from_pk1(
                    P, self.volumes[elem_idx], b_vec)
            # Reaction force = elastic gradient (wall must push back)
            forces[vi] = grad
        return forces

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_positions(self):
        return self.x.copy()

    def get_velocities(self):
        return self.v.copy()

    def get_total_energy(self, activation=None):
        """Compute total elastic energy."""
        if activation is None:
            act = self._last_activation
        elif np.isscalar(activation):
            act = np.full(len(self.tets), float(activation))
        else:
            act = np.asarray(activation, dtype=float)

        from .constitutive import total_energy
        E = 0.0
        for e in range(len(self.tets)):
            tet = self.tets[e]
            x_elem = self.x[tet]
            F = compute_deformation_gradient(x_elem, self.Dm_inv[e])
            E += self.volumes[e] * total_energy(
                F, self.d0[e], self.mu, self.kappa,
                self.sigma0, act[e], self.scale, self.kPE, self.e0)
        return E

    def mesh_info(self):
        """Print mesh statistics."""
        n_verts = len(self.x)
        n_elems = len(self.tets)
        total_vol = self.volumes.sum()
        total_mass = self.masses.sum()
        print(f"Mesh: {n_verts} vertices, {n_elems} elements")
        print(f"Volume: {total_vol:.6e} m^3, Mass: {total_mass:.4f} kg")
        print(f"Colors: {self.n_colors}")
        print(f"Material: mu={self.mu}, kappa={self.kappa}, "
              f"sigma0={self.sigma0}")
