# Validation Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove broken/tautological validation code, fix the failing test, and add two meaningful validation layers: (A) constitutive model direct stress verification, (B) fully-constrained uniform deformation reaction force test.

**Architecture:** Three-phase cleanup-then-build approach. Phase 1 removes broken code (`vbd_force_length`, `run_vbd_fl_validation`, the `vbd` CLI command). Phase 2 fixes the existing failing test. Phase 3 adds two new validation tests that verify the constitutive model and reaction force calculation independently of solver convergence.

**Tech Stack:** Python, NumPy, matplotlib (for output plots), the existing `vbd_muscle` package.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `run_level0_fem.py` | Remove `vbd_force_length()` and `run_vbd_fl_validation()`, remove `--skip-vbd` flag, renumber parts |
| Modify | `main.py` | Remove `vbd` command and its import, update docstring |
| Modify | `tests/test_phase1.py` | Fix `f_PE(1.0)` expectation, add constitutive stress test, add reaction force test |

---

## Task 1: Remove broken VBD quasi-static validation

The `vbd_force_length()` and `run_vbd_fl_validation()` in `run_level0_fem.py` produce unreliable results due to:
- sigma0/mu=60 ill-conditioning causing VBD non-convergence
- `compute_reaction_forces` returns elastic gradient (sign mismatch)
- 3D lateral expansion deviates from 1D DGF assumptions

**Files:**
- Modify: `run_level0_fem.py` — delete `vbd_force_length()` (lines 92-172) and `run_vbd_fl_validation()` (lines 175-208), remove `--skip-vbd` argparse block, simplify `main()`
- Modify: `main.py` — remove `run_vbd` function and `'vbd'` entry from commands dict, update module docstring

- [ ] **Step 1: Edit `run_level0_fem.py`**

Delete `vbd_force_length()` and `run_vbd_fl_validation()` functions entirely. Remove the VBD-related imports that become unused (`generate_box_mesh`, `assign_fiber_directions`, `VBDSolver`). Remove the `--skip-vbd` argparse block from `main()`. Simplify `main()` to only run: Part 1 (analytical curves) and Part 2 (activation dynamics).

The resulting `main()` should be:
```python
def main():
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("Level 0 Validation: Single Muscle Force-Length")
    print("=" * 60)

    print(f"\nParameters:")
    print(f"  F0 = {F0:.0f} N")
    print(f"  l_opt = {l_opt*100:.1f} cm")
    print(f"  PCSA = {PCSA*1e4:.2f} cm^2")
    print(f"  sigma0 = {sigma0/1e3:.0f} kPa")
    print(f"  mu = {mu/1e3:.1f} kPa, kappa = {kappa/1e3:.0f} kPa")
    print(f"  Cross-section side = {side*100:.2f} cm")

    # Part 1: Analytical DGF curves
    print("\n--- Part 1: Analytical DGF Curves ---")
    plot_analytical_curves()

    # Part 2: Activation dynamics
    run_activation_dynamics_validation()

    plt.show()
    print("\nDone.")
```

- [ ] **Step 2: Edit `main.py`**

Remove `run_vbd()` function. Remove `'vbd'` from the commands dict. Update module docstring to remove the `vbd` line. The commands dict becomes:
```python
commands = {
    'test': ('Phase 1 验证测试', run_tests),
    'curves': ('DGF 解析曲线', run_curves),
    'activation': ('激活动力学', run_activation),
    'demo': ('VBD 动态演示', run_demo),
}
```

- [ ] **Step 3: Run to verify nothing is broken**

Run: `uv run python main.py curves` — should show analytical curves
Run: `uv run python main.py activation` — should show activation dynamics
Run: `uv run python main.py` — should list commands (no `vbd`)

- [ ] **Step 4: Commit**

```bash
git add run_level0_fem.py main.py
git commit -m "Remove broken VBD quasi-static F-L validation

The vbd_force_length/run_vbd_fl_validation produced unreliable results
due to ill-conditioning (sigma0/mu=60), solver non-convergence, and
3D lateral effects deviating from 1D DGF assumptions."
```

---

## Task 2: Fix failing `f_PE(1.0)` test

The `passive_force_length` was updated to match OpenSim (offset by `min_norm_fiber_length=0.2`), so `f_PE(1.0)` is no longer exactly 0. The test expectation must be updated.

**Files:**
- Modify: `tests/test_phase1.py:84-85`

- [ ] **Step 1: Update `f_PE(1.0)` test expectation**

OpenSim's passive curve has `f_PE(1.0) ≈ 0.018`. Change the test to check it's small but not zero:

```python
# f_PE(1.0) should be small (OpenSim-aligned curve has small offset)
fp1 = passive_force_length(1.0)
ok &= report("f_PE(1.0) ~ 0", fp1 < 0.05, f"val={fp1:.6f}")
```

- [ ] **Step 2: Run tests to verify fix**

Run: `uv run python tests/test_phase1.py`
Expected: ALL TESTS PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase1.py
git commit -m "Fix f_PE(1.0) test for OpenSim-aligned passive curve

The passive_force_length was updated to match OpenSim's offset by
min_norm_fiber_length, so f_PE(1.0) ≈ 0.018 instead of exactly 0."
```

---

## Task 3: Add constitutive model direct stress verification (Layer A)

**Purpose:** Verify that `fiber_pk1()` produces stress consistent with DGF analytical formulas. This tests the constitutive model directly, completely bypassing the solver.

**Method:**
1. Construct a single tet at rest with known geometry
2. Apply uniform deformation F = diag(1/√λ, 1/√λ, λ) for various λ
3. Call `fiber_pk1(F, d0, sigma0, activation)` to get P_fiber
4. Extract the Cauchy stress fiber component: σ_zz = (1/J) * (P_fiber @ F^T)_{zz}
5. Compare with DGF analytical: σ_expected = sigma0 * (a * f_L(λ) + f_PE(λ)) * λ

**Key insight:** For uniaxial deformation F=diag(lat,lat,λ) with d0=[0,0,1]:
- Fd0 = [0, 0, λ], so l_tilde = λ
- fiber_pk1 returns P = (f_total/λ) * outer(Fd0, d0) = f_total * e_z ⊗ e_z
  where f_total = sigma0 * (a * f_L(λ) + f_PE(λ))
- Cauchy σ = (1/J) P F^T, the zz component = (1/J) * f_total * F_{zz} = (1/J) * f_total * λ
- For incompressible (J=1): σ_zz = sigma0 * (a * f_L(λ) + f_PE(λ)) * λ
- Note: the extra λ factor comes from the PK1→Cauchy conversion (P @ F^T)

**Files:**
- Modify: `tests/test_phase1.py` — add `test_fiber_pk1_stress()`

- [ ] **Step 1: Write the test**

Add to `tests/test_phase1.py`:

```python
def test_fiber_pk1_stress():
    """Verify fiber_pk1 Cauchy stress matches DGF analytical curves.

    For uniaxial incompressible deformation F = diag(1/sqrt(lam), 1/sqrt(lam), lam)
    with d0 = [0,0,1], the Cauchy stress zz component should equal:
        sigma_zz = sigma0 * (activation * f_L(lam) + f_PE(lam)) * lam
    The extra lam factor comes from the PK1 -> Cauchy conversion (P @ F^T).
    """
    print("\n=== Fiber PK1 stress vs DGF analytical ===")
    ok = True

    d0 = np.array([0.0, 0.0, 1.0])
    sigma0 = 300000.0

    lam_values = [0.5, 0.7, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
    activations = [0.0, 0.5, 1.0]

    for act in activations:
        for lam in lam_values:
            # Incompressible uniaxial deformation
            lat = 1.0 / np.sqrt(lam)
            F = np.diag([lat, lat, lam])
            J = np.linalg.det(F)  # should be ~1.0

            P_fiber = fiber_pk1(F, d0, sigma0, act)

            # Cauchy stress: sigma = (1/J) * P @ F^T
            cauchy = (1.0 / J) * P_fiber @ F.T
            sigma_zz = cauchy[2, 2]

            # DGF analytical (extra lam from PK1->Cauchy conversion)
            fL = active_force_length(lam)
            fPE = passive_force_length(lam)
            sigma_expected = sigma0 * (act * fL + fPE) * lam

            err = abs(sigma_zz - sigma_expected) / (abs(sigma_expected) + 1.0)
            ok &= report(
                f"fiber stress a={act:.1f} lam={lam:.2f}",
                err < 1e-10,
                f"got={sigma_zz:.2f}, expected={sigma_expected:.2f}, rel_err={err:.2e}")

    return ok
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run python tests/test_phase1.py`
Expected: All `fiber stress a=... lam=...` tests PASS. This confirms `fiber_pk1` correctly encodes DGF curves.

If any FAIL, the constitutive model has a bug that needs fixing before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase1.py
git commit -m "Add constitutive model direct stress verification test

Verifies fiber_pk1 Cauchy stress matches DGF analytical curves for
uniaxial incompressible deformation at multiple stretch ratios and
activation levels. Tests the constitutive model independently of
the solver."
```

---

## Task 4: Add fully-constrained reaction force verification (Layer B)

**Purpose:** Verify that `compute_reaction_forces()` on a mesh with ALL vertices prescribed to uniform deformation gives the correct total force. This tests the FEM assembly + reaction force pipeline without depending on solver convergence.

**Method:**
1. Create box mesh (side x side x l_opt)
2. Directly set ALL vertex positions to uniform stretch: x_i = F * X_i where F = diag(1/√λ, 1/√λ, λ)
3. Call `compute_reaction_forces()` for z=0 face vertices
4. Sum z-component of reaction forces → total_Fz
5. Compare with analytical: the total traction on the z=0 reference face = P_zz * A_ref
   where P is from `total_pk1(F)` and A_ref is the reference cross-section area

**Key insight:** For uniform deformation, every element sees the same F, so the total reaction force (sum of elastic gradients at z=0 face) should equal the PK1 traction integrated over the reference face. By FEM divergence, sum of gradients at z=0 vertices = P @ N * A_ref where N=[0,0,-1] is the outward normal of the z=0 face.

**Important:** `set_prescribed_positions()` does NOT update `solver.x` — it only stores positions for `step()`/`solve_static()`. We must directly set `solver.x` to the deformed positions.

**Files:**
- Modify: `tests/test_phase1.py` — add `test_reaction_force_uniform()`

- [ ] **Step 1: Write the test**

```python
def test_reaction_force_uniform():
    """Verify reaction forces under fully-prescribed uniform deformation.

    All vertices are directly placed at x_i = F * X_i (no solver run).
    The sum of elastic gradients at z=0 face vertices should equal
    the PK1 traction on the reference z=0 face: F_z = -P_zz * A_ref.
    (Negative because the outward normal of z=0 face is [0,0,-1].)
    """
    print("\n=== Reaction force (fully constrained uniform deformation) ===")
    ok = True

    from vbd_muscle.solver import VBDSolver
    from vbd_muscle.mesh import assign_fiber_directions

    l_opt = 0.10
    sigma0 = 300000.0
    mu = 5000.0
    kappa = 100 * mu
    F0 = 1000.0
    PCSA = F0 / sigma0
    side = np.sqrt(PCSA)

    nodes, tets = generate_box_mesh(side, side, l_opt, 3, 3, 6)
    fiber_dirs = assign_fiber_directions(nodes, tets)
    A_ref = side ** 2  # reference cross-section area

    lam_values = [0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

    for lam in lam_values:
        solver = VBDSolver(
            nodes, tets, fiber_dirs,
            mu=mu, kappa=kappa, sigma0=sigma0,
            density=1060.0, damping=0.01, dt=0.001,
            n_iterations=1,
            gravity=np.array([0.0, 0.0, 0.0]),
        )

        # Directly set ALL vertex positions to uniform deformation
        # (set_prescribed_positions does NOT update solver.x)
        lat = 1.0 / np.sqrt(max(lam, 0.01))
        center_x, center_y = side / 2.0, side / 2.0
        for vi in range(len(nodes)):
            solver.x[vi] = np.array([
                center_x + (nodes[vi, 0] - center_x) * lat,
                center_y + (nodes[vi, 1] - center_y) * lat,
                nodes[vi, 2] * lam,
            ])

        # Compute reaction forces at z=0 face
        z_min_verts = np.where(np.abs(nodes[:, 2]) < 1e-10)[0]
        rf = solver.compute_reaction_forces(z_min_verts, activation=1.0)
        total_Fz = sum(f[2] for f in rf.values())

        # Analytical: PK1 stress for this uniform F
        F_mat = np.diag([lat, lat, lam])
        d0 = np.array([0.0, 0.0, 1.0])
        P = total_pk1(F_mat, d0, mu, kappa, sigma0, 1.0)

        # Expected z-force on z=0 face:
        # By FEM divergence, sum of elastic gradients at z=0 face =
        # integral of P @ N dA over the reference z=0 face, where N=[0,0,-1].
        # So F_z_expected = P_zz * (-1) * A_ref = -P_zz * A_ref
        F_expected = -P[2, 2] * A_ref

        rel_err = abs(total_Fz - F_expected) / (abs(F_expected) + 1.0)
        ok &= report(
            f"reaction force lam={lam:.2f}",
            rel_err < 1e-6,
            f"Fz={total_Fz:.2f} N, expected={F_expected:.2f} N, rel_err={rel_err:.2e}")

    return ok
```

- [ ] **Step 2: Run test**

Run: `uv run python tests/test_phase1.py`
Expected: All `reaction force lam=...` tests PASS.

If FAIL, investigate whether:
- The sign convention in `compute_reaction_forces` is correct
- The deformed area calculation is correct
- The Cauchy stress extraction is correct

Debug by printing intermediate values (P, cauchy, F_expected, total_Fz) and comparing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase1.py
git commit -m "Add fully-constrained reaction force verification test

Sets all vertices to uniform deformation, then verifies
compute_reaction_forces returns forces consistent with PK1 stress
times reference cross-section area. Tests FEM assembly + reaction
force pipeline independently of solver convergence."
```

---

## Task 5: Register new tests in main() and update CLI

**Files:**
- Modify: `tests/test_phase1.py:379-401` — add new tests to `main()`
- Modify: `main.py` — update docstring

- [ ] **Step 1: Add new tests to test_phase1.py main()**

```python
def main():
    print("=" * 60)
    print("Phase 1 Verification Tests")
    print("=" * 60)

    all_ok = True
    all_ok &= test_dgf_values()
    all_ok &= test_dgf_derivatives()
    all_ok &= test_neo_hookean_gradient()
    all_ok &= test_total_gradient()
    all_ok &= test_hessian_consistency()
    all_ok &= test_fiber_pk1_stress()          # NEW
    all_ok &= test_reaction_force_uniform()     # NEW
    all_ok &= test_box_mesh()
    all_ok &= test_activation_dynamics()
    all_ok &= test_vbd_basic()

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_ok else 1
```

- [ ] **Step 2: Run full test suite**

Run: `uv run python tests/test_phase1.py`
Expected: ALL TESTS PASSED (10 test groups)

Run: `uv run python main.py test`
Expected: Same result via CLI

- [ ] **Step 3: Final commit**

```bash
git add tests/test_phase1.py main.py
git commit -m "Register new validation tests in test suite"
```

---

## Summary of changes

| What | Before | After |
|------|--------|-------|
| `vbd_force_length()` | Broken, unreliable | Deleted |
| `run_vbd_fl_validation()` | Broken, unreliable | Deleted |
| `main.py vbd` command | Calls broken validation | Removed |
| `f_PE(1.0)` test | FAIL (expects 0.0) | PASS (expects < 0.05) |
| Constitutive stress test | Missing | Tests `fiber_pk1` against DGF at 11 stretches x 3 activations |
| Reaction force test | Missing | Tests `compute_reaction_forces` with fully-prescribed uniform deformation at 6 stretches |
| Test count | 8 (1 failing) | 10 (all passing) |
