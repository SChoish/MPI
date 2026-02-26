# MPI: Multi-step Proximal Policy Improvement in Offline RL

Offline reinforcement learning must balance two goals: policy updates should stay near dataset-supported actions to keep value estimates reliable, yet meaningful gains often require moving beyond the behavior distribution. This repository implements **Multi-step Proximal Policy Improvement (MPI)**, a plug-in refinement mechanism that composes sequential **re-centered** proximal steps on a policy manifold, enabling controlled improvement beyond dataset support while preserving local stability.

**MPI** interprets many behavior-anchored offline actor updates as a **single proximal policy improvement (SPI)** step—an implicit discretization of a manifold gradient flow induced by a critic-defined energy. MPI then advances the same flow through multiple re-centered proximal steps per iteration, corresponding to a finer implicit (JKO-style) discretization. The framework accommodates multiple policy geometries (e.g. Fisher–Rao, Wasserstein-2) and works with deterministic and diagonal-Gaussian policies. Experiments on D4RL show that MPI consistently improves strong baselines (e.g. TD3+BC, ReBRAC, IQL), often reducing conservatism without destabilizing training.

---

## Quick Start

### Installation

```bash
pip install -r requirements/requirements.txt
pip install geomloss PyYAML
# For JAX-based algorithms (ReBRAC, FQL)
pip install jax jaxlib flax optax ott-jax
```

Optional environment variables (e.g. for headless runs):

```bash
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
```

### Run

```bash
# PyTorch (IQL + MPI)
python -m algorithms.offline.mpi_main \
  --config_path configs/offline/mpi/halfcheetah/medium_v2_iql.yaml

# JAX (ReBRAC + MPI)
python -m algorithms.offline.mpi_jax \
  --config_path configs/offline/mpi/halfcheetah/medium_v2_rebrac.yaml
```

Outputs: checkpoints and logs under `results/` and `logs/` (if enabled).

---

## Theory (from the paper)

### Geometric view of offline actor updates

- **Policy manifold** $\mathcal{M}$: the policy class is treated as a statistical manifold with a chosen metric (e.g. Fisher–Rao or Wasserstein-2), inducing a geodesic distance $d_{\mathcal{M}}$.
- **Energy** $\mathcal{E}[\pi]$: typically defined via the critic (e.g. $\mathbb{E}_{a\sim\pi}[-\hat{Q}(s,a)]$ or $-\mathbb{E}[A(s,a)]$).
- **Single Proximal Policy Improvement (SPI)**: one implicit Euler (JKO) step on $\mathcal{M}$:
  $\pi_{k+1} \in \arg\min_{\pi\in\mathcal{M}} \left( \mathcal{E}[\pi] + \frac{1}{2\tau}\,d_{\mathcal{M}}^2(\pi,\pi_k) \right).$
  Many behavior-anchored offline actor objectives (e.g. TD3+BC, ReBRAC, IQL-style extraction) can be interpreted as one such SPI step from the behavior or a base policy.

### Multi-step Proximal Policy Improvement (MPI)

- **Idea**: instead of a single proximal step per iteration, compose **K re-centered** proximal steps with the same energy and geometry:
  - $\pi_0 = \pi_{\mathrm{base}}$ (base algorithm’s actor after its usual update),
  - $\pi_i \in \arg\min_{\pi\in\mathcal{M}} \left( \mathcal{E}[\pi] + \frac{1}{2\tau_i}\,d_{\mathcal{M}}^2(\pi,\pi_{i-1}) \right)$, $i=1,\ldots,K$.
- This corresponds to a **finer implicit discretization** of the same gradient flow, allowing **controlled advancement** beyond dataset support while each step remains proximal to the previous iterate.
- **Monotone descent**: the (estimated) energy decreases along $\pi_0,\pi_1,\ldots,\pi_K$ (up to optimization error).

### Implementation (plug-in refinement)

At each training iteration:

1. Run the base algorithm’s **critic update** and **actor update** → obtain $\pi_0$ (base policy).
2. Apply **MPI**: from $\pi_0$, solve K re-centered proximal subproblems with the current $\hat{\mathcal{E}}$ and $d_{\mathcal{M}}$ → obtain $\{\pi_i\}_{i=1}^K$.
3. Use the refined policies (e.g. $\pi_1$ or $\pi_2$) for rollout or evaluation; base algorithm’s critic and training logic are unchanged.

So: **$\pi_{\mathrm{base}}$** = base algorithm actor; **$\pi_1,\pi_2,\ldots$** = MPI-refined policies (paper table reports $\pi_{\mathrm{base}}$, $\pi_1$, $\pi_2$ on D4RL).

---

## Supported algorithms and geometry

| Base algorithm (PyTorch) | Base algorithm (JAX) |
|--------------------------|----------------------|
| IQL, TD3+BC, CQL, AWAC, SAC-N, EDAC | ReBRAC, FQL |

- **Geometry**: Wasserstein-2 (and Sinkhorn approximation where needed); diagonal-Gaussian and deterministic policies use closed-form $d_{\mathcal{M}}$ where applicable.
- **Energy** $\hat{\mathcal{E}}$ is algorithm-specific (e.g. $-\hat{Q}(s,\pi(s))$, $-\hat{A}$, or entropy-regularized Q terms). Config key `energy_function_type` can switch between Q-based and advantage-based energy for IQL/AWAC.

---

## Project structure

```
MPI/
├── algorithms/
│   ├── networks/           # Shared networks (PyTorch & JAX): actors, critics, MLP, policy_call
│   └── offline/
│       ├── mpi_main.py     # PyTorch: multi-actor training (IQL, TD3+BC, CQL, AWAC, SAC-N, EDAC)
│       ├── mpi_jax.py      # JAX: multi-actor training (ReBRAC, FQL)
│       ├── mpi_policies.py # Policy protocol and adapters
│       ├── utils_pytorch.py
│       ├── utils_jax.py
│       └── iql.py, td3_bc.py, cql.py, awac.py, sac_n.py, edac.py, rebrac.py, fql.py, ...
├── configs/
│   └── offline/
│       ├── mpi/            # Algorithm/env/task YAML configs
│       └── *_mpi_base.yaml
└── README.md
```

- **Actor0** = base algorithm’s actor (one SPI-type update from behavior/base).
- **Actor1, Actor2, …** = MPI refinement steps (re-centered proximal updates); their number and step sizes are set via config (e.g. `num_actors`, `w2_weights`).

---

## Configuration

- **Algorithm**: `algorithm: iql | td3_bc | cql | awac | sac_n | edac` (PyTorch), or ReBRAC/FQL (JAX).
- **MPI**: `num_actors`, `w2_weights` (one weight per refinement step from Actor1 onward).
- **Sinkhorn** (when W2 is approximated): `sinkhorn_K`, `sinkhorn_blur`, `sinkhorn_backend`.
- **Wandb**: `use_wandb`, `project`, `group`, `name`; or disable with `--no_wandb` / `use_wandb: false`.

Example (IQL + MPI, 3 actors, first refinement weight 100):

```yaml
algorithm: iql
num_actors: 3
w2_weights: [100.0, 100.0]
# ... env, seed, eval_freq, etc.
```

---

## Implementation notes

- **Critic**: updated only with the base actor (Actor0); no change to the base algorithm’s critic update.
- **Distance** $d_{\mathcal{M}}$: for diagonal Gaussian policies, W2 is computed in closed form; otherwise Sinkhorn (e.g. GeomLoss for PyTorch, OTT for JAX) is used.
- **Energy**: each algorithm implements its own energy (e.g. `compute_energy_function`); Actor1+ losses are energy + (weighted) $d_{\mathcal{M}}^2(\pi_i,\pi_{i-1})$.
- **Evaluation**: you can evaluate $\pi_{\mathrm{base}}$ or any $\pi_k$ ($k\ge 1$); the paper table reports $\pi_{\mathrm{base}}$, $\pi_1$, $\pi_2$ on D4RL locomotion.

---

## Troubleshooting

- **Import errors**: run as `python -m algorithms.offline.mpi_main` (or `mpi_jax`) from the repo root.
- **GeomLoss**: `pip install geomloss` (PyTorch).
- **OTT-jax**: `pip install ott-jax` (JAX).
- **Headless**: `export MUJOCO_GL=egl`.
- **Wandb**: set `use_wandb: false` in config or pass `--no_wandb`.

---

## References

- **MPI**: Multi-step Proximal Policy Improvement in Offline Reinforcement Learning (geometric view: policy manifold, SPI, re-centered proximal steps, JKO-style discretization).
- **Baselines**: TD3+BC, ReBRAC, IQL, CQL, AWAC, SAC-N, EDAC, FQL.
- **Benchmarks**: D4RL (e.g. MuJoCo locomotion, AntMaze).
- **D4RL**: [Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/Farama-Foundation/D4RL).

---

## License

See [LICENSE](LICENSE).
