# Is One Step Enough for Offline Policy Improvement?

Offline reinforcement learning must balance two goals: policy updates should stay near dataset-supported actions to keep value estimates reliable, yet meaningful gains often require moving beyond the behavior distribution. This repository implements **Multi-step Proximal Policy Improvement (MPI)**, which studies how policy improvement is composed through sequential **re-centered** proximal steps. The paper parameterizes a chain by a **nominal total horizon $T$** and **$K$ stages**, with local horizon **$h=T/K$**: increasing $K$ at fixed $T$ subdivides the horizon, whereas adding stages at a common $h$ extends it to $T=Kh$. The first actor uses dataset actions or a base extraction loss; each later actor uses its freshly updated predecessor as a detached reference.

MPI develops ideas first explored in [**POGO**](https://github.com/SChoish/POGO), an earlier study of transport-map policies and JKO/Sinkhorn-based policy flows. The present formulation turns that geometric exploration into a re-centered proximal framework that applies to standard behavior-regularized actor updates across multiple policy geometries and base algorithms.

**Paper authors:** [Soohyun Choi](https://github.com/SChoish)<sup>*</sup>,
[Seonvin Cho](https://github.com/seonvin0319)<sup>*</sup>, and Prof. Songnam Hong<sup>†</sup>  
<sup>*</sup>Equal contribution. <sup>†</sup>Corresponding author.

**Paper:** [Is One Step Enough for Offline Policy Improvement? (arXiv:2609.03842 v2)](https://arxiv.org/abs/2609.03842v2).

**MPI** connects TD3+BC to a **single proximal improvement (SPI)** objective under deterministic Wasserstein-2 geometry; advantage-weighted extraction has a local KL/Fisher–Rao connection. Under an ideal fixed critic, sequential re-centering can reach endpoints unavailable to any single proximal step. For smooth ideal updates at fixed $K$ and small $T$, subdivision reduces the leading local discretization error relative to the corresponding flow by $1/K$. These results concern improvement composition, not a general ordering of policy returns. The paper studies implicit-proximal (I-MPI) and explicit-target (E-MPI) deterministic updates, and Gaussian Q+BC–W2 and AWR–Fisher–Rao extraction with IQL critics.

The local horizon $h$ is a nominal coefficient of the critic-induced update, not a measure of actual action displacement. Small $h$ penalizes local movement more strongly; larger $h$ can increase discretization error and exposure to critic approximation error. MPI therefore studies a stability–improvement trade-off rather than guaranteeing monotonic improvement in true return under an imperfect offline critic.

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
- **Single proximal improvement (SPI)**: one metric proximal step, corresponding to an implicit Euler step in Euclidean coordinates:
  $\pi^+ \in \arg\min_{\pi\in\mathcal{M}} \left( \mathcal{E}[\pi] + \frac{1}{2h}\,d_{\mathcal{M}}^2(\pi,\nu) \right).$
  With a fixed critic and normalization, TD3+BC has the same minimizers as this objective with a deterministic reference equal to the conditional mean of dataset actions and $h=\alpha/2$. The connection for IQL/AWAC advantage weighting is through a KL-regularized distributional objective and is local in Fisher–Rao geometry.

### Multi-step Proximal Policy Improvement (MPI)

- **Ideal composition**: from an initial reference $\pi_0$, compose **$K$ re-centered** proximal steps with a fixed critic-induced energy and geometry:
  $\pi_k \in \arg\min_{\pi\in\mathcal{M}} \left( \mathcal{E}[\pi] + \frac{1}{2h_k}\,d_{\mathcal{M}}^2(\pi,\pi_{k-1}) \right)$, $k=1,\ldots,K$, with $T=\sum_{k=1}^K h_k$.
- **Horizon subdivision**: at fixed $T$, use $h=T/K$. Under the paper's local smoothness assumptions, ideal explicit and implicit updates reduce the leading flow discretization error by $1/K$. The analysis holds the critic and normalization fixed; live training need not do so across stages.
- **Additional refinement**: at a common local horizon $h$, increasing $K$ also increases $T=Kh$; this is not subdivision of a fixed horizon.
- **Estimated-energy descent**: for a fixed energy, feasible preceding policies, and sufficiently accurate proximal solves, estimated energy decreases up to optimization error. This does not guarantee monotonic true-return improvement.

### Implementation (plug-in refinement)

At each training iteration (actor updates follow the base algorithm's schedule):

1. Run the base algorithm’s **critic update** and scheduled **actor update** → update Actor0 (base policy).
2. Update the remaining persistent actors in order, taking an optimizer step on each re-centered proximal loss with its freshly updated predecessor detached as reference. These training updates approximate the ideal proximal subproblems.
3. Evaluate the actors; the paper deploys the final actor. Later actors do not supply critic bootstrap actions, and the base algorithm's value-learning procedure is retained.

**Indexing:** this repository calls the first trained actor **Actor0** ($\pi_{\mathrm{base}}$) and the later actors **Actor1, Actor2, …**. In v2, $K$ counts all trained stages, including the first: paper stage $k$ corresponds to repository Actor$(k-1)$, and the final actor is Actor$(K-1)$. For example, I-MPI-4 denotes a four-actor chain, including its first actor.

**Experiments in v2:** the main fixed-$T$ TD3+BC sweep uses nine D4RL locomotion tasks (HalfCheetah, Hopper, Walker2d × medium, medium-replay, expert), four seeds, and the final checkpoint after $10^6$ critic updates. Depth $K$ uses $K$ actor optimizer calls per delayed update. Subdivision broadens the observed range of useful total horizons; a common-small-$h$ comparison separately tests additional refinement. Matched fixed-reference controls show both gains and losses from re-centering across tasks, and Gaussian IQL extraction shows depth-dependent gains and failures. The supported algorithms below describe repository capabilities, not the scope of v2's experiments.

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

`num_actors` counts all actors (the paper's $K$). These entry points expose base-objective coefficients and `w2_weights`, not a direct $T$ or $h$ option; changing `num_actors` alone does not enforce the paper's equal-$h=T/K$ schedule. Matching that schedule requires coordinating the first-stage and later-stage coefficients with their energy and distance normalizations.

Example (IQL + MPI, 3 actors, first refinement weight 100):

```yaml
algorithm: iql
num_actors: 3
w2_weights: [100.0, 100.0]
# ... env, seed, eval_freq, etc.
```

---

## Implementation notes

- **Critic**: retains the base algorithm's value-learning procedure. For TD3+BC, Actor0 supplies bootstrap actions and later actors only affect extraction; IQL's value targets are actor-independent. At fixed $T$, changing $K$ also changes the first-stage coefficient, so TD3+BC critic targets can differ across depths.
- **Distance** $d_{\mathcal{M}}$: for diagonal Gaussian policies, W2 is computed in closed form; otherwise Sinkhorn (e.g. GeomLoss for PyTorch, OTT for JAX) is used.
- **Energy**: each algorithm implements its own energy (e.g. `compute_energy_function`); Actor1+ losses are energy + (weighted) $d_{\mathcal{M}}^2(\pi_i,\pi_{i-1})$.
- **Evaluation**: you can evaluate $\pi_{\mathrm{base}}$ or any later actor; v2 reports final-stage policies by total horizon $T$ and depth $K$.

---

## Troubleshooting

- **Import errors**: run as `python -m algorithms.offline.mpi_main` (or `mpi_jax`) from the repo root.
- **GeomLoss**: `pip install geomloss` (PyTorch).
- **OTT-jax**: `pip install ott-jax` (JAX).
- **Headless**: `export MUJOCO_GL=egl`.
- **Wandb**: set `use_wandb: false` in config or pass `--no_wandb`.

---

## References

- **MPI paper**: [Is One Step Enough for Offline Policy Improvement?](https://arxiv.org/abs/2609.03842v2) (nominal horizon subdivision, additional refinement, and re-centered improvement composition).
- **Baselines**: TD3+BC, ReBRAC, IQL, CQL, AWAC, SAC-N, EDAC, FQL.
- **Benchmarks**: D4RL (e.g. MuJoCo locomotion, AntMaze).
- **D4RL**: [Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/Farama-Foundation/D4RL).

---

## License

See [LICENSE](LICENSE).
