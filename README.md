# TREX: Topologically-guided Reinforcement Learning with EXploration

## Overview

TREX is a reinforcement learning algorithm designed for solving AC (Andrews-Curtis) problems. It combines Active Symbolic Closure (ASC) for feasibility constraints with Topological Neuro-Symbolic Compression (TNSC) for guided exploration.

### AC Problem Generator

The `ac_solver` module (located at `trex/ac_solver/`) provides the AC (Andrews-Curtis) problem generator and environment. The AC problem is an unsolved conjecture in combinatorial group theory about finding paths between group presentations.

#### AC Environment (`ac_solver/envs/`)

- **ACEnv**: Gymnasium-compatible environment for AC problems
  - **State Space**: Balanced presentations with two generators (relators)
  - **Action Space**: 12 discrete AC moves
  - **Reward**: Sparse terminal reward when presentation becomes trivial (sum of relator lengths = 2)
  - **Observation**: NumPy array representation of presentation (relators padded with zeros)
  
- **AC Moves** (`ac_solver/envs/ac_moves.py`): Implementation of AC operations
  - **Concatenation**: `r_i → r_i · r_j^±1` (4 moves)
  - **Conjugation**: `r_i → x_j^±1 · r_i · x_j^∓1` (8 moves)
  - Total: 12 moves per state
  
- **Presentation Utilities** (`ac_solver/envs/utils.py`): Helper functions
  - `is_array_valid_presentation()`: Check presentation validity
  - `is_presentation_trivial()`: Check if presentation is trivial
  - `generate_trivial_states()`: Generate all 8 trivial target states
  - `simplify_relator()`: Simplify relators by removing inverses
  - `convert_relators_to_presentation()`: Convert relators to array format

#### Search Algorithms (`ac_solver/search/`)

- **Breadth-First Search** (`breadth_first.py`): BFS for AC problems
- **Greedy Search** (`greedy.py`): Greedy search algorithm
- **Miller-Schupp Series** (`miller_schupp/`): Benchmark presentations
  - Data files with solved/unsolved presentations
  - Used for training and evaluation

#### PPO Baseline (`ac_solver/agents/`)

- Standard PPO implementation for comparison with TREX
- Includes training loop, agent class, and environment setup utilities

## Algorithm Components

### A. ASC: Active Symbolic Closure

ASC ensures that rollouts stay within the feasible set **F** by performing local admissibility checks at each step:

- **Local Validity Checking**: Each action is validated before sampling to ensure it doesn't violate feasibility constraints
- **Prefix-level Pruning**: Actions that would exceed `max_relator_length` or fail to change the presentation are marked as invalid
- **Implementation**: Invalid actions are masked by setting their logits to `-inf`, ensuring they are never sampled

### B. TNSC: Topological Neuro-Symbolic Compression

TNSC provides dense guidance signals within the feasible region using two topological potentials:

#### Width Potential (Ψ_P)
- Prefers actions that reduce the total relator length
- Helps reduce the effective branching factor by guiding toward simpler states
- Formula: `Ψ_P(a) = -(new_total_len - base_total_len)`

#### Depth Potential (Ψ_H)
- Measures distance to the closest trivial target presentation
- Uses Euclidean distance in embedding space (can be upgraded to Poincaré ball)
- Formula: `Ψ_H(a) = -min_distance(new_state, trivial_targets)`

#### Combined Potential
- Total potential: `Ψ_total = λ_width · Ψ_P + λ_depth · Ψ_H`
- Configurable via `--trex-width-coef` and `--trex-depth-coef` arguments

### C. Topologically Guided Sampling

The sampling distribution is modified to incorporate topological guidance:

```
P_sample(a_t|s_t) ∝ π_θ_old(a_t|s_t) · exp(λ · Ψ_total(s_t, a_t))
```

- Base policy logits are augmented with the topological potential scaled by `λ` (`--trex-lambda`)
- Combined with ASC validity masking to ensure trajectories stay in **F**
- Implementation: `logits = base_logits + λ · Ψ_total` with invalid actions masked

### D. Hybrid Reward

Reward function combines terminal success with weak process signals:

```
R(τ_i) = 1[τ_i ∈ T*] + β · Σ_t 1[Valid_S(·) = ⊤]
```

- **Terminal Reward**: Standard environment reward indicating successful solution
- **Process Reward**: Weak signal based on valid transitions (controlled by `--trex-beta-valid`)
- Helps provide learning signal even for early-stopped trajectories

### E. Policy Update

TREX uses a PPO-style update with KL penalty:

```
L_TREX(θ) = (1/G) Σ_i [min(r_i A_i, clip(r_i, 1-ε, 1+ε) A_i) - δ D_KL(π_θ(τ_i)|π_ref(τ_i))]
```

- **PPO Clip**: Standard clipped surrogate objective
- **KL Penalty**: Regularization term to prevent policy from deviating too far
- **Group-Relative Advantage**: Optional normalization of advantages within trajectory groups (via `--trex-group-adv`)
- **Adaptive Beta**: KL penalty coefficient adapts based on target KL threshold

## File Structure

```
trex/
├── __init__.py          # Module exports
├── config.py            # Command-line argument parsing
├── env_setup.py         # Environment initialization and setup
├── guidance.py          # ASC validity masking and TNSC potentials
├── policy.py            # TREXPolicy class (actor-critic network)
├── train.py             # Main training entry point
├── training.py          # TREX training loop implementation
├── utils.py             # Helper functions
├── README.md            # This documentation file
├── VERIFICATION.md      # Implementation verification report
├── eval/                # Evaluation scripts
│   ├── __init__.py
│   ├── evaluate.py      # Main evaluation script
│   ├── utils.py         # Evaluation utilities
│   ├── eval.sh          # Shell script for evaluation
│   └── README.md        # Evaluation documentation
└── ac_solver/           # AC (Andrews-Curtis) problem generator
    ├── __init__.py      # AC solver module exports
    ├── envs/            # AC environment implementation
    │   ├── __init__.py
    │   ├── ac_env.py    # ACEnv and ACEnvConfig classes
    │   ├── ac_moves.py  # AC move operations (12 moves)
    │   └── utils.py     # Presentation utilities and helpers
    ├── search/          # Classical search algorithms
    │   ├── __init__.py
    │   ├── breadth_first.py  # BFS for AC problems
    │   ├── greedy.py    # Greedy search algorithm
    │   └── miller_schupp/    # Miller-Schupp series
    │       ├── __init__.py
    │       ├── miller_schupp.py  # MS presentation generator
    │       └── data/     # Benchmark presentation data
    │           ├── all_presentations.txt
    │           ├── greedy_solved_presentations.txt
    │           └── ...
    └── agents/          # PPO baseline (for comparison)
        ├── __init__.py
        ├── ppo_agent.py # PPO agent class
        ├── ppo.py       # PPO training script
        ├── training.py  # PPO training loop
        ├── args.py      # Argument parsing
        ├── environment.py  # Environment setup
        └── utils.py     # Helper functions
```

## Usage

### Basic Training

```bash
python -m trex.train --use-trex
```

### Key Arguments

#### TREX-Specific Arguments

- `--use-trex`: Enable TREX features (required for full functionality)
- `--trex-lambda`: Global scale λ for topological potential (default: 1.0)
- `--trex-width-coef`: Weight for width potential Ψ_P (default: 1.0)
- `--trex-depth-coef`: Weight for depth potential Ψ_H (default: 1.0)
- `--trex-beta-valid`: Coefficient β for weak process reward (default: 0.0)
- `--trex-group-adv`: Enable group-relative advantage normalization (default: False)

#### Standard RL Arguments

- `--num-envs`: Number of parallel environments (default: 4)
- `--num-steps`: Steps per rollout (default: 2000)
- `--total-timesteps`: Total training timesteps (default: 200000)
- `--learning-rate`: Learning rate (default: 2.5e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--gae-lambda`: GAE lambda (default: 0.95)
- `--clip-coef`: PPO clipping coefficient (default: 0.2)
- `--ent-coef`: Entropy coefficient (default: 0.01)
- `--vf-coef`: Value function coefficient (default: 0.5)
- `--target-kl`: Target KL divergence threshold (default: 0.01)

#### Environment Arguments

- `--states-type`: Type of initial states ("solved", "unsolved", or "all", default: "all")
- `--horizon-length`: Maximum episode length (default: 2000)
- `--fixed-init-state`: Use a fixed initial state instead of loading from file

### Example: Full TREX Training

```bash
python -m trex.train \
    --use-trex \
    --trex-lambda 1.5 \
    --trex-width-coef 1.0 \
    --trex-depth-coef 1.0 \
    --trex-beta-valid 0.1 \
    --trex-group-adv \
    --num-envs 8 \
    --num-steps 2000 \
    --total-timesteps 1000000 \
    --learning-rate 2.5e-4 \
    --wandb-log \
    --wandb-project-name "TREX-AC-Solver"
```

## Implementation Details

### Validity Masking

The ASC component checks each action by:
1. Simulating the AC move on the current state
2. Checking if the move changes the presentation or lengths
3. Marking actions as invalid if they exceed `max_relator_length` or produce no change

### Potential Computation

For each state-action pair:
1. Compute width potential based on length reduction
2. Compute depth potential based on distance to trivial targets
3. Normalize both potentials across valid actions
4. Combine with configurable weights

### Guided Sampling

During rollout:
1. Compute validity masks and potentials for current states
2. Apply masking and potential augmentation to policy logits
3. Sample actions from the modified distribution
4. Track valid transitions for hybrid reward

### Training Loop

1. **Rollout Phase**: Collect trajectories with TREX-guided sampling
2. **Advantage Computation**: Compute GAE advantages (optionally group-normalized)
3. **Policy Update**: PPO clip + KL penalty with adaptive beta
4. **Value Update**: Standard value function regression

## Dependencies

- PyTorch
- NumPy
- Gymnasium
- WandB (optional, for logging)
- tqdm (for progress bars)

## Evaluation

TREX evaluation follows the structure of EMPO-main for consistency. Use the evaluation script to test trained models:

### Basic Evaluation

```bash
python -m trex.eval.evaluate \
    --checkpoint_path out/trex_checkpoint.pt \
    --output_dir ./eval_outputs \
    --num_episodes 100 \
    --states_type all \
    --use_trex_guidance \
    --save_trajectories \
    --save_failed
```

### Evaluation Arguments

- `--checkpoint_path`: Path to trained TREX checkpoint (.pt file)
- `--output_dir`: Directory to save evaluation results (default: ./eval_outputs)
- `--num_episodes`: Number of problems to evaluate (default: 100)
- `--max_steps`: Maximum steps per episode (default: 2000)
- `--states_type`: Type of initial states ("solved", "unsolved", or "all")
- `--use_trex_guidance`: Enable TREX guidance during evaluation (default: True)
- `--eval_deterministic`: Use greedy (deterministic) policy instead of sampling
- `--save_trajectories`: Save full action trajectories
- `--save_failed`: Save failed cases for analysis

### Evaluation Outputs

The evaluation script generates:
- `eval_results_<timestamp>.jsonl`: Detailed results for each problem
- `metrics_<timestamp>.json`: Aggregated metrics (success rate, path lengths, etc.)
- `failed_cases_<timestamp>.jsonl`: Failed problems (if `--save_failed` is used)

### Metrics

Evaluation metrics include:
- **Success Rate**: Percentage of problems solved
- **Average Path Length**: Mean number of steps for solved problems
- **Min/Max Path Length**: Shortest and longest solution paths
- **Per-problem Results**: Success status, path length, trajectory, final state

### Using the Shell Script

```bash
bash trex/eval/eval.sh <checkpoint_path> [output_dir] [num_episodes] [states_type]
```

Example:
```bash
bash trex/eval/eval.sh out/trex_checkpoint.pt ./eval_outputs 100 all
```

## Notes

- **Depth Potential**: Currently uses Euclidean distance as a placeholder. Can be upgraded to Poincaré ball embedding for hyperbolic geometry as specified in the original algorithm description.
- **Hybrid Reward**: Accumulates valid transitions until first violation, exactly as specified. The cumulative count is added to the reward when violation occurs or episode ends.
- **Group-Relative Advantage**: Optional normalization controlled by `--trex-group-adv` flag. When enabled, advantages are normalized within trajectory groups.
- **AC Generator**: The `ac_solver` module is included as a submodule providing the AC problem environment, classical search algorithms, and PPO baseline for comparison.

## References

TREX algorithm components:
- ASC: Active Symbolic Closure for feasibility constraints
- TNSC: Topological Neuro-Symbolic Compression for guided exploration
- Topologically Guided Sampling for improved exploration
- Hybrid Reward shaping for better learning signals
- TREX-style PPO updates with KL regularization
