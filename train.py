"""
This file trains TREX on AC Environment.
It sets up the training environment, initializes the TREXPolicy, and runs the TREX training loop.

Run this script directly to start training TREX:

```
python -m trex.train --use-trex
```

To see the entire list of command line arguments you may pass, check args.py
"""

import numpy as np
import torch
import random
from torch.optim import Adam
from trex.policy import TREXPolicy
from trex.config import parse_args
from trex.env_setup import get_env
from trex.training import trex_training_loop


def train_trex():
    """
    Main training function for TREX algorithm.
    """
    args = parse_args()
    
    # Ensure TREX is enabled
    if not args.use_trex:
        print("Warning: --use-trex is False. TREX features will be limited.")
        print("Consider running with --use-trex flag for full TREX functionality.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    (
        envs,
        initial_states,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
    ) = get_env(args)

    policy = TREXPolicy(envs, args.nodes_counts).to(device)
    optimizer = Adam(policy.parameters(), lr=args.learning_rate, eps=args.epsilon)

    trex_training_loop(
        envs,
        args,
        device,
        optimizer,
        policy,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
        initial_states,
    )

    envs.close()


if __name__ == "__main__":
    train_trex()
