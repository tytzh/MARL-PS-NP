"""Run the original V2 training with CPU and autograd optimizations.

Original source files and checkpoints are preserved. Each run writes to a
separate directory under fast_results.
"""

import os
from pathlib import Path
import runpy
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parent
EPISODES = 100  # Smoke test; use 400 for the original training budget.


def configure(output=None):
    # These small dense networks benefit from avoiding thread-pool overhead.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    for folder in (ROOT, ROOT / "initialization", ROOT / "training", ROOT / "agents"):
        sys.path.insert(0, str(folder))

    import torch
    torch.set_num_threads(1)

    import constants
    old_root = constants.path_main
    for name, value in list(vars(constants).items()):
        if isinstance(value, str) and value.startswith(old_root):
            setattr(constants, name, str(ROOT) + value[len(old_root):])
    if output is None:
        output = ROOT.parent / "fast_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    import ddpg_agents
    # Targets are updated by soft_update, never by an optimizer. Disabling
    # their gradients avoids constructing and backpropagating target graphs.
    for cls in (ddpg_agents.Agent_b, ddpg_agents.Agent_c):
        original_init = cls.__init__

        def initialize(self, *args, _initialize=original_init, **kwargs):
            _initialize(self, *args, **kwargs)
            self.actor_target.requires_grad_(False)
            self.critic_target.requires_grad_(False)

        cls.__init__ = initialize

    import training_utils
    training_utils.get_total_s_and_next = get_total_s_and_next_fast
    return output


def get_total_s_and_next_fast(obs_ls, obs_next_ls, d_obs, n_exist, n_knn,
                            EVs_exist, EV_exist_next, list_EVs, n_total):
    """Preserve the original neighbor order, stopping once enough are found."""
    current, following = [], []
    indices = EVs_exist[0]
    for i, ev_id in enumerate(indices):
        selected = []
        if n_knn > 1:
            neighbors = list_EVs[ev_id].idx_ev_near
            for j, other_id in enumerate(indices):
                if other_id != ev_id and other_id in neighbors:
                    selected.append(j)
                    if len(selected) == n_knn - 1:
                        break
        state = obs_ls[i].copy()
        next_state = obs_next_ls[i].copy()
        for j in selected:
            state.extend(obs_ls[j])
            next_state.extend(obs_next_ls[j])
        current.append(state + [0] * (n_total - len(state)))
        following.append(next_state + [0] * (n_total - len(next_state)))
    return current, following


if __name__ == "__main__":
    configure()
    runpy.run_path(str(ROOT / "training_core.py"),
                  init_globals={"EPISODES": EPISODES}, run_name="__main__")
