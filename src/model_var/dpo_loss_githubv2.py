"""Standard Direct Preference Optimization (DPO) loss.

Reference: Rafailov et al., "Direct Preference Optimization: Your Language
Model is Secretly a Reward Model", 2023. Implementation follows the common
open-source version (v2) with optional label smoothing / IPO variant.

Used by ``solver_dpo.py``:

    loss_dpo, chosen_reward, rejected_reward = DPOLoss().compute_loss(
        policy_chosen_logps, policy_rejected_logps,
        reference_chosen_logps, reference_rejected_logps,
    )
"""

import torch.nn.functional as F


class DPOLoss:
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0, ipo: bool = False):
        """
        Args:
            beta: temperature of the DPO objective.
            label_smoothing: conservative label-smoothing weight (0 = vanilla DPO).
            ipo: use the IPO (identity preference optimization) variant.
        """
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def compute_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ):
        """Return (loss, chosen_reward, rejected_reward), all averaged over the batch."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()
