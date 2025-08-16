# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Group Relative Policy Optimisation (GRPO) experiment implementation."""

from .grpo_utils import (
    compute_group_normalized_rewards, 
    compute_naive_policy_gradient_loss, 
    compute_grpo_clip_loss,
    compute_policy_gradient_loss,
    masked_mean,
    grpo_microbatch_train_step
)

__all__ = [
    "compute_group_normalized_rewards", 
    "compute_naive_policy_gradient_loss", 
    "compute_grpo_clip_loss",
    "compute_policy_gradient_loss",
    "masked_mean",
    "grpo_microbatch_train_step"
]