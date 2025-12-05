"""
TreePO Sampler Algorithm: Tree-based Policy Optimization sampling.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from ..registry import SamplerAlgorithm, register_sampler_algorithm
from . import TreeNode, backpropagation, extract_boxed_answer, get_ancestors
from .sample_schema import TreePOSampleSchema


@register_sampler_algorithm("treepo")
class TreePOSamplerAlgorithm(SamplerAlgorithm):
    """TreePO sampler algorithm implementation."""

    def __init__(self, ralo, config):
        super().__init__(ralo, config)
        self.schema = TreePOSampleSchema()

    def create_sample_schema(self) -> TreePOSampleSchema:
        """Create TreePO sample schema."""
        return self.schema

    def run(self) -> None:
        """Run the sampler (legacy method)."""
        self.ralo._treepo_run_sampler()

    def process_problem(self, problem: Dict[str, Any], services: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a problem using TreePO and return samples.

        Args:
            problem: Problem dictionary
            services: Dictionary with 'model', 'sampling', 'orchestrator' services

        Returns:
            List of sample dictionaries
        """
        import math
        import numpy as np
        from collections import defaultdict, deque
        from vllm import SamplingParams

        sampling_service = services["sampling"]
        model_service = services["model"]
        ralo = self.ralo

        # Get TreePO parameters
        generation_length = ralo.TreePO_kwargs.get("generation_length", 500)
        object_depth = ralo.TreePO_kwargs.get("depth", 8)
        budget_coefficient = ralo.TreePO_kwargs.get("budget_coefficient", 2)
        sampling_batch_size = ralo.TreePO_kwargs.get("sampling_batch_size", 16)
        rollout_number = ralo.rollout_num

        # Execute prompt template
        execute_prompt_template = lambda prompt: ralo.rollout_prompt_fn(prompt)

        # Reward function
        def gen_reward(answer, item):
            reward = {}
            for reward_fn in ralo.reward_fns:
                name = reward_fn.__name__
                if name.endswith("_tok"):
                    reward[name] = reward_fn(answer, item)
                else:
                    reward[name] = reward_fn(answer, item)
            reward["total"] = sum(reward.values())
            return reward

        # Initialize tree search
        step = {
            "main_prompt": execute_prompt_template(problem),
            "prompt_in_this_step": execute_prompt_template(problem),
            "finished_reason": None,
            "current_answer": "",
            "entropy_in_this_step": 0,
            "mean_entropy": 0,
            "mean_length": 0,
            "depth": 0,
            "token_ids": [],
            "gen_logps": [],
        }

        head_node = TreeNode(step, budget_coefficient)
        queue = deque([head_node])
        finished_nodes = []

        loop1_counter = 1
        chunk_decrease = 2

        # Main tree search loop
        while True:
            if loop1_counter % 3 == 0:
                if sampling_batch_size > 1:
                    sampling_batch_size = sampling_batch_size // chunk_decrease

            loop1_counter += 1

            # Fallback to parents if queue is empty
            if len(queue) == 0 and len(finished_nodes) < rollout_number and len(finished_nodes) > 0:
                parents_to_fallback = set()
                for node in finished_nodes:
                    if node.parent:
                        parents_to_fallback.add(node.parent)

                for parent_node in parents_to_fallback:
                    if parent_node not in queue:
                        queue.append(parent_node)

            # Check termination condition
            if len(finished_nodes) >= rollout_number or len(queue) == 0:
                finished_nodes.sort(key=lambda node: node.depth, reverse=True)
                finished_nodes = finished_nodes[:rollout_number]
                break

            # Group nodes by budget
            chunk_map = defaultdict(list)
            node_chunk_map = defaultdict(list)

            for _ in range(len(queue)):
                node = queue.popleft()
                chunk_map[node.budget].append(node.item["prompt_in_this_step"])
                node_chunk_map[node.budget].append(node)

            # Process each budget group
            for budget in chunk_map:
                if len(finished_nodes) >= rollout_number:
                    break

                node_for_budget = node_chunk_map[budget]
                chunk_for_budget = chunk_map[budget]

                # Batch processing
                if len(chunk_for_budget) < sampling_batch_size:
                    loop_size = 1
                else:
                    loop_size = len(chunk_for_budget) // sampling_batch_size + 1

                for iteration in range(loop_size):
                    if len(finished_nodes) >= rollout_number:
                        break

                    if iteration + 1 == loop_size:
                        chunks = chunk_for_budget[sampling_batch_size * iteration :]
                        node_chunks = node_for_budget[sampling_batch_size * iteration :]
                    else:
                        chunks = chunk_for_budget[
                            sampling_batch_size * iteration : sampling_batch_size * (iteration + 1)
                        ]
                        node_chunks = node_for_budget[
                            sampling_batch_size * iteration : sampling_batch_size * (iteration + 1)
                        ]

                    # Progressive logprobs strategy: start small, increase if needed
                    # This minimizes overhead while ensuring we get the sampled token's logprob
                    # Strategy: Try logprobs=10 first (most cases), then 50, 100, 200, 400 if needed
                    # Note: Retrying with larger logprobs will generate different results due to sampling,
                    # but this is acceptable as we prioritize getting accurate logprobs
                    logprobs_values = [10, 50, 100, 200, 400]  # Progressive search range
                    vllm_outputs = None
                    used_logprobs = None
                    missing_logprobs_count = 0
                    
                    for logprobs_val in logprobs_values:
                        # Create sampling params with current logprobs value
                        sampling_params = sampling_service.create_sampling_params(
                            n=budget,
                            temperature=ralo.gen_temperature,
                            max_tokens=int(generation_length),
                            logprobs=logprobs_val,
                            include_stop_str_in_output=True,
                        )
                        
                        # Generate using vLLM
                        vllm_outputs = sampling_service.generate(chunks, sampling_params, use_tqdm=False)
                        
                        # Check if all sampled tokens have their logprobs available
                        all_tokens_have_logprobs = True
                        missing_logprobs_count = 0
                        for i, v_output in enumerate(vllm_outputs):
                            for generation_output in v_output.outputs:
                                for token_id, step_logprobs in zip(
                                    generation_output.token_ids, generation_output.logprobs
                                ):
                                    if step_logprobs is not None and token_id not in step_logprobs:
                                        all_tokens_have_logprobs = False
                                        missing_logprobs_count += 1
                        
                        if all_tokens_have_logprobs:
                            used_logprobs = logprobs_val
                            break  # Found sufficient logprobs, no need to increase
                        elif missing_logprobs_count == 0:
                            # All logprobs are None (shouldn't happen with logprobs > 0, but handle gracefully)
                            used_logprobs = logprobs_val
                            break
                    
                    # If still missing some logprobs after trying all values, use the last (largest) one
                    if used_logprobs is None:
                        used_logprobs = logprobs_values[-1]
                    
                    # Log if we had to increase logprobs (for monitoring/debugging)
                    if used_logprobs > logprobs_values[0] and missing_logprobs_count > 0:
                        print(f"[TreePO] Increased logprobs from {logprobs_values[0]} to {used_logprobs} "
                              f"(missing {missing_logprobs_count} token logprobs)")

                    # Process generation outputs
                    for i, parent_node in enumerate(node_chunks):
                        if len(finished_nodes) >= rollout_number:
                            break

                        v_output = vllm_outputs[i]

                        for generation_output in v_output.outputs:
                            if len(finished_nodes) >= rollout_number:
                                break

                            # Extract token log probs and compute entropy
                            # Get logprob for the actual sampled token (the one in token_ids)
                            token_log_probs = []
                            token_probs = []
                            for token_id, step_logprobs in zip(
                                generation_output.token_ids, generation_output.logprobs
                            ):
                                if step_logprobs is None:
                                    # No logprobs available for this step
                                    token_log_probs.append(-float('inf'))
                                    token_probs.append(0.0)
                                elif token_id in step_logprobs:
                                    # Sampled token's logprob is available - this is what we want
                                    logprob = step_logprobs[token_id].logprob
                                    token_log_probs.append(logprob)
                                    token_probs.append(math.exp(logprob))
                                else:
                                    # Sampled token not in logprobs dict even after increasing logprobs
                                    # This can occur if token is out of vocabulary or special token
                                    # Use a very low probability as fallback
                                    token_log_probs.append(-float('inf'))
                                    token_probs.append(0.0)

                            sum_entropy = -sum(p * math.log(p) for p in token_probs if p > 0)
                            mean_length = len(generation_output.token_ids)

                            sample_step = {
                                "finished_reason": generation_output.finish_reason,
                                "text": generation_output.text,
                                "token_ids": generation_output.token_ids,
                                "mean_length": mean_length,
                                "entropy_in_this_step": sum_entropy / mean_length if mean_length > 0 else 0,
                                "mean_entropy": 0,
                                "depth": parent_node.depth,
                                "gen_logps": token_log_probs,
                            }

                            # Accumulate from parent
                            sample_step["main_prompt"] = parent_node.item["main_prompt"]
                            sample_step["prompt_in_this_step"] = (
                                parent_node.item["prompt_in_this_step"] + sample_step["text"]
                            )
                            sample_step["current_answer"] = parent_node.item["current_answer"] + sample_step["text"]
                            sample_step["mean_length"] = parent_node.item["mean_length"] + sample_step["mean_length"]
                            sample_step["gen_logps"] = parent_node.item["gen_logps"] + sample_step["gen_logps"]
                            sample_step["mean_entropy"] = (
                                parent_node.item["mean_entropy"] + sample_step["entropy_in_this_step"]
                            )
                            sample_step["token_ids"] = parent_node.item["token_ids"] + sample_step["token_ids"]

                            child_node = TreeNode(sample_step, budget_coefficient, parent_node.depth)
                            child_node.parent = parent_node

                            # Check termination conditions
                            is_natural_stop = sample_step["finished_reason"] == "stop"
                            is_boxed_stop = extract_boxed_answer(sample_step["current_answer"]) is not None
                            is_max_depth = child_node.depth >= object_depth

                            if is_natural_stop or is_boxed_stop or is_max_depth:
                                child_node.item["reward"] = gen_reward(child_node.item["current_answer"], problem)
                                child_node.item["depth"] = child_node.depth
                                backpropagation(child_node)
                                finished_nodes.append(child_node)
                            else:
                                queue.append(child_node)
                                parent_node.add_child(child_node)

                        if parent_node.budget != 0:
                            queue.append(parent_node)

        # Compute advantages
        all_rewards = [node.item["reward"]["total"] for node in finished_nodes]
        std_dev = np.std(all_rewards) if all_rewards else 1.0
        if std_dev == 0:
            std_dev = 1.0

        trajectory_sub_advantages = {node: [] for node in finished_nodes}

        for trajectory_node in finished_nodes:
            ancestors = get_ancestors(trajectory_node)
            if ancestors:
                root_node = ancestors[-1]
                if root_node not in ancestors:
                    ancestors.append(root_node)

            for ancestor in ancestors:
                mean_reward = np.mean(ancestor.children_rewards)
                sub_advantage = trajectory_node.item["reward"]["total"] - mean_reward
                trajectory_sub_advantages[trajectory_node].append(sub_advantage)

        # Create finished samples
        finished_samples = []
        for trajectory_node in finished_nodes:
            sub_advantages = trajectory_sub_advantages[trajectory_node]

            if not sub_advantages:
                final_advantage = 0
            else:
                mean_of_sub_advantages = np.mean(sub_advantages)
                final_advantage = mean_of_sub_advantages / std_dev

            trajectory_node.item["advantage"] = final_advantage
            trajectory_node.item["mean_entropy"] /= trajectory_node.item["depth"]
            finished_samples.append(trajectory_node.item)

        # Store finished_nodes for tree structure access
        # This allows external code to access the tree structure
        self._last_finished_nodes = finished_nodes
        self._last_head_node = head_node if 'head_node' in locals() else None

        return finished_samples

    def format_samples_for_upload(self, samples: List[Dict[str, Any]], problem: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format samples for upload to orchestrator.

        Args:
            samples: List of sample dictionaries
            problem: Original problem dictionary (optional)

        Returns:
            Formatted dictionary for upload
        """
        from torch.nn.utils.rnn import pad_sequence

        if not samples:
            return {}

        # Get prompt from first sample or problem
        if problem is not None:
            prompt_item = problem
        else:
            prompt_item = samples[0].get("prompt", {})

        prompt_ids = self.ralo.tokenizer(
            self.ralo.rollout_prompt_fn(prompt_item), return_tensors="pt", add_special_tokens=False
        )["input_ids"]

        curr_ans_ids = [x["token_ids"] for x in samples]
        gen_logps = [x["gen_logps"] for x in samples]
        entropies = [x.get("mean_entropy", 0.0) for x in samples]
        lengths = [x.get("mean_length", 0) for x in samples]
        answers = [x.get("text", "") for x in samples]

        corrects_all = [1 if max(0, x.get("reward", {}).get("correct_fn", 0)) > 0 else 0 for x in samples]
        curr_rewards = torch.tensor([x.get("advantage", 0.0) for x in samples], dtype=torch.float32)

        # Create batch inputs
        plen = prompt_ids.shape[1]
        tensor_list = [torch.tensor(lst) for lst in curr_ans_ids]
        output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=self.ralo.tokenizer.pad_token_id)
        Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)

        data = {
            "prompt": prompt_item,
            "group_answers": answers,
            "plen": plen,
            "inputs": merged_ids,
            "gen_logps": gen_logps,
            "rewards": curr_rewards,
            "entropy": entropies,
            "sample_count": len(samples),
            "mean_length": lengths,
            "Accuracy": sum(corrects_all) / max(1, len(corrects_all)),
            "corrects": corrects_all,
        }

        return data

