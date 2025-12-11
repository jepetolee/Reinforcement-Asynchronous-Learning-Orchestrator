"""
DAPO Sampler Algorithm: Decoupled Clip and Dynamic sAmpling Policy Optimization.
"""

from typing import Any, Dict, List
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ..registry import SamplerAlgorithm, register_sampler_algorithm
from .sample_schema import DAPOSampleSchema


@register_sampler_algorithm("dapo")
class DAPOSamplerAlgorithm(SamplerAlgorithm):
    """DAPO sampler algorithm implementation."""

    def __init__(self, ralo, config):
        super().__init__(ralo, config)
        self.schema = DAPOSampleSchema()

    def create_sample_schema(self) -> DAPOSampleSchema:
        """Create DAPO sample schema."""
        return self.schema

    def run(self) -> None:
        """Run the sampler."""
        # Reuse the standard RLVR sampling loop implemented on RALO
        self.ralo._rlvr_run_sampler()

    def process_problem(self, problem: Dict[str, Any], services: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a problem using DAPO and return samples.
        Implements Dynamic Sampling (generating a group of samples) and GRPO advantage calculation.

        Args:
            problem: Problem dictionary
            services: Dictionary with 'model', 'sampling', 'orchestrator' services

        Returns:
            List of sample dictionaries
        """
        sampling_service = services["sampling"]
        ralo = self.ralo

        # Get DAPO parameters
        dapo_config = getattr(ralo, "DAPO_kwargs", {})
        rollout_num = ralo.rollout_num
        # Default to global gen_max_tokens when DAPO generation_length is missing
        generation_length = dapo_config.get(
            "generation_length",
            getattr(ralo, "gen_max_tokens", 512),
        )
        # Safety: clamp invalid values
        if not generation_length or generation_length <= 0:
            generation_length = getattr(ralo, "gen_max_tokens", 512)
        # Debug trace to confirm effective generation length in logs
        try:
            print(f"[DAPO] generation_length={generation_length} rollout_num={rollout_num}")
        except Exception:
            pass
        temperature = ralo.gen_temperature
        top_p = getattr(ralo, 'gen_top_p', None)
        top_k = getattr(ralo, 'gen_top_k', None)
        min_p = getattr(ralo, 'gen_min_p', None)
        
        # Overlong Reward Shaping parameters
        length_penalty_coef = dapo_config.get("length_penalty_coef", 0.0)

        # Prepare prompt
        prompt_text = ralo.rollout_prompt_fn(problem)
        
        # 1. Dynamic Sampling: Generate multiple samples (group)
        # We generate rollout_num samples for the single prompt
        sampling_params = sampling_service.create_sampling_params(
            n=rollout_num,
            temperature=temperature,
            max_tokens=int(generation_length),
            top_p=top_p if top_p is not None else 1.0,
            top_k=top_k,
            min_p=min_p,
            logprobs=1, # We need logprobs for training
            include_stop_str_in_output=True,
        )
        try:
            print(f"[DAPO] sampling_params.max_tokens={getattr(sampling_params, 'max_tokens', None)}")
        except Exception:
            pass

        # Generate
        # Note: sampling_service.generate expects a list of prompts
        vllm_outputs = sampling_service.generate([prompt_text], sampling_params, use_tqdm=False)
        
        if not vllm_outputs:
            return []

        v_output = vllm_outputs[0] # Single prompt
        
        samples = []
        rewards_list = []
        lengths_list = []

        # 2. Process each generated output
        for generation_output in v_output.outputs:
            text = generation_output.text
            token_ids = generation_output.token_ids
            
            # Extract logprobs
            # generation_output.logprobs is a list of dicts {token_id: Logprob}
            # We need the logprob of the generated token
            gen_logps = []
            if generation_output.logprobs:
                for t_id, step_logprobs in zip(token_ids, generation_output.logprobs):
                    if step_logprobs and t_id in step_logprobs:
                        gen_logps.append(step_logprobs[t_id].logprob)
                    else:
                        # Fallback if logprob missing (shouldn't happen with logprobs=1)
                        gen_logps.append(-100.0) 
            else:
                # Mock if no logprobs returned (shouldn't happen)
                gen_logps = [-1.0] * len(token_ids)

            # Compute Reward
            # Combine text with prompt for reward function if needed, or just pass text/problem
            # ralo.reward_fns usually takes (answer, item)
            current_answer = text
            
            reward_dict = {}
            total_reward = 0.0
            for reward_fn in ralo.reward_fns:
                r_val = reward_fn(current_answer, problem)
                reward_dict[reward_fn.__name__] = r_val
                total_reward += r_val
            
            # 3. Overlong Reward Shaping (Hide and Seek)
            # Penalize length
            length = len(token_ids)
            shaped_reward = total_reward - (length_penalty_coef * length)
            
            reward_dict["total"] = shaped_reward
            reward_dict["raw_total"] = total_reward # Keep raw for logging

            samples.append({
                "prompt": problem, # Store original problem
                "text": text,
                "current_answer": current_answer,
                "token_ids": token_ids,
                "gen_logps": gen_logps,
                "rewards": reward_dict,
                "mean_length": length,
                "entropy": 0.0, # Placeholder, can compute if needed
                "finished_reason": generation_output.finish_reason
            })
            rewards_list.append(shaped_reward)
            lengths_list.append(length)

        # 4. Compute Advantages (GRPO)
        # Normalize rewards within the group
        rewards_arr = np.array(rewards_list)
        std_dev = np.std(rewards_arr)
        if std_dev < 1e-6:
            std_dev = 1.0 # Avoid division by zero
        mean_reward = np.mean(rewards_arr)
        
        advantages = (rewards_arr - mean_reward) / std_dev

        # Assign advantages to samples
        for i, sample in enumerate(samples):
            sample["advantage"] = float(advantages[i])

        return samples

    def format_samples_for_upload(self, samples: List[Dict[str, Any]], problem: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format samples for upload to orchestrator.
        
        Args:
            samples: List of sample dictionaries
            problem: Original problem dictionary (optional)

        Returns:
            Formatted dictionary for upload
        """
        if not samples:
            return {}

        # Get prompt from first sample or problem
        if problem is not None:
            prompt_item = problem
        else:
            prompt_item = samples[0].get("prompt", {})

        # Tokenize prompt
        prompt_text = self.ralo.rollout_prompt_fn(prompt_item)
        prompt_ids = self.ralo.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]

        curr_ans_ids = [x["token_ids"] for x in samples]
        gen_logps = [x["gen_logps"] for x in samples]
        entropies = [x.get("entropy", 0.0) for x in samples]
        lengths = [x.get("mean_length", 0) for x in samples]
        answers = [x.get("text", "") for x in samples]
        rewards = [x.get("rewards", {}).get("total", 0.0) for x in samples]
        advantages = [x.get("advantage", 0.0) for x in samples]

        # Check for correctness (assuming there's a correct_fn or similar in rewards)
        # This is heuristics based on ralo_cli conventions
        corrects_all = []
        for x in samples:
            r = x.get("rewards", {})
            # Try to find a correctness metric
            is_correct = 0
            if "correct_fn" in r:
                is_correct = 1 if r["correct_fn"] > 0 else 0
            elif "accuracy" in r:
                is_correct = 1 if r["accuracy"] > 0 else 0
            corrects_all.append(is_correct)

        # Create batch inputs
        plen = prompt_ids.shape[1]
        tensor_list = [torch.tensor(lst, dtype=torch.long) for lst in curr_ans_ids]
        # Pad sequences
        output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=self.ralo.tokenizer.pad_token_id)
        
        # Replicate prompt for batch
        Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
        
        # Merge prompt + output
        merged_ids = torch.cat([Qrep, output_ids], dim=1)

        data = {
            "prompt": prompt_item,
            "group_answers": answers,
            "plen": plen,
            "inputs": merged_ids,
            "gen_logps": gen_logps,
            "rewards": torch.tensor(advantages, dtype=torch.float32), # Trainer expects advantages in 'rewards' field for some algos, or we can use 'advantages' if trainer supports it.
            # Looking at TreePO trainer: "advantages = batch["rewards"].to(device).unsqueeze(1)"
            # So we store advantages in 'rewards' field for compatibility, or change trainer to look for advantages.
            # TreePO sampler puts 'advantage' into 'rewards' tensor?
            # TreePO sampler: curr_rewards = torch.tensor([x.get("advantage", 0.0) ...]) -> data["rewards"] = curr_rewards
            # So yes, "rewards" key in data dict actually holds the advantages.
            
            "raw_rewards": rewards, # Keep raw rewards just in case
            "entropy": entropies,
            "sample_count": len(samples),
            "mean_length": lengths,
            "Accuracy": sum(corrects_all) / max(1, len(corrects_all)),
            "corrects": corrects_all,
        }

        return data

