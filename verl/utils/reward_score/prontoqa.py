import re
import math
from typing import Any, Tuple, Dict


def parse_generation(prediction: str) -> str:
    """parse the generated texts to extract the final answer.

    Args:
        prediction (str): generated text

    Returns:
        List[str]: parsed texts
    """
    return prediction.rsplit("<answer>", 1)[-1].rsplit("</answer>", 1)[0].strip()


def format_reward(predict_str: str) -> float:
    try:
        think_begin_token_idx = predict_str.index("<think>")
    except ValueError:
        think_begin_token_idx = -1
    try:
        think_end_token_idx = predict_str.index("</think>")
    except ValueError:
        think_end_token_idx = -1
    try:
        answer_begin_token_idx = predict_str.index("<answer>")
    except ValueError:
        answer_begin_token_idx = -1
    try:
        answer_end_token_idx = predict_str.index("</answer>")
    except ValueError:
        answer_end_token_idx = -1
    """
    if "<think>" in predict_str and "</think>" in predict_str:
        return 0.0
    elif "<think>" in predict_str or "</think>" in predict_str:
        return -0.5
    elif "<answer>" in predict_str and "</answer>" in predict_str:
        return 0.0
    elif "<answer>" in predict_str or "</answer>" in predict_str:
        return -0.5
    else:
        return -1.0
    """
    if (
        think_begin_token_idx < think_end_token_idx
        and answer_begin_token_idx < answer_end_token_idx
        and think_end_token_idx < answer_begin_token_idx
    ):
        return 0.0
    else:
        return -1.0


def grade_language_repetition(
    given_answer: str,
    language: str = "zh",
    ngram: int = 2,
    tau: float = 1.0,
    steepness: float = 4.0,
) -> float:
    """
    Calculate a smoothed diversity reward based on distinct-n score for the given text,
    with temperature scaling to control the influence of the reward.

    Args:
        given_answer (str): The text to evaluate
        language (str): Language code, default "zh" for Chinese
        ngram (int): Size of n-grams to use, default 2
        tau (float): Temperature parameter in range [0, 1] to control reward scaling, default 1.0
                    - tau = 0: No diversity reward (always returns 0)
                    - tau = 1: Full diversity reward (returns value in [-1, 0])
                    - 0 < tau < 1: Scaled diversity reward

    Returns:
        float: A scaled reward value between -1 and 0, where values closer to 0 indicate higher diversity
    """
    # Ensure tau is in valid range
    tau = max(0.0, min(1.0, tau))

    # If tau is 0, diversity doesn't matter, return 0 reward
    if tau == 0:
        return 0.0

    # Check if input is empty
    if not given_answer or len(given_answer.strip()) == 0:
        return -1.0 * tau  # Minimum reward for empty text, scaled by tau

    # Chinese tokenization
    if language == "zh":
        try:
            import jieba

            tokens = list(jieba.cut(given_answer))
        except ImportError:
            # Fallback: simple character-based tokenization for Chinese
            tokens = list(given_answer)
    else:
        # For other languages, split by whitespace (simple approach)
        tokens = given_answer.split()

    # Generate n-grams
    ngrams = []
    for i in range(len(tokens) - ngram + 1):
        ngrams.append(tuple(tokens[i : i + ngram]))

    # Calculate distinct-n score
    if not ngrams:
        return -1.0 * tau  # Minimum reward if no n-grams could be formed, scaled by tau

    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))

    # Distinct-n score: ratio of unique n-grams to total n-grams
    distinct_n = unique_ngrams / total_ngrams if total_ngrams > 0 else 0

    # Smoothing function to map distinct-n (range 0-1) to reward (range -1 to 0)
    # Using a sigmoid-like function that gives more reward as diversity increases
    # and approaches 0 (max reward) as distinct_n approaches 1

    # Parameters to tune the smoothing function
    steepness = steepness  # Controls how steep the reward curve is
    midpoint = 0.5  # The distinct-n value that gives a reward of -0.5

    # Sigmoid-like function mapped to [-1, 0]
    raw_reward = -1 + 1 / (1 + math.exp(-(math.e**steepness) * (distinct_n - midpoint)))

    # Apply temperature scaling - scales the reward by tau
    scaled_reward = raw_reward * tau

    # Ensure the reward stays within [-1, 0]
    scaled_reward = max(-1, min(0, scaled_reward))

    return scaled_reward


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = parse_generation(predict_str)
    ground_truth = parse_generation(ground_truth)
    return 1.0 if answer.lower() == ground_truth.lower() else 0.0


def compute_score(
    prompt: str, predict_str: str, ground_truth: str
) -> Tuple[float, Dict[str, Any]]:
    acc_reward_score = acc_reward(predict_str, ground_truth)
    format_reward_score = format_reward(predict_str)
    language_repetition_score = grade_language_repetition(
        predict_str, language="en", ngram=1, tau=1.0, steepness=4.0
    )
    eval_result = {
        "input": prompt,
        "output": predict_str,
        "reference": ground_truth,
        "predicted_answer": parse_generation(predict_str),
        "reference_answer": parse_generation(ground_truth),
        # "meteor": meteor_score,
        "format_rewards": format_reward_score,
        "length_rewards": 0,
        "unk_error_rewards": 0,
        "repetition_rewards": language_repetition_score,
        "language_monotony_rewards": 0,
        "correctness_rewards": acc_reward_score,
        "soft_exact_match": acc_reward_score,
        "hard_exact_match": acc_reward_score,
    }

    rewards = format_reward_score + language_repetition_score + acc_reward_score

    return rewards, eval_result
