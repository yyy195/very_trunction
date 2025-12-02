

from mathruler.grader import extract_boxed_content, grade_answer
import re

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score_majority(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )

"""
{"input": "system\nYou are a helpful assistant.\nuser\nFind x. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.\nassistant\n", "output": "<think>So, we are given a right triangle with a 63 section between the two segments of the circle. We are also given that the hypotenuse is 12. We want to find the length of the unknown x.  We know that in a right triangle, we can use the Pythagorean theorem, which states that the square of one side is equal to the sum of the squares of the other two sides. So, we can set up an equation: 6^2 + 3^2 = x^2. We calculate: 36 + 9 = x^2. x^2 = 45. We take the square root of both sides to find x: x = 6. </think> \\boxed{x = 6}", "gts": "6", "score": 0.8999999761581421, "step": 3}
"""
gt = '7'
response = "<think>So, we are given a right triangle with a 63 section between the two segments of the circle. We are also given that the hypotenuse is 12. We want to find the length of the unknown x.  We know that in a right triangle, we can use the Pythagorean theorem, which states that the square of one side is equal to the sum of the squares of the other two sides. So, we can set up an equation: 6^2 + 3^2 = x^2. We calculate: 36 + 9 = x^2. x^2 = 45. We take the square root of both sides to find x: x = 6. </think> \\boxed{x = 6}"
box_content = extract_boxed_content(response)
format_match = format_reward(response)
print("Box:{box},Format:{format}".format(box=box_content,format=format_match))
score = compute_score_majority(response,gt)
print("\n answer:{}".format(score))