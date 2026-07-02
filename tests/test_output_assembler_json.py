"""OutputAssembler 结构化 JSON 解析路径的测试。

真实 MLLM 被要求以 JSON 回答交互任务;这里锁住:JSON 优先、方向对账
与正则路径同规则、解析失败时无损退回原有自由文本路径(旧行为不回归)。
"""
from __future__ import annotations

import unittest

from smot.output_assembler import UNVERIFIED_DIRECTION_CONFIDENCE, OutputAssembler


def _assemble(text: str, subject_id: int = 1, object_id: int = 2):
    return OutputAssembler().assemble_interaction(
        subject_id=subject_id,
        object_id=object_id,
        mllm_text=text,
        time_span=(0, 4),
        evidence_frames=(0, 2, 4),
    )


class StructuredJsonTest(unittest.TestCase):
    def test_plain_json(self):
        assertion = _assemble(
            '{"subject_id": 1, "object_id": 2, "predicate": "pushes", '
            '"sentence": "Person 1 pushes person 2."}'
        )
        self.assertEqual(assertion.predicate, "pushes")
        self.assertEqual(assertion.canonical_label, "pushes")  # 不在映射表,identity
        self.assertEqual((assertion.subject_id, assertion.object_id), (1, 2))
        self.assertEqual(assertion.confidence, 1.0)

    def test_fenced_json_with_prose(self):
        """模型常在 JSON 外包围栏/解释文字,解析必须能穿透。"""
        text = (
            "Sure! Here is the analysis:\n```json\n"
            '{"subject_id": 1, "object_id": 2, "predicate": "follows"}\n'
            "```\nHope this helps."
        )
        assertion = _assemble(text)
        self.assertEqual(assertion.predicate, "follows")
        self.assertEqual(assertion.canonical_label, "follow")  # 经映射表规范化
        self.assertEqual(assertion.confidence, 1.0)

    def test_reversed_direction_swaps(self):
        """模型声明的 subject/object 与候选边相反 -> 以模型为准交换。"""
        assertion = _assemble(
            '{"subject_id": 2, "object_id": 1, "predicate": "pushes"}'
        )
        self.assertEqual((assertion.subject_id, assertion.object_id), (2, 1))
        self.assertEqual(assertion.confidence, 1.0)

    def test_string_ids_accepted(self):
        assertion = _assemble(
            '{"subject_id": "1", "object_id": "2", "predicate": "greets"}'
        )
        self.assertEqual((assertion.subject_id, assertion.object_id), (1, 2))
        self.assertEqual(assertion.confidence, 1.0)

    def test_mismatched_ids_dampen_confidence(self):
        assertion = _assemble(
            '{"subject_id": 3, "object_id": 4, "predicate": "pushes"}'
        )
        self.assertEqual((assertion.subject_id, assertion.object_id), (1, 2))
        self.assertEqual(assertion.confidence, UNVERIFIED_DIRECTION_CONFIDENCE)

    def test_missing_ids_dampen_confidence(self):
        assertion = _assemble('{"predicate": "pushes"}')
        self.assertEqual(assertion.predicate, "pushes")
        self.assertEqual(assertion.confidence, UNVERIFIED_DIRECTION_CONFIDENCE)

    def test_invalid_json_falls_back_to_text_path(self):
        """老的自由文本路径必须原样保留(Mock 与不守指令的模型走它)。"""
        assertion = _assemble("subject_id=1 approaches object_id=2.")
        self.assertEqual(assertion.predicate, "approaches")
        self.assertEqual(assertion.canonical_label, "approach")
        self.assertEqual(assertion.confidence, 1.0)

    def test_json_without_predicate_falls_back(self):
        """有 JSON 但缺 predicate 键 -> 当作没有结构化输出。"""
        assertion = _assemble(
            'Result: {"note": "unsure"} but subject_id=1 approaches object_id=2.'
        )
        self.assertEqual(assertion.canonical_label, "approach")
        self.assertEqual(assertion.confidence, 1.0)

    def test_braces_in_prose_do_not_break_parsing(self):
        assertion = _assemble("The pair {1,2} approaches steadily.")
        self.assertEqual(assertion.canonical_label, "approach")


if __name__ == "__main__":
    unittest.main()
