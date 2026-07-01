"""三种 MLLM 任务类型对应的 prompt 模板构造函数。

对应 §4/§6:冻结的 MLLM 没有单独的输出头,任务完全靠 prompt 的措辞来
区分(instance / interaction / video 三选一)。这里的函数负责把
FactSelector 渲染出的 transcript 文本,包装成带有明确任务说明的完整
prompt 文本,再由 Pipeline 塞进 MLLMRequest 发给(真实或 mock 的)MLLM。
"""
from __future__ import annotations


def build_instance_prompt(track_id: int, transcript_text: str) -> str:
    """构造"描述单个目标行为"的 prompt。注意这里显式写出
    "track_id=<n>"这个 token——MockMLLMAdapter 会依赖这个固定格式,
    用正则从 prompt 文本里把 track_id 解析出来。
    """
    return (
        f"Describe the behavior of track_id={track_id} based on the following "
        f"motion facts: {transcript_text}"
    )


def build_interaction_prompt(subject_id: int, object_id: int, transcript_text: str) -> str:
    """构造"描述两个目标之间交互"的 prompt。同样显式写出
    "subject_id=<n>"和"object_id=<n>",供 MockMLLMAdapter 解析。
    """
    return (
        f"Describe the interaction between subject_id={subject_id} and "
        f"object_id={object_id} based on the following motion facts: {transcript_text}"
    )


def build_video_prompt(involved_ids: tuple[int, ...], transcript_text: str) -> str:
    """构造"概括整段视频"的 prompt,把所有涉及到的 track_id 列在
    方括号里。
    """
    ids_text = ", ".join(str(i) for i in involved_ids)
    return (
        f"Summarize the video involving track ids [{ids_text}] based on the "
        f"following motion facts: {transcript_text}"
    )
