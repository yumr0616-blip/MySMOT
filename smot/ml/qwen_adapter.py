"""冻结 Qwen3.5 的 MLLMAdapter 实现 + soft token 注入机制。

对应 §4 的冻结 MLLM:模型主体与视觉塔全部 requires_grad=False + eval(),
没有独立输出头,三种任务(instance/interaction/video)只靠 prompt 措辞
区分——QwenMLLMAdapter 实现的就是 smot.mllm.MLLMAdapter 这个 Protocol,
对 Pipeline 完全透明地替换 MockMLLMAdapter。

soft token 注入方式(训练与推理共用,这是本模块最重要的设计决策):
不走 inputs_embeds 手动拼接——Qwen3-VL 系架构的视觉特征融合是模型
forward 内部按 input_ids 定位占位符完成的(还包括 DeepStack 式的多层
注入),绕过它等于静默丢掉部分视觉通路。改为:

  1. 在 tokenized prompt 末尾追加 m 个占位 token(append_placeholder_tokens);
  2. 在 embedding 层挂一个 forward hook(soft_token_injection),把这
     m 个位置的嵌入替换成 projector 输出的 soft 向量(替换发生在
     autograd 图内,训练时梯度可以经 soft 向量回传给 projector/KFA);
  3. input_ids 照常进 model.generate()/forward(),模型内部的全部多模态
     逻辑原样生效。
生成的增量步(kv-cache 每步只喂 1 个新 token)序列长度不会覆盖注入
区间,hook 自动跳过,因此对 generate 也安全。
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch

from smot.ml.frames import FrameProvider, annotate_boxes, color_for_track, provider_for
from smot.mllm import MLLMRequest

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"

# 三种任务附加在 transcript 之后的输出指令。interaction 要求结构化 JSON
# (OutputAssembler 的结构化解析路径消费它;模型不守指令时 assembler 会
# 自动退回自由文本谓词抽取,所以这里不需要重试逻辑)。
_TASK_INSTRUCTIONS = {
    "instance": (
        "Answer with exactly one concise English sentence describing the "
        "highlighted target's visible behavior."
    ),
    "interaction": (
        "Decide who acts on whom. Respond with ONLY a JSON object and no "
        'other text: {"subject_id": <int, the actor>, "object_id": <int, '
        'the target>, "predicate": "<short verb phrase>", "sentence": '
        '"<one English sentence>"}. Use the integer ids shown on the box '
        "labels."
    ),
    "video": (
        "Answer with one or two concise English sentences summarizing what "
        "happens in the video."
    ),
}


def load_frozen_qwen(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    quantize_4bit: bool = False,
):
    """加载并冻结 Qwen3.5(bf16;可选 bitsandbytes 4-bit)。

    推理适配器、梯度门禁、训练循环共用这一份加载逻辑,保证"冻结边界"
    在所有入口一致:全部参数 requires_grad=False + eval()。可训练的只有
    外部的 KFA/projector 模块,它们不在这里。
    """
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    kwargs: dict = {"dtype": torch.bfloat16, "device_map": device}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def append_placeholder_tokens(inputs, m: int, pad_id: int) -> int:
    """在 tokenized prompt 末尾追加 m 个占位 token,返回注入起始位置。

    占位 token 的具体 id 无所谓(其嵌入会被 hook 整体替换),这里用
    调用方传入的 pad/eos id。所有与序列长度对齐的字段(attention_mask、
    mm_token_type_ids)同步延长,否则模型侧的形状校验会炸。
    """
    ids = inputs["input_ids"]
    start = ids.shape[1]
    filler = torch.full((ids.shape[0], m), pad_id, dtype=ids.dtype, device=ids.device)
    inputs["input_ids"] = torch.cat([ids, filler], dim=1)
    if inputs.get("attention_mask") is not None:
        mask = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat(
            [mask, torch.ones_like(filler)], dim=1
        )
    if inputs.get("mm_token_type_ids") is not None:
        mm = inputs["mm_token_type_ids"]
        inputs["mm_token_type_ids"] = torch.cat(
            [mm, torch.zeros_like(filler)], dim=1
        )
    return start


@contextlib.contextmanager
def soft_token_injection(model, soft: Optional[torch.Tensor], start_pos: int):
    """在作用域内把 embedding 层输出的 [start_pos, start_pos+m) 位置替换
    成 soft 向量(m = soft.shape[0])。soft 为 None 或空时不做任何事。

    替换用 clone + 切片赋值完成,留在 autograd 图内:训练时 loss 对这
    几个位置的梯度会流向 soft 向量(进而回到 projector / KFA),而冻结
    的 embedding 权重本身不会收到梯度。序列长度不覆盖注入区间的前向
    (generate 的增量步)自动跳过。
    """
    if soft is None or soft.shape[0] == 0:
        yield
        return
    end_pos = start_pos + soft.shape[0]

    def hook(_module, _args, output):
        if output.shape[1] >= end_pos:
            patched = output.clone()
            patched[:, start_pos:end_pos] = soft.to(
                dtype=output.dtype, device=output.device
            )
            return patched
        return output

    handle = model.get_input_embeddings().register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class QwenMLLMAdapter:
    """实现 smot.mllm.MLLMAdapter Protocol 的真实(冻结)多模态适配器。

    model/processor 可以直接注入(与训练循环共享同一份权重,避免 8GB
    显存里放两份模型),不注入时按 model_id 自行加载。
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        max_new_tokens: int = 96,
        quantize_4bit: bool = False,
        model=None,
        processor=None,
    ):
        if model is None or processor is None:
            model, processor = load_frozen_qwen(
                model_id, device=device, quantize_4bit=quantize_4bit
            )
        self._model = model
        self._processor = processor
        self._max_new_tokens = max_new_tokens
        # 帧提供者按 video_path 缓存(逐视频顺序处理,只留最近一个,
        # 避免长跑批时句柄/缓存无限增长)。
        self._provider_path: Optional[str] = None
        self._provider: Optional[FrameProvider] = None

    # ------------------------------------------------------------------
    # MLLMAdapter Protocol
    # ------------------------------------------------------------------

    def generate(self, request: MLLMRequest) -> str:
        images = self._render_frames(request)
        text = self._compose_text(request, has_images=bool(images))
        content: list[dict] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        soft = None
        start = 0
        if request.soft_tokens:
            soft = torch.tensor(
                request.soft_tokens, dtype=torch.float32, device=self._model.device
            )
            d_llm = self._model.get_input_embeddings().embedding_dim
            if soft.shape[-1] != d_llm:
                raise ValueError(
                    f"soft token 维度 {soft.shape[-1]} 与模型嵌入维度 {d_llm} 不一致"
                )
            start = append_placeholder_tokens(
                inputs, soft.shape[0], self._processor.tokenizer.eos_token_id
            )

        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode(), soft_token_injection(self._model, soft, start):
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
            )
        reply = self._processor.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )[0]
        return reply.strip()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _render_frames(self, request: MLLMRequest) -> list:
        """按 frame_refs 取帧并画框。没有 video_path(合成 fixture)或
        没有关键帧(video 任务)时返回空列表,退化为纯文本请求。"""
        if not request.video_path or not request.frame_refs:
            return []
        if self._provider_path != request.video_path:
            self._provider = provider_for(request.video_path)
            self._provider_path = request.video_path
        boxes_by_t = {t: dict(entries) for t, entries in request.frame_boxes}
        images = []
        for t in request.frame_refs:
            image = self._provider.frame(t)
            boxes = boxes_by_t.get(t)
            if boxes:
                image = annotate_boxes(image, boxes)
            images.append(image)
        return images

    def _compose_text(self, request: MLLMRequest, has_images: bool) -> str:
        """transcript + 颜色图例(与画框颜色同源)+ 任务输出指令。"""
        parts = [request.transcript_text]
        if has_images and request.frame_boxes:
            track_ids = sorted(
                {tid for _t, entries in request.frame_boxes for tid, _box in entries}
            )
            if track_ids:
                legend = ", ".join(
                    f"id={tid} is the {color_for_track(tid)} box" for tid in track_ids
                )
                parts.append(f"Box color legend: {legend}.")
        instruction = _TASK_INSTRUCTIONS.get(request.prompt_type)
        if instruction is None:
            raise ValueError(f"unknown prompt_type: {request.prompt_type!r}")
        parts.append(instruction)
        return "\n".join(parts)
