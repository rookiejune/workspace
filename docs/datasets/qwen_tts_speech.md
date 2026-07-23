# Qwen TTS Speech

zhuyin.datasets.qwen_tts_speech 把任意 canonical text dataset 转成 Qwen
CustomVoice TTS speech store。这个入口只负责 workspace 级组装；通用 speaker 展开逻辑在
anydataset.dataset，Qwen 推理在 anydataset.provider.QwenTTSProvider 和
anytrain.tts.qwen。

## 设计

物理 store 使用 expanded speaker grid：

text_index=0, speaker=vivian
text_index=0, speaker=ryan
text_index=1, speaker=vivian
text_index=1, speaker=ryan

这样 materializer 的 resume、sharding、失败重跑和质量过滤仍然按单条 waveform 处理。
每条 expanded sample 的 text meta 保存：

- TextMeta.SOURCE_INDEX：原 text dataset 中的样本下标。

每条 expanded sample 的 text views 保存：

- TextView.SPEAKERS：当前 Qwen speaker id。它是 TTS 生成条件，不写入 text meta。

公开读取入口按 text 聚合：

    from zhuyin.datasets.qwen_tts_speech import qwen_tts_speaker_grid

    dataset = qwen_tts_speaker_grid(
        root="/data/qwen_tts_speaker_grid",
        speaker_ids=("vivian", "ryan"),
    )
    sample = dataset[0]

聚合后的 sample 中：

- text item 保留原文和 TextMeta.SOURCE_INDEX，不写 speaker id。
- audio item 的 AudioView.WAVEFORM 仍是 (waveform_tensor, sample_rate)。其中
  waveform_tensor 以 speaker 为第 0 维；组内不同长度会 pad 到最大长度。
- audio item 的 AudioView.SPEAKERS 是 speaker id tuple，定义 waveform 第 0 维顺序。
- audio item 的 AudioView.SPEAKER_LENGTHS 是每个 speaker waveform 的真实长度。
- audio item 的 AudioMeta.SPEAKER_ID 是同一个 speaker id tuple。

## 生成

生成 expanded store 使用：

    from zhuyin.datasets.qwen_tts_speech import materialize_qwen_tts_speaker_grid

    materialize_qwen_tts_speaker_grid(
        text_dataset_factory=text_dataset_factory,
        speaker_ids=("vivian", "ryan"),
        output_dir="/data/qwen_tts_speaker_grid",
        split="train",
    )

Fudan smoke 入口会生成 2 条文本 × 2 个 speaker，并把 grouped wav 导出到
`$DYNAMIC_HOME/debug/qwen_tts_speaker_grid_smoke/wavs`:

    jobs/fudan/prepare_qwen_tts_speaker_grid_smoke.sh

speaker_ids 的顺序是数据契约的一部分。读取 grouped view 时必须传入同一顺序；如果
store 中的 SOURCE_INDEX 或 speaker id 与预期不一致，loader 会明确报错，而不是静默
重排。

## 边界

- SpeakerCartesianDataset 是通用 anydataset wrapper，不依赖 Qwen。
- materialize_qwen_tts_speaker_grid() 写 expanded store，不返回 grouped sample。
- qwen_tts_speaker_grid() 是只读 grouped view，面向按 text 使用多个 speaker waveform
  的训练或检查代码。
- 每个 view 保持唯一值类型；grouped view 不会把 AudioView.WAVEFORM 改成 dict。
- 如果后续要按单 speaker 过滤质量，直接消费 expanded store；如果要按 text 聚合统计，
  再通过 grouped view 或显式聚合报告处理。
