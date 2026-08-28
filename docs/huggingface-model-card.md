---
language:
  - zh
  - en
  - ja
  - es
  - ar
license: other
license_name: bilibili-model-license-and-third-party-licenses
license_link: LICENSE
library_name: indextts
pipeline_tag: text-to-speech
tags:
  - indextts
  - indextts-2.5
  - text-to-speech
  - voice-cloning
  - multilingual
  - emotion-controllable
---

# IndexTTS 2.5 · T8star-Aix 完整运行模型

这是供以下两个项目共用的 **IndexTTS 2.5 完整推理模型包**：

- [T8star-Aix IndexTTS 2.5 Desktop](https://github.com/T8mars/indextts25-desktop-t8)
- [T8star-Aix ComfyUI 节点](https://github.com/T8mars/comfyui-indextts25-t8)

桌面版和 ComfyUI 使用的是同一套模型文件，无需分别下载。桌面启动器可以直接选择 ComfyUI 的
`models/TTS/IndexTTS-2.5` 目录，共享一份约 7.72 GiB 的模型。

本仓库解决官方 `IndexTeam/IndexTTS-2.5` 模型仓库未包含 `bpe.model`，以及首次运行还会从多个
仓库下载 Wav2Vec2-BERT、CAMPPlus 和 BigVGAN 的问题。这里把实际推理需要的 26 个文件集中在一个
目录中，方便自动下载、断点续传、离线运行和 SHA-256 校验。

> 本仓库不是 IndexTTS 官方发布渠道。所有权重保持源文件内容不变，仅重新整理目录。
> 原始权利人不对此第三方整理仓库及其衍生集成提供认可、担保或保证，并对其不承担责任。

## 自动下载与模型版本

仓库根目录的 `model-bundle.json` 是机器可读的完整文件清单，`model-bundle.sig` 是对应的
Ed25519 签名。T8star-Aix 桌面启动器会先验证签名，再信任清单中的仓库版本、文件路径、大小和
SHA-256；只下载缺失或损坏的文件，网络中断后可以继续下载。

模型更新由 `bundleVersion` 判断，不使用仓库 HEAD 判断。因此仅修改 README 不会误报模型更新。
程序更新仍通过 GitHub Release 独立发布，模型不会被打进程序增量包。

## ComfyUI 模型放置路径

节点代码和模型是两个不同目录：

```text
ComfyUI/
├─ custom_nodes/
│  └─ comfyui-indextts25-T8/       ← GitHub 节点代码
└─ models/
   └─ TTS/
      └─ IndexTTS-2.5/             ← 本 HF 仓库的模型文件
         ├─ config.yaml
         ├─ bpe.model
         ├─ gpt.pth
         ├─ codec.pth
         ├─ s2mel.pth
         ├─ feat1.pt
         ├─ feat2.pt
         ├─ wav2vec2bert_stats.pt
         ├─ multilingual_zh_ja_yue_char_del.tiktoken
         ├─ qwen0.6bemo4-merge/
         └─ hf_cache/
            ├─ w2v-bert-2.0/
            ├─ campplus_cn_common.bin
            └─ bigvgan/
```

Windows 完整路径示例：

```text
D:\ComfyUI\models\TTS\IndexTTS-2.5\config.yaml
D:\ComfyUI\models\TTS\IndexTTS-2.5\bpe.model
D:\ComfyUI\models\TTS\IndexTTS-2.5\hf_cache\bigvgan\bigvgan_generator.pt
```

请确认 `config.yaml` 和 `bpe.model` 是 `IndexTTS-2.5` 的直接子文件。不要多套一层：

```text
ComfyUI/models/TTS/IndexTTS-2.5/IndexTTS-2.5-Comfy/config.yaml  ← 错误
```

## 桌面版模型放置路径

桌面启动器支持两种方式：

1. 点击“**Hugging Face 自动下载／修复完整模型**”，选择父目录，启动器会创建 `IndexTTS-2.5`；
2. 选择已有完整目录，包括 ComfyUI 的 `ComfyUI/models/TTS/IndexTTS-2.5`。

模型、保存音色、生成音频和用户设置都独立于桌面程序更新，更新应用不会删除这些内容。

## 命令行下载

```powershell
pip install -U huggingface_hub
hf download t8star/IndexTTS-2.5-Comfy --local-dir "ComfyUI/models/TTS/IndexTTS-2.5"
```

也可以在仓库页面选择 **Files and versions → Download**。解压后确保 `config.yaml`、`bpe.model`、
`gpt.pth` 和 `hf_cache` 位于同一个 `IndexTTS-2.5` 目录。

## 已包含内容

- IndexTTS 2.5 主模型、语速适配模型、Qwen 文本情感模型与多语言分词文件
- 官方 2.5 仓库缺失、但推理必需的 `bpe.model`
- `hf_cache/w2v-bert-2.0`：Transformers 权重、配置和预处理配置
- `hf_cache/campplus_cn_common.bin`：音色特征模型
- `hf_cache/bigvgan`：22.05 kHz BigVGAN 配置和生成器权重

没有收录训练优化器、判别器、重复的 Fairseq `conformer_shaw.pt` 或当前 IndexTTS 2.5 推理代码
不会读取的 MaskGCT 独立权重。

## 来源与固定版本

| 内容 | 来源 | 固定版本 |
|---|---|---|
| IndexTTS 2.5 主模型 | [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) | `c39ce5ba981572cb187443877ff559dfb246ce63` |
| `bpe.model` | [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | `740dcaff396282ffb241903d150ac011cd4b1ede` |
| Wav2Vec2-BERT 2.0 | [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) | `da985ba0987f70aaeb84a80f2851cfac8c697a7b` |
| CAMPPlus | [funasr/campplus](https://huggingface.co/funasr/campplus) | `e4b6ede7ce16997aff4ae69fbca1f0175e2afede` |
| BigVGAN | [nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x) | `633ff708ed5b74903e86ff1298cf4a98e921c513` |

详细归属和许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。下载或使用即表示你已阅读并
接受本仓库 `LICENSE` 中的 bilibili Model Use License Agreement 及各第三方模型许可证。

## English

This is the shared, inference-only IndexTTS 2.5 model bundle for the T8star-Aix Desktop package and
ComfyUI node. Both applications use the same byte-identical files; the Desktop launcher can directly
select `ComfyUI/models/TTS/IndexTTS-2.5`, so a second copy is unnecessary.

The repository combines the pinned official IndexTTS 2.5 files, the required `bpe.model`, and the exact
Wav2Vec2-BERT, CAMPPlus, and BigVGAN runtime files. `model-bundle.json` and `model-bundle.sig` provide a
signed, machine-readable version and per-file SHA-256 list for resumable automatic download and repair.

This is an unofficial packaging mirror and is not endorsed, warranted, or guaranteed by the original
right-holders. Review `LICENSE` and `THIRD_PARTY_NOTICES.md` before downloading or using the files.
