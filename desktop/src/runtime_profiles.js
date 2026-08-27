"use strict";

const RUNTIME_PROFILES = Object.freeze({
  low_vram: Object.freeze({
    label: "6–8GB 省显存",
    accelerationMode: "off",
    precisionMode: "float16",
    referenceDevice: "cpu",
    reuseDefaultEmotion: true,
    description: "FP16 + 参考编码器放 CPU；优先降低常驻显存，保持普通推理路径。"
  }),
  balanced: Object.freeze({
    label: "10–16GB 均衡",
    accelerationMode: "auto_safe",
    precisionMode: "auto",
    referenceDevice: "auto",
    reuseDefaultEmotion: true,
    description: "自动精度与参考设备，只启用环境已经支持的安全融合核。"
  }),
  max_speed: Object.freeze({
    label: "16GB+ 速度优先",
    accelerationMode: "gpt_accel",
    precisionMode: "auto",
    referenceDevice: "same",
    reuseDefaultEmotion: true,
    description: "参考编码器同显卡并启用 GPT 加速；不兼容采样会自动回退普通路径。"
  }),
  compatibility: Object.freeze({
    label: "稳定兼容",
    accelerationMode: "off",
    precisionMode: "auto",
    referenceDevice: "auto",
    reuseDefaultEmotion: false,
    description: "关闭可选加速并保留独立情感编码，适合排查兼容性或比较音质。"
  })
});

function recommendRuntimeProfile(hardware = {}) {
  if (!hardware.cudaAvailable) return "compatibility";
  const vramGb = Number(hardware.vramGb || 0);
  if (vramGb > 0 && vramGb < 10) return "low_vram";
  if (vramGb >= 16) return "max_speed";
  return "balanced";
}

function resolveRuntimeProfile(profileName, recommendedProfile = "compatibility") {
  const resolvedName = profileName === "recommended" ? recommendedProfile : profileName;
  const profile = RUNTIME_PROFILES[resolvedName];
  if (!profile) throw new Error("Unknown runtime profile.");
  return { name: resolvedName, ...profile };
}

function hardwareSummary(hardware = {}) {
  if (!hardware.cudaAvailable) {
    return "未检测到可用的 NVIDIA CUDA 显卡；建议使用稳定兼容方案（CPU 推理会较慢）。";
  }
  const name = String(hardware.deviceName || "NVIDIA CUDA GPU");
  const vramGb = Number(hardware.vramGb || 0);
  const bf16 = hardware.nativeBf16 ? "原生 BF16" : "FP16 优先";
  const memory = vramGb > 0 ? `${vramGb.toFixed(1)}GB` : "显存容量未知";
  return `${name} · ${memory} · ${bf16}`;
}

module.exports = {
  RUNTIME_PROFILES,
  hardwareSummary,
  recommendRuntimeProfile,
  resolveRuntimeProfile
};
