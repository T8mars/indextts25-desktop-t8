const assert = require("node:assert/strict");
const {
  RUNTIME_PROFILES,
  hardwareSummary,
  recommendRuntimeProfile,
  resolveRuntimeProfile
} = require("../src/runtime_profiles");

assert.equal(recommendRuntimeProfile({ cudaAvailable: false, vramGb: 32 }), "compatibility");
assert.equal(recommendRuntimeProfile({ cudaAvailable: true, vramGb: 8 }), "low_vram");
assert.equal(recommendRuntimeProfile({ cudaAvailable: true, vramGb: 12 }), "balanced");
assert.equal(recommendRuntimeProfile({ cudaAvailable: true, vramGb: 24 }), "balanced");

const recommended = resolveRuntimeProfile("recommended", "low_vram");
assert.equal(recommended.name, "low_vram");
assert.equal(recommended.precisionMode, "float16");
assert.equal(recommended.referenceDevice, "cpu");
assert.equal(recommended.accelerationMode, "off");
assert.equal(recommended.reuseDefaultEmotion, true);

const speed = resolveRuntimeProfile("max_speed");
assert.equal(speed.accelerationMode, "gpt_accel");
assert.equal(speed.referenceDevice, "same");
assert.throws(() => resolveRuntimeProfile("unknown"), /Unknown runtime profile/);
assert.ok(Object.isFrozen(RUNTIME_PROFILES.low_vram));
assert.match(
  hardwareSummary({ cudaAvailable: true, deviceName: "RTX Test", vramGb: 12, nativeBf16: true }),
  /RTX Test · 12\.0GB · 原生 BF16/
);
assert.match(hardwareSummary({ cudaAvailable: false }), /未检测到/);

console.log("Runtime profile policy OK");
