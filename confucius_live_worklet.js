class T8ConfuciusCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.source = [];
    this.base = 0;
    this.seen = 0;
    this.outputIndex = 0;
    this.pending = [];
    this.ratio = sampleRate / 16000;
    this.cutoff = 0.9 * Math.min(1, 16000 / sampleRate);
    this.radius = 16;
    this.stopped = false;
    this.port.onmessage = (event) => {
      if (event.data?.type === "flush" && !this.stopped) {
        this.stopped = true;
        this.produce(true);
        this.emit(true);
        this.port.postMessage({type: "flushed", totalSamples: this.outputIndex});
      }
    };
  }

  read(index) {
    if (index < 0 || index >= this.seen) return 0;
    return this.source[index - this.base] ?? 0;
  }

  sample(center) {
    const origin = Math.floor(center);
    let total = 0;
    let weights = 0;
    for (let i = origin - this.radius + 1; i <= origin + this.radius; i++) {
      const distance = center - i;
      const x = Math.PI * distance * this.cutoff;
      const sinc = Math.abs(x) < 1e-9 ? 1 : Math.sin(x) / x;
      const window = 0.5 + 0.5 * Math.cos(Math.PI * distance / this.radius);
      const weight = this.cutoff * sinc * window;
      total += this.read(i) * weight;
      weights += weight;
    }
    return weights ? total / weights : 0;
  }

  emit(force = false) {
    while (this.pending.length >= 640 || (force && this.pending.length)) {
      const count = Math.min(640, this.pending.length);
      const pcm = new Float32Array(this.pending.splice(0, count));
      this.port.postMessage({type: "pcm", samples: pcm}, [pcm.buffer]);
    }
  }

  produce(flush = false) {
    const limit = flush ? Math.round(this.seen / this.ratio) : Infinity;
    while (this.outputIndex < limit) {
      const center = this.outputIndex * this.ratio;
      if (!flush && center + this.radius >= this.seen) break;
      this.pending.push(this.sample(center));
      this.outputIndex++;
    }
    this.emit(flush);
    const oldest = Math.max(0, Math.floor(this.outputIndex * this.ratio) - this.radius - 2);
    if (oldest > this.base) {
      this.source.splice(0, oldest - this.base);
      this.base = oldest;
    }
  }

  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (!channels?.length) return true;
    const length = channels[0].length;
    for (let i = 0; i < length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i] / channels.length;
      this.source.push(value);
    }
    this.seen += length;
    this.produce(false);
    return true;
  }
}

registerProcessor("t8-confucius-capture", T8ConfuciusCapture);
