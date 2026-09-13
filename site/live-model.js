/* FlyHamlet's full-connectome LIF and closed-loop arena, computed in this browser.
 * Port of flyhamlet/sim.py, arena.py and typewriter.py. No replay data is read.
 * The browser uses xoshiro128** instead of NumPy PCG64: a seed reproduces a browser
 * session, but does not reproduce the Python stream. Poisson inputs have the same
 * Bernoulli-per-step law. Float32 arithmetic preserves the model's update order.
 * Resting cells are omitted after the original rest-epsilon rule, and exactly
 * stationary float32 states are retained without recomputing identical values.
 * All neurons and all CSR edges remain available for subsequent inputs.
 */
(function (root) {
  'use strict';

  const TAU = 2 * Math.PI;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const modulo = (x, m) => ((x % m) + m) % m;
  // Python round: preserve ties-to-even for configurations other than the default.
  const roundEven = x => {
    const lo = Math.floor(x);
    return x - lo === 0.5 ? lo + (lo % 2) : Math.round(x);
  };

  class SeededRandom {
    constructor(seed) {
      let s = Number(seed) >>> 0;
      const splitmix = () => {
        s = (s + 0x9e3779b9) >>> 0;
        let z = s;
        z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
        z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
        return (z ^ (z >>> 15)) >>> 0;
      };
      this.s = [splitmix(), splitmix(), splitmix(), splitmix()];
    }
    next() {
      const s = this.s;
      const rotl = (x, k) => (x << k) | (x >>> (32 - k));
      const result = Math.imul(rotl(Math.imul(s[1], 5), 7), 9) >>> 0;
      const t = s[1] << 9;
      s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
      s[2] ^= t; s[3] = rotl(s[3], 11);
      return result / 4294967296;
    }
    // Geometric skips sample independent Bernoulli trials without scanning zeros.
    bernoulliIndices(length, probability, callback) {
      if (!(probability > 0) || !length) return;
      if (probability >= 1) {
        for (let i = 0; i < length; ++i) callback(i);
        return;
      }
      const logFailure = Math.log1p(-probability);
      let index = -1;
      while (true) {
        index += 1 + Math.floor(Math.log1p(-this.next()) / logFailure);
        if (index >= length) return;
        callback(index);
      }
    }
  }

  class LIFNetwork {
    constructor(connectome, cfg, seed = 0) {
      this.n = connectome.n;
      this.cfg = cfg;
      this.dt = Number(cfg.dt_ms);
      if (!(this.dt > 0)) throw new Error('The neural timestep must be positive.');
      this.indptr = connectome.indptr;
      this.indices = connectome.indices;
      this.weights = connectome.weights;
      if (this.indptr.length !== this.n + 1 || this.indices.length !== this.weights.length ||
          this.indptr[this.n] !== this.indices.length) throw new Error('Incomplete connectome arrays.');
      this.rng = new SeededRandom(seed);
      const FloatArray = cfg.dtype === 'float64' ? Float64Array : Float32Array;
      this.f = cfg.dtype === 'float64' ? x => x : Math.fround;
      const f = this.f;
      this.v0 = f(cfg.v_rest_mV); this.vReset = f(cfg.v_reset_mV); this.vThreshold = f(cfg.v_thresh_mV);
      this.poissonWeight = f(cfg.poisson_weight_mV ?? 68.75);
      this.a = f(Math.exp(-this.dt / cfg.tau_mem_ms));
      this.cc = f(Math.exp(-this.dt / cfg.tau_syn_ms));
      this.b = f(cfg.tau_syn_ms / (cfg.tau_syn_ms - cfg.tau_mem_ms) *
        (Math.exp(-this.dt / cfg.tau_syn_ms) - Math.exp(-this.dt / cfg.tau_mem_ms)));
      this.restEps = cfg.rest_eps_mV ?? 1e-5;
      this.refractoryDefault = Math.max(0, roundEven(cfg.t_refractory_ms / this.dt) - 1);
      this.delaySteps = Math.max(1, roundEven(cfg.delay_ms / this.dt));
      this.v = new FloatArray(this.n).fill(this.v0);
      this.g = new FloatArray(this.n);
      this.current = new FloatArray(this.n);
      this.resumeAt = new Float64Array(this.n);
      this.refractorySteps = new Uint32Array(this.n).fill(this.refractoryDefault);
      this.active = new Uint32Array(this.n);
      this.isActive = new Uint8Array(this.n);
      this.activeCount = 0;
      this.queue = Array.from({length: this.delaySteps}, () => []);
      this.poisson = [];
      this.stepIndex = 0;
      this.totalSpikes = 0;
      const bg = cfg.background || {};
      if (bg.enabled && bg.rate_hz > 0) {
        if (bg.super_classes?.length && !connectome.backgroundIndices)
          throw new Error('Background target annotations are missing.');
        this.background = {indices: connectome.backgroundIndices || null,
          length: connectome.backgroundIndices?.length ?? this.n,
          p: bg.rate_hz * this.dt * 1e-3, weight: f(bg.weight_mV)};
      }
    }

    activate(index) {
      if (!this.isActive[index]) {
        this.isActive[index] = 1;
        this.active[this.activeCount++] = index;
      }
    }
    setVoltage(index, value) { this.v[index] = value; this.activate(index); }
    injectCurrent(indices, value) {
      for (const index of indices) { this.current[index] = value; this.activate(index); }
    }
    injectPoisson(indices, rateHz = 0, weight = this.poissonWeight, to = 'v', refractoryFree = true) {
      const input = {indices, p: rateHz * this.dt * 1e-3, weight: this.f(weight), to};
      this.poisson.push(input);
      if (refractoryFree) for (const index of indices) this.refractorySteps[index] = 0;
      return input;
    }
    setPoissonRate(input, rateHz) { input.p = rateHz * this.dt * 1e-3; }
    addInput(index, value, to = 'g') {
      // Frozen is determined before thresholding in this timestep. Newly spiking
      // cells accept input here, then discard it in the reset slot, just as Python.
      if (this.stepIndex < this.resumeAt[index]) return;
      const state = to === 'v' ? this.v : this.g;
      state[index] = this.f(state[index] + value);
      this.activate(index);
    }

    step() {
      const f = this.f, v = this.v, g = this.g, step = this.stepIndex;
      const spikes = [];
      let position = 0;
      while (position < this.activeCount) {
        const i = this.active[position];
        const ie = this.current[i];
        let fixedPoint = false;
        if (step >= this.resumeAt[i]) {
          const oldV = v[i], oldG = g[i];
          const ss = f(this.v0 + ie);
          const vi = f(f(ss + f(this.a * f(v[i] - ss))) + f(this.b * g[i]));
          const gi = f(this.cc * g[i]);
          if (ie === 0 && Math.abs(vi - this.v0) < this.restEps && Math.abs(gi) < this.restEps) {
            v[i] = this.v0; g[i] = 0;
          } else { v[i] = vi; g[i] = gi; }
          if (v[i] > this.vThreshold) spikes.push(i);
          // Float32 decay can settle at an exactly stationary representable value
          // slightly outside rest_eps. Retire that fixed point without changing
          // its stored state; any later synaptic input activates it again.
          fixedPoint = ie === 0 && v[i] === oldV && g[i] === oldG && v[i] <= this.vThreshold;
        }
        // Refractory timestamps survive removal from the active list, so incoming
        // synapses remain blocked for the full refractory interval.
        if (fixedPoint || (ie === 0 && v[i] === this.v0 && g[i] === 0)) {
          this.isActive[i] = 0;
          this.active[position] = this.active[--this.activeCount];
        } else ++position;
      }
      // Python flatnonzero returns ascending neuron indices; preserve that order
      // because per-synapse float32 additions are sensitive to summation order.
      spikes.sort((a, b) => a - b);
      const slot = step % this.delaySteps;
      const due = this.queue[slot];
      for (const pre of due) {
        for (let p = this.indptr[pre]; p < this.indptr[pre + 1]; ++p) {
          const post = this.indices[p];
          if (step >= this.resumeAt[post]) {
            g[post] = f(g[post] + this.weights[p]);
            this.activate(post);
          }
        }
      }
      for (const input of this.poisson) {
        this.rng.bernoulliIndices(input.indices.length, input.p, position => {
          // sim.py uses poisson_weight_mV for all voltage injections.
          this.addInput(input.indices[position], input.to === 'v' ? this.poissonWeight : input.weight, input.to);
        });
      }
      if (this.background) {
        const bg = this.background;
        let count = 0;
        this.rng.bernoulliIndices(bg.length, bg.p, () => ++count);
        // Background draws Binomial(m,p) events and chooses targets WITH replacement
        // in the Python model (distinct from the eye inputs).
        for (let k = 0; k < count; ++k) {
          const position = Math.floor(this.rng.next() * bg.length);
          this.addInput(bg.indices ? bg.indices[position] : position, bg.weight);
        }
      }
      for (const i of spikes) {
        v[i] = this.vReset; g[i] = 0;
        this.resumeAt[i] = step + this.refractorySteps[i] + 1;
      }
      this.queue[slot] = spikes;
      this.stepIndex++;
      this.totalSpikes += spikes.length;
      return spikes;
    }
    get timeS() { return this.stepIndex * this.dt * 1e-3; }
  }

  function rayToWall(x, y, phi, width, height) {
    const c = Math.cos(phi), s = Math.sin(phi);
    let distance = Infinity;
    if (c > 1e-9) distance = Math.min(distance, (width - x) / c);
    else if (c < -1e-9) distance = Math.min(distance, -x / c);
    if (s > 1e-9) distance = Math.min(distance, (height - y) / s);
    else if (s < -1e-9) distance = Math.min(distance, -y / s);
    return Math.max(0, distance);
  }

  class WindowedRate {
    constructor(steps, dtMs, neuronCount) {
      this.buffer = new Uint32Array(Math.max(1, steps));
      this.position = 0; this.total = 0;
      this.scale = 1 / (this.buffer.length * dtMs * 1e-3 * Math.max(1, neuronCount));
    }
    push(count) {
      this.total += count - this.buffer[this.position];
      this.buffer[this.position] = count;
      this.position = (this.position + 1) % this.buffer.length;
    }
    get hz() { return this.total * this.scale; }
  }

  class FlyArena {
    constructor(connectome, manifest, seed) {
      this.seed = Number(seed) >>> 0;
      this.manifest = manifest;
      this.cfg = manifest.config;
      const a = this.cfg.arena, m = a.motor;
      this.width = a.width_mm; this.height = a.height_mm;
      this.controlSteps = Math.max(1, roundEven(a.control_interval_ms / this.cfg.sim.dt_ms));
      this.logSteps = Math.max(1, roundEven(a.log_interval_ms / this.cfg.sim.dt_ms));
      this.logEveryControl = Math.max(1, Math.floor(this.logSteps / this.controlSteps));
      this.net = new LIFNetwork(connectome, this.cfg.sim, this.seed);
      const windowSteps = Math.max(1, roundEven(m.window_ms / this.net.dt));
      this.groupNames = ['turn_L', 'turn_R', 'fwd_L', 'fwd_R', 'bwd_L', 'bwd_R', 'GF'];
      this.groupCounts = new Uint32Array(this.groupNames.length);
      this.groups = {};
      this.groupMask = new Uint8Array(connectome.n);
      this.groupNames.forEach((name, bit) => {
        const indices = manifest.targets[name];
        if (!indices?.length) throw new Error(`Missing annotated motor target ${name}.`);
        this.groups[name] = new WindowedRate(windowSteps, this.net.dt, indices.length);
        for (const index of indices) this.groupMask[index] |= 1 << bit;
      });
      this.eyeInputs = ['eye_L', 'eye_R'].map(name => {
        if (!manifest.targets[name]?.length) throw new Error(`Missing annotated sensory target ${name}.`);
        return this.net.injectPoisson(manifest.targets[name], 0);
      });
      this.eyeRates = [0, 0];
      this.eyeRays = a.looming.eye_ray_angles_deg.map(angle => angle * Math.PI / 180);
      const rng = new SeededRandom(this.seed); // arena and network have separate streams, as in Python
      this.fly = a.start === 'center'
        ? {x: this.width / 2, y: this.height / 2, heading: 0, v: 0, omega: 0}
        : {x: this.width * (0.1 + 0.8 * rng.next()), y: this.height * (0.1 + 0.8 * rng.next()),
          heading: -Math.PI + TAU * rng.next(), v: 0, omega: 0};
      this.layout = manifest.layout;
      if (this.layout?.length !== a.typewriter.rows * a.typewriter.cols)
        throw new Error('The exported Python key layout is incomplete.');
      this.lastKey = this.keyAt(this.fly.x, this.fly.y);
      this.controlIndex = 0; this.wallSteps = 0; this.keyCount = 0;
      // The reference registers zero-rate inputs initially and senses after the
      // first control interval. Keep that precise initialization schedule.
    }
    keyAt(x, y) {
      const tw = this.cfg.arena.typewriter;
      const col = clamp(Math.trunc(x / this.width * tw.cols), 0, tw.cols - 1);
      const row = clamp(Math.trunc(y / this.height * tw.rows), 0, tw.rows - 1);
      return row * tw.cols + col;
    }
    sense() {
      const f = this.fly, loom = this.cfg.arena.looming;
      for (let eye = 0; eye < 2; ++eye) {
        const sign = eye === 0 ? 1 : -1;
        let distance = Infinity;
        for (const angle of this.eyeRays) distance = Math.min(distance,
          rayToWall(f.x, f.y, f.heading + sign * angle, this.width, this.height));
        const proximity = Math.max(0, 1 - distance / loom.wall_distance_mm);
        this.eyeRates[eye] = loom.max_rate_hz * proximity ** loom.exponent;
        this.net.setPoissonRate(this.eyeInputs[eye], this.eyeRates[eye]);
      }
    }
    sample() {
      const f = this.fly, g = this.groups;
      return {t_s: this.net.timeS, x: f.x, y: f.y, heading_deg: f.heading * 180 / Math.PI,
        v_mm_s: f.v, omega_deg_s: f.omega * 180 / Math.PI,
        loom_L_hz: this.eyeRates[0], loom_R_hz: this.eyeRates[1],
        turnDN_L_hz: g.turn_L.hz, turnDN_R_hz: g.turn_R.hz,
        fwdDN_hz: 0.5 * (g.fwd_L.hz + g.fwd_R.hz),
        bwdDN_hz: 0.5 * (g.bwd_L.hz + g.bwd_R.hz), GF_hz: g.GF.hz,
        neural_spikes: this.net.totalSpikes, active_neurons: this.net.activeCount};
    }
    stepControl() {
      const counts = this.groupCounts;
      for (let step = 0; step < this.controlSteps; ++step) {
        counts.fill(0);
        for (const spike of this.net.step()) {
          const mask = this.groupMask[spike];
          if (mask) for (let bit = 0; bit < counts.length; ++bit) if (mask & (1 << bit)) counts[bit]++;
        }
        for (let bit = 0; bit < counts.length; ++bit) this.groups[this.groupNames[bit]].push(counts[bit]);
      }
      const m = this.cfg.arena.motor, g = this.groups, f = this.fly;
      const turn = g.turn_L.hz - g.turn_R.hz;
      const fwd = 0.5 * (g.fwd_L.hz + g.fwd_R.hz);
      const bwd = 0.5 * (g.bwd_L.hz + g.bwd_R.hz);
      f.omega = clamp(m.turn_sign * m.turn_gain * Math.PI / 180 * turn,
        -m.max_turn_deg_s * Math.PI / 180, m.max_turn_deg_s * Math.PI / 180);
      f.v = clamp(m.base_speed_mm_s + m.speed_gain * fwd - m.backward_gain * bwd,
        -m.max_speed_mm_s, m.max_speed_mm_s);
      const dt = this.controlSteps * this.net.dt * 1e-3;
      f.heading = modulo(f.heading + f.omega * dt + Math.PI, TAU) - Math.PI;
      const nx = f.x + f.v * Math.cos(f.heading) * dt;
      const ny = f.y + f.v * Math.sin(f.heading) * dt;
      if (!(0 <= nx && nx <= this.width && 0 <= ny && ny <= this.height)) this.wallSteps++;
      f.x = clamp(nx, 0, this.width); f.y = clamp(ny, 0, this.height);
      this.sense();
      this.controlIndex++;
      const keyIndex = this.keyAt(f.x, f.y);
      let key = null;
      if (keyIndex !== this.lastKey) {
        key = [this.net.timeS, keyIndex, this.layout[keyIndex]];
        this.lastKey = keyIndex; this.keyCount++;
      }
      return {sample: this.controlIndex % this.logEveryControl === 0 ? this.sample() : null, key};
    }
    get metadata() {
      const tw = this.cfg.arena.typewriter;
      return {W: this.width, H: this.height, rows: tw.rows, cols: tw.cols, layout: this.layout,
        seed: this.seed, n: this.net.n, edgeCount: this.net.indices.length,
        dt_ms: this.net.dt, rng: 'xoshiro128** (browser; differs from NumPy PCG64)'};
    }
  }

  const api = {SeededRandom, LIFNetwork, FlyArena, WindowedRate, rayToWall};
  root.FlyHamletLive = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
