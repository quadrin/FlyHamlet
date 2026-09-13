/* A supervised two-letter decoder around the unchanged full-connectome LIF model.
 * Learning occurs only in this external decoder, never in a biological synapse.
 * Cues are arbitrary left/right LC4/LPLC2 stimulation, not representations of text.
 * Every decision is committed before its target can update the decoder.
 */
(function (root) {
  'use strict';
  const live = root.FlyHamletLive || (typeof require === 'function' ? require('./live-model.js') : null);
  if (!live) throw new Error('Load live-model.js before learning-model.js.');
  const {LIFNetwork, SeededRandom} = live;
  const GROUP_NAMES = Object.freeze(['turn_L', 'turn_R', 'fwd_L', 'fwd_R', 'bwd_L', 'bwd_R', 'GF']);
  const PHASES = Object.freeze(['baseline', 'train', 'evaluation']);
  const LETTERS = Object.freeze(['T', 'O']);
  const DEFAULTS = Object.freeze({baselineTrials: 20, trainTrials: 60, evaluationTrials: 40,
    trialMs: 200, warmupMs: 50, cueRateHz: 100, featureScaleHz: 100,
    learningRate: 1, l2: 0.001});
  const clone = value => JSON.parse(JSON.stringify(value));

  // Independent domains keep class shuffling, neural noise and tie-breaking apart.
  function deriveSeed(seed, domain, index = 0) {
    let x = ((Number(seed) >>> 0) ^ Math.imul(domain + 1, 0x9e3779b9) ^ Math.imul(index + 1, 0x85ebca6b)) >>> 0;
    x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
    x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
    return (x ^ (x >>> 16)) >>> 0;
  }

  function makeTrialPlan(seed, counts) {
    const result = [];
    PHASES.forEach((phase, phaseIndex) => {
      const count = counts[`${phase}Trials`];
      if (!Number.isInteger(count) || count < 0 || count % 2) throw new Error('Each phase needs an even, nonnegative trial count.');
      const rng = new SeededRandom(deriveSeed(seed, 1, phaseIndex));
      const targets = Array.from({length: count}, (_, i) => LETTERS[i % 2]);
      for (let i = count - 1; i > 0; --i) {
        const j = Math.floor(rng.next() * (i + 1));
        [targets[i], targets[j]] = [targets[j], targets[i]];
      }
      targets.forEach((target, phaseIndex) => result.push(Object.freeze({phase, phaseTrial: phaseIndex + 1,
        phaseTotal: count, target, neuralSeed: deriveSeed(seed, 2, result.length),
        decisionSeed: deriveSeed(seed, 3, result.length)})));
    });
    if (!result.length) throw new Error('At least one trial is required.');
    return Object.freeze(result);
  }

  class LinearDecoder {
    constructor(featureCount = GROUP_NAMES.length, {learningRate = 1, l2 = 0.001} = {}) {
      if (!Number.isInteger(featureCount) || featureCount < 1) throw new Error('Invalid decoder feature count.');
      if (!(learningRate > 0) || !Number.isFinite(learningRate) || !(l2 >= 0) || !Number.isFinite(l2)) throw new Error('Invalid decoder learning rule.');
      this.weights = new Float64Array(featureCount + 1); // last coefficient is bias
      this.learningRate = learningRate;
      this.l2 = l2;
      this.updateCount = 0;
    }
    check(features) {
      if (features.length !== this.weights.length - 1 || Array.from(features).some(x => !Number.isFinite(x)))
        throw new Error('The decoder accepts only its fixed-length numeric neural feature vector.');
    }
    probability(features) {
      this.check(features);
      let score = this.weights[this.weights.length - 1];
      for (let i = 0; i < features.length; ++i) score += this.weights[i] * features[i];
      return score >= 0 ? 1 / (1 + Math.exp(-score)) : Math.exp(score) / (1 + Math.exp(score));
    }
    predict(features, tieDraw = 0.5) {
      const probabilityO = this.probability(features);
      // One independent, prerecorded draw is used by both decoders on this trial.
      const prediction = probabilityO === 0.5 ? (tieDraw < 0.5 ? 'T' : 'O') : (probabilityO > 0.5 ? 'O' : 'T');
      return {prediction, probabilityO};
    }
    update(features, target) {
      if (!LETTERS.includes(target)) throw new Error('Unknown training target.');
      const error = (target === 'O' ? 1 : 0) - this.probability(features);
      for (let i = 0; i < features.length; ++i)
        this.weights[i] += this.learningRate * (error * features[i] - this.l2 * this.weights[i]);
      this.weights[this.weights.length - 1] += this.learningRate * error;
      this.updateCount++;
    }
    snapshot() { return Array.from(this.weights); }
  }

  // Independent movement/press semantics: crossing a key never emits text.
  // A display controller may assist movement, but cannot change the chosen letter.
  class ExplicitTypewriter {
    constructor(layout = 'abcdefghijklmnopqrstuvwxyz ', startIndex = null) {
      this.layout = Array.from(layout);
      if (!this.layout.length || new Set(this.layout).size !== this.layout.length)
        throw new Error('The keyboard layout must contain distinct symbols.');
      this.keyIndex = null;
      this.output = '';
      this.events = [];
      if (startIndex !== null) this.moveTo(startIndex);
    }
    moveTo(key) {
      const index = typeof key === 'number' ? key : this.layout.indexOf(String(key).toLowerCase());
      if (!Number.isInteger(index) || index < 0 || index >= this.layout.length) throw new Error('Unknown key.');
      this.keyIndex = index;
      return index;
    }
    move(key) { return this.moveTo(key); }
    press() {
      if (this.keyIndex === null) throw new Error('Move to a key before pressing.');
      const letter = this.layout[this.keyIndex];
      this.output += letter;
      const event = {index: this.events.length + 1, keyIndex: this.keyIndex, letter};
      this.events.push(event);
      return {...event};
    }
    reset() { this.output = ''; this.events = []; this.keyIndex = null; }
  }

  function wilsonInterval(correct, total, z = 1.959963984540054) {
    if (!total) return null;
    const p = correct / total, zz = z * z, denominator = 1 + zz / total;
    const center = (p + zz / (2 * total)) / denominator;
    const margin = z * Math.sqrt(p * (1 - p) / total + zz / (4 * total * total)) / denominator;
    return [Math.max(0, center - margin), Math.min(1, center + margin)];
  }

  function summarizeTrials(trials, predictionField = 'prediction') {
    const byClass = Object.fromEntries(LETTERS.map(letter => [letter, {correct: 0, total: 0, accuracy: null}]));
    let correct = 0;
    const confusion = {T: {T: 0, O: 0}, O: {T: 0, O: 0}};
    for (const trial of trials) {
      const prediction = trial[predictionField];
      const success = trial.target === prediction;
      correct += Number(success);
      byClass[trial.target].correct += Number(success);
      byClass[trial.target].total++;
      confusion[trial.target][prediction]++;
    }
    for (const row of Object.values(byClass)) if (row.total) row.accuracy = row.correct / row.total;
    return {correct, total: trials.length, accuracy: trials.length ? correct / trials.length : null,
      balancedAccuracy: LETTERS.every(letter => byClass[letter].total) ? (byClass.T.accuracy + byClass.O.accuracy) / 2 : null,
      byClass, confusion, wilson95: wilsonInterval(correct, trials.length)};
  }

  class LearningExperiment {
    constructor(connectome, manifest, seed, options = {}) {
      this.seed = Number(seed) >>> 0;
      this.connectome = connectome;
      this.manifest = manifest;
      this.dtMs = Number(manifest.config.sim.dt_ms);
      this.protocol = Object.freeze(Object.fromEntries(Object.entries(DEFAULTS).map(([key, value]) => [key, options[key] ?? value])));
      const p = this.protocol;
      for (const key of ['trialMs', 'cueRateHz', 'featureScaleHz', 'learningRate'])
        if (!(p[key] > 0) || !Number.isFinite(p[key])) throw new Error(`Invalid ${key}.`);
      if (!(p.warmupMs >= 0 && p.warmupMs < p.trialMs) || !(this.dtMs > 0)) throw new Error('Invalid neural/readout timing.');
      this.stepsPerTrial = p.trialMs / this.dtMs;
      this.warmupSteps = p.warmupMs / this.dtMs;
      if (!Number.isInteger(this.stepsPerTrial) || !Number.isInteger(this.warmupSteps)) throw new Error('Trial timings must be exact multiples of the neural timestep.');
      if (p.cueRateHz * this.dtMs / 1000 > 1) throw new Error('Cue probability exceeds one event per timestep.');
      this.plan = makeTrialPlan(this.seed, p);
      this.decoder = new LinearDecoder(GROUP_NAMES.length, p);
      this.control = new LinearDecoder(GROUP_NAMES.length, p);
      this.initialWeights = this.decoder.snapshot();
      this.frozenWeights = null;
      this.trials = [];
      this.totalSteps = 0;
      this.trialIndex = 0;
      this.net = null;
      this.counts = new Uint32Array(GROUP_NAMES.length);
      this.groupForNeuron = new Int8Array(connectome.n).fill(-1);
      const sensory = new Set();
      for (const name of ['eye_L', 'eye_R']) {
        const targets = manifest.targets[name];
        if (!targets?.length) throw new Error(`Missing sensory target ${name}.`);
        for (const index of targets) {
          if (!Number.isInteger(index) || index < 0 || index >= connectome.n || sensory.has(index))
            throw new Error('Sensory groups must contain valid, disjoint neuron indices.');
          sensory.add(index);
        }
      }
      this.groupSizes = GROUP_NAMES.map((name, group) => {
        const targets = manifest.targets[name];
        if (!targets?.length) throw new Error(`Missing motor target ${name}.`);
        for (const index of targets) {
          if (!Number.isInteger(index) || index < 0 || index >= connectome.n || sensory.has(index) || this.groupForNeuron[index] !== -1)
            throw new Error('Readout groups must be disjoint and exclude all stimulated sensory neurons.');
          this.groupForNeuron[index] = group;
        }
        return targets.length;
      });
      this.metadata = Object.freeze({seed: this.seed, n: connectome.n, edgeCount: connectome.indices.length,
        groupNames: [...GROUP_NAMES], groupSizes: [...this.groupSizes], cueMapping: {T: 'eye_L', O: 'eye_R'},
        layout: [...manifest.layout], model: 'fixed full-connectome LIF; external supervised logistic decoder',
        rng: 'xoshiro128**; separate target, neural-noise and decision domains'});
    }
    get done() { return this.trialIndex >= this.plan.length; }
    get simTimeS() { return this.totalSteps * this.dtMs / 1000; }
    get rates() {
      const observedSteps = this.net ? Math.max(0, this.net.stepIndex - this.warmupSteps) : 0;
      const seconds = observedSteps * this.dtMs / 1000;
      return Array.from(this.counts, (count, group) => seconds ? count / (seconds * this.groupSizes[group]) : 0);
    }
    get progress() {
      const item = this.done ? this.plan[this.plan.length - 1] : this.plan[this.trialIndex];
      return {phase: this.done ? 'complete' : item.phase, trial: Math.min(this.trialIndex + 1, this.plan.length),
        phaseTrial: item.phaseTrial, phaseTotal: item.phaseTotal, completed: this.trials.length,
        totalTrials: this.plan.length, simTime: this.simTimeS, simTimeS: this.simTimeS,
        stage: this.done ? 'complete' : ((this.net?.stepIndex ?? 0) < this.warmupSteps ? 'cue' : 'readout'),
        cueLetter: this.done ? null : item.target, rates: this.rates, updateCount: this.decoder.updateCount};
    }
    startTrial() {
      const item = this.plan[this.trialIndex];
      if (item.phase === 'evaluation' && this.frozenWeights === null)
        this.frozenWeights = Object.freeze(this.decoder.snapshot());
      // Only dynamic neural state and noise stream reset; no connectome edge is edited.
      this.net = new LIFNetwork(this.connectome, this.manifest.config.sim, item.neuralSeed);
      this.net.injectPoisson(this.manifest.targets[item.target === 'T' ? 'eye_L' : 'eye_R'], this.protocol.cueRateHz);
      this.counts.fill(0);
    }
    step() {
      if (this.done) return null;
      if (!this.net) this.startTrial();
      const inReadout = this.net.stepIndex >= this.warmupSteps;
      const spikes = this.net.step();
      this.totalSteps++;
      if (inReadout) for (const neuron of spikes) {
        const group = this.groupForNeuron[neuron];
        if (group >= 0) this.counts[group]++;
      }
      if (this.net.stepIndex < this.stepsPerTrial) return null;
      const item = this.plan[this.trialIndex];
      const rates = this.rates;
      const features = rates.map(rate => Math.min(1, rate / this.protocol.featureScaleHz));
      const tieDraw = new SeededRandom(item.decisionSeed).next();
      const before = this.decoder.snapshot();
      // Prediction sees seven downstream rates only; the target is first used in
      // the learning rule below, after both choices have been committed.
      const decision = this.decoder.predict(features, tieDraw);
      const control = this.control.predict(features, tieDraw);
      const updateCountBefore = this.decoder.updateCount;
      if (item.phase === 'train') this.decoder.update(features, item.target);
      const record = {trial: this.trialIndex + 1, ...item, prediction: decision.prediction,
        correct: decision.prediction === item.target, probabilityO: decision.probabilityO,
        controlPrediction: control.prediction, controlCorrect: control.prediction === item.target,
        features, rates, groupSpikeCounts: Array.from(this.counts), spikes: this.net.totalSpikes,
        simTime: this.simTimeS, simTimeS: this.simTimeS, updateCountBefore,
        updateCountAfter: this.decoder.updateCount, weightsBefore: before, weightsAfter: this.decoder.snapshot(),
        feedbackApplied: item.phase === 'train'};
      this.trials.push(Object.freeze(record));
      this.trialIndex++;
      this.net = null;
      return clone(record);
    }
    result() {
      const metrics = {};
      for (const phase of PHASES) {
        const trials = this.trials.filter(trial => trial.phase === phase);
        metrics[phase] = {...summarizeTrials(trials), control: summarizeTrials(trials, 'controlPrediction')};
      }
      return clone({format: 'flyhamlet-learning-v1', complete: this.done, seed: this.seed,
        config: this.protocol, metadata: this.metadata, simTimeS: this.simTimeS,
        neuralConfig: this.manifest.config.sim, provenance: this.manifest.provenance || null,
        connectivityIntegrity: this.manifest.integrity || null,
        featureDefinition: 'Per-neuron mean spike rate during the final trialMs-warmupMs; clamp(rateHz/featureScaleHz,0,1).',
        intervention: 'Supervised external linear decoder; fixed connectome; sensory cues supplied; assisted movement and independent press.',
        inferenceLimit: 'Cue classification/copying, not recall or biological learning. Seeds are repeated simulations of one anatomical connectome. No shuffled-connectome or conventional-controller comparison is included.',
        ciDefinition: 'Two-sided 95% Wilson interval for trial accuracy; one simulated run, not a biological-population interval.',
        initialWeights: this.initialWeights, frozenWeights: this.frozenWeights,
        finalWeights: this.decoder.snapshot(), controlWeights: this.control.snapshot(),
        updateCount: this.decoder.updateCount, metrics, trials: this.trials,
        outputs: Object.fromEntries(PHASES.map(phase => [phase, this.trials.filter(t => t.phase === phase).map(t => t.prediction).join('')]))});
    }
  }

  const api = {LearningExperiment, LinearDecoder, ExplicitTypewriter, summarizeTrials, wilsonInterval,
    makeTrialPlan, deriveSeed, GROUP_NAMES, PHASES, LETTERS, DEFAULTS};
  root.FlyHamletLearning = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
