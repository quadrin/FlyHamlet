/* Delayed-symbol decoding in a fixed LIF network, with no external history.
 * A cue is removed before every observation window. Each window's independent
 * frozen decoder sees only contemporaneous downstream population spike rates.
 * The same cue/noise trials are paired across retained, reset and rewired states.
 */
(function (root) {
  'use strict';
  const live = root.FlyHamletLive || (typeof require === 'function' ? require('./live-model.js') : null);
  const recall = root.FlyHamletRecall || (typeof require === 'function' ? require('./recall-model.js') : null);
  if (!live || !recall) throw new Error('Load live-model.js and recall-model.js before memory-model.js.');
  const {LIFNetwork, SeededRandom} = live;
  const {ALPHABET, createCodebook, createPools, fitRidge, predictLinear, deriveSeed, fingerprint} = recall;
  const CONDITIONS = Object.freeze(['retain', 'reset', 'rewired']);
  const DEFAULTS = Object.freeze({trainPerClass: 8, evaluationPerClass: 8, cueMs: 200,
    delaysMs: Object.freeze([0, 10, 25, 50, 100, 200]), windowMs: 20,
    poolCount: 128, cueRateHz: 100, ridge: 0.01, codebookSeed: 7919, poolSeed: 104729});
  const clone = value => JSON.parse(JSON.stringify(value));

  function makeMemoryPlan(seed, protocol) {
    const plan = [];
    ['train', 'evaluation'].forEach((phase, phaseIndex) => {
      const perClass = protocol[phase === 'train' ? 'trainPerClass' : 'evaluationPerClass'];
      if (!Number.isInteger(perClass) || perClass < 1) throw new Error('Each phase needs a positive integer number of trials per class.');
      const cues = Array.from({length: perClass}, () => [...ALPHABET]).flat();
      const rng = new SeededRandom(deriveSeed(seed, 53, phaseIndex));
      for (let i = cues.length - 1; i > 0; --i) {
        const j = Math.floor(rng.next() * (i + 1)); [cues[i], cues[j]] = [cues[j], cues[i]];
      }
      cues.forEach((cue, index) => plan.push(Object.freeze({phase, cue, phaseTrial: index + 1,
        phaseTotal: cues.length, pairedTrial: plan.length + 1,
        neuralSeed: deriveSeed(seed, 59, plan.length)})));
    });
    return Object.freeze(plan);
  }
  function wilson(correct, total) {
    if (!total) return null;
    const z = 1.959963984540054, zz = z * z, p = correct / total, d = 1 + zz / total;
    const center = (p + zz / (2 * total)) / d;
    const radius = z * Math.sqrt(p * (1 - p) / total + zz / (4 * total * total)) / d;
    return [Math.max(0, center - radius), Math.min(1, center + radius)];
  }
  function summarizeMemoryTrials(trials, delaysMs = DEFAULTS.delaysMs) {
    const metrics = {};
    for (const condition of CONDITIONS) {
      const rows = trials.filter(trial => trial.condition === condition && trial.phase === 'evaluation');
      const summarize = (delayMs, cueVisible = false) => {
        const byClass = Object.fromEntries(ALPHABET.map(code => [code, {correct: 0, total: 0, accuracy: null}]));
        const confusion = Object.fromEntries(ALPHABET.map(code => [code, Object.fromEntries(ALPHABET.map(prediction => [prediction, 0]))]));
        let correct = 0, total = 0, silentTrials = 0, readoutSilentTrials = 0;
        let populationSpikes = 0, activeNeurons = 0, readoutRateHz = 0;
        for (const trial of rows) {
          const window = cueVisible ? trial.cueReadout : trial.windows.find(item => item.delayMs === delayMs);
          if (!window || !ALPHABET.includes(window.prediction)) continue;
          const success = window.prediction === trial.cue;
          correct += Number(success); total++;
          byClass[trial.cue].correct += Number(success); byClass[trial.cue].total++;
          confusion[trial.cue][window.prediction]++;
          silentTrials += Number(window.silent); readoutSilentTrials += Number(window.readoutSilent);
          populationSpikes += window.populationSpikes; activeNeurons += window.activeNeurons;
          readoutRateHz += window.readoutPopulationRateHz;
        }
        for (const row of Object.values(byClass)) if (row.total) row.accuracy = row.correct / row.total;
        return {delayMs, correct, total, accuracy: total ? correct / total : null,
          balancedAccuracy: ALPHABET.every(code => byClass[code].total) ? ALPHABET.reduce((sum, code) => sum + byClass[code].accuracy, 0) / ALPHABET.length : null,
          chance: 1 / ALPHABET.length, byClass, confusion, wilson95: wilson(correct, total), silentTrials, readoutSilentTrials,
          meanPopulationSpikes: total ? populationSpikes / total : null,
          meanActiveNeurons: total ? activeNeurons / total : null,
          meanReadoutRateHz: total ? readoutRateHz / total : null};
      };
      const cueDelay = rows[0]?.cueReadout?.delayMs ?? -DEFAULTS.windowMs;
      metrics[condition] = {delays: delaysMs.map(delay => summarize(delay)),
        cueReadout: {...summarize(cueDelay, true), diagnostic: true, inputActive: true}};
    }
    return metrics;
  }

  class MemoryExperiment {
    constructor(graph, manifest, seed, options = {}, control = {}) {
      this.graph = graph; this.manifest = manifest; this.seed = Number(seed) >>> 0;
      const protocol = Object.fromEntries(Object.entries(DEFAULTS).map(([key, value]) => [key, options[key] ?? value]));
      if (!Array.isArray(protocol.delaysMs) || !protocol.delaysMs.length ||
          protocol.delaysMs.some(value => !Number.isFinite(value) || value < 0) ||
          new Set(protocol.delaysMs).size !== protocol.delaysMs.length)
        throw new Error('Readout delays must be distinct, nonnegative finite numbers.');
      protocol.delaysMs = Object.freeze([...protocol.delaysMs].sort((a, b) => a - b));
      this.protocol = Object.freeze(protocol);
      const p = this.protocol;
      for (const key of ['cueMs', 'windowMs', 'cueRateHz', 'ridge'])
        if (!(p[key] > 0) || !Number.isFinite(p[key])) throw new Error(`Invalid ${key}.`);
      if (!Number.isInteger(p.poolCount) || p.poolCount < 1) throw new Error('Invalid pool count.');
      this.dtMs = Number(manifest.config.sim.dt_ms);
      if (!(this.dtMs > 0) || !Number.isFinite(this.dtMs) || p.cueRateHz * this.dtMs / 1000 > 1)
        throw new Error('Invalid neural timestep or cue probability.');
      const background = manifest.config.sim.background;
      if (background?.enabled && background.rate_hz > 0)
        throw new Error('This cue-off assay requires background input to be disabled.');
      const integerSteps = value => {
        const steps = value / this.dtMs;
        if (!Number.isInteger(steps)) throw new Error('All cue/window timings must be exact neural timestep multiples.');
        return steps;
      };
      this.cueSteps = integerSteps(p.cueMs);
      this.windowSteps = integerSteps(p.windowMs);
      if (this.windowSteps > this.cueSteps) throw new Error('The cue-on diagnostic window must fit inside the cue period.');
      this.windows = p.delaysMs.map(delayMs => {
        const startStep = this.cueSteps + integerSteps(delayMs);
        return {delayMs, startStep, endStep: startStep + this.windowSteps,
          startMs: p.cueMs + delayMs, endMs: p.cueMs + delayMs + p.windowMs, inputActive: false};
      });
      this.cueWindow = {delayMs: -p.windowMs, startStep: this.cueSteps - this.windowSteps, endStep: this.cueSteps,
        startMs: p.cueMs - p.windowMs, endMs: p.cueMs, inputActive: true};
      this.observationWindows = [this.cueWindow, ...this.windows];
      this.stepsPerTrial = Math.max(...this.windows.map(window => window.endStep));
      if (!control.graph || control.graph.n !== graph.n || control.graph.indices.length !== graph.indices.length)
        throw new Error('A rewired control graph with the same neurons and edge slots is required.');
      this.controlGraph = control.graph; this.rewireMetadata = clone(control.metadata || {});
      this.plan = makeMemoryPlan(this.seed, p);
      // Reuse the fixed sensory codebook from the recall assay. START is unused;
      // all eye cells (including that unused partition) are excluded from features.
      const allCodes = createCodebook(manifest, p.codebookSeed);
      this.codebook = Object.fromEntries(ALPHABET.map(code => [code, allCodes[code]]));
      const excluded = new Set(Object.values(allCodes).flat());
      for (const index of excluded)
        if (!Number.isInteger(index) || index < 0 || index >= graph.n) throw new Error('Invalid sensory neuron.');
      const pools = createPools(graph.n, excluded, p.poolCount, p.poolSeed);
      this.poolForNeuron = pools.poolForNeuron; this.poolSizes = pools.sizes;
      this.readoutNeuronCount = graph.n - excluded.size;
      if (!this.readoutNeuronCount) throw new Error('No nonstimulated readout neurons remain.');
      this.conditionIndex = 0; this.planIndex = 0; this.trials = []; this.totalSteps = 0;
      this.trialStep = 0; this.net = null; this.cueInput = null; this.cueOffsetApplied = false;
      this.cueSpikes = 0; this.postCueSpikes = 0; this.resetSeed = null;
      this.counts = this.observationWindows.map(() => new Uint32Array(p.poolCount));
      this.populationSpikes = new Uint32Array(this.observationWindows.length);
      this.activeSets = this.observationWindows.map(() => new Set());
      this.trainingRows = this.windows.map(() => []); this.cueTrainingRows = []; this.trainingLabels = [];
      this.models = {}; this.fitCount = 0;
      this.metadata = {seed: this.seed, n: graph.n, edgeCount: graph.indices.length, layout: [...manifest.layout],
        labels: [...ALPHABET], codebook: clone(this.codebook), unusedStartCode: [...allCodes.START],
        sensoryExcludedCount: excluded.size, readoutNeuronCount: this.readoutNeuronCount,
        poolSizes: Array.from(this.poolSizes), conditions: [...CONDITIONS],
        pooling: 'Fixed index-hash pools of nonstimulated neurons; all annotated eye cells excluded.',
        featureDefinition: 'Contemporaneous pooled mean spikes per neuron per second in one post-cue window; no previous window, cue label or external history enters a decoder.',
        inputTiming: 'Cue is removed before the first post-cue step. Pending spikes and delayed synaptic events are retained in retain/rewired; reset replaces all neural dynamic state and its RNG at cue offset.',
        windowDependence: 'All delays are measured in the same trial; some windows overlap. They are correlated observations, not independent replications.',
        pairedDesign: 'Exactly the same balanced cue order and neural noise seeds are used for every condition; training and evaluation use distinct trials.',
        model: 'Fixed full-connectome LIF delayed-symbol decoding; external history absent',
        rng: 'xoshiro128**; independent cue-shuffle, neural-noise and reset domains'};
    }
    get done() { return this.conditionIndex >= CONDITIONS.length; }
    get condition() { return CONDITIONS[this.conditionIndex] ?? null; }
    get simTimeS() { return this.totalSteps * this.dtMs / 1000; }
    get progress() {
      const item = this.plan[Math.min(this.planIndex, this.plan.length - 1)];
      const activeWindows = this.windows.filter(window => this.trialStep >= window.startStep && this.trialStep < window.endStep);
      return {condition: this.condition, phase: this.done ? 'complete' : item.phase,
        conditionIndex: Math.min(this.conditionIndex + 1, CONDITIONS.length), conditionTotal: CONDITIONS.length,
        conditionTrial: Math.min(this.planIndex + 1, this.plan.length), phaseTrial: item.phaseTrial,
        phaseTotal: item.phaseTotal, completed: this.trials.length, totalTrials: this.plan.length * CONDITIONS.length,
        simTimeS: this.simTimeS, simTime: this.simTimeS, trialTimeMs: this.trialStep * this.dtMs,
        stage: this.done ? 'complete' : (this.trialStep < this.cueSteps ? 'cue' : (activeWindows.length ? 'readout' : 'delay')),
        cue: this.done ? null : item.cue, inputActive: !this.done && this.trialStep < this.cueSteps,
        cueActive: !this.done && this.trialStep < this.cueSteps,
        readoutDelaysMs: activeWindows.map(window => window.delayMs), updateCount: this.fitCount};
    }
    startTrial() {
      const item = this.plan[this.planIndex];
      const graph = this.condition === 'rewired' ? this.controlGraph : this.graph;
      this.net = new LIFNetwork(graph, this.manifest.config.sim, item.neuralSeed);
      this.cueInput = this.net.injectPoisson(this.codebook[item.cue], this.protocol.cueRateHz);
      this.trialStep = 0; this.cueOffsetApplied = false; this.cueSpikes = 0; this.postCueSpikes = 0;
      this.resetSeed = null; this.populationSpikes.fill(0);
      this.counts.forEach(counts => counts.fill(0)); this.activeSets.forEach(set => set.clear());
    }
    removeCue() {
      if (this.condition === 'reset') {
        this.resetSeed = deriveSeed(this.plan[this.planIndex].neuralSeed, 61);
        this.net = new LIFNetwork(this.graph, this.manifest.config.sim, this.resetSeed);
        this.cueInput = null;
      } else this.net.setPoissonRate(this.cueInput, 0);
      this.cueOffsetApplied = true;
    }
    step() {
      if (this.done) return null;
      if (!this.net) this.startTrial();
      if (this.trialStep === this.cueSteps && !this.cueOffsetApplied) this.removeCue();
      const spikes = this.net.step();
      if (this.trialStep < this.cueSteps) this.cueSpikes += spikes.length;
      else this.postCueSpikes += spikes.length;
      for (let index = 0; index < this.observationWindows.length; ++index) {
        const window = this.observationWindows[index];
        if (this.trialStep < window.startStep || this.trialStep >= window.endStep) continue;
        if (!window.inputActive && !this.cueOffsetApplied) throw new Error('A readout window overlapped the active cue.');
        this.populationSpikes[index] += spikes.length;
        for (const neuron of spikes) {
          this.activeSets[index].add(neuron);
          const pool = this.poolForNeuron[neuron]; if (pool >= 0) this.counts[index][pool]++;
        }
      }
      this.trialStep++; this.totalSteps++;
      if (this.trialStep < this.stepsPerTrial) return null;
      return this.finishTrial();
    }
    finishTrial() {
      const item = this.plan[this.planIndex], condition = this.condition;
      const seconds = this.protocol.windowMs / 1000;
      const observations = this.observationWindows.map((window, index) => {
        const counts = Array.from(this.counts[index]);
        const readoutSpikes = counts.reduce((sum, count) => sum + count, 0);
        return {delayMs: window.delayMs, startMs: window.startMs, endMs: window.endMs, inputActive: window.inputActive,
          features: counts.map((count, pool) => this.poolSizes[pool] ? count / (this.poolSizes[pool] * seconds) : 0),
          poolSpikeCounts: counts, populationSpikes: this.populationSpikes[index],
          activeNeurons: this.activeSets[index].size, silent: this.populationSpikes[index] === 0,
          readoutSpikes, readoutSilent: readoutSpikes === 0,
          populationRateHz: this.populationSpikes[index] / (this.graph.n * seconds),
          readoutPopulationRateHz: readoutSpikes / (this.readoutNeuronCount * seconds),
          prediction: null, correct: null, decoderFingerprint: null};
      });
      const [cueReadout, ...windows] = observations;
      const before = this.fitCount;
      if (item.phase === 'train') {
        windows.forEach((window, index) => this.trainingRows[index].push(window.features));
        this.cueTrainingRows.push(cueReadout.features);
        this.trainingLabels.push(item.cue);
        if (item.phaseTrial === item.phaseTotal) this.fitReadouts(condition);
      } else {
        // All predictions are computed from features and frozen weights before
        // the evaluator compares them with the cue that was removed earlier.
        const predictions = observations.map((window, index) => {
          const readout = index === 0 ? this.models[condition].cueReadout : this.models[condition].readouts[index - 1];
          return {...predictLinear(readout.weights, window.features, ALPHABET), fingerprint: readout.fingerprint};
        });
        predictions.forEach((decision, index) => Object.assign(observations[index], {
          prediction: decision.prediction, scores: decision.scores,
          decoderFingerprint: decision.fingerprint, correct: decision.prediction === item.cue}));
      }
      const record = {trial: this.trials.length + 1, condition, ...item, windows, cueReadout,
        cueOffsetMs: this.protocol.cueMs, inputsOffFromMs: this.protocol.cueMs,
        cueSpikes: this.cueSpikes, postCueSpikes: this.postCueSpikes,
        spikes: this.cueSpikes + this.postCueSpikes, resetSeed: this.resetSeed,
        simTimeS: this.simTimeS, decoderVersion: item.phase === 'evaluation' ? 1 : 0,
        updateCountBefore: before, updateCountAfter: this.fitCount, fitApplied: this.fitCount !== before,
        feedbackApplied: false};
      this.trials.push(record);
      this.net = null; this.cueInput = null; this.trialStep = 0;
      this.planIndex++;
      if (this.planIndex === this.plan.length) {
        this.conditionIndex++; this.planIndex = 0;
        this.trainingRows = this.windows.map(() => []); this.cueTrainingRows = []; this.trainingLabels = [];
      }
      return clone(record);
    }
    fitReadouts(condition) {
      const fitOne = (rows, delayMs) => {
        const fit = fitRidge(rows, this.trainingLabels, this.protocol.ridge, ALPHABET);
        const weights = Object.freeze([...fit.weights]);
        return {delayMs, weights, frozenWeights: weights,
          fingerprint: fingerprint(weights), dimensions: fit.dimensions, labels: [...ALPHABET]};
      };
      const readouts = this.trainingRows.map((rows, index) => fitOne(rows, this.windows[index].delayMs));
      const cueReadout = fitOne(this.cueTrainingRows, this.cueWindow.delayMs);
      this.models[condition] = {fitCount: 1, trainingExamples: this.trainingLabels.length, readouts, cueReadout};
      this.fitCount++;
    }
    result() {
      const models = Object.fromEntries(Object.entries(this.models).map(([condition, model]) => [condition,
        {...model, cueReadout: {...model.cueReadout, finalFingerprint: fingerprint(model.cueReadout.weights)},
          readouts: model.readouts.map(readout => ({...readout, finalFingerprint: fingerprint(readout.weights)}))}]));
      return clone({format: 'flyhamlet-memory-v1', complete: this.done, seed: this.seed,
        config: this.protocol, metadata: this.metadata, neuralConfig: this.manifest.config.sim,
        provenance: this.manifest.provenance || null, connectivityIntegrity: this.manifest.integrity || null,
        rewireMetadata: this.rewireMetadata, simTimeS: this.simTimeS, fitCount: this.fitCount,
        readoutFitCount: this.fitCount * this.observationWindows.length, models, trials: this.trials,
        metrics: summarizeMemoryTrials(this.trials, this.protocol.delaysMs),
        cueDiagnostic: 'An additional independently fitted decoder reads the final windowMs of cue-ON activity. It diagnoses initial encoding and is never combined with post-cue features or memory scores.',
        intervention: 'Stimulate a fixed sensory code, remove input, and decode separate post-cue spike windows with independent supervised ridge readouts; no external history.',
        conditionDefinitions: {retain: 'Original graph, neural state retained at cue offset.',
          reset: 'Original graph, all dynamic neural state, delayed events and RNG replaced with rest at cue offset.',
          rewired: 'One fixed rewired graph, neural state retained; readouts retrained with the same budget.'},
        windowDependence: 'Delays come from the same trials and some 20 ms windows overlap; delay points are correlated.',
        chance: 1 / ALPHABET.length,
        inferenceLimit: 'This measures linearly decodable symbol information in post-cue spiking under this stimulation and LIF model. Failure does not show that biological flies cannot remember; success does not establish phrase memory. One anatomy and one fixed rewired null do not establish population-wide biological effects.',
        confidenceInterval: '95% Wilson interval for held-out trial accuracy at each condition/delay, not an interval over biological animals or independent delay samples.'});
    }
  }
  const api = {MemoryExperiment, makeMemoryPlan, summarizeMemoryTrials, CONDITIONS, DEFAULTS, LABELS: ALPHABET};
  root.FlyHamletMemory = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
