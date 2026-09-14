/* Two-letter state decoding in a fixed LIF network, with no external history.
 * Two letter cues are presented in order, input is removed, and frozen per-position
 * decoders read one quantized snapshot of membrane voltage and synaptic current from
 * nonstimulated neurons. The same pair/noise trials are paired across the original
 * dynamics, an explicit slower-dynamics hypothesis, and a reset control.
 */
(function (root) {
  'use strict';
  const live = root.FlyHamletLive || (typeof require === 'function' ? require('./live-model.js') : null);
  const recall = root.FlyHamletRecall || (typeof require === 'function' ? require('./recall-model.js') : null);
  if (!live || !recall) throw new Error('Load live-model.js and recall-model.js before sequence-model.js.');
  const {LIFNetwork, SeededRandom} = live;
  const {createCodebook, createPools, fitRidge, predictLinear, deriveSeed, fingerprint} = recall;
  const LETTERS = Object.freeze(['t', 'o']);
  const PAIRS = Object.freeze(['tt', 'to', 'ot', 'oo']);
  const POSITIONS = Object.freeze(['first', 'second']);
  const CONDITIONS = Object.freeze(['original', 'slow', 'reset']);
  const DEFAULTS = Object.freeze({trainPerPair: 16, evaluationPerPair: 16, letterMs: 100, gapMs: 25,
    delaysMs: Object.freeze([25, 100, 200]), poolCount: 128, cueRateHz: 100, ridge: 0.01,
    stateResolutionMv: 0.01, slowTimeFactor: 10, slowWeightScale: 0.1,
    codebookSeed: 7919, poolSeed: 104729});
  const clone = value => JSON.parse(JSON.stringify(value));

  function makeSequencePlan(seed, protocol) {
    const plan = [];
    ['train', 'evaluation'].forEach((phase, phaseIndex) => {
      const perPair = protocol[phase === 'train' ? 'trainPerPair' : 'evaluationPerPair'];
      if (!Number.isInteger(perPair) || perPair < 1) throw new Error('Each phase needs a positive integer number of trials per pair.');
      const pairs = Array.from({length: perPair}, () => [...PAIRS]).flat();
      const rng = new SeededRandom(deriveSeed(seed, 71, phaseIndex));
      for (let i = pairs.length - 1; i > 0; --i) {
        const j = Math.floor(rng.next() * (i + 1)); [pairs[i], pairs[j]] = [pairs[j], pairs[i]];
      }
      pairs.forEach((pair, index) => plan.push(Object.freeze({phase, pair, first: pair[0], second: pair[1],
        phaseTrial: index + 1, phaseTotal: pairs.length, pairedTrial: plan.length + 1,
        neuralSeed: deriveSeed(seed, 73, plan.length)})));
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
  // Slower dynamics: membrane and synaptic time constants are multiplied by one
  // factor, and every signed synaptic weight by one scale. With scale = 1/factor
  // the time integral of each synaptic event's voltage response is unchanged
  // while its peak amplitude falls by the factor. Delay and refractory period are
  // not scaled. These are imposed engineering parameters, not fly physiology.
  function slowNeuralConfig(sim, factor) {
    if (!(factor > 0) || !Number.isFinite(factor)) throw new Error('Invalid slow time factor.');
    return {...sim, tau_mem_ms: sim.tau_mem_ms * factor, tau_syn_ms: sim.tau_syn_ms * factor};
  }
  function prepareSlowGraph(graph, scale) {
    if (!(scale > 0) || !Number.isFinite(scale)) throw new Error('Invalid slow weight scale.');
    const weights = new Float32Array(graph.weights.length);
    let maxAbsBefore = 0, maxAbsAfter = 0;
    for (let i = 0; i < weights.length; ++i) {
      weights[i] = Math.fround(graph.weights[i] * scale);
      maxAbsBefore = Math.max(maxAbsBefore, Math.abs(graph.weights[i]));
      maxAbsAfter = Math.max(maxAbsAfter, Math.abs(weights[i]));
    }
    return {graph: Object.freeze({n: graph.n, indptr: graph.indptr, indices: graph.indices, weights}),
      metadata: {weightScale: scale, edgeSlots: weights.length, maxAbsWeightMv: maxAbsBefore, maxAbsScaledWeightMv: maxAbsAfter,
        topology: 'Identical row pointers and target indices; only weight magnitudes change.',
        rationale: 'Each synaptic event keeps its integrated voltage response when weights are scaled by 1/factor and time constants by factor.'}};
  }
  function fitScaler(rows) {
    if (!rows.length) throw new Error('A scaler needs training rows.');
    const dimensions = rows[0].length;
    const mean = new Float64Array(dimensions), scale = new Float64Array(dimensions);
    for (const row of rows) {
      if (row.length !== dimensions) throw new Error('Scaler rows must have equal dimensions.');
      for (let i = 0; i < dimensions; ++i) mean[i] += row[i] / rows.length;
    }
    let constant = 0;
    for (let i = 0; i < dimensions; ++i) {
      let variance = 0;
      for (const row of rows) variance += (row[i] - mean[i]) ** 2 / rows.length;
      const std = Math.sqrt(variance);
      // Exactly constant training features carry no information and are zeroed
      // rather than amplified. Anything else is standardized to unit variance.
      if (std <= 1e-12 * Math.max(1, Math.abs(mean[i]))) { scale[i] = 0; constant++; } else scale[i] = 1 / std;
    }
    return {mean: Array.from(mean), scale: Array.from(scale), dimensions, constantFeatures: constant,
      fingerprint: fingerprint([...mean, ...scale])};
  }
  function applyScaler(scaler, measurements) {
    if (measurements.length !== scaler.mean.length) throw new Error('Scaler dimensions do not match the measurement.');
    return Array.from(measurements, (value, i) => (value - scaler.mean[i]) * scaler.scale[i]);
  }
  function summarizeSequenceTrials(trials, delaysMs = DEFAULTS.delaysMs) {
    const metrics = {};
    for (const condition of CONDITIONS) {
      const rows = trials.filter(trial => trial.condition === condition && trial.phase === 'evaluation');
      const summarize = (delayMs, inputActive) => {
        const byPair = Object.fromEntries(PAIRS.map(pair => [pair, {correct: 0, total: 0, accuracy: null}]));
        const confusion = Object.fromEntries(PAIRS.map(pair => [pair, Object.fromEntries(PAIRS.map(prediction => [prediction, 0]))]));
        const positions = Object.fromEntries(POSITIONS.map(position => [position, {correct: 0, total: 0}]));
        let correct = 0, total = 0, silentTrials = 0, nonrestVoltage = 0, nonrestCurrent = 0, spikesSinceCueOff = 0;
        for (const trial of rows) {
          const snapshot = inputActive ? trial.cueEnd : trial.snapshots.find(item => item.delayMs === delayMs);
          if (!snapshot || !PAIRS.includes(snapshot.pairPrediction)) continue;
          const success = snapshot.pairPrediction === trial.pair;
          correct += Number(success); total++;
          byPair[trial.pair].correct += Number(success); byPair[trial.pair].total++;
          confusion[trial.pair][snapshot.pairPrediction]++;
          for (const position of POSITIONS) {
            positions[position].correct += Number(snapshot.predictions[position].correct); positions[position].total++;
          }
          silentTrials += Number(snapshot.silent); nonrestVoltage += snapshot.nonrestVoltageNeurons;
          nonrestCurrent += snapshot.nonrestCurrentNeurons; spikesSinceCueOff += snapshot.spikesSinceCueOff;
        }
        for (const row of Object.values(byPair)) if (row.total) row.accuracy = row.correct / row.total;
        const score = (c, t) => ({correct: c, total: t, accuracy: t ? c / t : null, wilson95: wilson(c, t)});
        return {delayMs, inputActive, ...score(correct, total), chance: 1 / PAIRS.length, lastLetterOnly: 1 / LETTERS.length,
          balancedAccuracy: PAIRS.every(pair => byPair[pair].total) ? PAIRS.reduce((sum, pair) => sum + byPair[pair].accuracy, 0) / PAIRS.length : null,
          first: {...score(positions.first.correct, positions.first.total), chance: 1 / LETTERS.length},
          second: {...score(positions.second.correct, positions.second.total), chance: 1 / LETTERS.length},
          byPair, confusion, silentTrials,
          meanNonrestVoltageNeurons: total ? nonrestVoltage / total : null,
          meanNonrestCurrentNeurons: total ? nonrestCurrent / total : null,
          meanSpikesSinceCueOff: total ? spikesSinceCueOff / total : null};
      };
      metrics[condition] = {cueEnd: {...summarize(0, true), diagnostic: true}, delays: delaysMs.map(delay => summarize(delay, false))};
    }
    return metrics;
  }

  class SequenceExperiment {
    constructor(graph, manifest, seed, options = {}, prepared = null) {
      this.graph = graph; this.manifest = manifest; this.seed = Number(seed) >>> 0;
      const protocol = Object.fromEntries(Object.entries(DEFAULTS).map(([key, value]) => [key, options[key] ?? value]));
      if (!Array.isArray(protocol.delaysMs) || !protocol.delaysMs.length ||
          protocol.delaysMs.some(value => !Number.isFinite(value) || value <= 0) ||
          new Set(protocol.delaysMs).size !== protocol.delaysMs.length)
        throw new Error('Snapshot delays must be distinct positive finite numbers.');
      protocol.delaysMs = Object.freeze([...protocol.delaysMs].sort((a, b) => a - b));
      this.protocol = Object.freeze(protocol);
      const p = this.protocol;
      for (const key of ['letterMs', 'cueRateHz', 'ridge', 'stateResolutionMv', 'slowTimeFactor', 'slowWeightScale'])
        if (!(p[key] > 0) || !Number.isFinite(p[key])) throw new Error(`Invalid ${key}.`);
      if (!(p.gapMs >= 0) || !Number.isFinite(p.gapMs)) throw new Error('Invalid gapMs.');
      if (!Number.isInteger(p.poolCount) || p.poolCount < 1) throw new Error('Invalid pool count.');
      this.dtMs = Number(manifest.config.sim.dt_ms);
      if (!(this.dtMs > 0) || !Number.isFinite(this.dtMs) || p.cueRateHz * this.dtMs / 1000 > 1)
        throw new Error('Invalid neural timestep or cue probability.');
      const background = manifest.config.sim.background;
      if (background?.enabled && background.rate_hz > 0)
        throw new Error('This cue-off assay requires background input to be disabled.');
      const integerSteps = value => {
        const steps = Math.round(value / this.dtMs);
        if (Math.abs(steps * this.dtMs - value) > 1e-9) throw new Error('All cue and snapshot timings must be exact neural timestep multiples.');
        return steps;
      };
      this.letterSteps = integerSteps(p.letterMs); this.gapSteps = integerSteps(p.gapMs);
      this.secondStartStep = this.letterSteps + this.gapSteps;
      this.cueSteps = this.secondStartStep + this.letterSteps;
      this.cueOffMs = this.cueSteps * this.dtMs;
      this.snapshotPlan = p.delaysMs.map(delayMs => ({delayMs, step: this.cueSteps + integerSteps(delayMs),
        timeMs: this.cueOffMs + delayMs, inputActive: false}));
      this.cueEndPlan = {delayMs: 0, step: this.cueSteps, timeMs: this.cueOffMs, inputActive: true};
      this.observationPlan = [this.cueEndPlan, ...this.snapshotPlan];
      this.stepsPerTrial = Math.max(...this.snapshotPlan.map(item => item.step));
      this.slowConfig = Object.freeze(slowNeuralConfig(manifest.config.sim, p.slowTimeFactor));
      if (prepared && (prepared.graph?.n !== graph.n || prepared.graph.weights?.length !== graph.weights.length ||
          prepared.metadata?.weightScale !== p.slowWeightScale))
        throw new Error('The prepared slow graph must match this graph and the protocol weight scale.');
      const slow = prepared || prepareSlowGraph(graph, p.slowWeightScale);
      this.slowGraph = slow.graph; this.slowMetadata = clone(slow.metadata);
      this.plan = makeSequencePlan(this.seed, p);
      // Reuse the fixed sensory codebook from the recall assay. Only the t and o
      // partitions are stimulated; all eye cells are excluded from measurements.
      const allCodes = createCodebook(manifest, p.codebookSeed);
      this.codebook = Object.fromEntries(LETTERS.map(code => [code, allCodes[code]]));
      const excluded = new Set(Object.values(allCodes).flat());
      for (const index of excluded)
        if (!Number.isInteger(index) || index < 0 || index >= graph.n) throw new Error('Invalid sensory neuron.');
      const pools = createPools(graph.n, excluded, p.poolCount, p.poolSeed);
      this.poolForNeuron = pools.poolForNeuron; this.poolSizes = pools.sizes;
      this.readoutNeuronCount = graph.n - excluded.size;
      if (!this.readoutNeuronCount) throw new Error('No nonstimulated readout neurons remain.');
      this.measurementCount = 2 * p.poolCount;
      this.conditionIndex = 0; this.planIndex = 0; this.trials = []; this.totalSteps = 0;
      this.trialStep = 0; this.net = null; this.inputs = null; this.cueOffsetApplied = false;
      this.cueSpikes = 0; this.postCueSpikes = 0; this.resetSeed = null; this.observations = [];
      this.trainingRows = this.observationPlan.map(() => []); this.trainingLabels = [];
      this.models = {}; this.fitCount = 0;
      this.metadata = {seed: this.seed, n: graph.n, edgeCount: graph.indices.length, layout: [...manifest.layout],
        letters: [...LETTERS], pairs: [...PAIRS], positions: [...POSITIONS], codebook: clone(this.codebook),
        unusedCodes: Object.fromEntries(Object.entries(allCodes).filter(([code]) => !LETTERS.includes(code)).map(([code, cells]) => [code, [...cells]])),
        sensoryExcludedCount: excluded.size, readoutNeuronCount: this.readoutNeuronCount,
        poolSizes: Array.from(this.poolSizes), measurementCount: this.measurementCount, conditions: [...CONDITIONS],
        cueOffMs: this.cueOffMs, trialMs: this.stepsPerTrial * this.dtMs,
        pooling: 'Fixed index-hash pools of nonstimulated neurons; all annotated eye cells excluded.',
        measurementDefinition: `Per neuron, membrane voltage minus rest and synaptic current are rounded to the nearest ${p.stateResolutionMv} mV; each pool reports the mean of those rounded values. Features are the ${p.poolCount} voltage pools followed by the ${p.poolCount} current pools.`,
        quantizationRationale: 'Float32 decay can freeze microvolt-scale deviations that never return to rest. A fixed measurement resolution keeps such numerical remnants out of the decoders.',
        scaling: 'Each snapshot has one scaler with training-only means and standard deviations; exactly constant training features are zeroed. Scalers and decoder weights are frozen before evaluation.',
        featureDefinition: 'One current-state snapshot of nonstimulated neurons; no earlier snapshot, cue label, clock, trial index or previous output enters a decoder.',
        inputTiming: `First letter ${p.letterMs} ms, gap ${p.gapMs} ms, second letter ${p.letterMs} ms; both inputs are removed before the first post-cue step. Reset replaces all neural state and its RNG at cue offset.`,
        pairedDesign: 'Exactly the same balanced pair order and neural noise seeds are used for every condition; training and evaluation use distinct trials.',
        slowModel: {timeFactor: p.slowTimeFactor, weightScale: p.slowWeightScale, tauMemMs: this.slowConfig.tau_mem_ms,
          tauSynMs: this.slowConfig.tau_syn_ms, unscaled: ['delay_ms', 't_refractory_ms', 'poisson_weight_mV', 'thresholds'],
          status: 'Imposed engineering hypothesis; not identified from fruit-fly memory data.'},
        model: 'Fixed full-connectome LIF two-letter state decoding; external history absent',
        rng: 'xoshiro128**; independent pair-shuffle, neural-noise and reset domains'};
    }
    get done() { return this.conditionIndex >= CONDITIONS.length; }
    get condition() { return CONDITIONS[this.conditionIndex] ?? null; }
    get simTimeS() { return this.totalSteps * this.dtMs / 1000; }
    stageAt(step) {
      if (step < this.letterSteps) return 'first';
      if (step < this.secondStartStep) return 'gap';
      if (step < this.cueSteps) return 'second';
      return 'delay';
    }
    get progress() {
      const item = this.plan[Math.min(this.planIndex, this.plan.length - 1)];
      const stage = this.done ? 'complete' : this.stageAt(this.trialStep);
      const cue = stage === 'first' ? item.first : stage === 'second' ? item.second : null;
      return {condition: this.condition, phase: this.done ? 'complete' : item.phase,
        conditionIndex: Math.min(this.conditionIndex + 1, CONDITIONS.length), conditionTotal: CONDITIONS.length,
        conditionTrial: Math.min(this.planIndex + 1, this.plan.length), phaseTrial: item.phaseTrial,
        phaseTotal: item.phaseTotal, completed: this.trials.length, totalTrials: this.plan.length * CONDITIONS.length,
        simTimeS: this.simTimeS, simTime: this.simTimeS, trialTimeMs: this.trialStep * this.dtMs, trialMs: this.stepsPerTrial * this.dtMs,
        stage, pair: this.done ? null : item.pair, cue, inputActive: cue !== null,
        snapshotsTaken: this.observations.length, updateCount: this.fitCount};
    }
    startTrial() {
      const item = this.plan[this.planIndex];
      const original = this.condition === 'original';
      this.net = new LIFNetwork(original ? this.graph : this.slowGraph, original ? this.manifest.config.sim : this.slowConfig, item.neuralSeed);
      // Both letter inputs exist in every trial so that unstimulated eye cells are
      // treated identically across trials; only the presented letter has a rate.
      this.inputs = Object.fromEntries(LETTERS.map(letter => [letter, this.net.injectPoisson(this.codebook[letter], 0)]));
      this.trialStep = 0; this.cueOffsetApplied = false; this.cueSpikes = 0; this.postCueSpikes = 0;
      this.resetSeed = null; this.observations = [];
    }
    setLetter(letter) {
      for (const code of LETTERS) this.net.setPoissonRate(this.inputs[code], code === letter ? this.protocol.cueRateHz : 0);
    }
    removeCue() {
      if (this.condition === 'reset') {
        this.resetSeed = deriveSeed(this.plan[this.planIndex].neuralSeed, 79);
        this.net = new LIFNetwork(this.slowGraph, this.slowConfig, this.resetSeed);
        this.inputs = null;
      } else this.setLetter(null);
      this.cueOffsetApplied = true;
    }
    // Read the current state of every nonstimulated neuron, including retired
    // fixed points, at a fixed resolution. Pool means use exact integer quanta.
    measure(plan) {
      const net = this.net, res = this.protocol.stateResolutionMv, count = this.protocol.poolCount;
      const voltageQuanta = new Float64Array(count), currentQuanta = new Float64Array(count);
      let nonrestVoltage = 0, nonrestCurrent = 0, maxAbsVoltage = 0, maxAbsCurrent = 0;
      for (let neuron = 0; neuron < net.n; ++neuron) {
        const pool = this.poolForNeuron[neuron];
        if (pool < 0) continue;
        const dv = net.v[neuron] - net.v0, g = net.g[neuron];
        const qv = Math.round(dv / res), qg = Math.round(g / res);
        if (qv !== 0) { nonrestVoltage++; voltageQuanta[pool] += qv; maxAbsVoltage = Math.max(maxAbsVoltage, Math.abs(qv)); }
        if (qg !== 0) { nonrestCurrent++; currentQuanta[pool] += qg; maxAbsCurrent = Math.max(maxAbsCurrent, Math.abs(qg)); }
      }
      const measurements = new Array(2 * count);
      for (let pool = 0; pool < count; ++pool) {
        const size = this.poolSizes[pool];
        measurements[pool] = size ? voltageQuanta[pool] * res / size : 0;
        measurements[count + pool] = size ? currentQuanta[pool] * res / size : 0;
      }
      return {delayMs: plan.delayMs, timeMs: plan.timeMs, inputActive: plan.inputActive, measurements,
        nonrestVoltageNeurons: nonrestVoltage, nonrestCurrentNeurons: nonrestCurrent,
        maxAbsVoltageMv: maxAbsVoltage * res, maxAbsCurrentMv: maxAbsCurrent * res,
        silent: nonrestVoltage === 0 && nonrestCurrent === 0,
        spikesSinceCueOff: plan.inputActive ? 0 : this.postCueSpikes, activeNeurons: net.activeCount,
        features: null, predictions: null, pairPrediction: null, pairCorrect: null};
    }
    step() {
      if (this.done) return null;
      if (!this.net) this.startTrial();
      const item = this.plan[this.planIndex];
      if (this.trialStep === 0) this.setLetter(item.first);
      else if (this.trialStep === this.letterSteps) this.setLetter(null);
      else if (this.trialStep === this.secondStartStep) this.setLetter(item.second);
      else if (this.trialStep === this.cueSteps && !this.cueOffsetApplied) {
        this.observations.push(this.measure(this.cueEndPlan));
        this.removeCue();
      }
      const spikes = this.net.step();
      if (this.trialStep < this.cueSteps) this.cueSpikes += spikes.length;
      else { if (!this.cueOffsetApplied) throw new Error('A post-cue step ran with input still active.'); this.postCueSpikes += spikes.length; }
      this.trialStep++; this.totalSteps++;
      for (const plan of this.snapshotPlan) if (plan.step === this.trialStep) this.observations.push(this.measure(plan));
      if (this.trialStep < this.stepsPerTrial) return null;
      return this.finishTrial();
    }
    finishTrial() {
      const item = this.plan[this.planIndex], condition = this.condition;
      if (this.observations.length !== this.observationPlan.length) throw new Error('A trial finished with missing snapshots.');
      const before = this.fitCount;
      if (item.phase === 'train') {
        this.observations.forEach((snapshot, index) => this.trainingRows[index].push(snapshot.measurements));
        this.trainingLabels.push({first: item.first, second: item.second});
        if (item.phaseTrial === item.phaseTotal) this.fitReadouts(condition);
      } else {
        // Every prediction is computed from the snapshot, its frozen scaler and
        // frozen weights before the evaluator compares it with the removed cues.
        this.observations.forEach((snapshot, index) => {
          const readout = this.models[condition].readouts[index];
          snapshot.features = applyScaler(readout.scaler, snapshot.measurements);
          snapshot.predictions = Object.fromEntries(POSITIONS.map(position => {
            const fit = readout.positions[position];
            const decision = predictLinear(fit.weights, snapshot.features, LETTERS);
            return [position, {prediction: decision.prediction, scores: decision.scores,
              correct: decision.prediction === item[position], decoderFingerprint: fit.fingerprint}];
          }));
          snapshot.scalerFingerprint = readout.scaler.fingerprint;
          snapshot.pairPrediction = snapshot.predictions.first.prediction + snapshot.predictions.second.prediction;
          snapshot.pairCorrect = snapshot.pairPrediction === item.pair;
        });
      }
      const [cueEnd, ...snapshots] = this.observations;
      const record = {trial: this.trials.length + 1, condition, ...item, cueEnd, snapshots,
        cueOffsetMs: this.cueOffMs, inputsOffFromMs: this.cueOffMs,
        firstLetterMs: [0, this.protocol.letterMs], secondLetterMs: [this.secondStartStep * this.dtMs, this.cueOffMs],
        cueSpikes: this.cueSpikes, postCueSpikes: this.postCueSpikes, spikes: this.cueSpikes + this.postCueSpikes,
        resetSeed: this.resetSeed, simTimeS: this.simTimeS, decoderVersion: item.phase === 'evaluation' ? 1 : 0,
        updateCountBefore: before, updateCountAfter: this.fitCount, fitApplied: this.fitCount !== before,
        feedbackApplied: false};
      this.trials.push(record);
      this.net = null; this.inputs = null; this.trialStep = 0; this.observations = [];
      this.planIndex++;
      if (this.planIndex === this.plan.length) {
        this.conditionIndex++; this.planIndex = 0;
        this.trainingRows = this.observationPlan.map(() => []); this.trainingLabels = [];
      }
      return clone(record);
    }
    fitReadouts(condition) {
      const readouts = this.trainingRows.map((rows, index) => {
        const scaler = fitScaler(rows);
        const standardized = rows.map(row => applyScaler(scaler, row));
        const positions = Object.fromEntries(POSITIONS.map(position => {
          const fit = fitRidge(standardized, this.trainingLabels.map(label => label[position]), this.protocol.ridge, LETTERS);
          const weights = Object.freeze([...fit.weights]);
          return [position, {weights, frozenWeights: weights, fingerprint: fingerprint(weights), dimensions: fit.dimensions, labels: [...LETTERS]}];
        }));
        const plan = this.observationPlan[index];
        return {delayMs: plan.delayMs, timeMs: plan.timeMs, inputActive: plan.inputActive, scaler, positions};
      });
      this.models[condition] = {fitCount: 1, trainingExamples: this.trainingLabels.length, readouts};
      this.fitCount++;
    }
    result() {
      const models = Object.fromEntries(Object.entries(this.models).map(([condition, model]) => [condition,
        {...model, readouts: model.readouts.map(readout => ({...readout,
          scaler: {...readout.scaler, finalFingerprint: fingerprint([...readout.scaler.mean, ...readout.scaler.scale])},
          positions: Object.fromEntries(POSITIONS.map(position => [position,
            {...readout.positions[position], finalFingerprint: fingerprint(readout.positions[position].weights)}]))}))}]));
      return clone({format: 'flyhamlet-sequence-v1', complete: this.done, seed: this.seed,
        config: this.protocol, metadata: this.metadata, neuralConfig: this.manifest.config.sim,
        slowNeuralConfig: this.slowConfig, slowGraph: this.slowMetadata,
        provenance: this.manifest.provenance || null, connectivityIntegrity: this.manifest.integrity || null,
        simTimeS: this.simTimeS, fitCount: this.fitCount,
        readoutFitCount: this.fitCount * this.observationPlan.length * POSITIONS.length, models, trials: this.trials,
        metrics: summarizeSequenceTrials(this.trials, this.protocol.delaysMs),
        cueEndDiagnostic: 'The delay-0 snapshot is read at the final cue step, before input removal or reset. It diagnoses encoding while input is still on and is never combined with post-cue snapshots.',
        intervention: 'Stimulate two fixed sensory codes in order, remove input, and decode both letters from one quantized voltage/current snapshot with independent frozen ridge readouts; no external history.',
        conditionDefinitions: {original: 'Original graph and time constants, neural state retained at cue offset.',
          slow: 'Slower time constants with rescaled weights, neural state retained at cue offset.',
          reset: 'Slower model with identical cues; all dynamic neural state and RNG replaced with rest at cue offset.'},
        chance: {pair: 1 / PAIRS.length, position: 1 / LETTERS.length, lastLetterOnlyPair: 1 / LETTERS.length},
        snapshotDependence: 'All delays are read in the same trials; delay points are correlated observations.',
        inferenceLimit: 'This measures linearly decodable letter-order information in quantized subthreshold and synaptic state under this stimulation. Reading internal voltages is more permissive than spike decoding. The slower model is an imposed hypothesis; success there does not describe fruit-fly physiology, and no result here demonstrates autonomous typing or memory for a text.',
        confidenceInterval: '95% Wilson interval for held-out trial accuracy at each condition/snapshot, not an interval over biological animals or independent delay samples.'});
    }
  }
  const api = {SequenceExperiment, makeSequencePlan, summarizeSequenceTrials, prepareSlowGraph, slowNeuralConfig,
    fitScaler, applyScaler, wilson, CONDITIONS, POSITIONS, LETTERS, PAIRS, DEFAULTS};
  root.FlyHamletSequence = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
