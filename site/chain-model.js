/* Self-driven phrase recall from continuous neural state, with no external history.
 * The network runs without reset across a whole episode. After each letter cue and a
 * short gap, one quantized snapshot of nonstimulated membrane voltage and synaptic
 * current is decoded into the next letter, and that letter becomes the next cue.
 * Conditions: original dynamics, the slower-dynamics hypothesis, and a reset control
 * of the slower model whose state is cleared at every cue onset.
 */
(function (root) {
  'use strict';
  const live = root.FlyHamletLive || (typeof require === 'function' ? require('./live-model.js') : null);
  const recall = root.FlyHamletRecall || (typeof require === 'function' ? require('./recall-model.js') : null);
  const sequence = root.FlyHamletSequence || (typeof require === 'function' ? require('./sequence-model.js') : null);
  if (!live || !recall || !sequence) throw new Error('Load live-model.js, recall-model.js and sequence-model.js before chain-model.js.');
  const {LIFNetwork} = live;
  const {ALPHABET, INPUT_CODES, OUTPUT_CODES, START, END, createCodebook, createPools, fitRidge, predictLinear,
    deriveSeed, fingerprint, editDistance, oneHot, conventionalRollout} = recall;
  const {prepareSlowGraph, slowNeuralConfig, fitScaler, applyScaler, wilson} = sequence;
  const CONDITIONS = Object.freeze(['original', 'slow', 'reset']);
  const PHASES = Object.freeze(['train', 'diagnostic', 'recall']);
  const DEFAULTS = Object.freeze({phrase: 'to be or not to be', trainEpisodes: 6, diagnosticEpisodes: 3,
    recallEpisodes: 6, maxDecisions: 32, cueMs: 100, gapMs: 25, poolCount: 128, cueRateHz: 100, ridge: 0.01,
    stateResolutionMv: 0.01, slowTimeFactor: 10, slowWeightScale: 0.1, codebookSeed: 7919, poolSeed: 104729});
  const clone = value => JSON.parse(JSON.stringify(value));
  const SEED_DOMAINS = Object.freeze({train: 83, diagnostic: 89, recall: 97});

  function chainLength(output, reference) {
    let length = 0;
    while (length < output.length && length < reference.length && output[length] === reference[length]) length++;
    return length;
  }
  function summarizeChainEpisode(condition, phase, episode, output, stoppedBy, decisions, reference) {
    const prefix = chainLength(output, reference);
    const exact = output === reference && stoppedBy === END;
    return {condition, phase, episode, output, stoppedBy, decisions, exact, editDistance: editDistance(output, reference),
      chainLength: prefix, completed: exact, correctCharacters: prefix + Number(exact)};
  }
  // Mirrors the Sequence lab measurement: per-neuron rounding at a fixed resolution,
  // pool means from exact integer quanta, all neurons visited including retired fixed points.
  function measureState(net, poolForNeuron, poolSizes, count, resolution) {
    const voltageQuanta = new Float64Array(count), currentQuanta = new Float64Array(count);
    let nonrestVoltage = 0, nonrestCurrent = 0, maxAbsVoltage = 0, maxAbsCurrent = 0;
    for (let neuron = 0; neuron < net.n; ++neuron) {
      const pool = poolForNeuron[neuron];
      if (pool < 0) continue;
      const qv = Math.round((net.v[neuron] - net.v0) / resolution), qg = Math.round(net.g[neuron] / resolution);
      if (qv !== 0) { nonrestVoltage++; voltageQuanta[pool] += qv; maxAbsVoltage = Math.max(maxAbsVoltage, Math.abs(qv)); }
      if (qg !== 0) { nonrestCurrent++; currentQuanta[pool] += qg; maxAbsCurrent = Math.max(maxAbsCurrent, Math.abs(qg)); }
    }
    const measurements = new Array(2 * count);
    for (let pool = 0; pool < count; ++pool) {
      const size = poolSizes[pool];
      measurements[pool] = size ? voltageQuanta[pool] * resolution / size : 0;
      measurements[count + pool] = size ? currentQuanta[pool] * resolution / size : 0;
    }
    return {measurements, nonrestVoltageNeurons: nonrestVoltage, nonrestCurrentNeurons: nonrestCurrent,
      maxAbsVoltageMv: maxAbsVoltage * resolution, maxAbsCurrentMv: maxAbsCurrent * resolution,
      silent: nonrestVoltage === 0 && nonrestCurrent === 0, activeNeurons: net.activeCount};
  }
  function summarizeChainRun(episodes, trials, reference) {
    const metrics = {};
    for (const condition of CONDITIONS) {
      const recallEpisodes = episodes.filter(item => item.condition === condition && item.phase === 'recall');
      const lengths = recallEpisodes.map(item => item.chainLength);
      const diagnostic = trials.filter(trial => trial.condition === condition && trial.phase === 'diagnostic' && trial.prediction !== null);
      const byStep = {};
      for (const trial of diagnostic) {
        const row = byStep[trial.step] || (byStep[trial.step] = {step: trial.step, target: trial.target, correct: 0, total: 0});
        row.correct += Number(trial.correct); row.total++;
      }
      const correct = diagnostic.filter(trial => trial.correct).length;
      const activity = phase => {
        const rows = trials.filter(trial => trial.condition === condition && trial.phase === phase);
        return {decisions: rows.length, meanCueSpikes: rows.length ? rows.reduce((sum, trial) => sum + trial.cueSpikes, 0) / rows.length : null,
          meanNonrestVoltageNeurons: rows.length ? rows.reduce((sum, trial) => sum + trial.nonrestVoltageNeurons, 0) / rows.length : null,
          silentSnapshots: rows.filter(trial => trial.silent).length};
      };
      metrics[condition] = {
        recall: {episodes: recallEpisodes.length, exact: recallEpisodes.filter(item => item.exact).length,
          chainLengths: lengths, meanChainLength: lengths.length ? lengths.reduce((sum, value) => sum + value, 0) / lengths.length : null,
          maxChainLength: lengths.length ? Math.max(...lengths) : null,
          meanEditDistance: recallEpisodes.length ? recallEpisodes.reduce((sum, item) => sum + item.editDistance, 0) / recallEpisodes.length : null,
          ended: recallEpisodes.filter(item => item.stoppedBy === END).length, capped: recallEpisodes.filter(item => item.stoppedBy === 'cap').length,
          referenceLength: reference.length, activity: activity('recall')},
        diagnostic: {correct, total: diagnostic.length, accuracy: diagnostic.length ? correct / diagnostic.length : null,
          wilson95: wilson(correct, diagnostic.length), chance: 1 / OUTPUT_CODES.length,
          byStep: Object.values(byStep).sort((a, b) => a.step - b.step), activity: activity('diagnostic')}};
    }
    return metrics;
  }

  class ChainExperiment {
    constructor(graph, manifest, seed, options = {}, prepared = null) {
      this.graph = graph; this.manifest = manifest; this.seed = Number(seed) >>> 0;
      this.protocol = Object.freeze(Object.fromEntries(Object.entries(DEFAULTS).map(([key, value]) => [key, options[key] ?? value])));
      const p = this.protocol;
      if (typeof p.phrase !== 'string' || !p.phrase.length || Array.from(p.phrase).some(letter => !ALPHABET.includes(letter)))
        throw new Error('The training phrase must use the fixed seven-character alphabet.');
      for (const key of ['trainEpisodes', 'recallEpisodes', 'maxDecisions', 'poolCount'])
        if (!Number.isInteger(p[key]) || p[key] < 1) throw new Error(`Invalid ${key}.`);
      if (!Number.isInteger(p.diagnosticEpisodes) || p.diagnosticEpisodes < 0) throw new Error('Invalid diagnosticEpisodes.');
      for (const key of ['cueMs', 'cueRateHz', 'ridge', 'stateResolutionMv', 'slowTimeFactor', 'slowWeightScale'])
        if (!(p[key] > 0) || !Number.isFinite(p[key])) throw new Error(`Invalid ${key}.`);
      if (!(p.gapMs >= 0) || !Number.isFinite(p.gapMs)) throw new Error('Invalid gapMs.');
      this.dtMs = Number(manifest.config.sim.dt_ms);
      if (!(this.dtMs > 0) || !Number.isFinite(this.dtMs) || p.cueRateHz * this.dtMs / 1000 > 1)
        throw new Error('Invalid neural timestep or cue probability.');
      const background = manifest.config.sim.background;
      if (background?.enabled && background.rate_hz > 0) throw new Error('This assay requires background input to be disabled.');
      const integerSteps = value => {
        const steps = Math.round(value / this.dtMs);
        if (Math.abs(steps * this.dtMs - value) > 1e-9) throw new Error('Cue and gap timings must be exact neural timestep multiples.');
        return steps;
      };
      this.cueSteps = integerSteps(p.cueMs); this.gapSteps = integerSteps(p.gapMs);
      this.stepsPerDecision = this.cueSteps + this.gapSteps;
      this.decisionMs = this.stepsPerDecision * this.dtMs;
      this.slowConfig = Object.freeze(slowNeuralConfig(manifest.config.sim, p.slowTimeFactor));
      if (prepared && (prepared.graph?.n !== graph.n || prepared.graph.weights?.length !== graph.weights.length ||
          prepared.metadata?.weightScale !== p.slowWeightScale))
        throw new Error('The prepared slow graph must match this graph and the protocol weight scale.');
      const slow = prepared || prepareSlowGraph(graph, p.slowWeightScale);
      this.slowGraph = slow.graph; this.slowMetadata = clone(slow.metadata);
      this.codebook = createCodebook(manifest, p.codebookSeed);
      const excluded = new Set(Object.values(this.codebook).flat());
      for (const index of excluded)
        if (!Number.isInteger(index) || index < 0 || index >= graph.n) throw new Error('Invalid sensory neuron.');
      const pools = createPools(graph.n, excluded, p.poolCount, p.poolSeed);
      this.poolForNeuron = pools.poolForNeuron; this.poolSizes = pools.sizes;
      this.readoutNeuronCount = graph.n - excluded.size;
      if (!this.readoutNeuronCount) throw new Error('No nonstimulated readout neurons remain.');
      this.reference = p.phrase;
      this.trainingInputs = [START, ...p.phrase];
      this.trainingTargets = [...p.phrase, END];
      this.decisionsPerEpisode = this.trainingTargets.length;
      this.conditionIndex = 0; this.phase = 'train'; this.episode = 1; this.decisionStep = 1; this.cueStep = 0;
      this.net = null; this.inputs = null; this.cue = START; this.episodeSeed = null; this.cueSeed = null;
      this.cueSpikes = 0; this.episodeSpikes = 0; this.episodeOutput = ''; this.episodeDecisions = [];
      this.trainingRows = []; this.trainingLabels = []; this.models = {}; this.fitCount = 0;
      this.trials = []; this.episodes = []; this.totalSteps = 0;
      this.perConditionBudget = (p.trainEpisodes + p.diagnosticEpisodes) * this.decisionsPerEpisode + p.recallEpisodes * p.maxDecisions;
      this.totalTrialBudget = this.perConditionBudget * CONDITIONS.length;
      const comparatorRows = this.trainingInputs.map(code => oneHot(code));
      const comparatorFit = fitRidge(comparatorRows, this.trainingTargets, p.ridge);
      const rollout = conventionalRollout(comparatorFit, {historyLength: 1, maxDecisions: p.maxDecisions});
      this.comparator = {kind: 'Current-cue-only comparator: ridge on the one-hot current cue, own-feedback rollout, no state',
        deterministic: true, fitExamples: comparatorRows.length, weights: comparatorFit.weights, fingerprint: fingerprint(comparatorFit.weights),
        episode: summarizeChainEpisode('comparator', 'comparator', 1, rollout.output, rollout.stoppedBy, rollout.decisions, this.reference),
        note: 'The best a decoder can do from the current letter alone, without any memory of earlier letters.'};
      this.metadata = {seed: this.seed, n: graph.n, edgeCount: graph.indices.length, layout: [...manifest.layout],
        reference: this.reference, inputCodes: [...INPUT_CODES], outputCodes: [...OUTPUT_CODES], conditions: [...CONDITIONS], phases: [...PHASES],
        codebook: clone(this.codebook), sensoryExcludedCount: excluded.size, readoutNeuronCount: this.readoutNeuronCount,
        poolSizes: Array.from(this.poolSizes), measurementCount: 2 * p.poolCount, decisionMs: this.decisionMs,
        pooling: 'Fixed index-hash pools of nonstimulated neurons; all annotated eye cells excluded.',
        measurementDefinition: `Per neuron, membrane voltage minus rest and synaptic current are rounded to the nearest ${p.stateResolutionMv} mV; each pool reports the mean of those rounded values. Features are the ${p.poolCount} voltage pools followed by the ${p.poolCount} current pools.`,
        featureDefinition: 'One current-state snapshot read at the end of each cue-plus-gap period; no earlier snapshot, cue label, clock, step index or previous output enters the decoder.',
        continuity: 'In retained conditions one network runs through a whole episode without reset; the reset control replaces all neural state at every cue onset so each snapshot reflects only the current cue.',
        feedback: 'During recall the decoded letter is presented as the next sensory cue. END stops the episode; a fixed cap does not depend on the phrase length.',
        slowModel: {timeFactor: p.slowTimeFactor, weightScale: p.slowWeightScale, tauMemMs: this.slowConfig.tau_mem_ms, tauSynMs: this.slowConfig.tau_syn_ms,
          unscaled: ['delay_ms', 't_refractory_ms', 'poisson_weight_mV', 'thresholds'], status: 'Imposed engineering hypothesis; not identified from fruit-fly memory data.'},
        model: 'Fixed full-connectome LIF self-driven phrase recall from continuous state; external history absent',
        rng: 'xoshiro128**; independent episode and reset-cue noise domains'};
    }
    get done() { return this.conditionIndex >= CONDITIONS.length; }
    get condition() { return CONDITIONS[this.conditionIndex] ?? null; }
    get simTimeS() { return this.totalSteps * this.dtMs / 1000; }
    get episodeTotal() { return this.protocol[{train: 'trainEpisodes', diagnostic: 'diagnosticEpisodes', recall: 'recallEpisodes'}[this.phase]]; }
    get progress() {
      return {condition: this.condition, phase: this.done ? 'complete' : this.phase, conditionIndex: Math.min(this.conditionIndex + 1, CONDITIONS.length),
        conditionTotal: CONDITIONS.length, episode: this.episode, episodeTotal: this.done ? 0 : this.episodeTotal, step: this.decisionStep,
        cue: this.done ? null : this.cue, inputActive: !this.done && this.cueStep < this.cueSteps,
        stage: this.done ? 'complete' : (this.cueStep < this.cueSteps ? 'cue' : 'gap'), cueTimeMs: this.cueStep * this.dtMs, decisionMs: this.decisionMs,
        episodeOutput: this.episodeOutput, completed: this.trials.length, totalTrials: this.totalTrialBudget,
        simTimeS: this.simTimeS, simTime: this.simTimeS, updateCount: this.fitCount};
    }
    startCue() {
      const original = this.condition === 'original';
      if (this.decisionStep === 1) {
        this.episodeSeed = deriveSeed(this.seed, SEED_DOMAINS[this.phase], this.episode - 1);
        this.episodeSpikes = 0;
      }
      if (this.condition === 'reset' || this.decisionStep === 1) {
        this.cueSeed = this.condition === 'reset' ? deriveSeed(this.episodeSeed, 101, this.decisionStep - 1) : this.episodeSeed;
        this.net = new LIFNetwork(original ? this.graph : this.slowGraph, original ? this.manifest.config.sim : this.slowConfig, this.cueSeed);
        this.inputs = Object.fromEntries(INPUT_CODES.map(code => [code, this.net.injectPoisson(this.codebook[code], 0)]));
      }
      this.setCue(this.cue);
      this.cueStep = 0; this.cueSpikes = 0;
    }
    setCue(code) {
      for (const input of INPUT_CODES) this.net.setPoissonRate(this.inputs[input], input === code ? this.protocol.cueRateHz : 0);
    }
    step() {
      if (this.done) return null;
      if (this.cueStep === 0) this.startCue();
      else if (this.cueStep === this.cueSteps) this.setCue(null);
      const spikes = this.net.step();
      this.cueSpikes += spikes.length; this.episodeSpikes += spikes.length;
      this.cueStep++; this.totalSteps++;
      if (this.cueStep < this.stepsPerDecision) return null;
      return this.finishDecision();
    }
    finishDecision() {
      const condition = this.condition, phase = this.phase, episode = this.episode, step = this.decisionStep, cue = this.cue;
      const state = measureState(this.net, this.poolForNeuron, this.poolSizes, this.protocol.poolCount, this.protocol.stateResolutionMv);
      const before = this.fitCount;
      const record = {trial: this.trials.length + 1, condition, phase, episode, step, cue, target: null,
        episodeSeed: this.episodeSeed, cueSeed: this.cueSeed, snapshotMs: this.decisionMs, cueOffMs: this.protocol.cueMs, ...state,
        cueSpikes: this.cueSpikes, episodeSpikes: this.episodeSpikes, simTimeS: this.simTimeS, features: null, prediction: null, scores: null,
        correct: null, decoderVersion: phase === 'train' ? 0 : 1, decoderFingerprint: null, scalerFingerprint: null,
        feedbackApplied: phase === 'recall' && step > 1, updateCountBefore: before};
      let nextCue = null, episodeEnded = false;
      if (phase === 'train') {
        record.target = this.trainingTargets[step - 1];
        this.trainingRows.push(state.measurements); this.trainingLabels.push(record.target);
        nextCue = this.trainingInputs[step] ?? null;
        episodeEnded = step >= this.decisionsPerEpisode;
        if (episodeEnded && episode === this.protocol.trainEpisodes) this.fitReadout(condition);
      } else {
        const model = this.models[condition];
        record.features = applyScaler(model.scaler, state.measurements);
        const decision = predictLinear(model.weights, record.features, OUTPUT_CODES);
        record.prediction = decision.prediction; record.scores = decision.scores;
        record.decoderFingerprint = model.fingerprint; record.scalerFingerprint = model.scaler.fingerprint;
        if (phase === 'diagnostic') {
          record.target = this.trainingTargets[step - 1]; record.correct = decision.prediction === record.target;
          nextCue = this.trainingInputs[step] ?? null;
          episodeEnded = step >= this.decisionsPerEpisode;
        } else {
          this.episodeDecisions.push({step, cue, prediction: decision.prediction});
          if (decision.prediction !== END) this.episodeOutput += decision.prediction;
          episodeEnded = decision.prediction === END || step >= this.protocol.maxDecisions;
          if (episodeEnded) {
            this.episodes.push(summarizeChainEpisode(condition, phase, episode, this.episodeOutput,
              decision.prediction === END ? END : 'cap', clone(this.episodeDecisions), this.reference));
            this.totalTrialBudget -= this.protocol.maxDecisions - step;
          } else nextCue = decision.prediction;
        }
      }
      record.updateCountAfter = this.fitCount; record.fitApplied = this.fitCount !== before;
      this.trials.push(record);
      this.cueStep = 0;
      if (episodeEnded) this.nextEpisode(); else { this.decisionStep++; this.cue = nextCue; }
      return clone(record);
    }
    nextEpisode() {
      this.episode++; this.decisionStep = 1; this.cue = START; this.episodeOutput = ''; this.episodeDecisions = [];
      this.net = null; this.inputs = null;
      while (!this.done && this.episode > this.episodeTotal) {
        const index = PHASES.indexOf(this.phase);
        if (index + 1 < PHASES.length) { this.phase = PHASES[index + 1]; this.episode = 1; }
        else { this.conditionIndex++; this.phase = 'train'; this.episode = 1; this.trainingRows = []; this.trainingLabels = []; }
      }
    }
    fitReadout(condition) {
      const scaler = fitScaler(this.trainingRows);
      const fit = fitRidge(this.trainingRows.map(row => applyScaler(scaler, row)), this.trainingLabels, this.protocol.ridge, OUTPUT_CODES);
      const weights = Object.freeze([...fit.weights]);
      this.models[condition] = {fitCount: 1, trainingExamples: this.trainingLabels.length, scaler, weights, frozenWeights: weights,
        fingerprint: fingerprint(weights), dimensions: fit.dimensions, labels: [...OUTPUT_CODES]};
      this.fitCount++;
    }
    result() {
      const models = Object.fromEntries(Object.entries(this.models).map(([condition, model]) => [condition,
        {...model, finalFingerprint: fingerprint(model.weights),
          scaler: {...model.scaler, finalFingerprint: fingerprint([...model.scaler.mean, ...model.scaler.scale])}}]));
      return clone({format: 'flyhamlet-chain-v1', complete: this.done, seed: this.seed, reference: this.reference,
        config: this.protocol, metadata: this.metadata, neuralConfig: this.manifest.config.sim, slowNeuralConfig: this.slowConfig,
        slowGraph: this.slowMetadata, provenance: this.manifest.provenance || null, connectivityIntegrity: this.manifest.integrity || null,
        simTimeS: this.simTimeS, fitCount: this.fitCount, models, trials: this.trials, episodes: this.episodes, comparator: this.comparator,
        metrics: summarizeChainRun(this.episodes, this.trials, this.reference),
        conditionDefinitions: {original: 'Original graph and time constants; one network per episode, never reset between letters.',
          slow: 'Slower time constants with rescaled weights; one network per episode, never reset between letters.',
          reset: 'Slower model with identical cues; all dynamic neural state and RNG replaced with rest at every cue onset.'},
        intervention: 'Teacher-forced training episodes fit one scaler and one ridge readout per condition from continuous-state snapshots; frozen readouts then drive held-out teacher-forced diagnostics and autonomous own-feedback recall.',
        evaluation: 'Recall starts from START and feeds each decoded letter back as the next cue. Chain length is the longest correct prefix; END stops an episode and a fixed cap is independent of the phrase length. The reference is used only for training and post-hoc scoring.',
        inferenceLimit: 'Recall of one trained 18-character phrase by a fixed network plus an external readout. Reading internal voltages is more permissive than spike decoding, and the slower model is an imposed hypothesis. Success would show that this simulation can carry sequence position in its state; it would not show biological memory, language, or memory for a text.'});
    }
  }
  const api = {ChainExperiment, summarizeChainRun, summarizeChainEpisode, chainLength, measureState, CONDITIONS, PHASES, DEFAULTS, START, END};
  root.FlyHamletChain = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
