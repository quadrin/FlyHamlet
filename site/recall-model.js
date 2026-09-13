/* Phrase sequence recall with an external learned decoder and explicit memory.
 * The fixed LIF connectome encodes arbitrary sensory codes. Neither its synapses
 * nor the sensory codebook learn. Six previous/current neural feature vectors
 * are held in external software memory. This is not a model of fly language.
 */
(function (root) {
  'use strict';
  const live = root.FlyHamletLive || (typeof require === 'function' ? require('./live-model.js') : null);
  if (!live) throw new Error('Load live-model.js before recall-model.js.');
  const {LIFNetwork, SeededRandom} = live;
  const START = 'START', END = 'END';
  const ALPHABET = Object.freeze(['t', 'o', ' ', 'b', 'e', 'r', 'n']);
  const INPUT_CODES = Object.freeze([START, ...ALPHABET]);
  const OUTPUT_CODES = Object.freeze([...ALPHABET, END]);
  const DEFAULTS = Object.freeze({phrase: 'to be or not to be', trainEpisodes: 6,
    recallEpisodes: 3, ablatedEpisodes: 3, maxDecisions: 48, historyLength: 6,
    poolCount: 128, trialMs: 200, warmupMs: 50, cueRateHz: 100, ridge: 0.01,
    codebookSeed: 7919, poolSeed: 104729});
  const clone = value => JSON.parse(JSON.stringify(value));

  function deriveSeed(seed, domain, index = 0) {
    let x = ((Number(seed) >>> 0) ^ Math.imul(domain + 1, 0x9e3779b9) ^ Math.imul(index + 1, 0x85ebca6b)) >>> 0;
    x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
    x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
    return (x ^ (x >>> 16)) >>> 0;
  }
  function fingerprint(weights) {
    const bytes = new DataView(new ArrayBuffer(8));
    let hash = 0x811c9dc5;
    for (const weight of weights) {
      bytes.setFloat64(0, weight, true);
      for (let b = 0; b < 8; ++b) hash = Math.imul(hash ^ bytes.getUint8(b), 0x01000193) >>> 0;
    }
    return hash.toString(16).padStart(8, '0');
  }
  function editDistance(actual, expected) {
    let row = Array.from({length: expected.length + 1}, (_, i) => i);
    for (let i = 0; i < actual.length; ++i) {
      const next = [i + 1];
      for (let j = 0; j < expected.length; ++j)
        next.push(Math.min(next[j] + 1, row[j + 1] + 1, row[j] + Number(actual[i] !== expected[j])));
      row = next;
    }
    return row[expected.length];
  }
  function summarizeEpisode(phase, episode, output, stoppedBy, decisions, reference) {
    return {phase, episode, output, stoppedBy, decisions, exact: output === reference && stoppedBy === END,
      editDistance: editDistance(output, reference)};
  }

  // A priori disjoint sensory codes, shuffled once independently of the phrase,
  // labels, experimental seed, learned weights, or observed neural responses.
  function createCodebook(manifest, seed) {
    const indices = [...manifest.targets.eye_L, ...manifest.targets.eye_R];
    if (indices.length < INPUT_CODES.length || new Set(indices).size !== indices.length)
      throw new Error('Eight disjoint sensory codes need at least eight distinct annotated eye cells.');
    const rng = new SeededRandom(seed);
    for (let i = indices.length - 1; i > 0; --i) {
      const j = Math.floor(rng.next() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const codebook = Object.fromEntries(INPUT_CODES.map(code => [code, []]));
    indices.forEach((neuron, i) => codebook[INPUT_CODES[i % INPUT_CODES.length]].push(neuron));
    return codebook;
  }
  function createPools(n, excluded, count, seed) {
    const poolForNeuron = new Int32Array(n).fill(-1), sizes = new Uint32Array(count);
    for (let neuron = 0; neuron < n; ++neuron) {
      if (excluded.has(neuron)) continue;
      const pool = deriveSeed(seed, 17, neuron) % count;
      poolForNeuron[neuron] = pool; sizes[pool]++;
    }
    return {poolForNeuron, sizes};
  }
  function appendHistory(history, features, length) {
    history.push(Array.from(features));
    if (history.length > length) history.shift();
  }
  function historyVector(history, featureCount, historyLength, noPastHistory = false) {
    const vector = new Float64Array(featureCount * historyLength);
    const offset = historyLength - history.length;
    history.forEach((features, row) => {
      if (features.length !== featureCount || features.some(value => !Number.isFinite(value)))
        throw new Error('Invalid neural history feature vector.');
      if (!noPastHistory || row === history.length - 1) vector.set(features, (offset + row) * featureCount);
    });
    return vector;
  }
  function oneHot(code) {
    const index = INPUT_CODES.indexOf(code);
    if (index < 0) throw new Error('Unknown input code.');
    return INPUT_CODES.map((_, i) => Number(i === index));
  }

  // Dual ridge regression: 114 training examples by default, rather than a
  // costly inverse over 769 coefficients. Cholesky solves all output classes.
  // Scaling, pooling and history are fixed before training; there is no fit on
  // evaluation features. Bias is included and regularized with the same lambda.
  function fitRidge(examples, targets, ridge, labels = OUTPUT_CODES) {
    if (!examples.length || examples.length !== targets.length || !(ridge > 0) || !Number.isFinite(ridge))
      throw new Error('Invalid ridge training data.');
    const n = examples.length, dimensions = examples[0].length;
    if (examples.some(row => row.length !== dimensions || Array.from(row).some(value => !Number.isFinite(value))))
      throw new Error('Training vectors must have equal finite dimensions.');
    const targetIndices = targets.map(target => {
      const index = labels.indexOf(target);
      if (index < 0) throw new Error('Unknown training target.');
      return index;
    });
    const L = Array.from({length: n}, () => new Float64Array(n));
    for (let i = 0; i < n; ++i) for (let j = 0; j <= i; ++j) {
      let value = 1; // constant bias feature
      for (let feature = 0; feature < dimensions; ++feature) value += examples[i][feature] * examples[j][feature];
      if (i === j) value += ridge;
      for (let k = 0; k < j; ++k) value -= L[i][k] * L[j][k];
      if (i === j) {
        if (!(value > 0)) throw new Error('Ridge fit lost positive definiteness.');
        L[i][j] = Math.sqrt(value);
      } else L[i][j] = value / L[j][j];
    }
    const weights = new Float64Array((dimensions + 1) * labels.length);
    for (let label = 0; label < labels.length; ++label) {
      const y = new Float64Array(n), alpha = new Float64Array(n);
      for (let i = 0; i < n; ++i) {
        let value = Number(targetIndices[i] === label);
        for (let j = 0; j < i; ++j) value -= L[i][j] * y[j];
        y[i] = value / L[i][i];
      }
      for (let i = n - 1; i >= 0; --i) {
        let value = y[i];
        for (let j = i + 1; j < n; ++j) value -= L[j][i] * alpha[j];
        alpha[i] = value / L[i][i];
      }
      for (let i = 0; i < n; ++i) {
        for (let feature = 0; feature < dimensions; ++feature)
          weights[feature * labels.length + label] += examples[i][feature] * alpha[i];
        weights[dimensions * labels.length + label] += alpha[i];
      }
    }
    return {weights: Array.from(weights), dimensions, labels: [...labels], ridge};
  }
  function predictLinear(weights, features, labels = OUTPUT_CODES) {
    if (weights.length !== (features.length + 1) * labels.length)
      throw new Error('Decoder dimensions do not match its features.');
    const scores = labels.map((_, label) => {
      let score = weights[features.length * labels.length + label];
      for (let feature = 0; feature < features.length; ++feature)
        score += weights[feature * labels.length + label] * features[feature];
      return score;
    });
    let winner = 0;
    for (let i = 1; i < scores.length; ++i) if (scores[i] > scores[winner]) winner = i;
    return {prediction: labels[winner], scores};
  }

  // This autonomous actor deliberately has no reference, episode, trial counter,
  // expected next character, reference length, or evaluator. The caller supplies
  // the neural encoding of its own current cue. It outputs its own next cue/END.
  class RecallActor {
    constructor(weights, {featureCount, historyLength = 6, labels = OUTPUT_CODES, noPastHistory = false}) {
      this.weights = Object.freeze(Array.from(weights));
      this.featureCount = featureCount;
      this.historyLength = historyLength;
      this.labels = Object.freeze(Array.from(labels));
      this.noPastHistory = noPastHistory;
      this.history = [];
      this.cue = START;
      this.ended = false;
      if (this.weights.length !== (featureCount * historyLength + 1) * labels.length)
        throw new Error('Invalid actor weights.');
    }
    observe(features) {
      if (this.ended) throw new Error('An ended actor cannot accept another cue.');
      if (features.length !== this.featureCount) throw new Error('Wrong actor feature count.');
      appendHistory(this.history, features, this.historyLength);
      const vector = historyVector(this.history, this.featureCount, this.historyLength, this.noPastHistory);
      const decision = predictLinear(this.weights, vector, this.labels);
      this.cue = decision.prediction;
      this.ended = this.cue === END;
      return decision;
    }
  }

  // Reference-free rollout; the caller scores only after this returns. The
  // safety cap is fixed by configuration, independent of target phrase length.
  function conventionalRollout(fit, {historyLength, maxDecisions}) {
    const actor = new RecallActor(fit.weights, {featureCount: INPUT_CODES.length, historyLength});
    const decisions = [];
    let output = '';
    for (let step = 1; step <= maxDecisions; ++step) {
      const cue = actor.cue;
      const decision = actor.observe(oneHot(cue));
      decisions.push({step, cue, prediction: decision.prediction, scores: decision.scores});
      if (actor.ended) return {output, stoppedBy: END, decisions};
      output += decision.prediction;
    }
    return {output, stoppedBy: 'cap', decisions};
  }

  class RecallExperiment {
    constructor(connectome, manifest, seed, options = {}) {
      this.connectome = connectome; this.manifest = manifest; this.seed = Number(seed) >>> 0;
      this.protocol = Object.freeze(Object.fromEntries(Object.entries(DEFAULTS).map(([key, value]) => [key, options[key] ?? value])));
      const p = this.protocol;
      if (typeof p.phrase !== 'string' || !p.phrase.length || Array.from(p.phrase).some(letter => !ALPHABET.includes(letter)))
        throw new Error('The training phrase must use the fixed seven-character alphabet.');
      for (const key of ['trainEpisodes', 'maxDecisions', 'historyLength', 'poolCount'])
        if (!Number.isInteger(p[key]) || p[key] < 1) throw new Error(`Invalid ${key}.`);
      for (const key of ['recallEpisodes', 'ablatedEpisodes'])
        if (!Number.isInteger(p[key]) || p[key] < 0) throw new Error(`Invalid ${key}.`);
      if (!(p.ridge > 0) || !Number.isFinite(p.ridge)) throw new Error('Invalid ridge penalty.');
      this.dtMs = Number(manifest.config.sim.dt_ms);
      if (!(this.dtMs > 0) || !Number.isFinite(this.dtMs) || !(p.trialMs > 0) || !Number.isFinite(p.trialMs) ||
          !(p.warmupMs >= 0 && p.warmupMs < p.trialMs) || !(p.cueRateHz > 0) || p.cueRateHz * this.dtMs / 1000 > 1)
        throw new Error('Invalid neural timing or cue rate.');
      this.stepsPerCue = p.trialMs / this.dtMs; this.warmupSteps = p.warmupMs / this.dtMs;
      if (!Number.isInteger(this.stepsPerCue) || !Number.isInteger(this.warmupSteps))
        throw new Error('Cue timings must be exact multiples of the neural timestep.');
      this.codebook = createCodebook(manifest, p.codebookSeed);
      const excluded = new Set(Object.values(this.codebook).flat());
      for (const neuron of excluded)
        if (!Number.isInteger(neuron) || neuron < 0 || neuron >= connectome.n) throw new Error('Invalid sensory neuron index.');
      const pools = createPools(connectome.n, excluded, p.poolCount, p.poolSeed);
      this.poolForNeuron = pools.poolForNeuron; this.poolSizes = pools.sizes;
      this.counts = new Uint32Array(p.poolCount);
      this.reference = p.phrase;
      this.trainingInputs = [START, ...p.phrase];
      this.trainingTargets = [...p.phrase, END];
      this.trainingRows = []; this.trainingLabels = []; this.conventionalRows = [];
      this.trials = []; this.episodes = []; this.comparator = null;
      this.fitCount = 0; this.fit = null; this.frozenWeights = null; this.decoderFingerprint = null;
      this.totalSteps = 0; this.net = null; this.currentNeuralSeed = null;
      this.phase = 'train'; this.episode = 1; this.decisionStep = 1;
      this.history = []; this.codeHistory = []; this.episodeOutput = ''; this.episodeDecisions = [];
      this.actor = null;
      this.totalTrialBudget = p.trainEpisodes * this.trainingTargets.length +
        (p.recallEpisodes + p.ablatedEpisodes) * p.maxDecisions;
      this.metadata = {seed: this.seed, n: connectome.n, edgeCount: connectome.indices.length,
        layout: [...manifest.layout], inputCodes: [...INPUT_CODES], outputCodes: [...OUTPUT_CODES],
        codebook: clone(this.codebook), poolSizes: Array.from(this.poolSizes),
        pooling: 'Fixed index-hash pools of all nonstimulated neurons; no pruning of the simulated graph.',
        memory: `${p.historyLength} current/past neural feature vectors in external software memory`,
        model: 'Fixed full-connectome LIF encoder; supervised ridge decoder; external history',
        featureDefinition: 'Mean downstream spikes per neuron per second during the final trialMs-warmupMs; empty pools are zero.',
        rng: 'xoshiro128**; distinct domains for training, recall and ablated neural noise'};
    }
    get done() { return this.phase === 'complete'; }
    get simTimeS() { return this.totalSteps * this.dtMs / 1000; }
    get currentCue() { return this.phase === 'train' ? this.trainingInputs[this.decisionStep - 1] : this.actor?.cue ?? null; }
    get progress() {
      return {phase: this.phase, episode: this.episode, step: this.decisionStep,
        cue: this.done ? null : this.currentCue, completed: this.trials.length,
        totalTrials: this.totalTrialBudget, simTimeS: this.simTimeS, simTime: this.simTimeS,
        updateCount: this.fitCount, trainExamples: this.trainingRows.length,
        episodeTotal: this.phase === 'train' ? this.protocol.trainEpisodes :
          (this.phase === 'recall' ? this.protocol.recallEpisodes : this.protocol.ablatedEpisodes)};
    }
    startCue() {
      const domain = {train: 23, recall: 29, ablated: 31}[this.phase];
      // Episode/decision choose an independent noise seed, not a decoder input.
      // Fixed 65536 stride avoids dependence on previous rollout stopping times.
      this.currentNeuralSeed = deriveSeed(this.seed, domain, (this.episode - 1) * 65536 + this.decisionStep - 1);
      this.net = new LIFNetwork(this.connectome, this.manifest.config.sim, this.currentNeuralSeed);
      this.net.injectPoisson(this.codebook[this.currentCue], this.protocol.cueRateHz);
      this.counts.fill(0);
    }
    step() {
      if (this.done) return null;
      if (!this.net) this.startCue();
      const collect = this.net.stepIndex >= this.warmupSteps;
      const spikes = this.net.step(); this.totalSteps++;
      if (collect) for (const neuron of spikes) {
        const pool = this.poolForNeuron[neuron];
        if (pool >= 0) this.counts[pool]++;
      }
      if (this.net.stepIndex < this.stepsPerCue) return null;
      const duration = (this.stepsPerCue - this.warmupSteps) * this.dtMs / 1000;
      const features = Array.from(this.counts, (count, pool) => this.poolSizes[pool] ? count / (this.poolSizes[pool] * duration) : 0);
      const cue = this.currentCue, phase = this.phase, episode = this.episode, step = this.decisionStep;
      const record = {trial: this.trials.length + 1, phase, episode, step, cue,
        neuralSeed: this.currentNeuralSeed, features, poolSpikeCounts: Array.from(this.counts),
        spikes: this.net.totalSpikes, simTimeS: this.simTimeS, prediction: null,
        decoderVersion: this.fitCount, decoderFingerprint: this.decoderFingerprint,
        updateCountBefore: this.fitCount, feedbackApplied: false};
      this.net = null;
      if (phase === 'train') {
        appendHistory(this.history, features, this.protocol.historyLength);
        appendHistory(this.codeHistory, oneHot(cue), this.protocol.historyLength);
        this.trainingRows.push(historyVector(this.history, this.protocol.poolCount, this.protocol.historyLength));
        this.conventionalRows.push(historyVector(this.codeHistory, INPUT_CODES.length, this.protocol.historyLength));
        const target = this.trainingTargets[step - 1];
        this.trainingLabels.push(target); record.target = target;
        this.decisionStep++;
        if (this.decisionStep > this.trainingTargets.length) this.nextEpisode();
      } else {
        const decision = this.actor.observe(features);
        record.prediction = decision.prediction; record.scores = decision.scores;
        this.episodeDecisions.push({step, cue, prediction: decision.prediction, neuralSeed: record.neuralSeed});
        if (decision.prediction !== END) this.episodeOutput += decision.prediction;
        if (decision.prediction === END || step >= this.protocol.maxDecisions) {
          const stoppedBy = decision.prediction === END ? END : 'cap';
          this.episodes.push(summarizeEpisode(phase, episode, this.episodeOutput, stoppedBy,
            clone(this.episodeDecisions), this.reference));
          this.totalTrialBudget -= this.protocol.maxDecisions - step;
          this.nextEpisode();
        } else this.decisionStep++;
      }
      record.updateCountAfter = this.fitCount;
      record.fitApplied = record.updateCountAfter > record.updateCountBefore;
      this.trials.push(record);
      return clone(record);
    }
    fitDecoders() {
      this.fit = fitRidge(this.trainingRows, this.trainingLabels, this.protocol.ridge);
      this.frozenWeights = Object.freeze([...this.fit.weights]);
      this.decoderFingerprint = fingerprint(this.frozenWeights);
      this.fitCount++;
      const fit = fitRidge(this.conventionalRows, this.trainingLabels, this.protocol.ridge);
      const episodes = [];
      for (let episode = 1; episode <= this.protocol.recallEpisodes; ++episode) {
        const rollout = conventionalRollout(fit, this.protocol);
        episodes.push(summarizeEpisode('conventional', episode, rollout.output, rollout.stoppedBy,
          rollout.decisions, this.reference));
      }
      this.comparator = {kind: 'Conventional one-hot cue history, ridge decoder, own-feedback rollout',
        deterministic: true, fitExamples: this.trainingLabels.length, weights: fit.weights,
        fingerprint: fingerprint(fit.weights), featureCount: INPUT_CODES.length,
        historyLength: this.protocol.historyLength, episodes,
        note: 'Repeated conventional episodes are identical deterministic rollouts, not independent evidence.'};
    }
    nextEpisode() {
      this.episode++;
      if (this.phase === 'train' && this.episode > this.protocol.trainEpisodes) {
        this.fitDecoders(); this.phase = 'recall'; this.episode = 1;
      }
      if (this.phase === 'recall' && this.episode > this.protocol.recallEpisodes) {
        this.phase = 'ablated'; this.episode = 1;
      }
      if (this.phase === 'ablated' && this.episode > this.protocol.ablatedEpisodes) this.phase = 'complete';
      this.decisionStep = 1; this.history = []; this.codeHistory = [];
      this.episodeOutput = ''; this.episodeDecisions = [];
      if (!this.done && this.phase !== 'train')
        this.actor = new RecallActor(this.frozenWeights, {featureCount: this.protocol.poolCount,
          historyLength: this.protocol.historyLength, noPastHistory: this.phase === 'ablated'});
    }
    result() {
      const metrics = {};
      for (const phase of ['recall', 'ablated', 'conventional']) {
        const episodes = phase === 'conventional' ? this.comparator?.episodes || [] : this.episodes.filter(e => e.phase === phase);
        metrics[phase] = {exact: episodes.filter(e => e.exact).length, total: episodes.length,
          meanEditDistance: episodes.length ? episodes.reduce((sum, e) => sum + e.editDistance, 0) / episodes.length : null,
          ended: episodes.filter(e => e.stoppedBy === END).length, capped: episodes.filter(e => e.stoppedBy === 'cap').length};
      }
      return clone({format: 'flyhamlet-recall-v1', complete: this.done, seed: this.seed, reference: this.reference,
        config: this.protocol, metadata: this.metadata, neuralConfig: this.manifest.config.sim,
        provenance: this.manifest.provenance || null, connectivityIntegrity: this.manifest.integrity || null,
        simTimeS: this.simTimeS, fitCount: this.fitCount, updateCount: this.fitCount,
        initialWeights: null, frozenWeights: this.frozenWeights, finalWeights: this.fit?.weights || null,
        frozenFingerprint: this.decoderFingerprint,
        finalFingerprint: this.fit ? fingerprint(this.fit.weights) : null,
        trials: this.trials, episodes: this.episodes, comparator: this.comparator, metrics,
        trainingExamples: this.trainingLabels.length,
        intervention: 'Teacher-forced sensory encoding, batch supervised ridge fit, then own-feedback recall with six-vector external memory; connectome is fixed.',
        evaluation: 'Autonomous actors receive START then their own predictions. Learned END stops a rollout; a fixed safety cap is independent of the phrase length. Reference is used only for training and separate post-rollout scoring.',
        ablation: 'Same trained decoder; keep the current neural feature vector and mask all past history slots to zero. Neural cue feedback follows its own predictions.',
        inferenceLimit: 'Recall of one trained sequence by an engineered system, not language understanding or biological memory. Noise seeds are simulations of one anatomy. Conventional and ablated controls do not establish an advantage of the real connectome.'});
    }
  }
  const api = {RecallExperiment, RecallActor, fitRidge, predictLinear, conventionalRollout,
    historyVector, appendHistory, oneHot, createCodebook, createPools, deriveSeed, fingerprint,
    editDistance, summarizeEpisode, START, END, ALPHABET, INPUT_CODES, OUTPUT_CODES, DEFAULTS};
  root.FlyHamletRecall = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
