/* In-brain phrase memory: a plastic Kenyon-cell-to-MBON layer inside the fixed connectome.
 * Letters arrive as sparse olfactory-style codes on uniglomerular projection neurons. The
 * only thing that learns is the strength of existing Kenyon-cell-to-MBON synapses, moved by
 * a supervised three-factor rule during training. The readout is fixed: eight seeded groups
 * of MBONs vote by spike count. Recall feeds each decoded letter back as the next cue.
 */
(function (root) {
  'use strict';
  const live = root.FlyHamletLive || (typeof require === 'function' ? require('./live-model.js') : null);
  const recall = root.FlyHamletRecall || (typeof require === 'function' ? require('./recall-model.js') : null);
  const sequence = root.FlyHamletSequence || (typeof require === 'function' ? require('./sequence-model.js') : null);
  const chain = root.FlyHamletChain || (typeof require === 'function' ? require('./chain-model.js') : null);
  if (!live || !recall || !sequence || !chain) throw new Error('Load live-model.js, recall-model.js, sequence-model.js and chain-model.js before mb-model.js.');
  const {LIFNetwork, SeededRandom} = live;
  const {ALPHABET, INPUT_CODES, OUTPUT_CODES, START, END, fitRidge, deriveSeed, fingerprint, oneHot, conventionalRollout} = recall;
  const {prepareSlowGraph, slowNeuralConfig, wilson} = sequence;
  const {summarizeChainEpisode} = chain;
  const CONDITIONS = Object.freeze(['plastic', 'plasticReset', 'frozen']);
  const PHASES = Object.freeze(['train', 'diagnostic', 'recall']);
  const DEFAULTS = Object.freeze({phrase: 'to be or not to be', trainEpisodes: 12, diagnosticEpisodes: 3, recallEpisodes: 6,
    maxDecisions: 32, cueMs: 100, gapMs: 25, warmupMs: 250, cueRateHz: 100, slowTimeFactor: 10, slowWeightScale: 0.3,
    learningRateMvPerSpike: 1, maxWeightMv: 20, stateResolutionMv: 0.01, ridge: 0.01, codebookSeed: 7919, groupSeed: 104729});
  const clone = value => JSON.parse(JSON.stringify(value));
  const SEED_DOMAINS = Object.freeze({train: 107, diagnostic: 109, recall: 113});

  function shuffled(values, seed) {
    const copy = [...values], rng = new SeededRandom(seed);
    for (let i = copy.length - 1; i > 0; --i) { const j = Math.floor(rng.next() * (i + 1)); [copy[i], copy[j]] = [copy[j], copy[i]]; }
    return copy;
  }
  function partition(values, codes, seed) {
    const order = shuffled(values, seed);
    return Object.fromEntries(codes.map((code, k) => [code, order.filter((_, i) => i % codes.length === k).sort((a, b) => a - b)]));
  }
  function validateAnnotation(annotation, n) {
    for (const name of ['kenyonCells', 'mbons', 'uniglomerularPNs']) {
      const values = annotation?.[name];
      if (!Array.isArray(values) || !values.length) throw new Error(`Mushroom-body annotation needs ${name}.`);
      if (values.some(index => !Number.isInteger(index) || index < 0 || index >= n)) throw new Error(`Invalid ${name} index.`);
      if (new Set(values).size !== values.length) throw new Error(`Duplicate ${name} index.`);
    }
    const all = [...annotation.kenyonCells, ...annotation.mbons, ...annotation.uniglomerularPNs];
    if (new Set(all).size !== all.length) throw new Error('Kenyon cells, MBONs and projection neurons must be disjoint.');
    if (annotation.mbons.length < OUTPUT_CODES.length) throw new Error('At least eight MBONs are needed for eight output groups.');
    if (annotation.uniglomerularPNs.length < INPUT_CODES.length) throw new Error('At least eight projection neurons are needed for eight input codes.');
  }
  function summarizeMushroomBodyRun(episodes, trials, reference) {
    const metrics = {};
    for (const condition of CONDITIONS) {
      const rows = trials.filter(trial => trial.condition === condition);
      const recallEpisodes = episodes.filter(item => item.condition === condition && item.phase === 'recall');
      const lengths = recallEpisodes.map(item => item.chainLength);
      const byEpisode = {};
      for (const trial of rows.filter(trial => trial.phase === 'train')) {
        const row = byEpisode[trial.episode] || (byEpisode[trial.episode] = {episode: trial.episode, correct: 0, total: 0, updates: 0});
        row.correct += Number(trial.correct); row.total++; row.updates += Number(trial.learningApplied);
      }
      const diagnostic = rows.filter(trial => trial.phase === 'diagnostic');
      const correct = diagnostic.filter(trial => trial.correct).length;
      const activity = phase => {
        const items = rows.filter(trial => trial.phase === phase);
        const mean = key => items.length ? items.reduce((sum, trial) => sum + trial[key], 0) / items.length : null;
        return {decisions: items.length, meanKcFraction: mean('kcFraction'), meanMbonSpikes: mean('mbonSpikes'), meanAllSpikes: mean('allSpikes'),
          silentDecisions: items.filter(trial => trial.silent).length, tiedDecisions: items.filter(trial => trial.tie).length};
      };
      metrics[condition] = {
        training: {curve: Object.values(byEpisode).map(row => ({...row, accuracy: row.total ? row.correct / row.total : null})).sort((a, b) => a.episode - b.episode),
          updates: rows.filter(trial => trial.learningApplied).length, activity: activity('train')},
        diagnostic: {correct, total: diagnostic.length, accuracy: diagnostic.length ? correct / diagnostic.length : null,
          wilson95: wilson(correct, diagnostic.length), chance: 1 / OUTPUT_CODES.length, activity: activity('diagnostic')},
        recall: {episodes: recallEpisodes.length, exact: recallEpisodes.filter(item => item.exact).length, chainLengths: lengths,
          meanChainLength: lengths.length ? lengths.reduce((sum, value) => sum + value, 0) / lengths.length : null,
          maxChainLength: lengths.length ? Math.max(...lengths) : null,
          meanEditDistance: recallEpisodes.length ? recallEpisodes.reduce((sum, item) => sum + item.editDistance, 0) / recallEpisodes.length : null,
          ended: recallEpisodes.filter(item => item.stoppedBy === END).length, capped: recallEpisodes.filter(item => item.stoppedBy === 'cap').length,
          referenceLength: reference.length, activity: activity('recall')}};
    }
    return metrics;
  }

  class MushroomBodyExperiment {
    constructor(graph, manifest, annotation, seed, options = {}, prepared = null) {
      this.graph = graph; this.manifest = manifest; this.seed = Number(seed) >>> 0;
      this.protocol = Object.freeze(Object.fromEntries(Object.entries(DEFAULTS).map(([key, value]) => [key, options[key] ?? value])));
      const p = this.protocol;
      if (typeof p.phrase !== 'string' || !p.phrase.length || Array.from(p.phrase).some(letter => !ALPHABET.includes(letter)))
        throw new Error('The training phrase must use the fixed seven-character alphabet.');
      for (const key of ['trainEpisodes', 'recallEpisodes', 'maxDecisions'])
        if (!Number.isInteger(p[key]) || p[key] < 1) throw new Error(`Invalid ${key}.`);
      if (!Number.isInteger(p.diagnosticEpisodes) || p.diagnosticEpisodes < 0) throw new Error('Invalid diagnosticEpisodes.');
      for (const key of ['cueMs', 'cueRateHz', 'slowTimeFactor', 'slowWeightScale', 'learningRateMvPerSpike', 'maxWeightMv', 'stateResolutionMv', 'ridge'])
        if (!(p[key] > 0) || !Number.isFinite(p[key])) throw new Error(`Invalid ${key}.`);
      for (const key of ['gapMs', 'warmupMs']) if (!(p[key] >= 0) || !Number.isFinite(p[key])) throw new Error(`Invalid ${key}.`);
      this.dtMs = Number(manifest.config.sim.dt_ms);
      if (!(this.dtMs > 0) || !Number.isFinite(this.dtMs) || p.cueRateHz * this.dtMs / 1000 > 1) throw new Error('Invalid neural timestep or cue probability.');
      const background = manifest.config.sim.background;
      if (background?.enabled && background.rate_hz > 0) throw new Error('This assay requires background input to be disabled.');
      const integerSteps = value => {
        const steps = Math.round(value / this.dtMs);
        if (Math.abs(steps * this.dtMs - value) > 1e-9) throw new Error('Cue and gap timings must be exact neural timestep multiples.');
        return steps;
      };
      this.cueSteps = integerSteps(p.cueMs); this.gapSteps = integerSteps(p.gapMs); this.warmupSteps = integerSteps(p.warmupMs);
      this.stepsPerDecision = this.cueSteps + this.gapSteps; this.decisionMs = this.stepsPerDecision * this.dtMs;
      this.warmupThisDecision = 0;
      validateAnnotation(annotation, graph.n);
      this.annotation = {kenyonCells: [...annotation.kenyonCells], mbons: [...annotation.mbons], uniglomerularPNs: [...annotation.uniglomerularPNs]};
      this.slowConfig = Object.freeze(slowNeuralConfig(manifest.config.sim, p.slowTimeFactor));
      if (prepared && (prepared.graph?.n !== graph.n || prepared.graph.weights?.length !== graph.weights.length || prepared.metadata?.weightScale !== p.slowWeightScale))
        throw new Error('The prepared slow graph must match this graph and the protocol weight scale.');
      const slow = prepared || prepareSlowGraph(graph, p.slowWeightScale);
      this.baseWeights = slow.graph.weights; this.slowMetadata = clone(slow.metadata);
      // Fixed, seeded input codes on projection neurons and output groups of MBONs.
      this.codebook = partition(this.annotation.uniglomerularPNs, INPUT_CODES, p.codebookSeed);
      this.groups = partition(this.annotation.mbons, OUTPUT_CODES, p.groupSeed);
      this.groupOf = new Int8Array(graph.n).fill(-1);
      OUTPUT_CODES.forEach((code, g) => { for (const index of this.groups[code]) this.groupOf[index] = g; });
      this.isKC = new Uint8Array(graph.n);
      for (const index of this.annotation.kenyonCells) this.isKC[index] = 1;
      // Every existing Kenyon-cell-to-MBON synapse slot is plastic; nothing else ever changes.
      const slots = [], slotGroup = [], kcSlotStart = new Int32Array(graph.n).fill(-1), kcSlotEnd = new Int32Array(graph.n).fill(-1);
      for (const kc of this.annotation.kenyonCells) {
        kcSlotStart[kc] = slots.length;
        for (let slot = graph.indptr[kc]; slot < graph.indptr[kc + 1]; ++slot) {
          const g = this.groupOf[graph.indices[slot]];
          if (g >= 0) { slots.push(slot); slotGroup.push(g); }
        }
        kcSlotEnd[kc] = slots.length;
      }
      if (!slots.length) throw new Error('No Kenyon-cell-to-MBON synapses exist in this graph.');
      this.plasticSlots = Uint32Array.from(slots); this.plasticSlotGroup = Int8Array.from(slotGroup);
      this.kcSlotStart = kcSlotStart; this.kcSlotEnd = kcSlotEnd;
      this.initialPlasticWeights = Float32Array.from(this.plasticSlots, slot => this.baseWeights[slot]);
      for (const weight of this.initialPlasticWeights) if (weight < 0) throw new Error('Kenyon-cell-to-MBON synapses are expected to be excitatory.');
      this.weights = null; this.conditionGraph = null;
      this.reference = p.phrase; this.trainingInputs = [START, ...p.phrase]; this.trainingTargets = [...p.phrase, END];
      this.decisionsPerEpisode = this.trainingTargets.length;
      this.conditionIndex = -1; this.phase = 'train'; this.episode = 1; this.decisionStep = 1; this.cueStep = 0;
      this.net = null; this.inputs = null; this.cue = START; this.episodeSeed = null; this.cueSeed = null;
      this.kcCounts = new Uint16Array(graph.n); this.activeKCs = []; this.groupCounts = new Float64Array(OUTPUT_CODES.length);
      this.mbonSpikes = 0; this.kcSpikes = 0; this.allSpikes = 0; this.episodeOutput = ''; this.episodeDecisions = [];
      this.trials = []; this.episodes = []; this.models = {}; this.totalSteps = 0; this.updateCount = 0;
      this.perConditionBudget = (p.trainEpisodes + p.diagnosticEpisodes) * this.decisionsPerEpisode + p.recallEpisodes * p.maxDecisions;
      this.totalTrialBudget = this.perConditionBudget * CONDITIONS.length;
      const comparatorRows = this.trainingInputs.map(code => oneHot(code));
      const comparatorFit = fitRidge(comparatorRows, this.trainingTargets, p.ridge);
      const rollout = conventionalRollout(comparatorFit, {historyLength: 1, maxDecisions: p.maxDecisions});
      this.comparator = {kind: 'Current-cue-only comparator: ridge on the one-hot current cue, own-feedback rollout, no state',
        deterministic: true, fitExamples: comparatorRows.length, weights: comparatorFit.weights, fingerprint: fingerprint(comparatorFit.weights),
        episode: summarizeChainEpisode('comparator', 'comparator', 1, rollout.output, rollout.stoppedBy, rollout.decisions, this.reference),
        note: 'The best a decoder can do from the current letter alone, without any memory of earlier letters.'};
      this.beginCondition();
      this.metadata = {seed: this.seed, n: graph.n, edgeCount: graph.indices.length, layout: [...manifest.layout], reference: this.reference,
        inputCodes: [...INPUT_CODES], outputCodes: [...OUTPUT_CODES], conditions: [...CONDITIONS], phases: [...PHASES],
        codebook: clone(this.codebook), groups: clone(this.groups), annotationCounts: {kenyonCells: this.annotation.kenyonCells.length,
          mbons: this.annotation.mbons.length, uniglomerularPNs: this.annotation.uniglomerularPNs.length},
        plasticSlotCount: this.plasticSlots.length, plasticSlotsPerGroup: OUTPUT_CODES.map((_, g) => slotGroup.filter(value => value === g).length),
        initialPlasticWeightMv: {mean: this.initialPlasticWeights.reduce((sum, value) => sum + value, 0) / this.initialPlasticWeights.length,
          max: Math.max(...this.initialPlasticWeights)},
        decisionMs: this.decisionMs,
        inputDefinition: 'Each input code drives its own disjoint set of uniglomerular antennal-lobe projection neurons at cueRateHz for cueMs, then a gap with no input.',
        readoutDefinition: 'Fixed seeded partition of all annotated MBONs into eight groups, one per output code. The decision is the group with the most MBON spikes in the cue-plus-gap window; ties fall to the group with the highest mean quantized MBON voltage deviation, then the lowest group index. No parameter of the readout is learned.',
        learningRule: 'After each training decision, if the target group is not the strict spike-count winner: for every Kenyon cell that spiked in the window, its synapses onto target-group MBONs gain learningRateMvPerSpike per spike (capped at maxWeightMv) and its synapses onto the best non-target group lose the same amount (floored at zero). Only existing Kenyon-cell-to-MBON synapses change; the teacher is external.',
        continuity: 'plastic and frozen keep one network per episode; plasticReset replaces all neural state at every cue onset. Learned synaptic weights persist across episodes within a condition and are frozen after training.',
        warmup: 'A resting network barely answers its first cue, so the first cue of every episode is presented for warmupMs before its decision window. In plasticReset every cue gets the same warm-up with the current letter only, so its activity is comparable but carries no history. Spikes are counted only in the final cueMs plus gapMs of each decision.',
        slowModel: {timeFactor: p.slowTimeFactor, weightScale: p.slowWeightScale, tauMemMs: this.slowConfig.tau_mem_ms, tauSynMs: this.slowConfig.tau_syn_ms,
          calibration: 'Weight scale chosen so that projection-neuron cues activate a sparse Kenyon-cell population (about 3 to 9 percent) with letter-selective, position-dependent codes; the original weights saturate two thirds of all Kenyon cells for every letter.',
          status: 'Imposed engineering hypothesis; not identified from fruit-fly memory data.'},
        model: 'Fixed full-connectome LIF with plastic Kenyon-cell-to-MBON synapses; in-brain readout; external history absent',
        rng: 'xoshiro128**; independent episode and reset-cue noise domains'};
    }
    get done() { return this.conditionIndex >= CONDITIONS.length; }
    get condition() { return CONDITIONS[this.conditionIndex] ?? null; }
    get simTimeS() { return this.totalSteps * this.dtMs / 1000; }
    get learning() { return this.phase === 'train' && this.condition !== 'frozen'; }
    get episodeTotal() { return this.protocol[{train: 'trainEpisodes', diagnostic: 'diagnosticEpisodes', recall: 'recallEpisodes'}[this.phase]]; }
    get progress() {
      return {condition: this.condition, phase: this.done ? 'complete' : this.phase, conditionIndex: Math.min(this.conditionIndex + 1, CONDITIONS.length),
        conditionTotal: CONDITIONS.length, episode: this.episode, episodeTotal: this.done ? 0 : this.episodeTotal, step: this.decisionStep,
        cue: this.done ? null : this.cue, inputActive: !this.done && this.cueStep < this.cueSteps, learning: !this.done && this.learning,
        stage: this.done ? 'complete' : (this.cueStep < this.warmupThisDecision + this.cueSteps ? (this.cueStep < this.warmupThisDecision ? 'warmup' : 'cue') : 'gap'),
        cueTimeMs: this.cueStep * this.dtMs, decisionMs: this.decisionMs, decisionTotalMs: (this.warmupThisDecision + this.stepsPerDecision) * this.dtMs,
        groupCounts: Array.from(this.groupCounts), episodeOutput: this.episodeOutput, completed: this.trials.length, totalTrials: this.totalTrialBudget,
        simTimeS: this.simTimeS, simTime: this.simTimeS, updateCount: this.updateCount};
    }
    beginCondition() {
      this.conditionIndex++;
      if (this.done) { this.weights = null; this.conditionGraph = null; return; }
      // Each condition starts from the same scaled anatomical weights; only plastic slots can move.
      this.weights = new Float32Array(this.baseWeights);
      this.conditionGraph = {n: this.graph.n, indptr: this.graph.indptr, indices: this.graph.indices, weights: this.weights};
      this.phase = 'train'; this.episode = 1; this.decisionStep = 1; this.cue = START;
      this.models[this.condition] = {updates: 0, potentiatedSlots: 0, depressedSlots: 0, frozenAfterTraining: false};
    }
    startCue() {
      if (this.decisionStep === 1) this.episodeSeed = deriveSeed(this.seed, SEED_DOMAINS[this.phase], this.episode - 1);
      if (this.condition === 'plasticReset' || this.decisionStep === 1) {
        this.cueSeed = this.condition === 'plasticReset' ? deriveSeed(this.episodeSeed, 127, this.decisionStep - 1) : this.episodeSeed;
        this.net = new LIFNetwork(this.conditionGraph, this.slowConfig, this.cueSeed);
        this.inputs = Object.fromEntries(INPUT_CODES.map(code => [code, this.net.injectPoisson(this.codebook[code], 0)]));
      }
      this.warmupThisDecision = (this.condition === 'plasticReset' || this.decisionStep === 1) ? this.warmupSteps : 0;
      this.setCue(this.cue);
      this.cueStep = 0; this.kcSpikes = 0; this.mbonSpikes = 0; this.allSpikes = 0; this.groupCounts.fill(0);
      for (const kc of this.activeKCs) this.kcCounts[kc] = 0;
      this.activeKCs = [];
    }
    setCue(code) {
      for (const input of INPUT_CODES) this.net.setPoissonRate(this.inputs[input], input === code ? this.protocol.cueRateHz : 0);
    }
    step() {
      if (this.done) return null;
      if (this.cueStep === 0) this.startCue();
      else if (this.cueStep === this.warmupThisDecision + this.cueSteps) this.setCue(null);
      const spikes = this.net.step();
      if (this.cueStep >= this.warmupThisDecision) {
        this.allSpikes += spikes.length;
        for (const neuron of spikes) {
          if (this.isKC[neuron]) { if (this.kcCounts[neuron]++ === 0) this.activeKCs.push(neuron); this.kcSpikes++; }
          else { const g = this.groupOf[neuron]; if (g >= 0) { this.groupCounts[g]++; this.mbonSpikes++; } }
        }
      }
      this.cueStep++; this.totalSteps++;
      if (this.cueStep < this.warmupThisDecision + this.stepsPerDecision) return null;
      return this.finishDecision();
    }
    tieScores() {
      const res = this.protocol.stateResolutionMv, scores = new Float64Array(OUTPUT_CODES.length);
      OUTPUT_CODES.forEach((code, g) => {
        let sum = 0;
        for (const index of this.groups[code]) sum += Math.round((this.net.v[index] - this.net.v0) / res) * res;
        scores[g] = sum / this.groups[code].length;
      });
      return scores;
    }
    decide(counts, scores, exclude = -1) {
      let best = -1;
      for (let g = 0; g < counts.length; ++g) {
        if (g === exclude) continue;
        if (best < 0 || counts[g] > counts[best] || (counts[g] === counts[best] && scores[g] > scores[best])) best = g;
      }
      return best;
    }
    learn(target) {
      const counts = this.groupCounts, scores = this.tieScores();
      const rival = this.decide(counts, scores, target);
      if (counts[target] > counts[rival]) return {applied: false, potentiated: 0, depressed: 0, rival};
      const eta = this.protocol.learningRateMvPerSpike, max = this.protocol.maxWeightMv, w = this.weights;
      let potentiated = 0, depressed = 0, moved = 0;
      for (const kc of this.activeKCs) {
        const delta = eta * this.kcCounts[kc];
        for (let k = this.kcSlotStart[kc]; k < this.kcSlotEnd[kc]; ++k) {
          const g = this.plasticSlotGroup[k], slot = this.plasticSlots[k], before = w[slot];
          if (g === target) w[slot] = Math.fround(Math.min(max, before + delta));
          else if (g === rival) w[slot] = Math.fround(Math.max(0, before - delta));
          else continue;
          if (w[slot] > before) potentiated++; else if (w[slot] < before) depressed++;
          moved += Math.abs(w[slot] - before);
        }
      }
      this.updateCount++;
      const model = this.models[this.condition];
      model.updates++; model.potentiatedSlots += potentiated; model.depressedSlots += depressed;
      return {applied: true, potentiated, depressed, rival, meanAbsDeltaMv: potentiated + depressed ? moved / (potentiated + depressed) : 0};
    }
    finishDecision() {
      const condition = this.condition, phase = this.phase, episode = this.episode, step = this.decisionStep, cue = this.cue;
      const scores = this.tieScores();
      const winner = this.decide(this.groupCounts, scores);
      const prediction = OUTPUT_CODES[winner];
      const counts = Array.from(this.groupCounts);
      const tie = counts.filter(value => value === counts[winner]).length > 1;
      const record = {trial: this.trials.length + 1, condition, phase, episode, step, cue, target: null, prediction, correct: null,
        groupCounts: counts, tieScores: Array.from(scores), tie, silent: this.mbonSpikes === 0, mbonSpikes: this.mbonSpikes,
        kcSpikes: this.kcSpikes, activeKcCount: this.activeKCs.length, kcFraction: this.activeKCs.length / this.annotation.kenyonCells.length,
        kcActivity: this.activeKCs.map(kc => [kc, this.kcCounts[kc]]).sort((a, b) => a[0] - b[0]), allSpikes: this.allSpikes,
        episodeSeed: this.episodeSeed, cueSeed: this.cueSeed, windowMs: this.decisionMs, warmupMs: this.warmupThisDecision * this.dtMs, cueOffMs: this.protocol.cueMs, simTimeS: this.simTimeS,
        learningEnabled: this.learning, learningApplied: false, potentiated: 0, depressed: 0, rival: null, meanAbsDeltaMv: 0,
        feedbackApplied: phase === 'recall' && step > 1, updateCountBefore: this.updateCount};
      let nextCue = null, episodeEnded = false;
      if (phase !== 'recall') {
        record.target = this.trainingTargets[step - 1]; record.correct = prediction === record.target;
        if (this.learning) {
          const outcome = this.learn(OUTPUT_CODES.indexOf(record.target));
          record.learningApplied = outcome.applied; record.potentiated = outcome.potentiated; record.depressed = outcome.depressed;
          record.rival = OUTPUT_CODES[outcome.rival]; record.meanAbsDeltaMv = outcome.meanAbsDeltaMv || 0;
        }
        nextCue = this.trainingInputs[step] ?? null;
        episodeEnded = step >= this.decisionsPerEpisode;
      } else {
        this.episodeDecisions.push({step, cue, prediction, groupCounts: counts});
        if (prediction !== END) this.episodeOutput += prediction;
        episodeEnded = prediction === END || step >= this.protocol.maxDecisions;
        if (episodeEnded) {
          this.episodes.push(summarizeChainEpisode(condition, phase, episode, this.episodeOutput, prediction === END ? END : 'cap', clone(this.episodeDecisions), this.reference));
          this.totalTrialBudget -= this.protocol.maxDecisions - step;
        } else nextCue = prediction;
      }
      record.updateCountAfter = this.updateCount;
      this.trials.push(record);
      this.cueStep = 0;
      if (episodeEnded) this.nextEpisode(); else { this.decisionStep++; this.cue = nextCue; }
      return clone(record);
    }
    snapshotWeights() {
      return Float32Array.from(this.plasticSlots, slot => this.weights[slot]);
    }
    nextEpisode() {
      this.episode++; this.decisionStep = 1; this.cue = START; this.episodeOutput = ''; this.episodeDecisions = [];
      this.net = null; this.inputs = null;
      while (!this.done && this.episode > this.episodeTotal) {
        const index = PHASES.indexOf(this.phase);
        if (this.phase === 'train') {
          const model = this.models[this.condition], learned = this.snapshotWeights();
          model.frozenAfterTraining = true; model.learnedPlasticWeights = Array.from(learned);
          model.fingerprint = fingerprint(learned); model.frozenFingerprint = model.fingerprint;
          model.weightStats = {mean: learned.reduce((sum, value) => sum + value, 0) / learned.length, max: Math.max(...learned),
            atZero: learned.filter(value => value === 0).length, atMax: learned.filter(value => value === this.protocol.maxWeightMv).length,
            changed: learned.filter((value, i) => value !== this.initialPlasticWeights[i]).length};
        }
        if (index + 1 < PHASES.length) { this.phase = PHASES[index + 1]; this.episode = 1; }
        else this.beginCondition();
      }
    }
    result() {
      const models = Object.fromEntries(Object.entries(this.models).map(([condition, model]) => [condition, {...model,
        finalFingerprint: model.learnedPlasticWeights && this.condition === condition && this.weights ? fingerprint(this.snapshotWeights()) : model.frozenFingerprint ?? null}]));
      for (const condition of Object.keys(models)) if (models[condition].learnedPlasticWeights && models[condition].finalFingerprint === null) models[condition].finalFingerprint = models[condition].fingerprint;
      return clone({format: 'flyhamlet-mb-v1', complete: this.done, seed: this.seed, reference: this.reference, config: this.protocol,
        metadata: this.metadata, neuralConfig: this.manifest.config.sim, slowNeuralConfig: this.slowConfig, slowGraph: this.slowMetadata,
        provenance: this.manifest.provenance || null, connectivityIntegrity: this.manifest.integrity || null,
        annotation: {counts: this.metadata.annotationCounts, plasticSlots: Array.from(this.plasticSlots), plasticSlotGroup: Array.from(this.plasticSlotGroup)},
        initialPlasticWeights: Array.from(this.initialPlasticWeights), simTimeS: this.simTimeS, updateCount: this.updateCount, models,
        trials: this.trials, episodes: this.episodes, comparator: this.comparator,
        metrics: summarizeMushroomBodyRun(this.episodes, this.trials, this.reference),
        conditionDefinitions: {plastic: 'Slower model; Kenyon-cell-to-MBON synapses learn during training; one network per episode, never reset between letters.',
          plasticReset: 'Same learning rule and cues; all neural state replaced with rest at every cue onset, so no state carries between letters.',
          frozen: 'Same model and cues with learning disabled; the fixed MBON-group readout on unchanged synapses.'},
        intervention: 'Sparse projection-neuron letter codes drive the mushroom body of the slower model. A supervised three-factor rule moves only existing Kenyon-cell-to-MBON synapses during teacher-forced training. A fixed MBON-group vote is the readout; recall feeds the decoded letter back as the next cue.',
        evaluation: 'Diagnostic episodes are teacher-forced with learning off. Recall starts from START and feeds each decoded letter back as the next cue; END stops an episode and a fixed cap is independent of the phrase length. Chain length is the longest correct prefix. The reference is used only as the training teacher and for post-hoc scoring.',
        inferenceLimit: 'The learned transitions live in existing synapses of this simulation, but the teacher, the sparse codes, the slower time constants and the rescaled weights are imposed by the experimenter. Nothing here shows how flies learn, and a recited phrase is not memory for a text.'});
    }
  }
  const api = {MushroomBodyExperiment, summarizeMushroomBodyRun, validateAnnotation, partition, CONDITIONS, PHASES, DEFAULTS, START, END};
  root.FlyHamletMushroomBody = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
