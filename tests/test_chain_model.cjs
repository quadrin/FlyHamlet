/* Protocol tests use synthetic graphs and do not estimate biological memory.
 * Run: node --test tests/test_chain_model.cjs */
const test = require('node:test');
const assert = require('node:assert/strict');
const {predictLinear, fingerprint, START, END, OUTPUT_CODES} = require('../site/recall-model.js');
const {SequenceExperiment, applyScaler, fitScaler} = require('../site/sequence-model.js');
const {ChainExperiment, summarizeChainRun, summarizeChainEpisode, chainLength, measureState, CONDITIONS, DEFAULTS} = require('../site/chain-model.js');

function fixture(connected = true) {
  const n = 16, indices = [], weights = [], indptr = [0];
  for (let neuron = 0; neuron < n; ++neuron) {
    if (connected && neuron < 8) { indices.push(neuron + 8); weights.push(80); }
    indptr.push(indices.length);
  }
  const graph = {n, indptr: new Uint32Array(indptr), indices: new Uint32Array(indices), weights: new Float32Array(weights)};
  const manifest = {config: {sim: {dt_ms: 1, v_rest_mV: -52, v_reset_mV: -52, v_thresh_mV: -45,
    tau_mem_ms: 20, tau_syn_ms: 5, t_refractory_ms: 2.2, delay_ms: 2,
    poisson_weight_mV: 68.75, dtype: 'float32', rest_eps_mV: 1e-5, background: {enabled: false}}},
    targets: {eye_L: [0, 1, 2, 3], eye_R: [4, 5, 6, 7]}, layout: Array.from('abcdefghijklmnopqrstuvwxyz ')};
  return {graph, manifest};
}
const SMALL = {trainEpisodes: 2, diagnosticEpisodes: 1, recallEpisodes: 2, cueRateHz: 1000};
function make(seed = 42, options = {}, connected = true) {
  const {graph, manifest} = fixture(connected);
  return new ChainExperiment(graph, manifest, seed, {...SMALL, ...options});
}
function finish(exp) {
  const returned = [];
  while (!exp.done) { const record = exp.step(); if (record) returned.push(record); }
  return {result: exp.result(), returned};
}

test('phases, budgets and the current-cue-only comparator are laid out as declared', () => {
  const exp = make();
  assert.equal(exp.decisionsPerEpisode, 19);
  assert.equal(exp.perConditionBudget, 3 * 19 + 2 * 32);
  assert.equal(exp.comparator.episode.output, 'to be be be be be be be be be be');
  assert.equal(exp.comparator.episode.chainLength, 6);
  assert.equal(exp.comparator.deterministic, true);
  const {result} = finish(exp);
  assert.equal(result.fitCount, 3);
  assert.equal(result.episodes.length, 6);
  assert(result.trials.length <= 3 * exp.perConditionBudget && result.trials.length >= 3 * (3 * 19 + 2));
  assert.equal(result.trials.length, exp.totalTrialBudget, 'the budget shrinks by exactly the unused decisions of ended episodes');
  assert.equal(result.simTimeS, result.trials.length * 0.125);
  for (const condition of CONDITIONS) {
    const rows = result.trials.filter(row => row.condition === condition);
    assert.equal(rows.filter(row => row.phase === 'train').length, 38);
    assert.equal(rows.filter(row => row.phase === 'diagnostic').length, 19);
    assert.equal(result.metrics[condition].recall.episodes, 2);
    assert.equal(result.metrics[condition].diagnostic.total, 19);
  }
});

test('retained conditions run one network per episode, reset replaces it at every cue, and seeds pair across conditions', () => {
  const exp = make();
  const seen = [];
  let episodeKey = null, net = null;
  while (!exp.done) {
    const before = exp.net;
    const record = exp.step();
    if (exp.cueStep === 1) {
      const key = `${exp.condition}/${exp.phase}/${exp.episode}`;
      if (exp.condition !== 'reset') {
        if (key === episodeKey) assert.equal(exp.net, net, 'the same network continues within an episode');
        else assert.notEqual(exp.net, net, 'a fresh network starts each episode');
      } else assert.notEqual(exp.net, before, 'reset builds a new network for every cue');
      episodeKey = key; net = exp.net;
      seen.push({condition: exp.condition, phase: exp.phase, episode: exp.episode, step: exp.decisionStep, stepIndex: exp.net.stepIndex});
    }
    if (record && exp.condition === 'reset') assert.equal(record.decisionStep, undefined);
  }
  for (const item of seen) {
    if (item.condition === 'reset') assert.equal(item.stepIndex, 1, 'every reset cue starts from a resting network');
    else assert.equal(item.stepIndex, (item.step - 1) * 125 + 1, 'retained networks carry their clock across cues');
  }
  const {result} = {result: exp.result()};
  const original = result.trials.filter(row => row.condition === 'original' && row.phase !== 'recall');
  for (const condition of ['slow', 'reset']) {
    const rows = result.trials.filter(row => row.condition === condition && row.phase !== 'recall');
    assert.equal(rows.length, original.length);
    rows.forEach((row, i) => {
      for (const key of ['phase', 'episode', 'step', 'cue', 'target', 'episodeSeed']) assert.equal(row[key], original[i][key]);
      if (condition === 'reset') { assert.notEqual(row.cueSeed, row.episodeSeed); assert.equal(row.activeNeurons >= 0, true); }
      else assert.equal(row.cueSeed, row.episodeSeed);
    });
  }
  const resetSeeds = result.trials.filter(row => row.condition === 'reset').map(row => row.cueSeed);
  assert.equal(new Set(resetSeeds).size, resetSeeds.length, 'each reset cue has its own noise seed');
});

test('cues follow the phrase while teacher-forced and the actor\'s own predictions during recall', () => {
  const {result} = finish(make());
  const inputs = [START, ...DEFAULTS.phrase], targets = [...DEFAULTS.phrase, END];
  for (const row of result.trials) {
    if (row.phase !== 'recall') {
      assert.equal(row.cue, inputs[row.step - 1]); assert.equal(row.target, targets[row.step - 1]);
      assert.equal(row.feedbackApplied, false);
    }
  }
  for (const episode of result.episodes) {
    const rows = result.trials.filter(row => row.condition === episode.condition && row.phase === 'recall' && row.episode === episode.episode);
    rows.forEach((row, index) => {
      assert.equal(row.cue, index === 0 ? START : rows[index - 1].prediction);
      assert.equal(row.feedbackApplied, index > 0); assert.equal(row.target, null); assert.equal(row.correct, null);
    });
    assert.equal(episode.output, rows.map(row => row.prediction).filter(code => code !== END).join(''));
    assert.equal(episode.stoppedBy, rows.at(-1).prediction === END ? END : 'cap');
    assert(rows.length <= 32);
    assert.equal(episode.chainLength, chainLength(episode.output, DEFAULTS.phrase));
  }
  assert(result.episodes.some(episode => episode.condition === 'slow' && episode.exact), 'the slow fixture recalls the phrase');
});

test('one scaler and one readout per condition fit only training snapshots, then stay frozen for diagnostics and recall', () => {
  const {result} = finish(make());
  assert.equal(result.trials.filter(row => row.fitApplied).length, 3);
  for (const condition of CONDITIONS) {
    const model = result.models[condition];
    assert.equal(model.trainingExamples, 38); assert.equal(model.dimensions, 256);
    assert.equal(model.weights.length, 257 * 8);
    assert.deepEqual(model.weights, model.frozenWeights);
    assert.equal(model.fingerprint, model.finalFingerprint); assert.equal(fingerprint(model.weights), model.fingerprint);
    assert.equal(model.scaler.fingerprint, model.scaler.finalFingerprint);
    const rows = result.trials.filter(row => row.condition === condition);
    assert.deepEqual(fitScaler(rows.filter(row => row.phase === 'train').map(row => row.measurements)).mean, model.scaler.mean);
    for (const row of rows) {
      if (row.phase === 'train') { assert.equal(row.prediction, null); assert.equal(row.features, null); continue; }
      assert.equal(row.updateCountBefore, row.updateCountAfter); assert.equal(row.decoderVersion, 1);
      assert.deepEqual(row.features, applyScaler(model.scaler, row.measurements));
      const independent = predictLinear(model.weights, row.features, OUTPUT_CODES);
      assert.equal(row.prediction, independent.prediction); assert.deepEqual(row.scores, independent.scores);
      assert.equal(row.decoderFingerprint, model.fingerprint); assert.equal(row.scalerFingerprint, model.scaler.fingerprint);
    }
  }
});

test('the state measurement is identical to the Sequence lab measurement', () => {
  const {graph, manifest} = fixture();
  const sequence = new SequenceExperiment(graph, manifest, 9, {trainPerPair: 1, evaluationPerPair: 1, cueRateHz: 1000});
  for (let step = 0; step < 60; ++step) sequence.step();
  const expected = sequence.measure(sequence.cueEndPlan);
  const actual = measureState(sequence.net, sequence.poolForNeuron, sequence.poolSizes, 128, 0.01);
  assert.deepEqual(actual.measurements, expected.measurements);
  assert.equal(actual.nonrestVoltageNeurons, expected.nonrestVoltageNeurons);
  assert.equal(actual.nonrestCurrentNeurons, expected.nonrestCurrentNeurons);
  assert.equal(actual.maxAbsVoltageMv, expected.maxAbsVoltageMv);
  assert(actual.nonrestVoltageNeurons > 0);
});

test('sensory cells never enter measurements; with no signal every condition types the same constant string', () => {
  const exp = make(5, {}, false);
  for (let neuron = 0; neuron < 8; ++neuron) assert.equal(exp.poolForNeuron[neuron], -1);
  const {result} = finish(exp);
  for (const row of result.trials) assert(row.silent && row.measurements.every(value => value === 0));
  const outputs = new Set(result.episodes.map(episode => episode.output));
  assert.equal(outputs.size, 1);
  for (const condition of CONDITIONS) assert(result.metrics[condition].diagnostic.accuracy <= 5 / 19, 'a constant prediction cannot beat the most frequent target');
});

test('episode scoring counts the longest correct prefix and requires END for an exact match', () => {
  assert.equal(chainLength('to bx', 'to be'), 4); assert.equal(chainLength('', 'to be'), 0); assert.equal(chainLength('to be or', 'to be'), 5);
  const capped = summarizeChainEpisode('slow', 'recall', 1, 'to be', 'cap', [], 'to be');
  assert.equal(capped.exact, false); assert.equal(capped.chainLength, 5); assert.equal(capped.correctCharacters, 5);
  const ended = summarizeChainEpisode('slow', 'recall', 2, 'to be', END, [], 'to be');
  assert.equal(ended.exact, true); assert.equal(ended.correctCharacters, 6); assert.equal(ended.editDistance, 0);
  const metrics = summarizeChainRun([capped, ended], [], 'to be');
  assert.deepEqual(metrics.slow.recall.chainLengths, [5, 5]); assert.equal(metrics.slow.recall.exact, 1);
  assert.equal(metrics.slow.recall.ended, 1); assert.equal(metrics.slow.recall.capped, 1);
  assert.equal(metrics.original.recall.meanChainLength, null);
});

test('seeds reproduce, returned snapshots are isolated, and invalid protocols are rejected', () => {
  const {result, returned} = finish(make(7));
  assert.deepEqual(result, finish(make(7)).result);
  assert.deepEqual(result.metrics, summarizeChainRun(result.episodes, result.trials, result.reference));
  returned[0].measurements[0] = -10;
  assert.notEqual(result.trials[0].measurements[0], -10);
  for (const options of [{phrase: 'tox'}, {phrase: ''}, {trainEpisodes: 0}, {recallEpisodes: 0}, {maxDecisions: 0}, {diagnosticEpisodes: -1},
    {cueMs: 100.5}, {gapMs: -1}, {ridge: 0}, {cueRateHz: 1001}, {stateResolutionMv: 0}, {slowTimeFactor: 0}, {slowWeightScale: 0}])
    assert.throws(() => make(1, options), `${JSON.stringify(options)} should be rejected`);
  const {graph, manifest} = fixture();
  manifest.config.sim.background = {enabled: true, rate_hz: 1};
  assert.throws(() => new ChainExperiment(graph, manifest, 1), /background/);
});
