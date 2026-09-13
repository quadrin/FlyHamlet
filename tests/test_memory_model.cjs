/* Protocol tests use synthetic graphs and do not estimate biological memory.
 * Run: node --test tests/test_memory_model.cjs */
const test = require('node:test');
const assert = require('node:assert/strict');
const {LIFNetwork} = require('../site/live-model.js');
const {predictLinear, fingerprint} = require('../site/recall-model.js');
const {MemoryExperiment, makeMemoryPlan, summarizeMemoryTrials, DEFAULTS, LABELS} = require('../site/memory-model.js');

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
function make(seed = 42, options = {}, connected = true) {
  const {graph, manifest} = fixture(connected);
  return new MemoryExperiment(graph, manifest, seed,
    {trainPerClass: 1, evaluationPerClass: 1, cueRateHz: 1000, ...options},
    {graph, metadata: {testControl: true}});
}
function finish(exp) {
  const returned = [];
  while (!exp.done) { const record = exp.step(); if (record) returned.push(record); }
  return {result: exp.result(), returned};
}

test('train/evaluation classes are balanced and noise is independent of shuffled labels', () => {
  const plan = makeMemoryPlan(431, DEFAULTS);
  assert.equal(plan.length, 112);
  for (const phase of ['train', 'evaluation']) {
    const rows = plan.filter(row => row.phase === phase);
    assert.equal(rows.length, 56);
    for (const label of LABELS) assert.equal(rows.filter(row => row.cue === label).length, 8);
  }
  assert.equal(new Set(plan.map(row => row.neuralSeed)).size, plan.length);
  const alternate = makeMemoryPlan(431, {...DEFAULTS, trainPerClass: 4, evaluationPerClass: 12});
  assert.deepEqual(plan.map(row => row.neuralSeed), alternate.map(row => row.neuralSeed));
  assert(plan.some((row, i) => row.cue !== alternate[i].cue));
});

test('all conditions receive paired cues/noise, and retain/reset cue-on diagnostics match exactly', () => {
  const {result} = finish(make(79, {cueRateHz: 100}));
  const retained = result.trials.filter(row => row.condition === 'retain');
  for (const condition of ['reset', 'rewired']) {
    const rows = result.trials.filter(row => row.condition === condition);
    rows.forEach((row, i) => {
      assert.equal(row.cue, retained[i].cue); assert.equal(row.neuralSeed, retained[i].neuralSeed);
      assert.equal(row.pairedTrial, retained[i].pairedTrial);
      if (condition === 'reset') {
        assert.equal(row.cueSpikes, retained[i].cueSpikes);
        assert.deepEqual(row.cueReadout.features, retained[i].cueReadout.features);
        assert.deepEqual(row.cueReadout.poolSpikeCounts, retained[i].cueReadout.poolSpikeCounts);
      }
    });
  }
});

test('input turns off before post-cue sampling; queued retained activity is allowed but all late windows fall silent', () => {
  const exp = make();
  for (let step = 0; step < 200; ++step) exp.step();
  const net = exp.net;
  assert.equal(exp.trialStep, 200); assert.equal(exp.cueOffsetApplied, false);
  assert.equal(net.poisson[0].p, 1);
  exp.step();
  assert.equal(exp.net, net); assert.equal(exp.cueOffsetApplied, true);
  assert.equal(net.poisson[0].p, 0);
  let record = null;
  while (!record) record = exp.step();
  assert(record.postCueSpikes > 0, 'synaptic/state transients may survive cue removal');
  assert(record.windows.some(window => window.populationSpikes > 0));
  assert.equal(record.windows.at(-1).populationSpikes, 0);
  for (const window of record.windows) {
    assert.equal(window.inputActive, false);
    assert(window.startMs >= record.inputsOffFromMs);
    assert.equal(window.endMs - window.startMs, 20);
  }
  assert.equal(record.cueReadout.inputActive, true);
  assert.equal(record.cueReadout.startMs, 180); assert.equal(record.cueReadout.endMs, 200);
});

test('reset replaces all neural state at offset, with no input or pending spikes remaining', () => {
  const exp = make();
  while (exp.condition !== 'reset' || exp.trialStep < 200) exp.step();
  const old = exp.net;
  assert(old.totalSpikes > 0);
  exp.step();
  assert.notEqual(exp.net, old);
  assert.equal(exp.net.stepIndex, 1);
  assert.equal(exp.net.totalSpikes, 0); assert.equal(exp.net.activeCount, 0);
  assert.equal(exp.net.poisson.length, 0);
  assert(exp.net.queue.every(slot => slot.length === 0));
  assert([...exp.net.v].every(value => value === -52));
  assert([...exp.net.g].every(value => value === 0));
  assert([...exp.net.current].every(value => value === 0));
  assert([...exp.net.resumeAt].every(value => value === 0));
  assert.notDeepEqual(exp.net.rng.s, old.rng.s);
});

test('balanced zero-state reset is exactly chance at every delay, while cue-on encoding is positive', () => {
  const {result} = finish(make());
  assert.equal(result.metrics.retain.cueReadout.accuracy, 1);
  assert.equal(result.metrics.reset.cueReadout.accuracy, 1);
  for (const metric of result.metrics.reset.delays) {
    assert.equal(metric.accuracy, 1 / 7); assert.equal(metric.balancedAccuracy, 1 / 7);
    assert.equal(metric.silentTrials, 7); assert.equal(metric.readoutSilentTrials, 7);
  }
  for (const row of result.trials.filter(row => row.condition === 'reset')) {
    assert.equal(row.postCueSpikes, 0);
    for (const window of row.windows) {
      assert(window.features.every(value => value === 0)); assert.equal(window.activeNeurons, 0);
    }
  }
});

test('independent post-cue decoders and separate cue diagnostic fit only training data, then stay frozen', () => {
  const {result} = finish(make());
  assert.equal(result.fitCount, 3); assert.equal(result.readoutFitCount, 21);
  assert.equal(result.trials.filter(row => row.fitApplied).length, 3);
  for (const condition of ['retain', 'reset', 'rewired']) {
    const model = result.models[condition];
    assert.equal(model.trainingExamples, 7);
    assert.equal(model.readouts.length, 6);
    for (const readout of [model.cueReadout, ...model.readouts]) {
      assert.equal(readout.dimensions, 128, 'no external history or previous windows');
      assert.equal(readout.weights.length, 129 * 7);
      assert.deepEqual(readout.weights, readout.frozenWeights);
      assert.equal(readout.fingerprint, readout.finalFingerprint);
      assert.equal(fingerprint(readout.weights), readout.fingerprint);
    }
    const rows = result.trials.filter(row => row.condition === condition);
    for (const row of rows) {
      if (row.phase === 'train') {
        assert([row.cueReadout, ...row.windows].every(window => window.prediction === null));
        continue;
      }
      assert.equal(row.updateCountBefore, row.updateCountAfter);
      assert.equal(row.decoderVersion, 1); assert.equal(row.feedbackApplied, false);
      [row.cueReadout, ...row.windows].forEach((window, index) => {
        const fit = index === 0 ? model.cueReadout : model.readouts[index - 1];
        const independentlyPredicted = predictLinear(fit.weights, window.features, LABELS);
        assert.equal(window.prediction, independentlyPredicted.prediction);
        assert.deepEqual(window.scores, independentlyPredicted.scores);
        assert.equal(window.correct, window.prediction === row.cue);
        assert.equal(window.decoderFingerprint, fit.fingerprint);
      });
    }
  }
});

test('sensory cells never enter features, and no signal means no hidden label/history leakage', () => {
  const exp = make(491, {}, false);
  for (let neuron = 0; neuron < 8; ++neuron) assert.equal(exp.poolForNeuron[neuron], -1);
  assert.equal(exp.metadata.sensoryExcludedCount, 8);
  const {result} = finish(exp);
  assert(result.trials.some(row => row.cueReadout.populationSpikes > 0));
  for (const row of result.trials) for (const window of [row.cueReadout, ...row.windows])
    assert(window.features.every(value => value === 0));
  for (const condition of ['retain', 'reset', 'rewired']) {
    assert.equal(result.metrics[condition].cueReadout.accuracy, 1 / 7);
    assert(result.metrics[condition].delays.every(metric => metric.accuracy === 1 / 7));
  }
});

test('cue-on and every overlapping post-cue window independently reproduce with exact half-open boundaries', () => {
  const {graph, manifest} = fixture();
  const exp = new MemoryExperiment(graph, manifest, 745, {trainPerClass: 1, evaluationPerClass: 1,
    cueRateHz: 100}, {graph, metadata: {test: true}});
  let record = null; while (!record) record = exp.step();
  const net = new LIFNetwork(graph, manifest.config.sim, record.neuralSeed);
  const input = net.injectPoisson(exp.codebook[record.cue], 100);
  const windows = [record.cueReadout, ...record.windows];
  const counts = windows.map(() => new Array(128).fill(0));
  const populations = windows.map(() => 0), unique = windows.map(() => new Set());
  for (let step = 0; step < 420; ++step) {
    if (step === 200) net.setPoissonRate(input, 0);
    const spikes = net.step();
    windows.forEach((window, i) => {
      if (step >= window.startMs && step < window.endMs) {
        populations[i] += spikes.length;
        for (const neuron of spikes) {
          unique[i].add(neuron);
          const pool = exp.poolForNeuron[neuron]; if (pool >= 0) counts[i][pool]++;
        }
      }
    });
  }
  windows.forEach((window, i) => {
    assert.deepEqual(window.poolSpikeCounts, counts[i]);
    assert.equal(window.populationSpikes, populations[i]); assert.equal(window.activeNeurons, unique[i].size);
    assert.deepEqual(window.features, counts[i].map((count, pool) => exp.poolSizes[pool] ? count / (exp.poolSizes[pool] * .02) : 0));
  });
});

test('the supplied control graph is used without changing either graph', () => {
  const {graph, manifest} = fixture();
  const control = {...graph, weights: new Float32Array(graph.weights.length)};
  const original = [...graph.weights], beforeControl = [...control.weights];
  const exp = new MemoryExperiment(graph, manifest, 11, {trainPerClass: 1, evaluationPerClass: 1,
    cueRateHz: 1000}, {graph: control, metadata: {zeroWeightTest: true}});
  const {result} = finish(exp);
  assert.deepEqual([...graph.weights], original); assert.deepEqual([...control.weights], beforeControl);
  assert.equal(result.metrics.retain.cueReadout.accuracy, 1);
  assert.equal(result.metrics.rewired.cueReadout.accuracy, 1 / 7);
  assert.deepEqual(result.rewireMetadata, {zeroWeightTest: true});
});

test('one step means one neural timestep, all trials are retained, snapshots are isolated and seeds reproduce', () => {
  const exp = make(32, {cueRateHz: 100});
  assert.equal(exp.step(), null); assert.equal(exp.simTimeS, .001); assert.equal(exp.progress.trialTimeMs, 1);
  const {result, returned} = finish(exp);
  assert.equal(result.trials.length, 42);
  assert.equal(result.simTimeS, 17.64);
  assert.deepEqual(result.trials.map(row => row.trial), Array.from({length: 42}, (_, i) => i + 1));
  assert.equal(exp.progress.completed, exp.progress.totalTrials); assert.equal(exp.step(), null);
  assert.deepEqual(result, finish(make(32, {cueRateHz: 100})).result);
  assert.deepEqual(result.metrics, summarizeMemoryTrials(result.trials));
  returned[0].windows[0].features[0] = -10;
  result.trials[0].cueReadout.features[0] = -20;
  assert.notEqual(exp.result().trials[0].windows[0].features[0], -10);
  assert.notEqual(exp.result().trials[0].cueReadout.features[0], -20);
});

test('invalid timing, input, readout and control conditions are rejected', () => {
  for (const options of [{delaysMs: [-1]}, {delaysMs: [0, 0]}, {delaysMs: []}, {cueMs: 200.5},
    {windowMs: 201}, {trainPerClass: 0}, {evaluationPerClass: 0}, {ridge: 0}, {cueRateHz: 1001}])
    assert.throws(() => make(1, options));
  const {graph, manifest} = fixture();
  assert.throws(() => new MemoryExperiment(graph, manifest, 1), /rewired/);
  manifest.config.sim.background = {enabled: true, rate_hz: 1};
  assert.throws(() => new MemoryExperiment(graph, manifest, 1, {}, {graph}), /background/);
});
