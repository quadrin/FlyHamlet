/* Protocol tests use synthetic graphs and do not estimate biological memory.
 * Run: node --test tests/test_sequence_model.cjs */
const test = require('node:test');
const assert = require('node:assert/strict');
const {LIFNetwork} = require('../site/live-model.js');
const {predictLinear, fingerprint} = require('../site/recall-model.js');
const {SequenceExperiment, makeSequencePlan, summarizeSequenceTrials, prepareSlowGraph, slowNeuralConfig,
  fitScaler, applyScaler, wilson, DEFAULTS, LETTERS, PAIRS, CONDITIONS, POSITIONS} = require('../site/sequence-model.js');

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
  return new SequenceExperiment(graph, manifest, seed, {trainPerPair: 2, evaluationPerPair: 2, cueRateHz: 1000, ...options});
}
function finish(exp) {
  const returned = [];
  while (!exp.done) { const record = exp.step(); if (record) returned.push(record); }
  return {result: exp.result(), returned};
}

test('train/evaluation pairs are balanced and noise is independent of shuffled labels', () => {
  const plan = makeSequencePlan(431, DEFAULTS);
  assert.equal(plan.length, 128);
  for (const phase of ['train', 'evaluation']) {
    const rows = plan.filter(row => row.phase === phase);
    assert.equal(rows.length, 64);
    for (const pair of PAIRS) assert.equal(rows.filter(row => row.pair === pair).length, 16);
  }
  assert(plan.every(row => row.pair === row.first + row.second && LETTERS.includes(row.first) && LETTERS.includes(row.second)));
  assert.equal(new Set(plan.map(row => row.neuralSeed)).size, plan.length);
  const alternate = makeSequencePlan(431, {...DEFAULTS, trainPerPair: 8, evaluationPerPair: 24});
  assert.deepEqual(plan.map(row => row.neuralSeed), alternate.map(row => row.neuralSeed));
  assert(plan.some((row, i) => row.pair !== alternate[i].pair));
});

test('all conditions receive paired cues/noise, and slow/reset final-cue snapshots match exactly', () => {
  const {result} = finish(make(79, {cueRateHz: 100}));
  const original = result.trials.filter(row => row.condition === 'original');
  const slow = result.trials.filter(row => row.condition === 'slow');
  for (const condition of ['slow', 'reset']) {
    const rows = result.trials.filter(row => row.condition === condition);
    rows.forEach((row, i) => {
      assert.equal(row.pair, original[i].pair); assert.equal(row.neuralSeed, original[i].neuralSeed);
      assert.equal(row.pairedTrial, original[i].pairedTrial);
      if (condition === 'reset') {
        assert.equal(row.cueSpikes, slow[i].cueSpikes);
        assert.deepEqual(row.cueEnd.measurements, slow[i].cueEnd.measurements);
        assert.equal(row.cueEnd.nonrestVoltageNeurons, slow[i].cueEnd.nonrestVoltageNeurons);
      }
    });
  }
});

test('letters drive only their own code in order, and both inputs are off before the first post-cue step', () => {
  const exp = make();
  const item = exp.plan[0];
  const rates = [];
  for (let step = 0; step < 226; ++step) {
    exp.step();
    rates.push({step, t: exp.net.poisson[0].p, o: exp.net.poisson[1].p, off: exp.cueOffsetApplied});
  }
  const rateOf = (row, letter) => letter === 't' ? row.t : row.o;
  const other = letter => letter === 't' ? 'o' : 't';
  for (const row of rates.slice(0, 100)) { assert.equal(rateOf(row, item.first), 1); assert.equal(rateOf(row, other(item.first)), 0); assert.equal(row.off, false); }
  for (const row of rates.slice(100, 125)) { assert.equal(row.t, 0); assert.equal(row.o, 0); assert.equal(row.off, false); }
  for (const row of rates.slice(125, 225)) { assert.equal(rateOf(row, item.second), 1); assert.equal(rateOf(row, other(item.second)), 0); assert.equal(row.off, false); }
  assert.deepEqual(rates[225], {step: 225, t: 0, o: 0, off: true});
  assert.equal(exp.observations.length, 1, 'the final-cue snapshot is taken at input off');
  assert.equal(exp.observations[0].timeMs, 225); assert.equal(exp.observations[0].inputActive, true);
  let record = null;
  while (!record) record = exp.step();
  assert.deepEqual(record.snapshots.map(snapshot => [snapshot.delayMs, snapshot.timeMs, snapshot.inputActive]), [[25, 250, false], [100, 325, false], [200, 425, false]]);
  assert.deepEqual(record.firstLetterMs, [0, 100]); assert.deepEqual(record.secondLetterMs, [125, 225]);
  assert.equal(record.inputsOffFromMs, 225);
  assert(record.cueSpikes > 0);
});

test('reset replaces all neural state at input off, after the final-cue snapshot was taken', () => {
  const exp = make();
  while (exp.condition !== 'reset' || exp.trialStep < 225) exp.step();
  const old = exp.net;
  assert(old.totalSpikes > 0); assert.equal(exp.observations.length, 0);
  exp.step();
  assert.notEqual(exp.net, old);
  assert.equal(exp.observations.length, 1);
  assert(exp.observations[0].nonrestVoltageNeurons > 0, 'the diagnostic snapshot precedes the reset');
  assert.equal(exp.net.stepIndex, 1);
  assert.equal(exp.net.totalSpikes, 0); assert.equal(exp.net.activeCount, 0);
  assert.equal(exp.net.poisson.length, 0); assert.equal(exp.inputs, null);
  assert(exp.net.queue.every(slot => slot.length === 0));
  assert([...exp.net.v].every(value => value === -52));
  assert([...exp.net.g].every(value => value === 0));
  assert.notDeepEqual(exp.net.rng.s, old.rng.s);
  assert.equal(exp.net.cfg.tau_mem_ms, 200, 'the reset control uses the slower model');
});

test('balanced reset snapshots are exactly chance and silent, while slow-model encoding is decodable', () => {
  const {result} = finish(make());
  assert.equal(result.metrics.slow.cueEnd.second.accuracy, 1);
  assert.equal(result.metrics.reset.cueEnd.second.accuracy, 1);
  for (const metric of result.metrics.reset.delays) {
    assert.equal(metric.accuracy, 0.25); assert.equal(metric.balancedAccuracy, 0.25);
    assert.equal(metric.first.accuracy, 0.5); assert.equal(metric.second.accuracy, 0.5);
    assert.equal(metric.silentTrials, 8); assert.equal(metric.meanNonrestVoltageNeurons, 0);
    assert.equal(metric.meanSpikesSinceCueOff, 0);
  }
  for (const row of result.trials.filter(row => row.condition === 'reset')) {
    assert.equal(row.postCueSpikes, 0);
    for (const snapshot of row.snapshots) {
      assert(snapshot.measurements.every(value => value === 0)); assert.equal(snapshot.activeNeurons, 0);
      assert.equal(snapshot.silent, true);
      if (row.phase === 'evaluation') assert(snapshot.features.every(value => value === 0));
    }
  }
  for (const metric of result.metrics.slow.delays) assert(metric.meanNonrestVoltageNeurons > 0, 'slow dynamics retain subthreshold state');
});

test('per-snapshot scalers and per-position decoders fit only training data, then stay frozen', () => {
  const {result} = finish(make());
  assert.equal(result.fitCount, 3); assert.equal(result.readoutFitCount, 24);
  assert.equal(result.trials.filter(row => row.fitApplied).length, 3);
  for (const condition of CONDITIONS) {
    const model = result.models[condition];
    assert.equal(model.trainingExamples, 8);
    assert.equal(model.readouts.length, 4);
    const rows = result.trials.filter(row => row.condition === condition);
    model.readouts.forEach((readout, index) => {
      assert.equal(readout.scaler.dimensions, 256);
      assert.equal(readout.scaler.fingerprint, readout.scaler.finalFingerprint);
      const trainingRows = rows.filter(row => row.phase === 'train').map(row => (index === 0 ? row.cueEnd : row.snapshots[index - 1]).measurements);
      assert.deepEqual(fitScaler(trainingRows).mean, readout.scaler.mean);
      for (const position of POSITIONS) {
        const fit = readout.positions[position];
        assert.equal(fit.dimensions, 256, 'no external history or previous snapshots');
        assert.equal(fit.weights.length, 257 * 2);
        assert.deepEqual(fit.weights, fit.frozenWeights);
        assert.equal(fit.fingerprint, fit.finalFingerprint);
        assert.equal(fingerprint(fit.weights), fit.fingerprint);
      }
    });
    for (const row of rows) {
      if (row.phase === 'train') {
        assert([row.cueEnd, ...row.snapshots].every(snapshot => snapshot.predictions === null && snapshot.features === null));
        continue;
      }
      assert.equal(row.updateCountBefore, row.updateCountAfter);
      assert.equal(row.decoderVersion, 1); assert.equal(row.feedbackApplied, false);
      [row.cueEnd, ...row.snapshots].forEach((snapshot, index) => {
        const readout = model.readouts[index];
        assert.deepEqual(snapshot.features, applyScaler(readout.scaler, snapshot.measurements));
        assert.equal(snapshot.scalerFingerprint, readout.scaler.fingerprint);
        for (const position of POSITIONS) {
          const independent = predictLinear(readout.positions[position].weights, snapshot.features, LETTERS);
          assert.equal(snapshot.predictions[position].prediction, independent.prediction);
          assert.deepEqual(snapshot.predictions[position].scores, independent.scores);
          assert.equal(snapshot.predictions[position].correct, independent.prediction === row[position]);
          assert.equal(snapshot.predictions[position].decoderFingerprint, readout.positions[position].fingerprint);
        }
        assert.equal(snapshot.pairPrediction, snapshot.predictions.first.prediction + snapshot.predictions.second.prediction);
        assert.equal(snapshot.pairCorrect, snapshot.pairPrediction === row.pair);
      });
    }
  }
});

test('sensory cells never enter measurements, and no signal means exact chance everywhere', () => {
  const exp = make(491, {}, false);
  for (let neuron = 0; neuron < 8; ++neuron) assert.equal(exp.poolForNeuron[neuron], -1);
  assert.equal(exp.metadata.sensoryExcludedCount, 8);
  const {result} = finish(exp);
  assert(result.trials.every(row => row.cueSpikes > 0), 'sensory cells themselves do spike');
  for (const row of result.trials) for (const snapshot of [row.cueEnd, ...row.snapshots])
    assert(snapshot.measurements.every(value => value === 0) && snapshot.silent);
  for (const condition of CONDITIONS) {
    for (const metric of [result.metrics[condition].cueEnd, ...result.metrics[condition].delays]) {
      assert.equal(metric.accuracy, 0.25); assert.equal(metric.first.accuracy, 0.5); assert.equal(metric.second.accuracy, 0.5);
    }
  }
});

test('measurements round each neuron to the fixed resolution, so frozen float32 remnants read as rest', () => {
  const exp = make();
  exp.startTrial();
  const net = exp.net, pool = neuron => exp.poolForNeuron[neuron], size = neuron => exp.poolSizes[pool(neuron)];
  net.v[8] = net.v0 + 0.004;                 // below half a quantum
  net.v[9] = net.v0 + 0.006;                 // rounds to one quantum
  net.v[10] = Math.fround(net.v0 + 0.0038);  // a float32 fixed point the model could retire
  net.g[11] = -0.017;                        // rounds to minus two quanta
  net.v[0] = net.v0 + 5;                     // stimulated sensory cell: excluded
  const snapshot = exp.measure(exp.cueEndPlan);
  assert.equal(snapshot.nonrestVoltageNeurons, 1); assert.equal(snapshot.nonrestCurrentNeurons, 1);
  assert.equal(snapshot.maxAbsVoltageMv, 0.01); assert.equal(snapshot.maxAbsCurrentMv, 0.02);
  assert.equal(snapshot.measurements[pool(9)], 0.01 / size(9));
  assert.equal(snapshot.measurements[128 + pool(11)], -0.02 / size(11));
  assert.equal(snapshot.measurements.filter(value => value !== 0).length, 2);
  assert.equal(snapshot.silent, false);
  net.v[9] = net.v0; net.g[11] = 0;
  assert.equal(exp.measure(exp.cueEndPlan).silent, true);
  const coarse = make(42, {stateResolutionMv: 0.1});
  coarse.startTrial(); coarse.net.v[9] = coarse.net.v0 + 0.04;
  assert.equal(coarse.measure(coarse.cueEndPlan).silent, true);
});

test('every snapshot independently reproduces from a direct simulation with the same timing', () => {
  const {graph, manifest} = fixture();
  const exp = new SequenceExperiment(graph, manifest, 745, {trainPerPair: 1, evaluationPerPair: 1, cueRateHz: 100});
  const {result} = finish(exp);
  const replay = record => {
    const original = record.condition === 'original';
    const net = new LIFNetwork(original ? graph : exp.slowGraph, original ? manifest.config.sim : exp.slowConfig, record.neuralSeed);
    const inputs = Object.fromEntries(LETTERS.map(letter => [letter, net.injectPoisson(exp.codebook[letter], 0)]));
    const setLetter = letter => { for (const code of LETTERS) net.setPoissonRate(inputs[code], code === letter ? 100 : 0); };
    const measure = () => {
      const quanta = new Array(256).fill(0);
      for (let neuron = 0; neuron < graph.n; ++neuron) {
        const pool = exp.poolForNeuron[neuron]; if (pool < 0) continue;
        quanta[pool] += Math.round((net.v[neuron] - net.v0) / 0.01);
        quanta[128 + pool] += Math.round(net.g[neuron] / 0.01);
      }
      return quanta.map((value, i) => exp.poolSizes[i % 128] ? value * 0.01 / exp.poolSizes[i % 128] : 0);
    };
    const expected = {};
    for (let step = 0; step < 425; ++step) {
      if (step === 0) setLetter(record.first);
      if (step === 100) setLetter(null);
      if (step === 125) setLetter(record.second);
      if (step === 225) { expected[0] = measure(); setLetter(null); }
      net.step();
      for (const delay of [25, 100, 200]) if (step + 1 === 225 + delay) expected[delay] = measure();
    }
    assert.deepEqual(record.cueEnd.measurements, expected[0]);
    for (const snapshot of record.snapshots) assert.deepEqual(snapshot.measurements, expected[snapshot.delayMs]);
    return expected;
  };
  const replayed = result.trials.filter(record => record.condition !== 'reset').map(record => ({condition: record.condition, expected: replay(record)}));
  assert.equal(replayed.length, 16);
  assert(replayed.some(item => item.condition === 'slow' && item.expected[0].some(value => value !== 0) && item.expected[200].some(value => value !== 0)),
    'the slower model keeps a nonzero trace to replay');
});

test('the slower model rescales only time constants and weights, and never edits the original graph', () => {
  const {graph, manifest} = fixture();
  const original = [...graph.weights];
  const slow = prepareSlowGraph(graph, 0.1);
  assert.equal(slow.graph.indptr, graph.indptr); assert.equal(slow.graph.indices, graph.indices);
  assert.deepEqual([...slow.graph.weights], original.map(weight => Math.fround(weight * 0.1)));
  assert.deepEqual([...graph.weights], original);
  assert.equal(slow.metadata.weightScale, 0.1);
  const config = slowNeuralConfig(manifest.config.sim, 10);
  assert.equal(config.tau_mem_ms, 200); assert.equal(config.tau_syn_ms, 50);
  assert.equal(config.delay_ms, 2); assert.equal(config.t_refractory_ms, 2.2); assert.equal(config.v_thresh_mV, -45);
  const exp = new SequenceExperiment(graph, manifest, 11, {trainPerPair: 1, evaluationPerPair: 1, cueRateHz: 1000}, slow);
  assert.equal(exp.slowGraph, slow.graph);
  const {result} = finish(exp);
  assert.deepEqual([...graph.weights], original);
  assert.deepEqual(result.slowGraph, slow.metadata);
  assert.deepEqual(result.slowNeuralConfig, config);
  assert.throws(() => new SequenceExperiment(graph, manifest, 11, {}, prepareSlowGraph(graph, 0.5)), /weight scale/);
  const other = new SequenceExperiment(graph, manifest, 11, {slowWeightScale: 0.5}, prepareSlowGraph(graph, 0.5));
  assert.equal(other.slowMetadata.weightScale, 0.5);
});

test('one step means one neural timestep, all trials are retained, snapshots are isolated and seeds reproduce', () => {
  const exp = make(32, {cueRateHz: 100});
  assert.equal(exp.step(), null); assert.equal(exp.simTimeS, .001); assert.equal(exp.progress.trialTimeMs, 1);
  assert.equal(exp.progress.stage, 'first'); assert.equal(exp.progress.cue, exp.plan[0].first); assert.equal(exp.progress.inputActive, true);
  const {result, returned} = finish(exp);
  assert.equal(result.trials.length, 48);
  assert.equal(result.simTimeS, 20.4);
  assert.deepEqual(result.trials.map(row => row.trial), Array.from({length: 48}, (_, i) => i + 1));
  assert.equal(exp.progress.completed, exp.progress.totalTrials); assert.equal(exp.progress.stage, 'complete'); assert.equal(exp.step(), null);
  assert.deepEqual(result, finish(make(32, {cueRateHz: 100})).result);
  assert.deepEqual(result.metrics, summarizeSequenceTrials(result.trials));
  returned[0].snapshots[0].measurements[0] = -10;
  result.trials[0].cueEnd.measurements[0] = -20;
  assert.notEqual(exp.result().trials[0].snapshots[0].measurements[0], -10);
  assert.notEqual(exp.result().trials[0].cueEnd.measurements[0], -20);
});

test('scalers zero constant features, standardize the rest, and Wilson intervals are bounded', () => {
  const scaler = fitScaler([[1, 0, 2], [1, 2, 4], [1, 4, 6]]);
  assert.deepEqual(scaler.scale[0], 0); assert.equal(scaler.constantFeatures, 1);
  const rows = [[1, 0, 2], [1, 2, 4], [1, 4, 6]].map(row => applyScaler(scaler, row));
  assert(rows.every(row => row[0] === 0));
  for (const i of [1, 2]) {
    const mean = rows.reduce((sum, row) => sum + row[i], 0) / 3;
    const variance = rows.reduce((sum, row) => sum + (row[i] - mean) ** 2, 0) / 3;
    assert(Math.abs(mean) < 1e-12 && Math.abs(variance - 1) < 1e-12);
  }
  assert.throws(() => applyScaler(scaler, [1, 2]));
  assert.deepEqual(wilson(0, 0), null);
  const [low, high] = wilson(8, 32);
  assert(low > 0.12 && low < 0.25 && high > 0.25 && high < 0.45);
  assert.deepEqual(wilson(32, 32)[1], 1);
});

test('invalid timing, input, readout and model conditions are rejected', () => {
  for (const options of [{delaysMs: [0]}, {delaysMs: [-1]}, {delaysMs: [25, 25]}, {delaysMs: []}, {letterMs: 100.5},
    {gapMs: -1}, {trainPerPair: 0}, {evaluationPerPair: 0}, {ridge: 0}, {cueRateHz: 1001}, {stateResolutionMv: 0},
    {slowTimeFactor: 0}, {slowWeightScale: -1}, {poolCount: 0}])
    assert.throws(() => make(1, options), `${JSON.stringify(options)} should be rejected`);
  const {graph, manifest} = fixture();
  manifest.config.sim.background = {enabled: true, rate_hz: 1};
  assert.throws(() => new SequenceExperiment(graph, manifest, 1), /background/);
});
