/* Synthetic protocol tests, not an estimate of full-connectome recall accuracy.
 * Run with node --test tests/test_recall_model.cjs. */
const test = require('node:test');
const assert = require('node:assert/strict');
const {LIFNetwork} = require('../site/live-model.js');
const {RecallExperiment, RecallActor, INPUT_CODES, OUTPUT_CODES, DEFAULTS, START, END,
  fitRidge, conventionalRollout, appendHistory, historyVector, oneHot, fingerprint,
  editDistance} = require('../site/recall-model.js');

function fixture(connected = true) {
  const n = 16, indices = [], weights = [], offsets = [0];
  for (let neuron = 0; neuron < n; ++neuron) {
    if (connected && neuron < 8) { indices.push(neuron + 8); weights.push(80); }
    offsets.push(indices.length);
  }
  const connectome = {n, indptr: new Uint32Array(offsets), indices: new Uint32Array(indices), weights: new Float32Array(weights)};
  const manifest = {config: {sim: {dt_ms: 1, v_rest_mV: -52, v_reset_mV: -52, v_thresh_mV: -45,
    tau_mem_ms: 20, tau_syn_ms: 5, t_refractory_ms: 2.2, delay_ms: 2,
    poisson_weight_mV: 68.75, dtype: 'float32', rest_eps_mV: 1e-5, background: {enabled: false}}},
    targets: {eye_L: [0, 1, 2, 3], eye_R: [4, 5, 6, 7]}, layout: Array.from('abcdefghijklmnopqrstuvwxyz ')};
  return {connectome, manifest};
}
function make(seed = 42, options = {}, connected = true) {
  const {connectome, manifest} = fixture(connected);
  return new RecallExperiment(connectome, manifest, seed,
    {trainEpisodes: 2, recallEpisodes: 1, ablatedEpisodes: 1, cueRateHz: 1000, ...options});
}
function finish(experiment) {
  const records = [];
  while (!experiment.done) { const record = experiment.step(); if (record) records.push(record); }
  return {records, result: experiment.result()};
}
function fitConventional(historyLength) {
  const rows = [], targets = [], inputs = [START, ...DEFAULTS.phrase];
  const desired = [...DEFAULTS.phrase, END];
  for (let episode = 0; episode < 6; ++episode) {
    const history = [];
    for (let step = 0; step < desired.length; ++step) {
      appendHistory(history, oneHot(inputs[step]), historyLength);
      rows.push(historyVector(history, INPUT_CODES.length, historyLength));
      targets.push(desired[step]);
    }
  }
  return fitRidge(rows, targets, .01);
}

test('six-character history learns phrase and learned END; five-character context is ambiguous', () => {
  const six = conventionalRollout(fitConventional(6), {historyLength: 6, maxDecisions: 48});
  assert.equal(six.output, DEFAULTS.phrase);
  assert.equal(six.stoppedBy, END);
  assert.equal(six.decisions.length, DEFAULTS.phrase.length + 1);
  const five = conventionalRollout(fitConventional(5), {historyLength: 5, maxDecisions: 48});
  assert(five.output !== DEFAULTS.phrase || five.stoppedBy !== END);
});

test('actor has no reference or position; its own wrong prediction becomes its next cue', () => {
  const weights = new Array((8 * 6 + 1) * 8).fill(0);
  weights[48 * 8 + OUTPUT_CODES.indexOf('n')] = 1;
  const actor = new RecallActor(weights, {featureCount: 8, historyLength: 6});
  assert.equal(actor.cue, START);
  assert.equal(actor.observe(oneHot(actor.cue)).prediction, 'n');
  assert.equal(actor.cue, 'n');
  assert.equal(actor.observe(oneHot(actor.cue)).prediction, 'n');
  for (const field of ['reference', 'target', 'step', 'position', 'episode', 'maxDecisions']) assert(!(field in actor));
  const fit = {weights};
  const rollout = conventionalRollout(fit, {historyLength: 6, maxDecisions: 48});
  assert.equal(rollout.output, 'n'.repeat(48));
  assert.equal(rollout.stoppedBy, 'cap');
  assert(rollout.decisions.slice(1).every(record => record.cue === 'n'));
});

test('END is a decoder output, not an externally imposed phrase length', () => {
  const weights = new Array((8 * 6 + 1) * 8).fill(0);
  weights[48 * 8 + OUTPUT_CODES.indexOf(END)] = 1;
  const actor = new RecallActor(weights, {featureCount: 8, historyLength: 6});
  assert.equal(actor.observe(oneHot(START)).prediction, END);
  assert.equal(actor.ended, true);
  assert.throws(() => actor.observe(oneHot(START)), /ended/);
  const rollout = conventionalRollout({weights}, {historyLength: 6, maxDecisions: 48});
  assert.equal(rollout.output, ''); assert.equal(rollout.stoppedBy, END);
  assert.equal(rollout.decisions.length, 1);
});

test('no-past-history ablation preserves current features, masks previous slots, and keeps weights', () => {
  const history = [[1, 2], [3, 4]];
  assert.deepEqual([...historyVector(history, 2, 3)], [0, 0, 1, 2, 3, 4]);
  assert.deepEqual([...historyVector(history, 2, 3, true)], [0, 0, 0, 0, 3, 4]);
  const weights = new Array((2 * 3 + 1) * 8).fill(0);
  weights[0] = 1;
  const a = new RecallActor(weights, {featureCount: 2, historyLength: 3});
  const b = new RecallActor(weights, {featureCount: 2, historyLength: 3, noPastHistory: true});
  assert.deepEqual(a.weights, b.weights);
  a.observe([1, 2]); b.observe([1, 2]);
  assert.deepEqual(a.weights, b.weights);
});

test('synthetic neural signatures support autonomous phrase recall and a learned stop', () => {
  const {result} = finish(make());
  const recall = result.episodes.find(episode => episode.phase === 'recall');
  assert.equal(recall.output, DEFAULTS.phrase);
  assert.equal(recall.stoppedBy, END);
  assert.equal(recall.exact, true);
  assert.equal(recall.editDistance, 0);
  assert.equal(result.metrics.recall.exact, 1);
  assert(result.episodes.some(episode => episode.phase === 'ablated' && !episode.exact));
  assert.equal(result.comparator.episodes[0].output, DEFAULTS.phrase);
});

test('training is teacher-forced collection; only one fit occurs and all autonomous choices are reference-free', () => {
  const {result} = finish(make());
  assert.equal(result.fitCount, 1);
  assert.deepEqual(result.frozenWeights, result.finalWeights);
  assert.equal(result.finalFingerprint, fingerprint(result.finalWeights));
  assert.equal(result.frozenFingerprint, result.finalFingerprint);
  assert.equal(result.trials.filter(record => record.fitApplied).length, 1);
  const train = result.trials.filter(record => record.phase === 'train');
  assert.equal(train.length, 2 * 19);
  assert(train.every(record => record.prediction === null && record.decoderVersion === 0));
  for (let episode = 1; episode <= 2; ++episode) {
    const rows = train.filter(record => record.episode === episode);
    assert.deepEqual(rows.map(row => row.cue), [START, ...DEFAULTS.phrase]);
    assert.deepEqual(rows.map(row => row.target), [...DEFAULTS.phrase, END]);
  }
  for (const record of result.trials.filter(record => record.phase !== 'train')) {
    assert(!('target' in record));
    assert.equal(record.decoderVersion, 1);
    assert.equal(record.updateCountBefore, 1); assert.equal(record.updateCountAfter, 1);
    assert.equal(record.decoderFingerprint, result.frozenFingerprint);
    assert.equal(record.feedbackApplied, false);
    const previous = result.trials.find(row => row.phase === record.phase && row.episode === record.episode && row.step === record.step - 1);
    assert.equal(record.cue, previous ? previous.prediction : START);
  }
});

test('all stimulated sensory cells are excluded from pooled features and zero signals cannot leak the phrase', () => {
  const exp = make(51, {phrase: 'to', trainEpisodes: 1, maxDecisions: 6}, false);
  const sensory = Object.values(exp.codebook).flat();
  assert.equal(new Set(sensory).size, 8);
  assert.deepEqual([...sensory].sort((a,b) => a-b), [0,1,2,3,4,5,6,7]);
  for (const neuron of sensory) assert.equal(exp.poolForNeuron[neuron], -1);
  const {result} = finish(exp);
  assert(result.trials.some(record => record.spikes > 0));
  assert(result.trials.every(record => record.features.every(value => value === 0)));
  assert.equal(result.metrics.recall.exact, 0);
});

test('one step advances one neural timestep and every completed cue is emitted exactly once', () => {
  const exp = make(13, {phrase: 'to', trainEpisodes: 1, recallEpisodes: 0, ablatedEpisodes: 0});
  assert.equal(exp.simTimeS, 0);
  assert.equal(exp.step(), null);
  assert.equal(exp.simTimeS, .001);
  const {records, result} = finish(exp);
  assert.equal(records.length, 3);
  assert.deepEqual(records.map(record => record.trial), [1, 2, 3]);
  assert.equal(result.simTimeS, .6);
  assert.equal(exp.progress.phase, 'complete');
  assert.equal(exp.progress.totalTrials, exp.progress.completed);
  assert.equal(exp.step(), null);
});

test('a neural trial independently reproduces from its exported cue/seed, with the original graph unchanged', () => {
  const {connectome, manifest} = fixture();
  const before = [...connectome.weights];
  const exp = new RecallExperiment(connectome, manifest, 65, {phrase: 'to', trainEpisodes: 1,
    recallEpisodes: 1, ablatedEpisodes: 0, cueRateHz: 100, maxDecisions: 8});
  const {result} = finish(exp);
  assert.deepEqual([...connectome.weights], before);
  for (const record of result.trials) {
    const net = new LIFNetwork(connectome, manifest.config.sim, record.neuralSeed);
    net.injectPoisson(result.metadata.codebook[record.cue], result.config.cueRateHz);
    const counts = new Array(result.config.poolCount).fill(0);
    for (let step = 0; step < 200; ++step) {
      const spikes = net.step();
      if (step >= 50) for (const neuron of spikes) {
        const pool = exp.poolForNeuron[neuron]; if (pool >= 0) counts[pool]++;
      }
    }
    assert.deepEqual(record.poolSpikeCounts, counts);
    assert.equal(record.spikes, net.totalSpikes);
  }
});

test('seed and codebook reproducibility; exported snapshots do not mutate results', () => {
  const options = {phrase: 'to', trainEpisodes: 1, recallEpisodes: 1, ablatedEpisodes: 0,
    cueRateHz: 100, maxDecisions: 8};
  const a = make(451, options), b = make(451, options), c = make(452, options);
  const one = finish(a).result, two = finish(b).result, three = finish(c).result;
  assert.deepEqual(one, two);
  assert.deepEqual(one.metadata.codebook, three.metadata.codebook);
  assert.notDeepEqual(one.trials.map(t => t.spikes), three.trials.map(t => t.spikes));
  one.trials[0].features[0] = -100;
  assert.notEqual(a.result().trials[0].features[0], -100);
});

test('edit distance counts missing, extra and wrong letters, while exact recall also requires END', () => {
  assert.equal(editDistance('to be', 'to be'), 0);
  assert.equal(editDistance('to b', 'to be'), 1);
  assert.equal(editDistance('to bee', 'to be'), 1);
  assert.equal(editDistance('to bo', 'to be'), 1);
  assert.equal(editDistance('', 'to be'), 5);
});

test('invalid protocol parameters are rejected', () => {
  for (const options of [{phrase: 'Hamlet'}, {phrase: ''}, {trainEpisodes: 0}, {ridge: 0},
    {poolCount: 0}, {warmupMs: 200}, {trialMs: 200.5}, {historyLength: 0}, {cueRateHz: 1001}])
    assert.throws(() => make(1, options));
});
