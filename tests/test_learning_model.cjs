/* Run: node --test tests/test_learning_model.cjs (no dependencies or full assets).
 * Synthetic graphs verify protocol integrity, not biological or full-brain accuracy.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const {LIFNetwork} = require('../site/live-model.js');
const {LearningExperiment, LinearDecoder, ExplicitTypewriter, GROUP_NAMES, DEFAULTS,
  makeTrialPlan, summarizeTrials, wilsonInterval} = require('../site/learning-model.js');

function fixture(connected = true) {
  const n = 9;
  const indptr = new Uint32Array([0, ...(connected ? [1, 2] : [0, 0]), ...Array(7).fill(connected ? 2 : 0)]);
  const connectome = {n, indptr, indices: new Uint32Array(connected ? [2, 3] : []),
    weights: new Float32Array(connected ? [80, 80] : [])};
  const sim = {dt_ms: 1, v_rest_mV: -52, v_reset_mV: -52, v_thresh_mV: -45,
    tau_mem_ms: 20, tau_syn_ms: 5, t_refractory_ms: 2.2, delay_ms: 2,
    w_syn_mV: 0.275, poisson_weight_mV: 68.75, dtype: 'float32',
    rest_eps_mV: 1e-5, background: {enabled: false}};
  const targets = Object.fromEntries(GROUP_NAMES.map((name, i) => [name, [i + 2]]));
  const manifest = {config: {sim}, targets: {...targets, eye_L: [0], eye_R: [1]},
    layout: Array.from('abcdefghijklmnopqrstuvwxyz ')};
  return {connectome, manifest};
}
function experiment(seed = 123, options = {}, connected = true) {
  const {connectome, manifest} = fixture(connected);
  return new LearningExperiment(connectome, manifest, seed, options);
}
function finish(exp) {
  const returned = [];
  while (!exp.done) {
    const record = exp.step();
    if (record) returned.push(record);
  }
  return {result: exp.result(), returned};
}

test('each phase is shuffled and balanced; noise seeds are unique and independent of labels', () => {
  const plan = makeTrialPlan(991, DEFAULTS);
  assert.equal(plan.length, 120);
  for (const phase of ['baseline', 'train', 'evaluation']) {
    const trials = plan.filter(t => t.phase === phase);
    assert.equal(trials.filter(t => t.target === 'T').length, trials.length / 2);
    assert.deepEqual(trials.map(t => t.phaseTrial), trials.map((_, i) => i + 1));
  }
  assert.equal(new Set(plan.map(t => t.neuralSeed)).size, plan.length);
  assert(plan.every(t => t.neuralSeed !== t.decisionSeed));
  // Moving phase boundaries changes class shuffles, not noise or decision seeds.
  const boundaries = makeTrialPlan(991, {baselineTrials: 40, trainTrials: 40, evaluationTrials: 40});
  assert.deepEqual(plan.map(t => t.neuralSeed), boundaries.map(t => t.neuralSeed));
  assert.deepEqual(plan.map(t => t.decisionSeed), boundaries.map(t => t.decisionSeed));
  assert(plan.some((t, i) => t.target !== boundaries[i].target));
});

test('one step advances one neural timestep, resets trials, and yields every decision once', () => {
  const exp = experiment(11, {baselineTrials: 2, trainTrials: 2, evaluationTrials: 2});
  assert.equal(exp.progress.trial, 1);
  assert.equal(exp.progress.completed, 0);
  assert.equal(exp.simTimeS, 0);
  assert.equal(exp.step(), null);
  assert.equal(exp.simTimeS, 0.001);
  assert.equal(exp.net.stepIndex, 1);
  assert.equal(exp.net.totalSpikes, 0);
  for (let i = 1; i < 199; ++i) assert.equal(exp.step(), null);
  const first = exp.step();
  assert.equal(first.trial, 1);
  assert.equal(first.simTimeS, 0.2);
  assert.equal(exp.net, null);
  assert.equal(exp.step(), null);
  assert.equal(exp.net.stepIndex, 1);
  assert.equal(exp.net.totalSpikes, 0, 'the next trial does not inherit spikes or state');
  const {result, returned} = finish(exp);
  assert.equal(returned.length, 5);
  assert.equal(result.trials.length, 6);
  assert.equal(result.simTimeS, 1.2);
  assert.equal(exp.progress.phase, 'complete');
  assert.equal(exp.progress.cueLetter, null);
  assert.equal(exp.step(), null);
});

test('seven downstream features contain no direct cue, seed, or phase information', () => {
  const {result} = finish(experiment(332, {baselineTrials: 4, trainTrials: 4, evaluationTrials: 4}, false));
  assert(result.trials.some(t => t.spikes > 0), 'sensory neurons did receive and respond to cues');
  for (const trial of result.trials) {
    assert.deepEqual(trial.features, new Array(7).fill(0), 'disconnected motor neurons convey no cue');
    assert.deepEqual(trial.rates, new Array(7).fill(0));
    assert.deepEqual(trial.groupSpikeCounts, new Array(7).fill(0));
  }
  const decoder = new LinearDecoder();
  assert.throws(() => decoder.predict([1, 2]), /fixed-length/);
  assert.throws(() => decoder.predict([0, 0, 0, 0, 0, 0, NaN]), /numeric/);
});

test('sensory/readout overlap and duplicated motor membership are rejected', () => {
  for (const index of [0, 3]) {
    const {connectome, manifest} = fixture();
    manifest.targets.turn_L = [index];
    assert.throws(() => new LearningExperiment(connectome, manifest, 1), /disjoint/);
  }
});

test('baseline and evaluation are frozen; training predictions precede exactly one update', () => {
  const {result} = finish(experiment());
  const zero = new Array(8).fill(0);
  for (const trial of result.trials) {
    const decoder = new LinearDecoder();
    decoder.weights.set(trial.weightsBefore);
    assert.equal(decoder.probability(trial.features), trial.probabilityO,
      'exported prediction must be computed from the weights before feedback');
    if (trial.phase === 'train') {
      assert.equal(trial.updateCountAfter, trial.updateCountBefore + 1);
      assert.equal(trial.feedbackApplied, true);
      decoder.update(trial.features, trial.target);
      assert.deepEqual(trial.weightsAfter, decoder.snapshot());
    } else {
      assert.equal(trial.feedbackApplied, false);
      assert.equal(trial.updateCountAfter, trial.updateCountBefore);
      assert.deepEqual(trial.weightsBefore, trial.weightsAfter);
      assert.deepEqual(trial.weightsBefore, trial.phase === 'baseline' ? zero : result.frozenWeights);
    }
    if (trial.phase === 'baseline') assert.equal(trial.prediction, trial.controlPrediction);
  }
  assert.equal(result.updateCount, 60);
  assert.deepEqual(result.controlWeights, zero);
  assert.deepEqual(result.finalWeights, result.frozenWeights);
});

test('connectome edges stay unchanged, and any trial reproduces from rest with its exported noise seed', () => {
  const {connectome, manifest} = fixture();
  const before = {indptr: [...connectome.indptr], indices: [...connectome.indices], weights: [...connectome.weights]};
  const exp = new LearningExperiment(connectome, manifest, 701, {baselineTrials: 2, trainTrials: 4, evaluationTrials: 2});
  const {result} = finish(exp);
  assert.deepEqual({indptr: [...connectome.indptr], indices: [...connectome.indices], weights: [...connectome.weights]}, before);
  for (const trial of result.trials) {
    const net = new LIFNetwork(connectome, manifest.config.sim, trial.neuralSeed);
    net.injectPoisson(manifest.targets[trial.target === 'T' ? 'eye_L' : 'eye_R'], result.config.cueRateHz);
    const counts = new Array(7).fill(0);
    for (let step = 0; step < 200; ++step) {
      const spikes = net.step();
      if (step >= 50) for (const neuron of spikes) if (neuron >= 2) counts[neuron - 2]++;
    }
    assert.deepEqual(trial.groupSpikeCounts, counts);
    assert.deepEqual(trial.features, counts.map(count => Math.min(1, (count / 0.15) / 100)));
    assert.equal(trial.spikes, net.totalSpikes);
  }
});

test('external decoder learns a synthetic cue relay, on independent evaluation noise', () => {
  const {result} = finish(experiment(125));
  assert(result.metrics.evaluation.accuracy >= 0.9,
    `synthetic cue-relay evaluation accuracy ${result.metrics.evaluation.accuracy}`);
  assert(result.metrics.evaluation.accuracy > result.metrics.evaluation.control.accuracy);
  assert(result.metrics.evaluation.byClass.T.accuracy >= 0.9);
  assert(result.metrics.evaluation.byClass.O.accuracy >= 0.9);
});

test('same seed reproduces all trials and weights; a new seed changes neural observations', () => {
  const options = {baselineTrials: 2, trainTrials: 8, evaluationTrials: 4};
  const a = finish(experiment(501, options)).result;
  const b = finish(experiment(501, options)).result;
  const c = finish(experiment(502, options)).result;
  assert.deepEqual(a, b);
  assert.notDeepEqual(a.trials.map(t => t.groupSpikeCounts), c.trials.map(t => t.groupSpikeCounts));
});

test('mistakes remain in transcript/export; returned snapshots cannot alter model records', () => {
  const exp = experiment(203, {baselineTrials: 10, trainTrials: 0, evaluationTrials: 10}, false);
  const {result, returned} = finish(exp);
  assert(result.trials.some(t => !t.correct));
  assert.equal(result.trials.length, 20);
  assert.equal(result.outputs.baseline.length + result.outputs.evaluation.length, 20);
  assert.equal(result.outputs.evaluation, result.trials.filter(t => t.phase === 'evaluation').map(t => t.prediction).join(''));
  result.trials[0].features[0] = 100;
  returned[0].prediction = '?';
  assert.equal(exp.result().trials[0].features[0], 0);
  assert.notEqual(exp.result().trials[0].prediction, '?');
});

test('explicit movement is silent, distant keys are reachable, and repeated presses type ll', () => {
  const writer = new ExplicitTypewriter();
  assert.throws(() => writer.press(), /Move/);
  for (const letter of ['a', 'b', 'z', 'l']) writer.moveTo(letter);
  assert.equal(writer.output, '');
  writer.press(); writer.press();
  assert.equal(writer.output, 'll');
  writer.moveTo('a'); writer.press();
  writer.moveTo('z'); writer.press();
  writer.moveTo(' '); writer.press();
  assert.equal(writer.output, 'llaz ');
  writer.reset();
  for (const letter of writer.layout) { writer.moveTo(letter); writer.press(); }
  assert.equal(writer.output, 'abcdefghijklmnopqrstuvwxyz ');
  assert.equal(writer.events.length, 27);
  assert.throws(() => writer.moveTo('!'), /Unknown/);
});

test('metrics report class balance, control, confusion, and Wilson interval without inventing empty accuracy', () => {
  const trials = [{target: 'T', prediction: 'T'}, {target: 'T', prediction: 'O'},
    {target: 'O', prediction: 'O'}, {target: 'O', prediction: 'O'}];
  const metrics = summarizeTrials(trials);
  assert.equal(metrics.accuracy, 0.75);
  assert.equal(metrics.balancedAccuracy, 0.75);
  assert.deepEqual(metrics.confusion, {T: {T: 1, O: 1}, O: {T: 0, O: 2}});
  assert(metrics.wilson95[0] < 0.75 && metrics.wilson95[1] > 0.75);
  assert.deepEqual(wilsonInterval(0, 0), null);
  assert.equal(summarizeTrials([]).accuracy, null);
  assert.equal(summarizeTrials([]).balancedAccuracy, null);
});

test('invalid protocol timing, rates and unbalanced phase counts are rejected', () => {
  for (const options of [{trainTrials: 3}, {warmupMs: 200}, {trialMs: 200.5},
    {cueRateHz: 1001}, {learningRate: 0}, {l2: -1}, {featureScaleHz: 0}])
    assert.throws(() => experiment(1, options));
});
