/* Protocol tests use synthetic graphs and do not estimate biological learning.
 * Run: node --test tests/test_mb_model.cjs */
const test = require('node:test');
const assert = require('node:assert/strict');
const {START, END, OUTPUT_CODES, INPUT_CODES} = require('../site/recall-model.js');
const {prepareSlowGraph} = require('../site/sequence-model.js');
const {MushroomBodyExperiment, summarizeMushroomBodyRun, validateAnnotation, partition, CONDITIONS, DEFAULTS} = require('../site/mb-model.js');

// PNs 0-7, Kenyon cells 8-23, MBONs 24-39. PN k drives Kenyon cells 8+2k and 9+2k;
// every Kenyon cell contacts every MBON with a small weight, so only learning can
// make one MBON group win. There is no position information in this fixture.
function fixture() {
  const n = 40, indices = [], weights = [], indptr = [0];
  for (let i = 0; i < n; ++i) {
    if (i < 8) { indices.push(8 + 2 * i, 9 + 2 * i); weights.push(80, 80); }
    else if (i < 24) for (let m = 24; m < 40; ++m) { indices.push(m); weights.push(1); }
    indptr.push(indices.length);
  }
  const graph = {n, indptr: new Uint32Array(indptr), indices: new Uint32Array(indices), weights: new Float32Array(weights)};
  const manifest = {config: {sim: {dt_ms: 1, v_rest_mV: -52, v_reset_mV: -52, v_thresh_mV: -45, tau_mem_ms: 20, tau_syn_ms: 5,
    t_refractory_ms: 2.2, delay_ms: 2, poisson_weight_mV: 68.75, dtype: 'float32', rest_eps_mV: 1e-5, background: {enabled: false}}},
    targets: {eye_L: [0], eye_R: [1]}, layout: Array.from('abcdefghijklmnopqrstuvwxyz ')};
  const annotation = {kenyonCells: Array.from({length: 16}, (_, i) => 8 + i), mbons: Array.from({length: 16}, (_, i) => 24 + i),
    uniglomerularPNs: [0, 1, 2, 3, 4, 5, 6, 7]};
  return {graph, manifest, annotation};
}
const SMALL = {trainEpisodes: 3, diagnosticEpisodes: 1, recallEpisodes: 2, cueRateHz: 1000};
function make(seed = 42, options = {}) {
  const {graph, manifest, annotation} = fixture();
  return new MushroomBodyExperiment(graph, manifest, annotation, seed, {...SMALL, ...options});
}
function finish(exp) {
  const returned = [];
  while (!exp.done) { const record = exp.step(); if (record) returned.push(record); }
  return {result: exp.result(), returned};
}

test('codes, groups and plastic slots are fixed, seeded, disjoint and cover every Kenyon-cell-to-MBON synapse', () => {
  const exp = make();
  assert.deepEqual(Object.keys(exp.codebook), [...INPUT_CODES]); assert.deepEqual(Object.keys(exp.groups), [...OUTPUT_CODES]);
  const pns = Object.values(exp.codebook).flat(), mbons = Object.values(exp.groups).flat();
  assert.deepEqual([...pns].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual([...mbons].sort((a, b) => a - b), Array.from({length: 16}, (_, i) => 24 + i));
  assert(Object.values(exp.groups).every(group => group.length === 2));
  assert.equal(exp.plasticSlots.length, 256);
  assert.deepEqual(partition([1, 2, 3, 4], ['a', 'b'], 5), partition([1, 2, 3, 4], ['a', 'b'], 5));
  assert.notDeepEqual(partition([1, 2, 3, 4, 5, 6, 7, 8], ['a', 'b'], 5), partition([1, 2, 3, 4, 5, 6, 7, 8], ['a', 'b'], 6));
  for (let k = 0; k < exp.plasticSlots.length; ++k) {
    const slot = exp.plasticSlots[k], post = exp.graph.indices[slot];
    assert.equal(exp.groupOf[post], exp.plasticSlotGroup[k]);
  }
  assert.equal(exp.metadata.plasticSlotCount, 256);
  assert.deepEqual(exp.metadata.plasticSlotsPerGroup, Array(8).fill(32));
  assert.equal(exp.comparator.episode.chainLength, 6);
});

test('only Kenyon-cell-to-MBON slots change, only during training of plastic conditions, within bounds; the anatomy never changes', () => {
  const {graph, manifest, annotation} = fixture();
  const original = [...graph.weights];
  const exp = new MushroomBodyExperiment(graph, manifest, annotation, 42, {...SMALL, maxWeightMv: 0.9});
  const base = [...exp.baseWeights];
  const plastic = new Set(exp.plasticSlots);
  let previous = null;
  while (!exp.done) {
    const record = exp.step();
    if (!record) continue;
    if (exp.weights) for (let slot = 0; slot < exp.weights.length; ++slot) {
      if (!plastic.has(slot)) assert.equal(exp.weights[slot], base[slot], 'non-plastic synapses never move');
      else assert(exp.weights[slot] >= 0 && exp.weights[slot] <= 0.9, 'plastic weights stay within bounds');
    }
    if (record.condition === 'frozen') assert.equal(record.learningEnabled, false);
    if (record.phase !== 'train') assert.equal(record.learningEnabled, false);
    if (record.learningApplied) { assert.equal(record.phase, 'train'); assert.notEqual(record.condition, 'frozen'); assert(record.updateCountAfter > record.updateCountBefore); }
    else assert.equal(record.updateCountAfter, record.updateCountBefore);
    if (exp.weights && exp.condition !== record.condition) {
      assert.deepEqual(Array.from(exp.snapshotWeights()), Array.from(exp.initialPlasticWeights), 'each condition starts from the same scaled synapses');
    }
    previous = record;
  }
  assert.deepEqual([...graph.weights], original);
  assert.deepEqual([...exp.baseWeights], base);
  const result = exp.result();
  assert.deepEqual(result.models.frozen.learnedPlasticWeights, result.initialPlasticWeights);
  assert.equal(result.models.frozen.updates, 0); assert.equal(result.models.frozen.weightStats.changed, 0);
  assert(result.models.plastic.updates > 0); assert(result.models.plastic.weightStats.changed > 0);
  assert(result.models.plastic.weightStats.max <= 0.9);
  for (const condition of CONDITIONS) {
    assert.equal(result.models[condition].frozenAfterTraining, true);
    assert.equal(result.models[condition].fingerprint, result.models[condition].finalFingerprint, 'weights are frozen after training');
  }
});

test('the learning rule is reproducible from the recorded Kenyon-cell activity and decisions', () => {
  const {result} = finish(make(7, {maxWeightMv: 1.5}));
  const {graph} = fixture();
  const slots = result.annotation.plasticSlots, groups = result.annotation.plasticSlotGroup;
  for (const condition of ['plastic', 'plasticReset']) {
    const weights = Float64Array.from(result.initialPlasticWeights);
    const kcSlots = new Map();
    slots.forEach((slot, k) => {
      let pre = 0; while (graph.indptr[pre + 1] <= slot) pre++;
      if (!kcSlots.has(pre)) kcSlots.set(pre, []); kcSlots.get(pre).push(k);
    });
    for (const trial of result.trials.filter(row => row.condition === condition && row.phase === 'train')) {
      const target = OUTPUT_CODES.indexOf(trial.target);
      let rival = -1;
      for (let g = 0; g < 8; ++g) if (g !== target && (rival < 0 || trial.groupCounts[g] > trial.groupCounts[rival] ||
        (trial.groupCounts[g] === trial.groupCounts[rival] && trial.tieScores[g] > trial.tieScores[rival]))) rival = g;
      if (trial.groupCounts[target] > trial.groupCounts[rival]) { assert.equal(trial.learningApplied, false); continue; }
      assert.equal(trial.learningApplied, true); assert.equal(trial.rival, OUTPUT_CODES[rival]);
      for (const [kc, count] of trial.kcActivity) for (const k of kcSlots.get(kc) || []) {
        if (groups[k] === target) weights[k] = Math.fround(Math.min(1.5, weights[k] + DEFAULTS.learningRateMvPerSpike * count));
        else if (groups[k] === rival) weights[k] = Math.fround(Math.max(0, weights[k] - DEFAULTS.learningRateMvPerSpike * count));
      }
    }
    assert.deepEqual(Array.from(weights), result.models[condition].learnedPlasticWeights, `${condition} weights replay exactly`);
  }
});

test('decisions follow the MBON group vote with the declared tie-break, and recall feeds its own letters back', () => {
  const {result} = finish(make());
  const inputs = [START, ...DEFAULTS.phrase], targets = [...DEFAULTS.phrase, END];
  for (const row of result.trials) {
    let best = -1;
    for (let g = 0; g < 8; ++g) if (best < 0 || row.groupCounts[g] > row.groupCounts[best] || (row.groupCounts[g] === row.groupCounts[best] && row.tieScores[g] > row.tieScores[best])) best = g;
    assert.equal(row.prediction, OUTPUT_CODES[best]);
    assert.equal(row.silent, row.mbonSpikes === 0);
    assert.equal(row.mbonSpikes, row.groupCounts.reduce((sum, value) => sum + value, 0));
    assert.equal(row.kcSpikes, row.kcActivity.reduce((sum, [, count]) => sum + count, 0));
    assert.equal(row.activeKcCount, row.kcActivity.length);
    if (row.phase !== 'recall') { assert.equal(row.cue, inputs[row.step - 1]); assert.equal(row.target, targets[row.step - 1]); assert.equal(row.feedbackApplied, false); }
  }
  for (const episode of result.episodes) {
    const rows = result.trials.filter(row => row.condition === episode.condition && row.phase === 'recall' && row.episode === episode.episode);
    rows.forEach((row, index) => { assert.equal(row.cue, index === 0 ? START : rows[index - 1].prediction); assert.equal(row.feedbackApplied, index > 0); assert.equal(row.target, null); });
    assert.equal(episode.output, rows.map(row => row.prediction).filter(code => code !== END).join(''));
    assert(rows.length <= 32);
  }
  assert.equal(result.episodes.length, 6);
  assert(result.metrics.plastic.diagnostic.accuracy > result.metrics.frozen.diagnostic.accuracy, 'learning helps in the fixture');
});

test('plasticReset rebuilds and warms the network at every cue while plastic and frozen keep one per episode; seeds pair across conditions', () => {
  const exp = make(42, {warmupMs: 30});
  let net = null, key = null;
  while (!exp.done) {
    const record = exp.step();
    if (exp.cueStep === 1) {
      const episodeKey = `${exp.condition}/${exp.phase}/${exp.episode}`;
      if (exp.condition === 'plasticReset') { assert.equal(exp.net.stepIndex, 1); assert.equal(exp.warmupThisDecision, 30); }
      else {
        assert.equal(exp.warmupThisDecision, exp.decisionStep === 1 ? 30 : 0);
        if (episodeKey === key) assert.equal(exp.net, net); else assert.notEqual(exp.net, net);
      }
      key = episodeKey; net = exp.net;
    }
    if (record) assert.equal(record.warmupMs, record.condition === 'plasticReset' || record.step === 1 ? 30 : 0);
  }
  const result = exp.result();
  assert(Math.abs(result.simTimeS - result.trials.reduce((sum, row) => sum + (125 + row.warmupMs) / 1000, 0)) < 1e-9);
  const plastic = result.trials.filter(row => row.condition === 'plastic' && row.phase !== 'recall');
  for (const condition of ['plasticReset', 'frozen']) {
    const rows = result.trials.filter(row => row.condition === condition && row.phase !== 'recall');
    rows.forEach((row, i) => { for (const k of ['phase', 'episode', 'step', 'cue', 'target', 'episodeSeed']) assert.equal(row[k], plastic[i][k]); });
  }
  const resetSeeds = result.trials.filter(row => row.condition === 'plasticReset').map(row => row.cueSeed);
  assert.equal(new Set(resetSeeds).size, resetSeeds.length);
  assert.deepEqual(result.metrics, summarizeMushroomBodyRun(result.episodes, result.trials, result.reference));
  assert.deepEqual(finish(make(42, {warmupMs: 30})).result, result, 'seeds reproduce');
});

test('a prepared slow graph is used as-is, and invalid annotations or protocols are rejected', () => {
  const {graph, manifest, annotation} = fixture();
  const prepared = prepareSlowGraph(graph, 0.3);
  const exp = new MushroomBodyExperiment(graph, manifest, annotation, 1, SMALL, prepared);
  assert.equal(exp.baseWeights, prepared.graph.weights);
  assert.throws(() => new MushroomBodyExperiment(graph, manifest, annotation, 1, SMALL, prepareSlowGraph(graph, 0.5)), /weight scale/);
  assert.throws(() => validateAnnotation({...annotation, mbons: annotation.mbons.slice(0, 7)}, 40), /eight MBONs/);
  assert.throws(() => validateAnnotation({...annotation, kenyonCells: [8, 8]}, 40), /Duplicate/);
  assert.throws(() => validateAnnotation({...annotation, uniglomerularPNs: [0, 1, 2, 3, 4, 5, 6, 40]}, 40), /Invalid/);
  assert.throws(() => validateAnnotation({...annotation, uniglomerularPNs: [8, 9, 10, 11, 12, 13, 14, 15]}, 40), /disjoint/);
  for (const options of [{phrase: 'tox'}, {trainEpisodes: 0}, {recallEpisodes: 0}, {maxDecisions: 0}, {learningRateMvPerSpike: 0}, {maxWeightMv: 0},
    {cueMs: 100.5}, {gapMs: -1}, {warmupMs: -1}, {cueRateHz: 1001}, {slowWeightScale: 0}])
    assert.throws(() => make(1, options), `${JSON.stringify(options)} should be rejected`);
  const negative = fixture(); negative.graph.weights[negative.graph.indptr[8]] = -1;
  assert.throws(() => new MushroomBodyExperiment(negative.graph, negative.manifest, negative.annotation, 1, SMALL), /excitatory/);
  manifest.config.sim.background = {enabled: true, rate_hz: 1};
  assert.throws(() => new MushroomBodyExperiment(graph, manifest, annotation, 1, SMALL), /background/);
});
