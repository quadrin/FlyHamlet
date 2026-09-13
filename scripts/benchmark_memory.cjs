#!/usr/bin/env node
/* Developmental smoke check of the default browser post-cue neural-memory protocol on the
 * committed full connectome. No npm packages; Node 18+. These are technical
 * replicates of one anatomical brain, not a preregistered biological study.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {MemoryExperiment, DEFAULTS} = require('../site/memory-model.js');
const {prepareRewire} = require('../site/memory-rewire.js');

const repository = path.resolve(__dirname, '..');
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const arrayBytes = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
const graphHashes = graph => Object.fromEntries(['indptr', 'indices', 'weights']
  .map(name => [name, sha256(arrayBytes(graph[name]))]));

function argumentsForRun(argv) {
  let seeds = [42];
  let output = path.join(repository, 'results/memory_benchmark');
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--help' || argv[i] === '-h') {
      console.log('Usage: node scripts/benchmark_memory.cjs [--seeds 42] [--out results/memory_benchmark]\nRuns the unchanged default protocol; writes raw per-seed JSON and report.md.');
      return null;
    }
    if (argv[i] === '--seeds') {
      const supplied = argv[++i];
      if (!supplied || !/^\d+(,\d+)*$/.test(supplied)) throw new Error('--seeds requires comma-separated unsigned integers.');
      seeds = supplied.split(',').map(Number);
      if (seeds.some(seed => !Number.isSafeInteger(seed) || seed > 0xffffffff))
        throw new Error('Seeds must be between 0 and 4294967295.');
      if (new Set(seeds).size !== seeds.length) throw new Error('Duplicate seeds are not independent simulation runs.');
    } else if (argv[i] === '--out') {
      if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error('--out requires a directory path.');
      output = path.resolve(argv[++i]);
    } else throw new Error(`Unknown argument: ${argv[i]}`);
  }
  return {seeds, output};
}

function loadFullGraph() {
  const modelDirectory = path.join(repository, 'site/model');
  const manifest = JSON.parse(fs.readFileSync(path.join(modelDirectory, 'manifest.json')));
  assert.equal(manifest.format, 'flyhamlet-csr-v1');
  const graph = {n: manifest.n};
  for (const name of ['indptr', 'indices', 'weights']) {
    const descriptor = manifest.files[name];
    const file = typeof descriptor === 'string' ? descriptor : descriptor.url;
    const packed = fs.readFileSync(path.join(modelDirectory, file));
    const integrity = manifest.integrity[name];
    assert.equal(packed.length, integrity.bytes, `${name}: compressed size`);
    assert.equal(sha256(packed), integrity.sha256, `${name}: compressed SHA-256`);
    const raw = zlib.gunzipSync(packed);
    assert.equal(raw.length, integrity.rawBytes, `${name}: raw size`);
    assert.equal(sha256(raw), integrity.rawSha256, `${name}: raw SHA-256`);
    const Type = name === 'weights' ? Float32Array : Uint32Array;
    graph[name] = raw.byteOffset % 4 === 0
      ? new Type(raw.buffer, raw.byteOffset, raw.byteLength / 4)
      : new Type(Uint8Array.from(raw).buffer);
  }
  assert.equal(graph.indptr.length, manifest.n + 1);
  assert.equal(graph.indptr[0], 0);
  assert.equal(graph.indptr[manifest.n], manifest.edgeCount);
  assert.equal(graph.indices.length, manifest.edgeCount);
  assert.equal(graph.weights.length, manifest.edgeCount);
  for (let i = 0; i < manifest.n; i++) assert(graph.indptr[i] <= graph.indptr[i + 1], 'Nonmonotonic CSR row offsets');
  for (const index of graph.indices) if (index >= manifest.n) throw new Error('Invalid CSR target index.');
  for (const weight of graph.weights) if (!Number.isFinite(weight)) throw new Error('Nonfinite synaptic weight.');
  return {graph: Object.freeze(graph), manifest};
}

const CONDITIONS = ['retain', 'reset', 'rewired'];
const close = (actual, expected, message) => assert(Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)), message);
const percent = value => value === null ? 'n/a' : `${(100 * value).toFixed(1)}%`;

function predict(features, weights, labels) {
  assert.equal(weights.length, (features.length + 1) * labels.length);
  const scores = labels.map((_, label) => {
    let sum = weights[features.length * labels.length + label];
    features.forEach((value, index) => { sum += value * weights[index * labels.length + label]; });
    return sum;
  });
  let best = 0;
  for (let index = 1; index < scores.length; index++) if (scores[index] > scores[best]) best = index;
  return {prediction: labels[best], scores};
}

function summarize(records, selectWindow, labels) {
  const byClass = Object.fromEntries(labels.map(label => [label, {correct: 0, total: 0}]));
  let correct = 0, populationSpikes = 0, activeNeurons = 0, silent = 0, readoutSilent = 0, cueSpikes = 0, postCueSpikes = 0;
  for (const record of records) {
    const window = selectWindow(record);
    const success = window.prediction === record.cue;
    correct += Number(success);
    byClass[record.cue].correct += Number(success);
    byClass[record.cue].total++;
    populationSpikes += window.populationSpikes;
    activeNeurons += window.activeNeurons;
    silent += Number(window.silent);
    readoutSilent += Number(window.readoutSilent);
    cueSpikes += record.cueSpikes;
    postCueSpikes += record.postCueSpikes;
  }
  const total = records.length;
  return {correct, total, accuracy: total ? correct / total : null,
    balancedAccuracy: total ? labels.reduce((sum, label) => sum + byClass[label].correct / byClass[label].total, 0) / labels.length : null,
    byClass, silentTrials: silent, readoutSilentTrials: readoutSilent,
    meanPopulationSpikes: total ? populationSpikes / total : null,
    meanActiveNeurons: total ? activeNeurons / total : null,
    meanCueSpikes: total ? cueSpikes / total : null,
    meanPostCueSpikes: total ? postCueSpikes / total : null};
}

function assertMemoryIsolation(result) {
  assert(result.complete, 'Every planned condition and trial must finish');
  for (const [key, value] of Object.entries(DEFAULTS)) assert.deepEqual(result.config[key], value, `Default ${key} must be unchanged`);
  const p = result.config, labels = result.metadata.labels, poolSizes = result.metadata.poolSizes;
  assert.equal(labels.length, 7);
  assert.equal(poolSizes.length, p.poolCount);
  assert.equal(result.fitCount, 3);
  assert.equal(result.readoutFitCount, 3 * (p.delaysMs.length + 1));
  const trainCount = p.trainPerClass * labels.length, evaluationCount = p.evaluationPerClass * labels.length;
  assert.equal(result.trials.length, 3 * (trainCount + evaluationCount));
  const scores = {};
  for (const [conditionIndex, condition] of CONDITIONS.entries()) {
    const model = result.models[condition];
    assert.equal(model.fitCount, 1);
    assert.equal(model.trainingExamples, trainCount);
    assert.equal(model.readouts.length, p.delaysMs.length);
    for (const readout of [model.cueReadout, ...model.readouts]) {
      assert.deepEqual(readout.labels, labels);
      assert.deepEqual(readout.weights, readout.frozenWeights);
      assert.equal(readout.finalFingerprint, readout.fingerprint);
    }
    const records = result.trials.filter(record => record.condition === condition);
    for (const phase of ['train', 'evaluation']) {
      const phaseRecords = records.filter(record => record.phase === phase);
      assert.equal(phaseRecords.length, phase === 'train' ? trainCount : evaluationCount);
      for (const label of labels) assert.equal(phaseRecords.filter(record => record.cue === label).length,
        phase === 'train' ? p.trainPerClass : p.evaluationPerClass, 'Classes must remain balanced');
    }
    for (const record of records) {
      assert.equal(record.cueOffsetMs, p.cueMs);
      assert.equal(record.inputsOffFromMs, p.cueMs);
      assert.equal(record.windows.length, p.delaysMs.length);
      assert.equal(record.feedbackApplied, false);
      if (record.phase === 'evaluation') {
        assert.equal(record.decoderVersion, 1);
        assert.equal(record.updateCountBefore, conditionIndex + 1);
        assert.equal(record.updateCountAfter, conditionIndex + 1);
        assert.equal(record.fitApplied, false);
      }
      const observations = [{window: record.cueReadout, readout: model.cueReadout,
        start: p.cueMs - p.windowMs, active: true}, ...p.delaysMs.map(delay => ({
        window: record.windows.find(window => window.delayMs === delay),
        readout: model.readouts.find(readout => readout.delayMs === delay), start: p.cueMs + delay, active: false}))];
      for (const {window, readout, start, active} of observations) {
        assert(window && readout, 'Every declared observation has its own fitted decoder');
        assert.equal(window.startMs, start);
        assert.equal(window.endMs, start + p.windowMs);
        assert.equal(window.inputActive, active);
        if (!active) assert(window.startMs >= record.inputsOffFromMs, 'Every memory window starts after inputs are removed');
        assert.equal(window.features.length, p.poolCount);
        assert.equal(window.poolSpikeCounts.length, p.poolCount);
        let readoutSpikes = 0;
        window.poolSpikeCounts.forEach((count, pool) => {
          assert(Number.isSafeInteger(count) && count >= 0);
          readoutSpikes += count;
          const expected = poolSizes[pool] ? count / (poolSizes[pool] * (p.windowMs / 1000)) : 0;
          close(window.features[pool], expected, 'Window features must derive only from its own contemporary spike counts');
        });
        assert.equal(window.readoutSpikes, readoutSpikes);
        assert.equal(window.readoutSilent, readoutSpikes === 0);
        assert.equal(window.silent, window.populationSpikes === 0);
        assert(window.populationSpikes >= readoutSpikes);
        assert(window.activeNeurons >= 0 && window.activeNeurons <= result.metadata.n);
        if (record.phase === 'evaluation') {
          assert.equal(window.decoderFingerprint, readout.fingerprint);
          const decision = predict(window.features, readout.frozenWeights, labels);
          assert.equal(window.prediction, decision.prediction, 'Choice must follow only that window and frozen decoder');
          window.scores.forEach((score, index) => close(score, decision.scores[index], 'Reconstructed score must agree'));
          assert.equal(window.correct, decision.prediction === record.cue);
        } else assert.equal(window.prediction, null, 'Training observations are collected before fitting');
      }
    }
    const evaluation = records.filter(record => record.phase === 'evaluation');
    scores[condition] = {cueReadout: summarize(evaluation, record => record.cueReadout, labels),
      delays: p.delaysMs.map(delayMs => ({delayMs, ...summarize(evaluation,
        record => record.windows.find(window => window.delayMs === delayMs), labels)}))};
    const exported = result.metrics[condition];
    for (const [actual, reported] of [[scores[condition].cueReadout, exported.cueReadout],
      ...scores[condition].delays.map(score => [score, exported.delays.find(metric => metric.delayMs === score.delayMs)])]) {
      assert.equal(actual.correct, reported.correct);
      assert.equal(actual.total, reported.total);
      close(actual.accuracy, reported.accuracy, 'Accuracy must agree with independent scoring');
      close(actual.balancedAccuracy, reported.balancedAccuracy, 'Balanced accuracy must agree with independent scoring');
    }
  }
  const original = result.trials.filter(record => record.condition === 'retain');
  for (const condition of ['reset', 'rewired']) {
    const records = result.trials.filter(record => record.condition === condition);
    records.forEach((record, index) => {
      const matched = original[index];
      for (const key of ['phase', 'phaseTrial', 'cue', 'neuralSeed']) assert.equal(record[key], matched[key], 'Conditions must share the same input/noise schedule');
      if (condition === 'reset') {
        assert.equal(record.cueSpikes, matched.cueSpikes, 'Reset is applied only after an identical cue period');
        assert.deepEqual(record.cueReadout.poolSpikeCounts, matched.cueReadout.poolSpikeCounts);
      }
    });
  }
  return scores;
}

function report(results, manifest, rewire, generatedAt) {
  const rows = results.flatMap(result => CONDITIONS.flatMap(condition => {
    const score = result.validation.scores[condition];
    return [{name: 'Cue ON diagnostic', ...score.cueReadout}, ...score.delays.map(delay => ({name: `${delay.delayMs} ms after OFF`, ...delay}))]
      .map(value => `| ${result.seed} | ${condition} | ${value.name} | ${value.correct}/${value.total} (${percent(value.accuracy)}) | ${percent(value.balancedAccuracy)} | ${value.meanPopulationSpikes.toFixed(1)} | ${value.silentTrials}/${value.total} | ${value.readoutSilentTrials}/${value.total} |`);
  }));
  return `# Post-cue neural-memory benchmark: developmental smoke check

Generated ${generatedAt}. ${results.length} seeded run${results.length === 1 ? '' : 's'} of one FlyWire v783 anatomical graph (${manifest.n.toLocaleString('en-US')} neurons, ${manifest.edgeCount.toLocaleString('en-US')} connection slots) and one fixed rewired null (seed ${rewire.seed}). Seeds are technical simulations, not independent biological animals or independently drawn null graphs.

Each condition receives the same balanced ${DEFAULTS.trainPerClass * 7} training and ${DEFAULTS.evaluationPerClass * 7} evaluation trials over seven arbitrary sensory symbols. A ${DEFAULTS.cueMs} ms cue is switched OFF; six independent decoders read ${DEFAULTS.windowMs} ms spike windows beginning ${DEFAULTS.delaysMs.join(', ')} ms later. Each decoder receives only its own window's ${DEFAULTS.poolCount} pooled neural rates, with no external history, cue labels, previous observations, or corrections. State, delayed events, and noise state are reset at cue offset in the reset condition. The retained and rewired conditions keep their neural state. Chance classification is 1/7 (14.3%).

A seventh, separate decoder reads the final ${DEFAULTS.windowMs} ms while the cue is still ON. This diagnostic tests initial cue encoding and is never merged into a post-cue feature vector or memory score. Differences in cue-on accuracy indicate different initial encoding, complicating attribution of later differences specifically to retention.

The six delays are observations from the same trials; some windows overlap. They are correlated measurements, not six independent replications. Every condition/delay has its own ridge decoder fitted only on training observations, then frozen. No accuracy threshold or preferred biological outcome is enforced.

| Seed | Condition | Observation | Accuracy | Balanced accuracy | Mean population spikes / window | Fully silent trials | Readout silent trials |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
${rows.join('\n')}

Population activity includes all simulated cells. Readout silence refers only to nonstimulated neurons used by the pooled decoder. All memory observations occur after input removal; delayed synaptic events already in flight remain part of retained neural state. Silence or failed decoding in this LIF experiment does not show that biological flies cannot remember. Successful decoding is not a demonstration of phrase memory.

The null randomly permutes target slots while retaining every source slot and its signed weight. Incoming and outgoing connection-slot degrees are preserved for every neuron. It allows self-loops and parallel slots: ${rewire.selfLoopSlots.toLocaleString('en-US')} self-loop slots and ${rewire.parallelEdgeSlots.toLocaleString('en-US')} parallel slots, leaving ${rewire.distinctDirectedPairs.toLocaleString('en-US')} distinct directed pairs. Incoming weighted strength, per-neuron excitation/inhibition balance, spatial geometry, and numbers of distinct neighbors are not matched. One null realization cannot establish a population-wide effect of wiring.

Validation independently reconstructed every feature from its window's spike counts and every evaluation score from frozen weights, recomputed all accuracies and class-balanced scores, checked cue-OFF timestamps, matched input/noise schedules across controls, and confirmed identical retain/reset cue-period counts. The 21 fitted readouts stayed frozen during evaluation. Original compressed/raw assets passed manifest SHA-256 checks; hashes of both in-memory graphs were unchanged after each complete run.

Compute times: ${results.map(result => `seed ${result.seed}: ${result.validation.wallSeconds.toFixed(2)} s`).join('; ')}.

${results.map(result => `- [Seed ${result.seed}: full records, models, provenance, and integrity checks (gzip JSON)](seed-${result.seed}.json.gz)`).join('\n')}

\`\`\`sh
node scripts/benchmark_memory.cjs --seeds ${results.map(result => result.seed).join(',')} --out results/memory_benchmark
\`\`\`
`;
}

async function main() {
  const args = argumentsForRun(process.argv.slice(2));
  if (!args) return;
  const startedAt = new Date().toISOString();
  const {graph, manifest} = loadFullGraph();
  const beforeHashes = graphHashes(graph);
  console.log(`Verified full graph: ${manifest.n} neurons, ${manifest.edgeCount} connection slots. Preparing fixed rewired control.`);
  const preparedControl = await prepareRewire(graph, {seed: 1299709});
  const rewiredBefore = graphHashes(preparedControl.graph);
  const control = {graph: preparedControl.graph,
    metadata: {...preparedControl.metadata, rewiredIndicesSha256: rewiredBefore.indices}};
  assert.deepEqual(graphHashes(graph), beforeHashes, 'Preparing the control must not alter the anatomical graph');
  const sourceHashes = Object.fromEntries(['live-model.js', 'recall-model.js', 'memory-model.js', 'memory-rewire.js']
    .map(file => [file, sha256(fs.readFileSync(path.join(repository, 'site', file)))]));
  fs.mkdirSync(args.output, {recursive: true});
  const results = [];
  for (const file of fs.readdirSync(args.output).filter(name => /^seed-\d+\.json\.gz$/.test(name))) {
    const saved = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(args.output, file))).toString('utf8'));
    if (args.seeds.includes(saved.seed)) continue;
    if (saved.complete && JSON.stringify(saved.validation?.modelSourceSha256) === JSON.stringify(sourceHashes) &&
        JSON.stringify(saved.validation?.graphHashesBefore) === JSON.stringify(beforeHashes) &&
        JSON.stringify(saved.validation?.graphHashesAfter) === JSON.stringify(beforeHashes) &&
        JSON.stringify(saved.validation?.rewiredHashesBefore) === JSON.stringify(rewiredBefore) &&
        JSON.stringify(saved.validation?.rewiredHashesAfter) === JSON.stringify(rewiredBefore) &&
        Object.entries(DEFAULTS).every(([key, value]) => JSON.stringify(saved.config[key]) === JSON.stringify(value))) {
      assertMemoryIsolation(saved); results.push(saved);
    } else console.log(`Skipping saved ${file} from a different model, graph, or protocol revision.`);
  }
  console.log(`Rewired target SHA-256: ${rewiredBefore.indices}. Seeds: ${args.seeds.join(', ')}.`);
  for (const seed of args.seeds) {
    const started = performance.now();
    const experiment = new MemoryExperiment(graph, manifest, seed, {}, control);
    let phase = null;
    while (!experiment.done) {
      const record = experiment.step();
      if (record && `${record.condition}/${record.phase}` !== phase) {
        phase = `${record.condition}/${record.phase}`;
        console.log(`Seed ${seed}: ${phase}, ${((performance.now() - started) / 1000).toFixed(1)} s elapsed.`);
      }
    }
    const result = experiment.result();
    const wallSeconds = (performance.now() - started) / 1000;
    // Save raw observations even if a subsequent audit finds a problem; never
    // discard a scientific run merely because its outcome or validation fails.
    fs.writeFileSync(path.join(args.output, `seed-${seed}.json.gz`), zlib.gzipSync(`${JSON.stringify(result, null, 2)}\n`));
    const scores = assertMemoryIsolation(result);
    const afterHashes = graphHashes(graph), rewiredAfter = graphHashes(control.graph);
    assert.deepEqual(afterHashes, beforeHashes, 'The original graph must remain unchanged');
    assert.deepEqual(rewiredAfter, rewiredBefore, 'The fixed control graph must remain unchanged');
    result.validation = {kind: 'developmental smoke check; paired simulations of one anatomy and one fixed null',
      generatedAt: new Date().toISOString(), nodeVersion: process.version, wallSeconds, scores,
      cueOffVerified: true, evaluationWeightsFrozen: true, independentFeatureAndScoreReconstruction: true,
      pairedInputsVerified: true, connectomeUnchanged: true, rewiredGraphUnchanged: true,
      graphHashesBefore: beforeHashes, graphHashesAfter: afterHashes,
      rewiredHashesBefore: rewiredBefore, rewiredHashesAfter: rewiredAfter, modelSourceSha256: sourceHashes};
    fs.writeFileSync(path.join(args.output, `seed-${seed}.json.gz`), zlib.gzipSync(`${JSON.stringify(result, null, 2)}\n`));
    results.push(result); results.sort((left, right) => left.seed - right.seed);
    fs.writeFileSync(path.join(args.output, 'report.md'), report(results, manifest, control.metadata, startedAt));
    for (const condition of CONDITIONS) console.log(`Seed ${seed} ${condition}: cue ${percent(scores[condition].cueReadout.accuracy)}; OFF delays ${scores[condition].delays.map(score => `${score.delayMs}ms=${percent(score.accuracy)}`).join(', ')}.`);
    console.log(`Seed ${seed}: ${wallSeconds.toFixed(2)} s; all isolation and integrity checks passed.`);
  }
  console.log(`Completed ${args.seeds.length} requested runs; report covers ${results.length} saved seeds: ${path.join(args.output, 'report.md')}`);
}

if (require.main === module) main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1; });
module.exports = {predict, summarize, assertMemoryIsolation, report};
