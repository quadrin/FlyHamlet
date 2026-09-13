#!/usr/bin/env node
/* Developmental smoke check of the default browser learning protocol on the
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
const {LearningExperiment, DEFAULTS} = require('../site/learning-model.js');

const repository = path.resolve(__dirname, '..');
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const arrayBytes = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
const graphHashes = graph => Object.fromEntries(['indptr', 'indices', 'weights']
  .map(name => [name, sha256(arrayBytes(graph[name]))]));
const percentage = value => value === null ? 'n/a' : `${(value * 100).toFixed(1)}%`;
const interval = value => value ? `${percentage(value[0])}–${percentage(value[1])}` : 'n/a';

function argumentsForRun(argv) {
  let seeds = [42, 43, 44];
  let output = path.join(repository, 'results/learning_benchmark');
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--help' || argv[i] === '-h') {
      console.log('Usage: node scripts/benchmark_learning.cjs [--seeds 42,43,44] [--out results/learning_benchmark]\nRuns the unchanged default protocol; writes raw per-seed JSON and report.md.');
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

function assertLearningIsolation(result) {
  assert(result.complete, 'Every planned trial must finish');
  assert.deepEqual(result.config, DEFAULTS, 'The benchmark uses the unchanged default protocol');
  assert.equal(result.trials.length, DEFAULTS.baselineTrials + DEFAULTS.trainTrials + DEFAULTS.evaluationTrials);
  assert.equal(result.updateCount, DEFAULTS.trainTrials);
  assert.deepEqual(result.controlWeights, result.initialWeights, 'The untrained control must remain frozen');
  assert.deepEqual(result.finalWeights, result.frozenWeights, 'Final weights must equal evaluation-start weights');
  for (const trial of result.trials) {
    if (trial.phase === 'evaluation') {
      assert.equal(trial.feedbackApplied, false, 'No evaluation feedback may update the decoder');
      assert.equal(trial.updateCountBefore, DEFAULTS.trainTrials);
      assert.equal(trial.updateCountAfter, DEFAULTS.trainTrials);
      assert.deepEqual(trial.weightsBefore, result.frozenWeights, 'Evaluation pre-decision weights must stay frozen');
      assert.deepEqual(trial.weightsAfter, result.frozenWeights, 'Evaluation post-decision weights must stay frozen');
    } else if (trial.phase === 'baseline') {
      assert.equal(trial.feedbackApplied, false);
      assert.deepEqual(trial.weightsBefore, result.initialWeights);
      assert.deepEqual(trial.weightsAfter, result.initialWeights);
    } else {
      assert.equal(trial.feedbackApplied, true);
      assert.equal(trial.updateCountAfter, trial.updateCountBefore + 1);
    }
  }
}

function report(results, manifest, startedAt) {
  const rows = results.map(result => {
    const metric = result.metrics;
    return `| ${result.seed} | ${percentage(metric.baseline.accuracy)} | ${percentage(metric.train.accuracy)} | ${percentage(metric.evaluation.accuracy)} (${metric.evaluation.correct}/${metric.evaluation.total}) | ${interval(metric.evaluation.wilson95)} | ${percentage(metric.evaluation.control.accuracy)} | ${result.validation.wallSeconds.toFixed(2)} s |`;
  });
  return `# Two-letter learning benchmark: developmental smoke check

Generated ${startedAt}. ${results.length} technical replicate${results.length === 1 ? '' : 's'} of **one anatomical connectome** (FlyWire v783; ${manifest.n.toLocaleString('en-US')} neurons, ${manifest.edgeCount.toLocaleString('en-US')} connected pairs). This is a developmental smoke check, not a preregistered study or biological replication.

The unchanged default protocol supplies a left-eye cue for T or right-eye cue for O. Each trial resets neural state, presents 100 Hz Poisson stimulation for 200 ms, and reads seven downstream motor-group mean rates from the final 150 ms. An external logistic decoder receives 20 baseline, 60 supervised training, then 40 evaluation trials. Classes are balanced and independently shuffled within each phase. Evaluation contains fresh neural-noise seeds and no decoder updates. Cue information remains present: this tests classification/copying, not recall, autonomous typing, or biological synaptic learning.

| Seed | Baseline | Training, before each update | Evaluation | Evaluation 95% Wilson interval | Untrained control, evaluation | Compute time |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
${rows.join('\n')}

Intervals describe trial accuracy within each run. They are not population-level confidence intervals across brains. The untrained control uses unchanged zero weights with an independent, reproducible tie-breaking draw shared with the trained decoder on each trial. It receives the same neural features and targets for scoring; its weights never update. No shuffled-connectome, no-connectivity, or conventional-controller comparator is included, so these results do not establish an advantage of biological wiring.

All runs passed assertions that the evaluation weights equal their frozen start-of-evaluation values before and after **every** evaluation trial, no evaluation feedback is applied, exactly 60 supervised updates occur, and the control remains unchanged. Compressed and raw CSR files passed manifest SHA-256 checks. SHA-256 digests of every in-memory connectivity array remained identical after every run: no biological connection was edited.

Raw records (including per-trial neural seeds, feature vectors, decoder weights, correctness, provenance, protocol, and validation hashes):

${results.map(result => `- [Seed ${result.seed}](seed-${result.seed}.json)`).join('\n')}

Reproduce with Node.js:

\`\`\`sh
node scripts/benchmark_learning.cjs --seeds ${results.map(result => result.seed).join(',')} --out results/learning_benchmark
\`\`\`
`;
}

function main() {
  const args = argumentsForRun(process.argv.slice(2));
  if (!args) return;
  const startedAt = new Date().toISOString();
  const {graph, manifest} = loadFullGraph();
  const beforeHashes = graphHashes(graph);
  const modelSources = Object.fromEntries(['live-model.js', 'learning-model.js']
    .map(file => [file, sha256(fs.readFileSync(path.join(repository, 'site', file)))]));
  fs.mkdirSync(args.output, {recursive: true});
  const results = [];
  console.log(`Verified full graph: ${manifest.n} neurons, ${manifest.edgeCount} connected pairs. Seeds: ${args.seeds.join(', ')}.`);
  for (const seed of args.seeds) {
    const started = performance.now();
    const experiment = new LearningExperiment(graph, manifest, seed);
    while (!experiment.done) experiment.step();
    const result = experiment.result();
    const wallSeconds = (performance.now() - started) / 1000;
    assertLearningIsolation(result);
    const afterHashes = graphHashes(graph);
    assert.deepEqual(afterHashes, beforeHashes, 'The experiment must not alter any connectome array');
    result.validation = {kind: 'developmental smoke check; technical replicates of one anatomical connectome',
      generatedAt: new Date().toISOString(), nodeVersion: process.version, wallSeconds,
      evaluationWeightsFrozen: true, evaluationFeedbackDisabled: true,
      controlUnchanged: true, connectomeUnchanged: true,
      graphHashesBefore: beforeHashes, graphHashesAfter: afterHashes, modelSourceSha256: modelSources};
    fs.writeFileSync(path.join(args.output, `seed-${seed}.json`), `${JSON.stringify(result, null, 2)}\n`);
    results.push(result);
    fs.writeFileSync(path.join(args.output, 'report.md'), report(results, manifest, startedAt));
    const metrics = result.metrics;
    console.log(`Seed ${seed}: baseline ${percentage(metrics.baseline.accuracy)}, train ${percentage(metrics.train.accuracy)}, evaluation ${percentage(metrics.evaluation.accuracy)} [${interval(metrics.evaluation.wilson95)}], control ${percentage(metrics.evaluation.control.accuracy)}; ${wallSeconds.toFixed(2)} s; isolation checks passed.`);
  }
  console.log(`Wrote ${results.length} raw runs and report: ${path.join(args.output, 'report.md')}`);
}

try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
