#!/usr/bin/env node
/* Developmental smoke check of the default browser phrase-recall protocol on the
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
const {RecallExperiment, DEFAULTS} = require('../site/recall-model.js');

const repository = path.resolve(__dirname, '..');
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const arrayBytes = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
const graphHashes = graph => Object.fromEntries(['indptr', 'indices', 'weights']
  .map(name => [name, sha256(arrayBytes(graph[name]))]));

function argumentsForRun(argv) {
  let seeds = [42];
  let output = path.join(repository, 'results/recall_benchmark');
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--help' || argv[i] === '-h') {
      console.log('Usage: node scripts/benchmark_recall.cjs [--seeds 42] [--out results/recall_benchmark]\nRuns the unchanged default protocol; writes raw per-seed JSON and report.md.');
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

function editDistance(left, right) {
  let previous = Array.from({length: right.length + 1}, (_, index) => index);
  for (let row = 1; row <= left.length; row++) {
    const current = [row];
    for (let column = 1; column <= right.length; column++) {
      current[column] = Math.min(current[column - 1] + 1, previous[column] + 1,
        previous[column - 1] + Number(left[row - 1] !== right[column - 1]));
    }
    previous = current;
  }
  return previous[right.length];
}

function scoreEpisodes(episodes, reference) {
  const exact = episodes.filter(episode => episode.output === reference && episode.stoppedBy === 'END').length;
  const ended = episodes.filter(episode => episode.stoppedBy === 'END').length;
  return {count: episodes.length, exact, ended, capped: episodes.length - ended,
    exactRate: episodes.length ? exact / episodes.length : null,
    meanEditDistance: episodes.length ? episodes.reduce((sum, episode) => sum + editDistance(episode.output, reference), 0) / episodes.length : null};
}

function assertRecallIsolation(result) {
  assert(result.complete, 'Every planned episode must finish');
  assert.equal(typeof result.reference, 'string', 'The scoring reference must be explicit');
  for (const [key, value] of Object.entries(DEFAULTS))
    assert.deepEqual(result.config[key], value, `The benchmark must use the default ${key}`);
  assert.equal(result.fitCount, 1, 'The decoder must be fitted exactly once before autonomous recall');
  assert.equal(result.finalFingerprint, result.frozenFingerprint, 'Frozen decoder fingerprints must agree');
  assert.deepEqual(result.finalWeights, result.frozenWeights, 'The recall decoder must remain frozen');
  const recall = result.episodes.filter(episode => episode.phase === 'recall');
  const ablated = result.episodes.filter(episode => episode.phase === 'ablated');
  assert.equal(recall.length, result.config.recallEpisodes);
  assert.equal(ablated.length, result.config.ablatedEpisodes);
  for (const episode of [...recall, ...ablated, ...result.comparator.episodes]) {
    assert(['END', 'cap'].includes(episode.stoppedBy), 'Every episode has an explicit stopping reason');
    assert.equal(episode.exact, episode.output === result.reference && episode.stoppedBy === 'END',
      'Exact recall requires the literal phrase and a predicted END');
    assert.equal(episode.editDistance, editDistance(episode.output, result.reference),
      'Reported edit distance must agree with independent reconstruction');
  }
  for (const episode of result.comparator.episodes) {
    assert(episode.decisions.length > 0, 'The conventional comparator must export its decisions');
    assert.equal(episode.decisions[0].cue, 'START');
    const emitted = [];
    episode.decisions.forEach((decision, index) => {
      if (index) assert.equal(decision.cue, episode.decisions[index - 1].prediction,
        'The conventional comparator must also consume only its own prior prediction');
      if (decision.prediction === 'END') assert.equal(index, episode.decisions.length - 1);
      else emitted.push(decision.prediction);
    });
    assert.equal(emitted.join(''), episode.output);
    assert.equal(episode.decisions.at(-1).prediction === 'END', episode.stoppedBy === 'END');
    assert(episode.decisions.length <= result.config.maxDecisions);
  }
  for (const episode of [...recall, ...ablated]) {
    const records = result.trials.filter(record => record.phase === episode.phase && record.episode === episode.episode);
    assert(records.length > 0, 'Each episode exports its actual decisions');
    assert.equal(records[0].cue, 'START', 'Every autonomous episode begins with START');
    const emitted = [];
    for (let index = 0; index < records.length; index++) {
      const record = records[index];
      assert(!Object.hasOwn(record, 'target'), 'Autonomous records must not contain a teacher target');
      if (index > 0) assert.equal(record.cue, records[index - 1].prediction,
        'Autonomous input must be the previous self-generated prediction');
      if (record.prediction === 'END') assert.equal(index, records.length - 1, 'No actions may follow END');
      else emitted.push(record.prediction);
      if (Object.hasOwn(record, 'feedbackApplied')) assert.equal(record.feedbackApplied, false);
      assert.equal(record.decoderVersion, 1);
      assert.equal(record.updateCountBefore, 1);
      assert.equal(record.updateCountAfter, 1);
      assert.equal(record.decoderFingerprint, result.frozenFingerprint);
      if (Object.hasOwn(record, 'weightsBefore')) assert.deepEqual(record.weightsBefore, result.frozenWeights);
      if (Object.hasOwn(record, 'weightsAfter')) assert.deepEqual(record.weightsAfter, result.frozenWeights);
    }
    assert.equal(emitted.join(''), episode.output, 'Output must be reconstructed from the committed choices');
    assert.equal(records.at(-1).prediction === 'END', episode.stoppedBy === 'END');
    if (typeof episode.decisions === 'number') assert.equal(records.length, episode.decisions);
    if (Array.isArray(episode.decisions)) assert.deepEqual(episode.decisions,
      records.map(({step, cue, prediction, neuralSeed}) => ({step, cue, prediction, neuralSeed})),
      'Per-episode audit decisions must match the raw trial log');
    assert(records.length <= result.config.maxDecisions, 'The safety cap must be respected');
  }
  return {recall: scoreEpisodes(recall, result.reference), ablated: scoreEpisodes(ablated, result.reference),
    comparator: scoreEpisodes(result.comparator.episodes, result.reference)};
}

function report(results, manifest, startedAt) {
  const rows = results.flatMap(result => Object.entries(result.validation.scores).map(([condition, score]) =>
    `| ${result.seed} | ${condition} | ${score.exact}/${score.count} | ${score.ended}/${score.count} | ${score.capped} | ${score.meanEditDistance?.toFixed(2) ?? 'n/a'} |`));
  const timing = results.map(result => `Seed ${result.seed}: ${result.validation.wallSeconds.toFixed(2)} s`).join('; ');
  return `# Phrase-recall benchmark: developmental smoke check

Generated ${startedAt}. ${results.length} seeded run${results.length === 1 ? '' : 's'} of **one anatomical connectome** (FlyWire v783; ${manifest.n.toLocaleString('en-US')} neurons, ${manifest.edgeCount.toLocaleString('en-US')} connected pairs). Repeated seeds and rollouts are technical simulations, not independent biological brains. This is a developmental smoke check, not a preregistered study.

Reference phrase: **${results[0].reference}**. Each run uses the unchanged default protocol: ${results[0].config.trainEpisodes} teacher-forced training episodes, then ${results[0].config.recallEpisodes} autonomous rollouts and ${results[0].config.ablatedEpisodes} rollouts with past history masked. During autonomous recall, the next cue comes exclusively from the preceding emitted character, beginning with START; neither a next-letter cue nor a correction is supplied. END is a learned output. Exact success requires both the literal reference phrase and a predicted END. Incorrect outputs and decisions remain in the raw records.

The full connectome is fixed. Learning occurs in an external decoder, with an external history buffer of ${results[0].config.historyLength} feature vectors. The ablation keeps the current ${results[0].config.poolCount}-component neural vector and masks all ${results[0].config.historyLength - 1} past vectors; it uses the same trained weights and self-generated cues. Symbol cues are arbitrary experimenter-designed codes, not a claim that the fly recognizes written characters. The conventional comparator learns from a separate history of explicit character one-hot vectors on the same training examples and then generates its own feedback. Any successful recall is a property of the combined model, external memory, coding scheme, and decoder; it does not demonstrate biological synaptic learning, Shakespeare comprehension, or recall from fly wiring alone.

| Seed | Condition | Exact phrase + END | Ended with END | Capped episodes | Mean edit distance |
| --- | --- | ---: | ---: | ---: | ---: |
${rows.join('\n')}

Compute time: ${timing}. No minimum accuracy or preferred outcome is enforced. Edit distances and stopping success above are independently recomputed from exported decisions. Repeated conventional episodes are identical deterministic rollouts, not independent evidence. The ablation and conventional comparator are limited controls, not a matched shuffled-connectome experiment or proof of an advantage from anatomical wiring.

All runs passed assertions that autonomous inputs match preceding self-generated choices, no autonomous record contains a teacher target, outputs reconstruct exactly from the chosen characters, END terminates generation, and final decoder weights equal their frozen values. Compressed and raw CSR files passed the manifest SHA-256 checks. In-memory hashes of every connectivity array were unchanged after each run.

Raw records include neural seeds, trial features, generated decisions, stopping reasons, model/protocol provenance, fitted weights, and validation hashes:

${results.map(result => `- [Seed ${result.seed}](seed-${result.seed}.json)`).join('\n')}

Reproduce with Node.js:

\`\`\`sh
node scripts/benchmark_recall.cjs --seeds ${results.map(result => result.seed).join(',')} --out results/recall_benchmark
\`\`\`
`;
}

function main() {
  const args = argumentsForRun(process.argv.slice(2));
  if (!args) return;
  const startedAt = new Date().toISOString();
  const {graph, manifest} = loadFullGraph();
  const beforeHashes = graphHashes(graph);
  const modelSources = Object.fromEntries(['live-model.js', 'recall-model.js']
    .map(file => [file, sha256(fs.readFileSync(path.join(repository, 'site', file)))]));
  fs.mkdirSync(args.output, {recursive: true});
  // Later replications can append seeds without recomputing earlier full runs.
  // Only matching source/protocol revisions are pooled into the report.
  const results = [];
  for (const file of fs.readdirSync(args.output).filter(name => /^seed-\d+\.json$/.test(name))) {
    const saved = JSON.parse(fs.readFileSync(path.join(args.output, file), 'utf8'));
    if (args.seeds.includes(saved.seed)) continue;
    if (saved.complete && saved.reference === DEFAULTS.phrase &&
        JSON.stringify(saved.validation?.graphHashesBefore) === JSON.stringify(beforeHashes) &&
        JSON.stringify(saved.validation?.graphHashesAfter) === JSON.stringify(beforeHashes) &&
        JSON.stringify(saved.validation?.modelSourceSha256) === JSON.stringify(modelSources) &&
        Object.entries(DEFAULTS).every(([key, value]) => JSON.stringify(saved.config[key]) === JSON.stringify(value))) {
      assertRecallIsolation(saved);
      results.push(saved);
    } else console.log(`Skipping saved ${file} from a different model/protocol revision.`);
  }
  console.log(`Verified full graph: ${manifest.n} neurons, ${manifest.edgeCount} connected pairs. Seeds: ${args.seeds.join(', ')}.`);
  for (const seed of args.seeds) {
    const started = performance.now();
    const experiment = new RecallExperiment(graph, manifest, seed);
    let lastPhase = null;
    while (!experiment.done) {
      const record = experiment.step();
      if (record && record.phase !== lastPhase) {
        lastPhase = record.phase;
        console.log(`Seed ${seed}: ${record.phase} phase, ${(performance.now() - started).toFixed(0)} ms elapsed.`);
      }
    }
    const result = experiment.result();
    const wallSeconds = (performance.now() - started) / 1000;
    const scores = assertRecallIsolation(result);
    const afterHashes = graphHashes(graph);
    assert.deepEqual(afterHashes, beforeHashes, 'The experiment must not alter any connectome array');
    result.validation = {kind: 'developmental smoke check; one anatomical connectome; external memory and decoder',
      generatedAt: new Date().toISOString(), nodeVersion: process.version, wallSeconds, scores,
      autonomousFeedbackVerified: true, targetsAbsentDuringRecall: true, finalWeightsFrozen: true,
      independentlyScored: true, connectomeUnchanged: true,
      graphHashesBefore: beforeHashes, graphHashesAfter: afterHashes, modelSourceSha256: modelSources};
    fs.writeFileSync(path.join(args.output, `seed-${seed}.json`), `${JSON.stringify(result, null, 2)}\n`);
    results.push(result);
    results.sort((left, right) => left.seed - right.seed);
    fs.writeFileSync(path.join(args.output, 'report.md'), report(results, manifest, startedAt));
    const summary = Object.entries(scores).map(([condition, score]) =>
      `${condition}: exact ${score.exact}/${score.count}, END ${score.ended}/${score.count}, edit ${score.meanEditDistance?.toFixed(2) ?? 'n/a'}`).join('; ');
    console.log(`Seed ${seed}: ${summary}; ${wallSeconds.toFixed(2)} s; isolation checks passed.`);
  }
  console.log(`Completed ${args.seeds.length} requested runs; report covers ${results.length} saved seeds: ${path.join(args.output, 'report.md')}`);
}

if (require.main === module) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
module.exports = {editDistance, scoreEpisodes, assertRecallIsolation, report};
