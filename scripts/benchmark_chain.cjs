#!/usr/bin/env node
/* Developmental smoke check of the default browser self-driven phrase-recall protocol on the
 * committed full connectome. No npm packages; Node 18+. These are technical replicates of
 * one anatomical brain, not a preregistered biological study.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {ChainExperiment, chainLength, DEFAULTS, CONDITIONS, START, END} = require('../site/chain-model.js');
const {prepareSlowGraph} = require('../site/sequence-model.js');
const {OUTPUT_CODES, editDistance} = require('../site/recall-model.js');
const {predict, standardize, scalerFromRows} = require('./benchmark_sequence.cjs');

const repository = path.resolve(__dirname, '..');
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const arrayBytes = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
const graphHashes = graph => Object.fromEntries(['indptr', 'indices', 'weights'].map(name => [name, sha256(arrayBytes(graph[name]))]));
const close = (actual, expected, message) => assert(Math.abs(actual - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), message);
const percent = value => value === null ? 'n/a' : `${(100 * value).toFixed(1)}%`;

function argumentsForRun(argv) {
  let seeds = [42];
  let output = path.join(repository, 'results/chain_benchmark');
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--help' || argv[i] === '-h') {
      console.log('Usage: node scripts/benchmark_chain.cjs [--seeds 42] [--out results/chain_benchmark]\nRuns the unchanged default protocol; writes raw per-seed JSON and report.md.');
      return null;
    }
    if (argv[i] === '--seeds') {
      const supplied = argv[++i];
      if (!supplied || !/^\d+(,\d+)*$/.test(supplied)) throw new Error('--seeds requires comma-separated unsigned integers.');
      seeds = supplied.split(',').map(Number);
      if (seeds.some(seed => !Number.isSafeInteger(seed) || seed > 0xffffffff)) throw new Error('Seeds must be between 0 and 4294967295.');
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
    graph[name] = raw.byteOffset % 4 === 0 ? new Type(raw.buffer, raw.byteOffset, raw.byteLength / 4) : new Type(Uint8Array.from(raw).buffer);
  }
  assert.equal(graph.indptr.length, manifest.n + 1);
  assert.equal(graph.indptr[manifest.n], manifest.edgeCount);
  assert.equal(graph.indices.length, manifest.edgeCount);
  assert.equal(graph.weights.length, manifest.edgeCount);
  for (let i = 0; i < manifest.n; i++) assert(graph.indptr[i] <= graph.indptr[i + 1], 'Nonmonotonic CSR row offsets');
  for (const index of graph.indices) if (index >= manifest.n) throw new Error('Invalid CSR target index.');
  for (const weight of graph.weights) if (!Number.isFinite(weight)) throw new Error('Nonfinite synaptic weight.');
  return {graph: Object.freeze(graph), manifest};
}

function assertChainIsolation(result) {
  assert(result.complete, 'Every planned condition, phase and episode must finish');
  for (const [key, value] of Object.entries(DEFAULTS)) assert.deepEqual(result.config[key], value, `Default ${key} must be unchanged`);
  const p = result.config, reference = result.reference, poolSizes = result.metadata.poolSizes;
  const targets = [...reference, END], inputs = [START, ...reference];
  assert.equal(result.fitCount, 3);
  assert.equal(result.slowNeuralConfig.tau_mem_ms, result.neuralConfig.tau_mem_ms * p.slowTimeFactor);
  assert.equal(result.slowGraph.weightScale, p.slowWeightScale);
  const scores = {};
  for (const [conditionIndex, condition] of CONDITIONS.entries()) {
    const model = result.models[condition];
    assert.equal(model.fitCount, 1);
    assert.equal(model.trainingExamples, p.trainEpisodes * targets.length);
    assert.deepEqual(model.labels, [...OUTPUT_CODES]);
    assert.equal(model.dimensions, 2 * p.poolCount);
    assert.deepEqual(model.weights, model.frozenWeights);
    assert.equal(model.fingerprint, model.finalFingerprint);
    assert.equal(model.scaler.fingerprint, model.scaler.finalFingerprint);
    const records = result.trials.filter(record => record.condition === condition);
    const training = records.filter(record => record.phase === 'train');
    assert.equal(training.length, model.trainingExamples);
    const rebuilt = scalerFromRows(training.map(record => record.measurements));
    rebuilt.mean.forEach((value, i) => close(model.scaler.mean[i], value, 'Scaler mean must come from training measurements only'));
    rebuilt.scale.forEach((value, i) => close(model.scaler.scale[i], value, 'Scaler scale must come from training measurements only'));
    const episodeSeeds = new Map();
    for (const record of records) {
      assert.equal(record.snapshotMs, p.cueMs + p.gapMs);
      assert.equal(record.cueOffMs, p.cueMs);
      assert.equal(record.measurements.length, 2 * p.poolCount);
      record.measurements.forEach((value, i) => {
        const size = poolSizes[i % p.poolCount];
        if (!size) assert.equal(value, 0);
        else close(value * size / p.stateResolutionMv, Math.round(value * size / p.stateResolutionMv), 'Pool means must be sums of fixed-resolution quanta');
      });
      assert.equal(record.silent, record.nonrestVoltageNeurons === 0 && record.nonrestCurrentNeurons === 0);
      const key = `${record.phase}/${record.episode}`;
      if (episodeSeeds.has(key)) assert.equal(record.episodeSeed, episodeSeeds.get(key), 'One noise seed per episode'); else episodeSeeds.set(key, record.episodeSeed);
      if (condition === 'reset') assert.notEqual(record.cueSeed, record.episodeSeed);
      else assert.equal(record.cueSeed, record.episodeSeed, 'Retained conditions keep one network per episode');
      if (record.phase === 'train') {
        assert.equal(record.cue, inputs[record.step - 1]); assert.equal(record.target, targets[record.step - 1]);
        assert.equal(record.prediction, null); assert.equal(record.features, null); assert.equal(record.decoderVersion, 0);
        assert.equal(record.feedbackApplied, false);
        continue;
      }
      assert.equal(record.decoderVersion, 1);
      assert.equal(record.updateCountBefore, conditionIndex + 1); assert.equal(record.updateCountAfter, conditionIndex + 1);
      assert.equal(record.fitApplied, false);
      assert.equal(record.decoderFingerprint, model.fingerprint); assert.equal(record.scalerFingerprint, model.scaler.fingerprint);
      const features = standardize(model.scaler, record.measurements);
      features.forEach((value, i) => close(record.features[i], value, 'Features must be the frozen scaler applied to the measurement'));
      const decision = predict(features, model.frozenWeights, OUTPUT_CODES);
      assert.equal(record.prediction, decision.prediction, 'Choice must follow only that snapshot, its frozen scaler and frozen decoder');
      record.scores.forEach((score, i) => close(score, decision.scores[i], 'Reconstructed score must agree'));
      if (record.phase === 'diagnostic') {
        assert.equal(record.cue, inputs[record.step - 1]); assert.equal(record.target, targets[record.step - 1]);
        assert.equal(record.correct, record.prediction === record.target); assert.equal(record.feedbackApplied, false);
      } else {
        assert.equal(record.target, null); assert.equal(record.correct, null);
        assert.equal(record.feedbackApplied, record.step > 1);
      }
    }
    // Recall cues must be the actor's own previous predictions, never the reference.
    const recall = records.filter(record => record.phase === 'recall');
    const episodes = result.episodes.filter(item => item.condition === condition && item.phase === 'recall');
    assert.equal(episodes.length, p.recallEpisodes);
    for (const episode of episodes) {
      const rows = recall.filter(record => record.episode === episode.episode).sort((a, b) => a.step - b.step);
      assert.equal(rows.length, episode.decisions.length);
      assert(rows.length <= p.maxDecisions);
      let output = '';
      rows.forEach((record, index) => {
        assert.equal(record.step, index + 1);
        assert.equal(record.cue, index === 0 ? START : rows[index - 1].prediction, 'Each recall cue is the previous own prediction');
        if (record.prediction !== END) output += record.prediction;
      });
      const last = rows[rows.length - 1];
      assert.equal(episode.stoppedBy, last.prediction === END ? END : 'cap');
      if (last.prediction !== END) assert.equal(rows.length, p.maxDecisions);
      assert.equal(episode.output, output);
      assert.equal(episode.chainLength, chainLength(output, reference));
      assert.equal(episode.editDistance, editDistance(output, reference));
      assert.equal(episode.exact, output === reference && episode.stoppedBy === END);
    }
    const diagnostic = records.filter(record => record.phase === 'diagnostic');
    assert.equal(diagnostic.length, p.diagnosticEpisodes * targets.length);
    const correct = diagnostic.filter(record => record.correct).length;
    const lengths = episodes.map(item => item.chainLength);
    scores[condition] = {diagnosticAccuracy: diagnostic.length ? correct / diagnostic.length : null, diagnosticCorrect: correct, diagnosticTotal: diagnostic.length,
      chainLengths: lengths, meanChainLength: lengths.reduce((sum, value) => sum + value, 0) / lengths.length, maxChainLength: Math.max(...lengths),
      exact: episodes.filter(item => item.exact).length, ended: episodes.filter(item => item.stoppedBy === END).length,
      meanEditDistance: episodes.reduce((sum, item) => sum + item.editDistance, 0) / episodes.length,
      outputs: episodes.map(item => item.output),
      meanRecallCueSpikes: recall.reduce((sum, record) => sum + record.cueSpikes, 0) / recall.length,
      meanRecallNonrestVoltageNeurons: recall.reduce((sum, record) => sum + record.nonrestVoltageNeurons, 0) / recall.length};
    const exported = result.metrics[condition];
    close(scores[condition].diagnosticAccuracy, exported.diagnostic.accuracy, 'Diagnostic accuracy must agree with independent scoring');
    assert.deepEqual(exported.recall.chainLengths, lengths);
    close(scores[condition].meanEditDistance, exported.recall.meanEditDistance, 'Edit distance must agree');
    assert.equal(exported.recall.exact, scores[condition].exact);
  }
  const original = result.trials.filter(record => record.condition === 'original' && record.phase !== 'recall');
  for (const condition of ['slow', 'reset']) {
    const rows = result.trials.filter(record => record.condition === condition && record.phase !== 'recall');
    assert.equal(rows.length, original.length);
    rows.forEach((record, index) => {
      for (const key of ['phase', 'episode', 'step', 'cue', 'target', 'episodeSeed']) assert.equal(record[key], original[index][key], 'Teacher-forced phases share cues and episode seeds across conditions');
    });
  }
  return scores;
}

function report(results, manifest, slowGraph, generatedAt) {
  const rows = results.flatMap(result => CONDITIONS.map(condition => {
    const score = result.validation.scores[condition];
    return `| ${result.seed} | ${condition} | ${score.diagnosticCorrect}/${score.diagnosticTotal} (${percent(score.diagnosticAccuracy)}) | ${score.chainLengths.join(', ')} | ${score.meanChainLength.toFixed(1)} | ${score.exact}/${score.chainLengths.length} | ${score.meanEditDistance.toFixed(1)} | ${score.meanRecallCueSpikes.toFixed(0)} | ${score.meanRecallNonrestVoltageNeurons.toFixed(0)} |`;
  }));
  const outputs = results.flatMap(result => CONDITIONS.flatMap(condition => result.validation.scores[condition].outputs.map((output, index) => `| ${result.seed} | ${condition} | ${index + 1} | \`${output.replace(/ /g, '␣')}\` |`)));
  const comparator = results[0].comparator.episode;
  const slow = results[0].slowNeuralConfig;
  return `# Self-driven phrase recall benchmark: developmental smoke check

Generated ${generatedAt}. ${results.length} seeded run${results.length === 1 ? '' : 's'} of one FlyWire v783 anatomical graph (${manifest.n.toLocaleString('en-US')} neurons, ${manifest.edgeCount.toLocaleString('en-US')} connection slots). Seeds are technical simulations, not independent biological animals.

Reference phrase: \`${results[0].reference}\` (${results[0].reference.length} characters plus END). Each condition trains one scaler and one ridge readout on ${DEFAULTS.trainEpisodes} teacher-forced episodes (${DEFAULTS.trainEpisodes * (results[0].reference.length + 1)} snapshots), then runs ${DEFAULTS.diagnosticEpisodes} held-out teacher-forced diagnostic episodes and ${DEFAULTS.recallEpisodes} autonomous recall episodes. Every cue lasts ${DEFAULTS.cueMs} ms at ${DEFAULTS.cueRateHz} Hz, followed by a ${DEFAULTS.gapMs} ms gap; the snapshot of ${2 * DEFAULTS.poolCount} quantized voltage and current pool means (${DEFAULTS.stateResolutionMv} mV per neuron) is read at the end of the gap and the next cue starts immediately. In retained conditions one network runs through the whole episode without reset. During recall the decoded letter becomes the next sensory cue; END stops an episode and a cap of ${DEFAULTS.maxDecisions} decisions does not depend on the phrase length. Chain length is the longest correct prefix of the output.

The slower-dynamics condition multiplies the membrane and synaptic time constants by ${DEFAULTS.slowTimeFactor} (${slow.tau_mem_ms} ms and ${slow.tau_syn_ms} ms) and every signed synaptic weight by ${DEFAULTS.slowWeightScale}; delay, refractory period and thresholds are unchanged. It is an imposed engineering hypothesis. The reset control uses the slower model and identical cues but replaces all neural state at every cue onset, so each snapshot reflects only the current letter. A current-cue-only comparator (ridge on the one-hot cue, own-feedback rollout, no state) produces \`${comparator.output.replace(/ /g, '␣')}\` with chain length ${comparator.chainLength}: the best any decoder can do without memory of earlier letters.

| Seed | Condition | Diagnostic next-letter accuracy (teacher-forced, held out) | Recall chain lengths | Mean chain | Exact | Mean edit distance | Mean spikes per recall cue | Mean neurons off rest at recall snapshots |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
${rows.join('\n')}

| Seed | Condition | Episode | Autonomous output |
| --- | --- | ---: | --- |
${outputs.join('\n')}

Validation independently rebuilt every scaler from training measurements, every feature from its snapshot and frozen scaler, and every choice from frozen weights; confirmed that every recall cue was the actor's own previous prediction and never the reference; recomputed chain lengths, edit distances and diagnostic accuracies; and confirmed that teacher-forced phases share cues and episode seeds across conditions. The three fitted readouts stayed frozen. Original compressed/raw assets passed manifest SHA-256 checks; hashes of the anatomical graph and of the rescaled comparison graph (scaled weights SHA-256 ${slowGraph.scaledWeightsSha256}) were unchanged after each complete run. Reading internal voltages is more permissive than spike decoding; a long chain under slower dynamics concerns that hypothetical model, not fruit-fly physiology, and no result here is memory for a text.

Compute times: ${results.map(result => `seed ${result.seed}: ${result.validation.wallSeconds.toFixed(2)} s`).join('; ')}.

${results.map(result => `- [Seed ${result.seed}: full records, models, provenance, and integrity checks (gzip JSON)](seed-${result.seed}.json.gz)`).join('\n')}

\`\`\`sh
node scripts/benchmark_chain.cjs --seeds ${results.map(result => result.seed).join(',')} --out results/chain_benchmark
\`\`\`
`;
}

async function main() {
  const args = argumentsForRun(process.argv.slice(2));
  if (!args) return;
  const startedAt = new Date().toISOString();
  const {graph, manifest} = loadFullGraph();
  const beforeHashes = graphHashes(graph);
  console.log(`Verified full graph: ${manifest.n} neurons, ${manifest.edgeCount} connection slots. Preparing the rescaled comparison graph.`);
  const slow = prepareSlowGraph(graph, DEFAULTS.slowWeightScale);
  const slowBefore = graphHashes(slow.graph);
  const prepared = {graph: slow.graph, metadata: {...slow.metadata, scaledWeightsSha256: slowBefore.weights}};
  const sourceHashes = Object.fromEntries(['live-model.js', 'recall-model.js', 'sequence-model.js', 'chain-model.js']
    .map(file => [file, sha256(fs.readFileSync(path.join(repository, 'site', file)))]));
  fs.mkdirSync(args.output, {recursive: true});
  const results = [];
  for (const file of fs.readdirSync(args.output).filter(name => /^seed-\d+\.json\.gz$/.test(name))) {
    const saved = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(args.output, file))).toString('utf8'));
    if (args.seeds.includes(saved.seed)) continue;
    if (saved.complete && JSON.stringify(saved.validation?.modelSourceSha256) === JSON.stringify(sourceHashes) &&
        JSON.stringify(saved.validation?.graphHashesBefore) === JSON.stringify(beforeHashes) &&
        JSON.stringify(saved.validation?.slowHashesBefore) === JSON.stringify(slowBefore) &&
        Object.entries(DEFAULTS).every(([key, value]) => JSON.stringify(saved.config[key]) === JSON.stringify(value))) {
      assertChainIsolation(saved); results.push(saved);
    } else console.log(`Skipping saved ${file} from a different model, graph, or protocol revision.`);
  }
  console.log(`Scaled weights SHA-256: ${slowBefore.weights}. Seeds: ${args.seeds.join(', ')}.`);
  for (const seed of args.seeds) {
    const started = performance.now();
    const experiment = new ChainExperiment(graph, manifest, seed, {}, prepared);
    let phase = null;
    while (!experiment.done) {
      const record = experiment.step();
      if (record && `${record.condition}/${record.phase}` !== phase) {
        phase = `${record.condition}/${record.phase}`;
        console.log(`Seed ${seed}: ${phase}, ${((performance.now() - started) / 1000).toFixed(1)} s elapsed.`);
      }
      if (record?.phase === 'recall' && (record.prediction === END || record.step === experiment.protocol.maxDecisions))
        console.log(`  ${record.condition} recall episode ${record.episode}: "${experiment.episodes.at(-1).output}" (chain ${experiment.episodes.at(-1).chainLength})`);
    }
    const result = experiment.result();
    const wallSeconds = (performance.now() - started) / 1000;
    fs.writeFileSync(path.join(args.output, `seed-${seed}.json.gz`), zlib.gzipSync(`${JSON.stringify(result, null, 2)}\n`));
    const scores = assertChainIsolation(result);
    const afterHashes = graphHashes(graph), slowAfter = graphHashes(slow.graph);
    assert.deepEqual(afterHashes, beforeHashes, 'The original graph must remain unchanged');
    assert.deepEqual(slowAfter, slowBefore, 'The rescaled comparison graph must remain unchanged');
    result.validation = {kind: 'developmental smoke check; paired simulations of one anatomy under original and imposed slower dynamics',
      generatedAt: new Date().toISOString(), nodeVersion: process.version, wallSeconds, scores,
      ownFeedbackVerified: true, evaluationWeightsFrozen: true, independentScalerFeatureAndScoreReconstruction: true,
      pairedInputsVerified: true, connectomeUnchanged: true, slowGraphUnchanged: true,
      graphHashesBefore: beforeHashes, graphHashesAfter: afterHashes, slowHashesBefore: slowBefore, slowHashesAfter: slowAfter, modelSourceSha256: sourceHashes};
    fs.writeFileSync(path.join(args.output, `seed-${seed}.json.gz`), zlib.gzipSync(`${JSON.stringify(result, null, 2)}\n`));
    results.push(result); results.sort((left, right) => left.seed - right.seed);
    fs.writeFileSync(path.join(args.output, 'report.md'), report(results, manifest, prepared.metadata, startedAt));
    for (const condition of CONDITIONS) console.log(`Seed ${seed} ${condition}: diagnostic ${percent(scores[condition].diagnosticAccuracy)}; chains ${scores[condition].chainLengths.join(',')} (mean ${scores[condition].meanChainLength.toFixed(1)}); exact ${scores[condition].exact}.`);
    console.log(`Seed ${seed}: ${wallSeconds.toFixed(2)} s; all isolation and integrity checks passed.`);
  }
  console.log(`Completed ${args.seeds.length} requested runs; report covers ${results.length} saved seeds: ${path.join(args.output, 'report.md')}`);
}

if (require.main === module) main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1; });
module.exports = {assertChainIsolation, report};
