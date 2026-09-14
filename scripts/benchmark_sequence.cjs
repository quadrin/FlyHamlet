#!/usr/bin/env node
/* Developmental smoke check of the default browser two-letter state-decoding protocol on
 * the committed full connectome. No npm packages; Node 18+. These are technical
 * replicates of one anatomical brain, not a preregistered biological study.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {SequenceExperiment, prepareSlowGraph, DEFAULTS, CONDITIONS, POSITIONS, PAIRS, LETTERS} = require('../site/sequence-model.js');

const repository = path.resolve(__dirname, '..');
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const arrayBytes = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
const graphHashes = graph => Object.fromEntries(['indptr', 'indices', 'weights']
  .map(name => [name, sha256(arrayBytes(graph[name]))]));

function argumentsForRun(argv) {
  let seeds = [42];
  let output = path.join(repository, 'results/sequence_benchmark');
  let resolution = DEFAULTS.stateResolutionMv;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--help' || argv[i] === '-h') {
      console.log('Usage: node scripts/benchmark_sequence.cjs [--seeds 42] [--out results/sequence_benchmark] [--resolution 0.01]\nRuns the default protocol (optionally at another state resolution in mV); writes raw per-seed JSON and report.md.');
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
    } else if (argv[i] === '--resolution') {
      resolution = Number(argv[++i]);
      if (!(resolution > 0) || !Number.isFinite(resolution)) throw new Error('--resolution requires a positive number of millivolts.');
    } else throw new Error(`Unknown argument: ${argv[i]}`);
  }
  return {seeds, output, resolution};
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

const close = (actual, expected, message) => assert(Math.abs(actual - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), message);
const percent = value => value === null ? 'n/a' : `${(100 * value).toFixed(1)}%`;

function standardize(scaler, measurements) {
  assert.equal(measurements.length, scaler.mean.length);
  return measurements.map((value, index) => (value - scaler.mean[index]) * scaler.scale[index]);
}
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
function scalerFromRows(rows) {
  const dimensions = rows[0].length;
  const mean = Array.from({length: dimensions}, (_, i) => rows.reduce((sum, row) => sum + row[i] / rows.length, 0));
  const scale = mean.map((m, i) => {
    const std = Math.sqrt(rows.reduce((sum, row) => sum + (row[i] - m) ** 2 / rows.length, 0));
    return std <= 1e-12 * Math.max(1, Math.abs(m)) ? 0 : 1 / std;
  });
  return {mean, scale};
}

function summarize(records, selectSnapshot) {
  const byPair = Object.fromEntries(PAIRS.map(pair => [pair, {correct: 0, total: 0}]));
  const positions = Object.fromEntries(POSITIONS.map(position => [position, 0]));
  let correct = 0, silent = 0, nonrestVoltage = 0, nonrestCurrent = 0, spikesSinceCueOff = 0, cueSpikes = 0, postCueSpikes = 0;
  for (const record of records) {
    const snapshot = selectSnapshot(record);
    const success = snapshot.pairPrediction === record.pair;
    correct += Number(success);
    byPair[record.pair].correct += Number(success);
    byPair[record.pair].total++;
    for (const position of POSITIONS) positions[position] += Number(snapshot.predictions[position].prediction === record[position]);
    silent += Number(snapshot.silent);
    nonrestVoltage += snapshot.nonrestVoltageNeurons;
    nonrestCurrent += snapshot.nonrestCurrentNeurons;
    spikesSinceCueOff += snapshot.spikesSinceCueOff;
    cueSpikes += record.cueSpikes;
    postCueSpikes += record.postCueSpikes;
  }
  const total = records.length;
  return {correct, total, accuracy: total ? correct / total : null,
    balancedAccuracy: total ? PAIRS.reduce((sum, pair) => sum + byPair[pair].correct / byPair[pair].total, 0) / PAIRS.length : null,
    firstAccuracy: total ? positions.first / total : null, secondAccuracy: total ? positions.second / total : null,
    byPair, silentTrials: silent,
    meanNonrestVoltageNeurons: total ? nonrestVoltage / total : null,
    meanNonrestCurrentNeurons: total ? nonrestCurrent / total : null,
    meanSpikesSinceCueOff: total ? spikesSinceCueOff / total : null,
    meanCueSpikes: total ? cueSpikes / total : null,
    meanPostCueSpikes: total ? postCueSpikes / total : null};
}

function assertSequenceIsolation(result, resolution = DEFAULTS.stateResolutionMv) {
  assert(result.complete, 'Every planned condition and trial must finish');
  for (const [key, value] of Object.entries(DEFAULTS))
    if (key !== 'stateResolutionMv') assert.deepEqual(result.config[key], value, `Default ${key} must be unchanged`);
  assert.equal(result.config.stateResolutionMv, resolution);
  const p = result.config, poolSizes = result.metadata.poolSizes;
  assert.equal(poolSizes.length, p.poolCount);
  assert.equal(result.metadata.measurementCount, 2 * p.poolCount);
  assert.equal(result.fitCount, 3);
  assert.equal(result.readoutFitCount, 3 * (p.delaysMs.length + 1) * 2);
  assert.equal(result.slowNeuralConfig.tau_mem_ms, result.neuralConfig.tau_mem_ms * p.slowTimeFactor);
  assert.equal(result.slowNeuralConfig.tau_syn_ms, result.neuralConfig.tau_syn_ms * p.slowTimeFactor);
  for (const key of ['delay_ms', 't_refractory_ms', 'v_thresh_mV', 'poisson_weight_mV', 'dt_ms'])
    assert.equal(result.slowNeuralConfig[key], result.neuralConfig[key], `${key} is not scaled`);
  assert.equal(result.slowGraph.weightScale, p.slowWeightScale);
  const trainCount = p.trainPerPair * PAIRS.length, evaluationCount = p.evaluationPerPair * PAIRS.length;
  assert.equal(result.trials.length, 3 * (trainCount + evaluationCount));
  const cueOffMs = 2 * p.letterMs + p.gapMs;
  const scores = {};
  for (const [conditionIndex, condition] of CONDITIONS.entries()) {
    const model = result.models[condition];
    assert.equal(model.fitCount, 1);
    assert.equal(model.trainingExamples, trainCount);
    assert.equal(model.readouts.length, p.delaysMs.length + 1);
    const records = result.trials.filter(record => record.condition === condition);
    for (const phase of ['train', 'evaluation']) {
      const phaseRecords = records.filter(record => record.phase === phase);
      assert.equal(phaseRecords.length, phase === 'train' ? trainCount : evaluationCount);
      for (const pair of PAIRS) assert.equal(phaseRecords.filter(record => record.pair === pair).length,
        phase === 'train' ? p.trainPerPair : p.evaluationPerPair, 'Pairs must remain balanced');
    }
    const training = records.filter(record => record.phase === 'train');
    model.readouts.forEach((readout, index) => {
      const expected = index === 0 ? {delayMs: 0, timeMs: cueOffMs, inputActive: true}
        : {delayMs: p.delaysMs[index - 1], timeMs: cueOffMs + p.delaysMs[index - 1], inputActive: false};
      assert.deepEqual({delayMs: readout.delayMs, timeMs: readout.timeMs, inputActive: readout.inputActive}, expected);
      const rows = training.map(record => (index === 0 ? record.cueEnd : record.snapshots[index - 1]).measurements);
      const rebuilt = scalerFromRows(rows);
      rebuilt.mean.forEach((value, i) => close(readout.scaler.mean[i], value, 'Scaler mean must come from training measurements only'));
      rebuilt.scale.forEach((value, i) => close(readout.scaler.scale[i], value, 'Scaler scale must come from training measurements only'));
      assert.equal(readout.scaler.fingerprint, readout.scaler.finalFingerprint);
      for (const position of POSITIONS) {
        const fit = readout.positions[position];
        assert.deepEqual(fit.labels, [...LETTERS]);
        assert.equal(fit.dimensions, 2 * p.poolCount);
        assert.deepEqual(fit.weights, fit.frozenWeights);
        assert.equal(fit.finalFingerprint, fit.fingerprint);
      }
    });
    for (const record of records) {
      assert.equal(record.cueOffsetMs, cueOffMs);
      assert.equal(record.inputsOffFromMs, cueOffMs);
      assert.deepEqual(record.firstLetterMs, [0, p.letterMs]);
      assert.deepEqual(record.secondLetterMs, [p.letterMs + p.gapMs, cueOffMs]);
      assert.equal(record.pair, record.first + record.second);
      assert.equal(record.snapshots.length, p.delaysMs.length);
      assert.equal(record.feedbackApplied, false);
      if (record.phase === 'evaluation') {
        assert.equal(record.decoderVersion, 1);
        assert.equal(record.updateCountBefore, conditionIndex + 1);
        assert.equal(record.updateCountAfter, conditionIndex + 1);
        assert.equal(record.fitApplied, false);
      }
      [record.cueEnd, ...record.snapshots].forEach((snapshot, index) => {
        const readout = model.readouts[index];
        assert.equal(snapshot.delayMs, readout.delayMs);
        assert.equal(snapshot.timeMs, readout.timeMs);
        assert.equal(snapshot.inputActive, readout.inputActive);
        if (!snapshot.inputActive) assert(snapshot.timeMs > record.inputsOffFromMs, 'Every memory snapshot follows input removal');
        assert.equal(snapshot.measurements.length, 2 * p.poolCount);
        for (const value of snapshot.measurements) assert(Number.isFinite(value));
        snapshot.measurements.forEach((value, i) => {
          const size = poolSizes[i % p.poolCount];
          if (!size) assert.equal(value, 0);
          else close(value * size / resolution, Math.round(value * size / resolution), 'Pool means must be sums of fixed-resolution quanta');
        });
        assert.equal(snapshot.silent, snapshot.nonrestVoltageNeurons === 0 && snapshot.nonrestCurrentNeurons === 0);
        if (snapshot.silent) assert(snapshot.measurements.every(value => value === 0));
        assert(snapshot.nonrestVoltageNeurons <= result.metadata.readoutNeuronCount);
        if (snapshot.inputActive) assert.equal(snapshot.spikesSinceCueOff, 0);
        else assert(snapshot.spikesSinceCueOff <= record.postCueSpikes);
        if (record.phase === 'evaluation') {
          const features = standardize(readout.scaler, snapshot.measurements);
          features.forEach((value, i) => close(snapshot.features[i], value, 'Features must be the frozen scaler applied to the measurement'));
          assert.equal(snapshot.scalerFingerprint, readout.scaler.fingerprint);
          for (const position of POSITIONS) {
            const decision = predict(features, readout.positions[position].frozenWeights, LETTERS);
            const stored = snapshot.predictions[position];
            assert.equal(stored.decoderFingerprint, readout.positions[position].fingerprint);
            assert.equal(stored.prediction, decision.prediction, 'Choice must follow only that snapshot, its frozen scaler and frozen decoder');
            stored.scores.forEach((score, i) => close(score, decision.scores[i], 'Reconstructed score must agree'));
            assert.equal(stored.correct, decision.prediction === record[position]);
          }
          assert.equal(snapshot.pairPrediction, snapshot.predictions.first.prediction + snapshot.predictions.second.prediction);
          assert.equal(snapshot.pairCorrect, snapshot.pairPrediction === record.pair);
        } else {
          assert.equal(snapshot.predictions, null, 'Training observations are collected before fitting');
          assert.equal(snapshot.features, null);
        }
      });
    }
    const evaluation = records.filter(record => record.phase === 'evaluation');
    scores[condition] = {cueEnd: summarize(evaluation, record => record.cueEnd),
      delays: p.delaysMs.map(delayMs => ({delayMs, ...summarize(evaluation,
        record => record.snapshots.find(snapshot => snapshot.delayMs === delayMs))}))};
    const exported = result.metrics[condition];
    for (const [actual, reported] of [[scores[condition].cueEnd, exported.cueEnd],
      ...scores[condition].delays.map(score => [score, exported.delays.find(metric => metric.delayMs === score.delayMs)])]) {
      assert.equal(actual.correct, reported.correct);
      assert.equal(actual.total, reported.total);
      close(actual.accuracy, reported.accuracy, 'Pair accuracy must agree with independent scoring');
      close(actual.balancedAccuracy, reported.balancedAccuracy, 'Balanced accuracy must agree with independent scoring');
      close(actual.firstAccuracy, reported.first.accuracy, 'First-letter accuracy must agree');
      close(actual.secondAccuracy, reported.second.accuracy, 'Second-letter accuracy must agree');
      assert.equal(actual.silentTrials, reported.silentTrials);
    }
  }
  const original = result.trials.filter(record => record.condition === 'original');
  const slow = result.trials.filter(record => record.condition === 'slow');
  for (const condition of ['slow', 'reset']) {
    const records = result.trials.filter(record => record.condition === condition);
    records.forEach((record, index) => {
      const matched = original[index];
      for (const key of ['phase', 'phaseTrial', 'pair', 'first', 'second', 'neuralSeed']) assert.equal(record[key], matched[key], 'Conditions must share the same input/noise schedule');
      if (condition === 'reset') {
        assert.equal(record.cueSpikes, slow[index].cueSpikes, 'Reset is applied only after an identical slow-model cue period');
        assert.deepEqual(record.cueEnd.measurements, slow[index].cueEnd.measurements);
        assert.equal(record.postCueSpikes, 0);
        for (const snapshot of record.snapshots) {
          assert(snapshot.silent, 'A reset network with no input remains at rest');
          assert.equal(snapshot.activeNeurons, 0);
        }
      }
    });
  }
  return scores;
}

function report(results, manifest, slowGraph, generatedAt, resolution) {
  const rows = results.flatMap(result => CONDITIONS.flatMap(condition => {
    const score = result.validation.scores[condition];
    return [{name: 'Final cue step (input on)', ...score.cueEnd}, ...score.delays.map(delay => ({name: `${delay.delayMs} ms after OFF`, ...delay}))]
      .map(value => `| ${result.seed} | ${condition} | ${value.name} | ${value.correct}/${value.total} (${percent(value.accuracy)}) | ${percent(value.firstAccuracy)} | ${percent(value.secondAccuracy)} | ${value.meanNonrestVoltageNeurons.toFixed(0)} | ${value.meanNonrestCurrentNeurons.toFixed(0)} | ${value.meanSpikesSinceCueOff.toFixed(1)} | ${value.silentTrials}/${value.total} |`);
  }));
  const slow = results[0].slowNeuralConfig;
  return `# Two-letter state-decoding benchmark: developmental smoke check

Generated ${generatedAt}. ${results.length} seeded run${results.length === 1 ? '' : 's'} of one FlyWire v783 anatomical graph (${manifest.n.toLocaleString('en-US')} neurons, ${manifest.edgeCount.toLocaleString('en-US')} connection slots). Seeds are technical simulations, not independent biological animals.

Each condition receives the same balanced ${DEFAULTS.trainPerPair * PAIRS.length} training and ${DEFAULTS.evaluationPerPair * PAIRS.length} evaluation trials over the four ordered pairs tt, to, ot and oo. A ${DEFAULTS.letterMs} ms letter cue, a ${DEFAULTS.gapMs} ms gap and a second ${DEFAULTS.letterMs} ms letter cue end at ${2 * DEFAULTS.letterMs + DEFAULTS.gapMs} ms; all input is then switched OFF. One quantized snapshot of membrane voltage and synaptic current in ${DEFAULTS.poolCount} pools of nonstimulated neurons is read ${DEFAULTS.delaysMs.join(', ')} ms later, at a measurement resolution of ${resolution} mV per neuron. Two frozen ridge readouts per snapshot decode the first and second letters from the same ${2 * DEFAULTS.poolCount} standardized features, with no external history, cue labels, clock, previous snapshot or correction. Exact-pair chance is 25%; remembering only the final letter gives 50% expected exact-pair accuracy and 50% first-letter accuracy.

The slower-dynamics condition multiplies the membrane and synaptic time constants by ${DEFAULTS.slowTimeFactor} (${slow.tau_mem_ms} ms and ${slow.tau_syn_ms} ms) and every signed synaptic weight by ${DEFAULTS.slowWeightScale}, so that each synaptic event keeps its integrated voltage response while its peak falls. Conduction delay, refractory period and thresholds are unchanged. This is an imposed engineering hypothesis, not measured fruit-fly physiology. The reset control uses the slower model and identical cues, then replaces all neural state at cue offset; its post-cue snapshots are exactly rest.

A separate diagnostic reads the same kind of snapshot at the final cue step, while input is still ON and before any reset. It tests encoding and is never combined with post-cue features. The delays are observations from the same trials and are correlated, not independent replications. Every condition/snapshot/position has its own scaler and ridge decoder fitted only on training observations, then frozen. No accuracy threshold or preferred outcome is enforced.

| Seed | Condition | Observation | Exact pair | First letter | Second letter | Mean neurons off rest (voltage) | Mean neurons off rest (current) | Mean spikes since OFF | Fully silent trials |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
${rows.join('\n')}

Neurons "off rest" are nonstimulated neurons whose quantized voltage deviation or synaptic current is nonzero at the snapshot. Spikes since OFF count all neurons between input removal and the snapshot. Reading internal voltages and currents is more permissive than the spike-only Memory lab; a decodable subthreshold trace is not a demonstration of biologically accessible memory, learned motor control, language or memory for a text. Success under slower dynamics concerns that hypothetical model.

Validation independently rebuilt every scaler from training measurements, every feature from its snapshot and frozen scaler, and every evaluation choice from frozen weights, recomputed all pair, first-letter, second-letter and class-balanced scores, checked input-off timestamps, matched pair/noise schedules across conditions, confirmed identical slow/reset cue-period spikes and final-cue snapshots, and confirmed that reset post-cue snapshots are exactly rest. The ${3 * (DEFAULTS.delaysMs.length + 1) * 2} fitted readouts stayed frozen during evaluation. Original compressed/raw assets passed manifest SHA-256 checks; hashes of the anatomical graph and of the rescaled comparison graph (scaled weights SHA-256 ${slowGraph.scaledWeightsSha256}) were unchanged after each complete run.

Compute times: ${results.map(result => `seed ${result.seed}: ${result.validation.wallSeconds.toFixed(2)} s`).join('; ')}.

${results.map(result => `- [Seed ${result.seed}: full records, models, provenance, and integrity checks (gzip JSON)](seed-${result.seed}.json.gz)`).join('\n')}

\`\`\`sh
node scripts/benchmark_sequence.cjs --seeds ${results.map(result => result.seed).join(',')} --out ${path.relative(repository, path.dirname(path.join(repository, 'results/sequence_benchmark/report.md')))}${resolution === DEFAULTS.stateResolutionMv ? '' : ` --resolution ${resolution}`}
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
  for (let i = 0; i < graph.weights.length; i += 997) assert.equal(slow.graph.weights[i], Math.fround(graph.weights[i] * DEFAULTS.slowWeightScale));
  assert.equal(slow.graph.indices, graph.indices); assert.equal(slow.graph.indptr, graph.indptr);
  const prepared = {graph: slow.graph, metadata: {...slow.metadata, scaledWeightsSha256: slowBefore.weights}};
  assert.deepEqual(graphHashes(graph), beforeHashes, 'Preparing the comparison graph must not alter the anatomical graph');
  const sourceHashes = Object.fromEntries(['live-model.js', 'recall-model.js', 'sequence-model.js']
    .map(file => [file, sha256(fs.readFileSync(path.join(repository, 'site', file)))]));
  fs.mkdirSync(args.output, {recursive: true});
  const results = [];
  for (const file of fs.readdirSync(args.output).filter(name => /^seed-\d+\.json\.gz$/.test(name))) {
    const saved = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(args.output, file))).toString('utf8'));
    if (args.seeds.includes(saved.seed)) continue;
    if (saved.complete && JSON.stringify(saved.validation?.modelSourceSha256) === JSON.stringify(sourceHashes) &&
        JSON.stringify(saved.validation?.graphHashesBefore) === JSON.stringify(beforeHashes) &&
        JSON.stringify(saved.validation?.graphHashesAfter) === JSON.stringify(beforeHashes) &&
        JSON.stringify(saved.validation?.slowHashesBefore) === JSON.stringify(slowBefore) &&
        JSON.stringify(saved.validation?.slowHashesAfter) === JSON.stringify(slowBefore) &&
        saved.config.stateResolutionMv === args.resolution &&
        Object.entries(DEFAULTS).every(([key, value]) => key === 'stateResolutionMv' || JSON.stringify(saved.config[key]) === JSON.stringify(value))) {
      assertSequenceIsolation(saved, args.resolution); results.push(saved);
    } else console.log(`Skipping saved ${file} from a different model, graph, or protocol revision.`);
  }
  console.log(`Scaled weights SHA-256: ${slowBefore.weights}. Resolution ${args.resolution} mV. Seeds: ${args.seeds.join(', ')}.`);
  for (const seed of args.seeds) {
    const started = performance.now();
    const experiment = new SequenceExperiment(graph, manifest, seed, {stateResolutionMv: args.resolution}, prepared);
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
    const scores = assertSequenceIsolation(result, args.resolution);
    const afterHashes = graphHashes(graph), slowAfter = graphHashes(slow.graph);
    assert.deepEqual(afterHashes, beforeHashes, 'The original graph must remain unchanged');
    assert.deepEqual(slowAfter, slowBefore, 'The rescaled comparison graph must remain unchanged');
    result.validation = {kind: 'developmental smoke check; paired simulations of one anatomy under original and imposed slower dynamics',
      generatedAt: new Date().toISOString(), nodeVersion: process.version, wallSeconds, scores,
      cueOffVerified: true, evaluationWeightsFrozen: true, independentScalerFeatureAndScoreReconstruction: true,
      pairedInputsVerified: true, resetSnapshotsAtRest: true, connectomeUnchanged: true, slowGraphUnchanged: true,
      graphHashesBefore: beforeHashes, graphHashesAfter: afterHashes,
      slowHashesBefore: slowBefore, slowHashesAfter: slowAfter, modelSourceSha256: sourceHashes};
    fs.writeFileSync(path.join(args.output, `seed-${seed}.json.gz`), zlib.gzipSync(`${JSON.stringify(result, null, 2)}\n`));
    results.push(result); results.sort((left, right) => left.seed - right.seed);
    fs.writeFileSync(path.join(args.output, 'report.md'), report(results, manifest, prepared.metadata, startedAt, args.resolution));
    for (const condition of CONDITIONS) console.log(`Seed ${seed} ${condition}: cue end ${percent(scores[condition].cueEnd.accuracy)}; OFF delays ${scores[condition].delays.map(score => `${score.delayMs}ms=${percent(score.accuracy)} (first ${percent(score.firstAccuracy)}, second ${percent(score.secondAccuracy)})`).join(', ')}.`);
    console.log(`Seed ${seed}: ${wallSeconds.toFixed(2)} s; all isolation and integrity checks passed.`);
  }
  console.log(`Completed ${args.seeds.length} requested runs; report covers ${results.length} saved seeds: ${path.join(args.output, 'report.md')}`);
}

if (require.main === module) main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1; });
module.exports = {predict, standardize, scalerFromRows, summarize, assertSequenceIsolation, report};
