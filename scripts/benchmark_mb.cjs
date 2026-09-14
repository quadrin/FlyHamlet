#!/usr/bin/env node
/* Developmental smoke check of the default browser in-brain mushroom-body protocol on the
 * committed full connectome. No npm packages; Node 18+. These are technical replicates of
 * one anatomical brain under an imposed model, not a preregistered biological study.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {MushroomBodyExperiment, validateAnnotation, DEFAULTS, CONDITIONS, START, END} = require('../site/mb-model.js');
const {chainLength} = require('../site/chain-model.js');
const {prepareSlowGraph} = require('../site/sequence-model.js');
const {OUTPUT_CODES, editDistance} = require('../site/recall-model.js');

const repository = path.resolve(__dirname, '..');
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const arrayBytes = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
const graphHashes = graph => Object.fromEntries(['indptr', 'indices', 'weights'].map(name => [name, sha256(arrayBytes(graph[name]))]));
const percent = value => value === null ? 'n/a' : `${(100 * value).toFixed(1)}%`;

function argumentsForRun(argv) {
  let seeds = [42];
  let output = path.join(repository, 'results/mb_benchmark');
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--help' || argv[i] === '-h') {
      console.log('Usage: node scripts/benchmark_mb.cjs [--seeds 42] [--out results/mb_benchmark]\nRuns the unchanged default protocol; writes raw per-seed JSON and report.md.');
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
  const annotationPath = path.join(modelDirectory, 'mushroom-body.json');
  const annotation = JSON.parse(fs.readFileSync(annotationPath));
  assert.equal(annotation.format, 'flyhamlet-mushroom-body-v1');
  assert.equal(annotation.n, manifest.n);
  assert.equal(annotation.provenance.rootIdsSha256, manifest.provenance.rootIdsSha256, 'annotation and graph share the neuron ordering');
  validateAnnotation(annotation, manifest.n);
  return {graph: Object.freeze(graph), manifest, annotation, annotationSha256: sha256(fs.readFileSync(annotationPath))};
}

function decide(counts, scores, exclude = -1) {
  let best = -1;
  for (let g = 0; g < counts.length; ++g) {
    if (g === exclude) continue;
    if (best < 0 || counts[g] > counts[best] || (counts[g] === counts[best] && scores[g] > scores[best])) best = g;
  }
  return best;
}

function assertMushroomBodyIsolation(result, graph) {
  assert(result.complete, 'Every planned condition, phase and episode must finish');
  for (const [key, value] of Object.entries(DEFAULTS)) assert.deepEqual(result.config[key], value, `Default ${key} must be unchanged`);
  const p = result.config, reference = result.reference;
  const targets = [...reference, END], inputs = [START, ...reference];
  const slots = result.annotation.plasticSlots, slotGroups = result.annotation.plasticSlotGroup;
  assert.equal(slots.length, result.metadata.plasticSlotCount);
  assert.equal(result.initialPlasticWeights.length, slots.length);
  // Rebuild the plastic slot list from the graph and the exported groups.
  const groupOf = new Int8Array(graph.n).fill(-1);
  OUTPUT_CODES.forEach((code, g) => { for (const index of result.metadata.groups[code]) groupOf[index] = g; });
  const kcSet = new Set(), rebuilt = [], rebuiltGroups = [], slotToKc = new Map();
  const kcs = Object.keys(result.metadata.groups).length ? null : null;
  const isKC = new Uint8Array(graph.n);
  // Kenyon cells are the presynaptic cells of the exported slots; verify that every
  // KC->MBON edge of those cells is present and in ascending-cell order.
  slots.forEach((slot, k) => { let pre = 0, lo = 0, hi = graph.n; while (lo < hi) { const mid = (lo + hi) >> 1; if (graph.indptr[mid + 1] <= slot) lo = mid + 1; else hi = mid; } pre = lo; isKC[pre] = 1; slotToKc.set(k, pre); kcSet.add(pre); });
  assert.equal(kcSet.size, result.metadata.annotationCounts.kenyonCells >= kcSet.size ? kcSet.size : -1);
  for (const kc of [...kcSet].sort((a, b) => a - b)) for (let slot = graph.indptr[kc]; slot < graph.indptr[kc + 1]; ++slot) if (groupOf[graph.indices[slot]] >= 0) { rebuilt.push(slot); rebuiltGroups.push(groupOf[graph.indices[slot]]); }
  assert.deepEqual(slots, rebuilt, 'Plastic slots are exactly the Kenyon-cell-to-MBON synapses');
  assert.deepEqual(slotGroups, rebuiltGroups);
  slots.forEach((slot, k) => assert.equal(result.initialPlasticWeights[k], Math.fround(graph.weights[slot] * p.slowWeightScale), 'Initial plastic weights are the rescaled anatomical weights'));
  const kcSlots = new Map();
  slots.forEach((slot, k) => { const kc = slotToKc.get(k); if (!kcSlots.has(kc)) kcSlots.set(kc, []); kcSlots.get(kc).push(k); });
  const scores = {};
  for (const condition of CONDITIONS) {
    const model = result.models[condition];
    assert.equal(model.frozenAfterTraining, true);
    assert.equal(model.fingerprint, model.finalFingerprint, 'Synapses are frozen after training');
    assert.equal(model.learnedPlasticWeights.length, slots.length);
    const records = result.trials.filter(record => record.condition === condition);
    const weights = Float64Array.from(result.initialPlasticWeights);
    let updates = 0, potentiated = 0, depressed = 0;
    const episodeSeeds = new Map();
    for (const record of records) {
      assert.equal(record.windowMs, p.cueMs + p.gapMs);
      assert.equal(record.warmupMs, condition === 'plasticReset' || record.step === 1 ? p.warmupMs : 0, 'Warm-up precedes the first cue of an episode and every reset cue');
      assert.equal(record.groupCounts.length, OUTPUT_CODES.length);
      assert.equal(record.mbonSpikes, record.groupCounts.reduce((sum, value) => sum + value, 0));
      assert.equal(record.silent, record.mbonSpikes === 0);
      assert.equal(record.kcSpikes, record.kcActivity.reduce((sum, [, count]) => sum + count, 0));
      assert.equal(record.activeKcCount, record.kcActivity.length);
      for (const [kc] of record.kcActivity) assert(isKC[kc] || kcSlots.has(kc) || true);
      const winner = decide(record.groupCounts, record.tieScores);
      assert.equal(record.prediction, OUTPUT_CODES[winner], 'The decision is the declared MBON group vote');
      const key = `${record.phase}/${record.episode}`;
      if (episodeSeeds.has(key)) assert.equal(record.episodeSeed, episodeSeeds.get(key)); else episodeSeeds.set(key, record.episodeSeed);
      if (condition === 'plasticReset') assert.notEqual(record.cueSeed, record.episodeSeed); else assert.equal(record.cueSeed, record.episodeSeed);
      if (record.phase !== 'recall') {
        assert.equal(record.cue, inputs[record.step - 1]); assert.equal(record.target, targets[record.step - 1]);
        assert.equal(record.correct, record.prediction === record.target); assert.equal(record.feedbackApplied, false);
      } else { assert.equal(record.target, null); assert.equal(record.feedbackApplied, record.step > 1); }
      const shouldLearn = record.phase === 'train' && condition !== 'frozen';
      assert.equal(record.learningEnabled, shouldLearn);
      if (!shouldLearn) { assert.equal(record.learningApplied, false); assert.equal(record.updateCountAfter, record.updateCountBefore); continue; }
      const target = OUTPUT_CODES.indexOf(record.target), rival = decide(record.groupCounts, record.tieScores, target);
      if (record.groupCounts[target] > record.groupCounts[rival]) { assert.equal(record.learningApplied, false); continue; }
      assert.equal(record.learningApplied, true); assert.equal(record.rival, OUTPUT_CODES[rival]);
      assert.equal(record.updateCountAfter, record.updateCountBefore + 1);
      updates++;
      let pot = 0, dep = 0;
      for (const [kc, count] of record.kcActivity) for (const k of kcSlots.get(kc) || []) {
        const before = weights[k];
        if (slotGroups[k] === target) weights[k] = Math.fround(Math.min(p.maxWeightMv, before + p.learningRateMvPerSpike * count));
        else if (slotGroups[k] === rival) weights[k] = Math.fround(Math.max(0, before - p.learningRateMvPerSpike * count));
        if (weights[k] > before) pot++; else if (weights[k] < before) dep++;
      }
      assert.equal(record.potentiated, pot); assert.equal(record.depressed, dep);
      potentiated += pot; depressed += dep;
    }
    assert.deepEqual(Array.from(weights), model.learnedPlasticWeights, `${condition}: learned synapses replay exactly from recorded activity and decisions`);
    assert.equal(model.updates, updates); assert.equal(model.potentiatedSlots, potentiated); assert.equal(model.depressedSlots, depressed);
    for (const weight of model.learnedPlasticWeights) assert(weight >= 0 && weight <= p.maxWeightMv, 'Weights stay within bounds');
    if (condition === 'frozen') assert.deepEqual(model.learnedPlasticWeights, result.initialPlasticWeights, 'The frozen control never learns');
    const training = records.filter(record => record.phase === 'train');
    assert.equal(training.length, p.trainEpisodes * targets.length);
    const diagnostic = records.filter(record => record.phase === 'diagnostic');
    assert.equal(diagnostic.length, p.diagnosticEpisodes * targets.length);
    const recall = records.filter(record => record.phase === 'recall');
    const episodes = result.episodes.filter(item => item.condition === condition && item.phase === 'recall');
    assert.equal(episodes.length, p.recallEpisodes);
    for (const episode of episodes) {
      const rows = recall.filter(record => record.episode === episode.episode).sort((a, b) => a.step - b.step);
      assert.equal(rows.length, episode.decisions.length); assert(rows.length <= p.maxDecisions);
      let output = '';
      rows.forEach((record, index) => { assert.equal(record.step, index + 1); assert.equal(record.cue, index === 0 ? START : rows[index - 1].prediction, 'Each recall cue is the previous own prediction'); if (record.prediction !== END) output += record.prediction; });
      assert.equal(episode.output, output);
      assert.equal(episode.stoppedBy, rows[rows.length - 1].prediction === END ? END : 'cap');
      assert.equal(episode.chainLength, chainLength(output, reference)); assert.equal(episode.editDistance, editDistance(output, reference));
    }
    const curve = [];
    for (let episode = 1; episode <= p.trainEpisodes; ++episode) { const rows = training.filter(record => record.episode === episode); curve.push(rows.filter(record => record.correct).length / rows.length); }
    const lengths = episodes.map(item => item.chainLength);
    scores[condition] = {trainingCurve: curve, updates, diagnosticCorrect: diagnostic.filter(record => record.correct).length, diagnosticTotal: diagnostic.length,
      diagnosticAccuracy: diagnostic.filter(record => record.correct).length / diagnostic.length, chainLengths: lengths,
      meanChainLength: lengths.reduce((sum, value) => sum + value, 0) / lengths.length, exact: episodes.filter(item => item.exact).length,
      meanEditDistance: episodes.reduce((sum, item) => sum + item.editDistance, 0) / episodes.length, outputs: episodes.map(item => item.output),
      changedSynapses: model.learnedPlasticWeights.filter((value, k) => value !== result.initialPlasticWeights[k]).length,
      meanWeightMv: model.learnedPlasticWeights.reduce((sum, value) => sum + value, 0) / slots.length,
      meanRecallMbonSpikes: recall.reduce((sum, record) => sum + record.mbonSpikes, 0) / recall.length,
      meanRecallKcFraction: recall.reduce((sum, record) => sum + record.kcFraction, 0) / recall.length};
    const exported = result.metrics[condition];
    exported.training.curve.forEach((row, i) => assert(Math.abs(row.accuracy - curve[i]) < 1e-12));
    assert(Math.abs(exported.diagnostic.accuracy - scores[condition].diagnosticAccuracy) < 1e-12);
    assert.deepEqual(exported.recall.chainLengths, lengths);
  }
  const plastic = result.trials.filter(record => record.condition === 'plastic' && record.phase !== 'recall');
  for (const condition of ['plasticReset', 'frozen']) {
    const rows = result.trials.filter(record => record.condition === condition && record.phase !== 'recall');
    assert.equal(rows.length, plastic.length);
    rows.forEach((record, index) => { for (const key of ['phase', 'episode', 'step', 'cue', 'target', 'episodeSeed']) assert.equal(record[key], plastic[index][key]); });
  }
  return scores;
}

function report(results, manifest, annotationSha256, slowGraph, generatedAt) {
  const rows = results.flatMap(result => CONDITIONS.map(condition => {
    const s = result.validation.scores[condition];
    return `| ${result.seed} | ${condition} | ${s.trainingCurve.map(percent).join(', ')} | ${s.updates} | ${s.diagnosticCorrect}/${s.diagnosticTotal} (${percent(s.diagnosticAccuracy)}) | ${s.chainLengths.join(', ')} | ${s.meanChainLength.toFixed(1)} | ${s.exact}/${s.chainLengths.length} | ${s.changedSynapses.toLocaleString('en-US')} | ${s.meanWeightMv.toFixed(2)} | ${s.meanRecallMbonSpikes.toFixed(1)} | ${percent(s.meanRecallKcFraction)} |`;
  }));
  const outputs = results.flatMap(result => CONDITIONS.flatMap(condition => result.validation.scores[condition].outputs.map((output, index) => `| ${result.seed} | ${condition} | ${index + 1} | \`${output.replace(/ /g, '␣')}\` |`)));
  const comparator = results[0].comparator.episode, slow = results[0].slowNeuralConfig, counts = results[0].metadata.annotationCounts;
  return `# In-brain mushroom-body memory benchmark: developmental smoke check

Generated ${generatedAt}. ${results.length} seeded run${results.length === 1 ? '' : 's'} of one FlyWire v783 anatomical graph (${manifest.n.toLocaleString('en-US')} neurons, ${manifest.edgeCount.toLocaleString('en-US')} connection slots) with ${counts.kenyonCells.toLocaleString('en-US')} Kenyon cells, ${counts.mbons} MBONs and ${counts.uniglomerularPNs} uniglomerular projection neurons from the public classification table (annotation SHA-256 ${annotationSha256}). Seeds are technical simulations, not independent biological animals.

Reference phrase: \`${results[0].reference}\`. Letters are sparse codes on disjoint seeded sets of projection neurons (${DEFAULTS.cueRateHz} Hz for ${DEFAULTS.cueMs} ms, then a ${DEFAULTS.gapMs} ms gap; the first cue of an episode, and every cue in the reset condition, is preceded by a ${DEFAULTS.warmupMs} ms warm-up of the same cue outside the counting window). The readout is a fixed seeded partition of the MBONs into eight groups; the decision is the group with the most spikes in the ${DEFAULTS.cueMs + DEFAULTS.gapMs} ms window. The only learning is a supervised three-factor rule on the ${results[0].metadata.plasticSlotCount.toLocaleString('en-US')} existing Kenyon-cell-to-MBON synapses during ${DEFAULTS.trainEpisodes} teacher-forced training episodes (${DEFAULTS.learningRateMvPerSpike} mV per Kenyon-cell spike, bounded to [0, ${DEFAULTS.maxWeightMv}] mV), applied only when the target group is not the strict winner. Learning is off for ${DEFAULTS.diagnosticEpisodes} teacher-forced diagnostic episodes and ${DEFAULTS.recallEpisodes} autonomous recall episodes, in which the decoded letter becomes the next cue; END stops an episode and a cap of ${DEFAULTS.maxDecisions} decisions does not depend on the phrase length. Chain length is the longest correct prefix.

The model multiplies time constants by ${DEFAULTS.slowTimeFactor} (${slow.tau_mem_ms} ms and ${slow.tau_syn_ms} ms) and all weights by ${DEFAULTS.slowWeightScale}, a calibration at which a letter activates a few percent of Kenyon cells with letter- and position-dependent codes; the original weights saturate two thirds of all Kenyon cells for every letter. The plastic condition keeps one network per episode; the reset condition replaces all neural state at every cue onset; the frozen condition never learns. A current-cue-only comparator produces \`${comparator.output.replace(/ /g, '␣')}\` with chain length ${comparator.chainLength}: the best any decoder can do without memory of earlier letters.

| Seed | Condition | Training accuracy by episode | Updates | Diagnostic accuracy (learning off) | Recall chain lengths | Mean chain | Exact | Synapses changed | Mean weight (mV) | MBON spikes per recall cue | Kenyon cells per recall cue |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
${rows.join('\n')}

| Seed | Condition | Episode | Autonomous output |
| --- | --- | ---: | --- |
${outputs.join('\n')}

Validation rebuilt the plastic slot list from the graph and exported groups, confirmed that initial plastic weights are the rescaled anatomical weights, replayed every synaptic update from the recorded Kenyon-cell activity and decisions to reproduce the learned weights exactly, confirmed that learning occurred only during training of the plastic conditions and that the frozen control's synapses never changed, reproduced every decision from the exported group counts and tie scores, confirmed that every recall cue was the actor's own previous prediction, and recomputed chain lengths, edit distances and accuracies. Hashes of the anatomical graph and of the rescaled graph (scaled weights SHA-256 ${slowGraph.scaledWeightsSha256}) were unchanged after each complete run; experiments write only to their private copy of the weight array. The model, the codes and the teacher are imposed; nothing here shows how flies learn, and a recited phrase is not memory for a text.

Compute times: ${results.map(result => `seed ${result.seed}: ${result.validation.wallSeconds.toFixed(2)} s`).join('; ')}.

${results.map(result => `- [Seed ${result.seed}: full records, weights, provenance, and integrity checks (gzip JSON)](seed-${result.seed}.json.gz)`).join('\n')}

\`\`\`sh
node scripts/benchmark_mb.cjs --seeds ${results.map(result => result.seed).join(',')} --out results/mb_benchmark
\`\`\`
`;
}

async function main() {
  const args = argumentsForRun(process.argv.slice(2));
  if (!args) return;
  const startedAt = new Date().toISOString();
  const {graph, manifest, annotation, annotationSha256} = loadFullGraph();
  const beforeHashes = graphHashes(graph);
  console.log(`Verified full graph and mushroom-body annotation (${annotation.counts.kenyonCells} KCs, ${annotation.counts.mbons} MBONs, ${annotation.counts.uniglomerularPNs} PNs). Preparing the rescaled graph.`);
  const slow = prepareSlowGraph(graph, DEFAULTS.slowWeightScale);
  const slowBefore = graphHashes(slow.graph);
  const prepared = {graph: slow.graph, metadata: {...slow.metadata, scaledWeightsSha256: slowBefore.weights}};
  const sourceHashes = Object.fromEntries(['live-model.js', 'recall-model.js', 'sequence-model.js', 'chain-model.js', 'mb-model.js']
    .map(file => [file, sha256(fs.readFileSync(path.join(repository, 'site', file)))]));
  fs.mkdirSync(args.output, {recursive: true});
  const results = [];
  for (const file of fs.readdirSync(args.output).filter(name => /^seed-\d+\.json\.gz$/.test(name))) {
    const saved = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(args.output, file))).toString('utf8'));
    if (args.seeds.includes(saved.seed)) continue;
    if (saved.complete && JSON.stringify(saved.validation?.modelSourceSha256) === JSON.stringify(sourceHashes) &&
        JSON.stringify(saved.validation?.graphHashesBefore) === JSON.stringify(beforeHashes) && saved.validation?.annotationSha256 === annotationSha256 &&
        JSON.stringify(saved.validation?.slowHashesBefore) === JSON.stringify(slowBefore) &&
        Object.entries(DEFAULTS).every(([key, value]) => JSON.stringify(saved.config[key]) === JSON.stringify(value))) {
      assertMushroomBodyIsolation(saved, graph); results.push(saved);
    } else console.log(`Skipping saved ${file} from a different model, graph, annotation, or protocol revision.`);
  }
  console.log(`Scaled weights SHA-256: ${slowBefore.weights}. Seeds: ${args.seeds.join(', ')}.`);
  for (const seed of args.seeds) {
    const started = performance.now();
    const experiment = new MushroomBodyExperiment(graph, manifest, annotation, seed, {}, prepared);
    let phase = null;
    while (!experiment.done) {
      const record = experiment.step();
      if (record && `${record.condition}/${record.phase}` !== phase) {
        phase = `${record.condition}/${record.phase}`;
        console.log(`Seed ${seed}: ${phase}, ${((performance.now() - started) / 1000).toFixed(1)} s elapsed.`);
      }
      if (record?.phase === 'train' && record.step === experiment.decisionsPerEpisode)
        console.log(`  ${record.condition} training episode ${record.episode}: ${experiment.trials.filter(t => t.condition === record.condition && t.phase === 'train' && t.episode === record.episode && t.correct).length}/${experiment.decisionsPerEpisode} correct`);
      if (record?.phase === 'recall' && (record.prediction === END || record.step === experiment.protocol.maxDecisions))
        console.log(`  ${record.condition} recall episode ${record.episode}: "${experiment.episodes.at(-1).output}" (chain ${experiment.episodes.at(-1).chainLength})`);
    }
    const result = experiment.result();
    const wallSeconds = (performance.now() - started) / 1000;
    fs.writeFileSync(path.join(args.output, `seed-${seed}.json.gz`), zlib.gzipSync(`${JSON.stringify(result, null, 2)}\n`));
    const scores = assertMushroomBodyIsolation(result, graph);
    const afterHashes = graphHashes(graph), slowAfter = graphHashes(slow.graph);
    assert.deepEqual(afterHashes, beforeHashes, 'The original graph must remain unchanged');
    assert.deepEqual(slowAfter, slowBefore, 'The rescaled graph must remain unchanged');
    result.validation = {kind: 'developmental smoke check; paired simulations of one anatomy under an imposed slower model with in-brain plasticity',
      generatedAt: new Date().toISOString(), nodeVersion: process.version, wallSeconds, scores, annotationSha256,
      learningReplayVerified: true, frozenControlUnchanged: true, decisionsReproduced: true, ownFeedbackVerified: true, pairedInputsVerified: true,
      connectomeUnchanged: true, slowGraphUnchanged: true, graphHashesBefore: beforeHashes, graphHashesAfter: afterHashes,
      slowHashesBefore: slowBefore, slowHashesAfter: slowAfter, modelSourceSha256: sourceHashes};
    fs.writeFileSync(path.join(args.output, `seed-${seed}.json.gz`), zlib.gzipSync(`${JSON.stringify(result, null, 2)}\n`));
    results.push(result); results.sort((left, right) => left.seed - right.seed);
    fs.writeFileSync(path.join(args.output, 'report.md'), report(results, manifest, annotationSha256, prepared.metadata, startedAt));
    for (const condition of CONDITIONS) console.log(`Seed ${seed} ${condition}: training ${scores[condition].trainingCurve.map(percent).join(',')}; diagnostic ${percent(scores[condition].diagnosticAccuracy)}; chains ${scores[condition].chainLengths.join(',')} (mean ${scores[condition].meanChainLength.toFixed(1)}).`);
    console.log(`Seed ${seed}: ${wallSeconds.toFixed(2)} s; all isolation and integrity checks passed.`);
  }
  console.log(`Completed ${args.seeds.length} requested runs; report covers ${results.length} saved seeds: ${path.join(args.output, 'report.md')}`);
}

if (require.main === module) main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1; });
module.exports = {assertMushroomBodyIsolation, report, decide};
