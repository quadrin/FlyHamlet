/* Run: node --test tests/test_memory_rewire.cjs. No full assets or dependencies.
 * These checks concern the graph null, not biological memory performance. */
const test = require('node:test');
const assert = require('node:assert/strict');
const {TargetShuffleRewire, prepareRewire, DEFAULT_SEED} = require('../site/memory-rewire.js');

function fixture() {
  return {n: 5, indptr: new Uint32Array([0, 4, 6, 9, 11, 14]),
    indices: new Uint32Array([0, 1, 2, 4, 0, 3, 1, 2, 4, 0, 2, 0, 2, 3]),
    weights: new Float32Array([.25, .5, 2, 1, -3, -1, 7, 5, 2, -.5, -8, 1, 2, 4])};
}
function finish(graph, seed = DEFAULT_SEED, chunk = 3) {
  const builder = new TargetShuffleRewire(graph, {seed});
  while (!builder.done) builder.advance(chunk);
  return {builder, ...builder.result()};
}
function incoming(graph) {
  const counts = new Array(graph.n).fill(0);
  for (const target of graph.indices) counts[target]++;
  return counts;
}
function countEdges(graph) {
  let self = 0, parallel = 0, rows = 0;
  for (let source = 0; source < graph.n; source++) {
    const targets = [...graph.indices.slice(graph.indptr[source], graph.indptr[source + 1])];
    const duplicateCount = targets.length - new Set(targets).size;
    self += targets.filter(target => target === source).length;
    parallel += duplicateCount;
    rows += Number(duplicateCount > 0);
  }
  return {self, parallel, rows};
}

test('rewiring preserves every stub degree and source weight list without mutating the graph', () => {
  const original = fixture();
  const before = Object.fromEntries(['indptr', 'indices', 'weights'].map(name => [name, [...original[name]]]));
  const {graph, metadata} = finish(original);
  assert.deepEqual(incoming(graph), incoming(original));
  assert.equal(graph.indptr, original.indptr, 'identical CSR rows preserve source stub degrees');
  assert.equal(graph.weights, original.weights, 'each source slot retains its exact signed weight');
  assert.notEqual(graph.indices, original.indices, 'rewiring owns a separate target buffer');
  assert.notDeepEqual([...graph.indices], [...original.indices]);
  for (const name of Object.keys(before)) assert.deepEqual([...original[name]], before[name]);
  assert.equal(metadata.degreePreservationVerified, true);
  assert.equal(metadata.sourceWeightsPreserved, true);
  assert.equal(metadata.incomingWeightedStrengthPreserved, false);
  assert.equal(metadata.distinctNeighborDegreesPreserved, false);
});

test('parallel edges and self loops are counted explicitly, rather than silently collapsed', () => {
  const original = fixture();
  let observedParallel = false, observedSelfLoop = false;
  for (let seed = 0; seed < 20; seed++) {
    const {graph, metadata} = finish(original, seed);
    const counts = countEdges(graph);
    assert.equal(metadata.parallelEdgeSlots, counts.parallel);
    assert.equal(metadata.parallelSourceRows, counts.rows);
    assert.equal(metadata.selfLoopSlots, counts.self);
    assert.equal(metadata.originalSelfLoopSlots, countEdges(original).self);
    assert.equal(metadata.distinctDirectedPairs, graph.indices.length - counts.parallel);
    assert.equal(graph.indices.length, original.indices.length);
    observedParallel ||= counts.parallel > 0;
    observedSelfLoop ||= counts.self > 0;
  }
  assert(observedParallel, 'this null allows non-simple graphs, rather than just relabeling neurons');
  assert(observedSelfLoop);
});

test('fixed seed is reproducible regardless of chunking; a changed graph seed changes the control', () => {
  const original = fixture();
  const first = finish(original, 67, 1);
  const second = finish(original, 67, 1000);
  const other = finish(original, 68, 2);
  assert.deepEqual(first.graph, second.graph);
  assert.deepEqual(first.metadata, second.metadata);
  assert.notDeepEqual(first.graph.indices, other.graph.indices);
  assert.equal(first.builder.progress.fraction, 1);
  assert.equal(first.builder.progress.loaded, first.builder.progress.total);
  assert.equal(first.builder.advance(10).fraction, 1);
});

test('each chunk is bounded even for a graph composed entirely of empty rows', () => {
  const empty = {n: 100, indptr: new Uint32Array(101), indices: new Uint32Array(0), weights: new Float32Array(0)};
  const builder = new TargetShuffleRewire(empty);
  let last = 0, chunks = 0;
  while (!builder.done) {
    const progress = builder.advance(3);
    assert(progress.loaded - last <= 3);
    assert(progress.loaded >= last);
    last = progress.loaded; chunks++;
  }
  assert(chunks >= 67);
  assert.equal(builder.progress.total, 200);
  assert.equal(builder.result().metadata.edgeSlots, 0);
  assert.equal(builder.result().metadata.distinctDirectedPairs, 0);
});

test('async preparation yields and exactly matches the synchronous control', async () => {
  const graph = fixture(), progress = [];
  let yields = 0;
  const result = await prepareRewire(graph, {seed: 99, batchOperations: 4,
    progress: item => progress.push({...item}), yieldControl: async () => { yields++; }});
  const expected = finish(graph, 99);
  assert.deepEqual(result.graph, expected.graph);
  assert.deepEqual(result.metadata, expected.metadata);
  assert(yields > 2);
  assert.equal(progress[0].loaded, 0);
  assert.equal(progress.at(-1).phase, 'complete');
  assert.equal(progress.at(-1).fraction, 1);
});

test('cancellation prevents publication of a partially rewired graph and leaves input intact', async () => {
  const graph = fixture(), before = [...graph.indices];
  let cancelled = false, yields = 0;
  await assert.rejects(prepareRewire(graph, {batchOperations: 1,
    shouldCancel: () => cancelled,
    yieldControl: async () => { yields++; cancelled = true; }}), error => error.name === 'AbortError');
  assert.equal(yields, 1);
  assert.deepEqual([...graph.indices], before);
  await assert.rejects(prepareRewire(graph, {shouldCancel: () => true}), error => error.name === 'AbortError');
  const builder = new TargetShuffleRewire(graph);
  assert.throws(() => builder.result(), /Finish preparing/);
});

test('malformed graph, seed and chunk configurations are rejected', () => {
  const graph = fixture();
  assert.throws(() => new TargetShuffleRewire({...graph, n: 0}), /CSR/);
  assert.throws(() => new TargetShuffleRewire(graph, {seed: -1}), /seed/);
  assert.throws(() => new TargetShuffleRewire(graph, {seed: 0x100000000}), /seed/);
  assert.throws(() => new TargetShuffleRewire(graph).advance(0), /chunk/);
  const badTarget = {...graph, indices: graph.indices.slice()}; badTarget.indices[0] = 5;
  assert.throws(() => finish(badTarget), /target/);
  const badRows = {...graph, indptr: new Uint32Array([0, 6, 4, 9, 11, 14])};
  assert.throws(() => finish(badRows), /offsets/);
});
