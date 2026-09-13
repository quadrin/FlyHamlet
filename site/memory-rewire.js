/* A degree-matched configuration-model control for the memory experiment.
 * Fisher–Yates permutes target stubs while retaining each source edge's weight.
 * This is a directed MULTIGRAPH: new self loops and parallel edges are allowed.
 * In/out stub degrees and source weight lists are exact; incoming weighted
 * strengths, distinct-neighbor degrees, and spatial geometry are not preserved.
 * The input graph is never edited. Returned indptr/weights share its storage
 * and must remain read-only, as they do in the fixed-connectome simulator.
 */
(function (root) {
  'use strict';
  const live = root.FlyHamletLive || (typeof require === 'function' ? require('./live-model.js') : null);
  if (!live) throw new Error('Load live-model.js before memory-rewire.js.');
  const DEFAULT_SEED = 1299709;
  const ALGORITHM = 'target-stub-fisher-yates-v1';

  class TargetShuffleRewire {
    constructor(graph, {seed = DEFAULT_SEED} = {}) {
      if (!graph || !Number.isInteger(graph.n) || graph.n < 1 || graph.n > 0x7fffffff ||
          !(graph.indptr instanceof Uint32Array) || !(graph.indices instanceof Uint32Array) ||
          !(graph.weights instanceof Float32Array || graph.weights instanceof Float64Array) ||
          graph.indptr.length !== graph.n + 1 || graph.indices.length !== graph.weights.length ||
          graph.indices.length > 0x7fffffff || graph.indptr[0] !== 0 ||
          graph.indptr[graph.n] !== graph.indices.length)
        throw new Error('Rewiring requires a complete typed CSR graph.');
      if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
        throw new Error('The rewiring seed must be an unsigned 32-bit integer.');
      this.original = graph;
      this.seed = seed;
      this.indices = graph.indices.slice();
      this.rng = new live.SeededRandom(seed);
      this.shuffleIndex = this.indices.length - 1;
      this.source = 0;
      this.edge = 0;
      this.verifyIndex = 0;
      this.degreeDelta = new Int32Array(graph.n);
      this.lastSource = new Int32Array(graph.n).fill(-1);
      this.lastDuplicateSource = -1;
      this.selfLoops = 0;
      this.originalSelfLoops = 0;
      this.parallelEdges = 0;
      this.parallelRows = 0;
      this.phase = 'shuffle';
      this.completedOperations = 0;
      this.totalOperations = Math.max(0, this.indices.length - 1) + this.indices.length + 2 * graph.n;
      this.output = null;
    }

    get done() { return this.phase === 'complete'; }

    get progress() {
      const message = this.phase === 'shuffle' ? 'Preparing the rewired comparison network' :
        this.done ? 'Rewired comparison verified' : 'Verifying comparison degrees and edge counts';
      return {phase: this.phase, message, loaded: this.completedOperations, total: this.totalOperations,
        fraction: this.completedOperations / this.totalOperations};
    }

    // Operations include swaps, edge audits, row transitions, and vertex checks.
    // Thus sparse/empty rows also yield regularly; work never hides an O(N) scan.
    advance(maxOperations = 50000) {
      if (!Number.isInteger(maxOperations) || maxOperations < 1)
        throw new Error('The preparation chunk size must be a positive integer.');
      let operations = 0;
      const graph = this.original;
      while (!this.done && operations < maxOperations) {
        if (this.phase === 'shuffle') {
          if (this.shuffleIndex <= 0) { this.phase = 'audit'; continue; }
          const i = this.shuffleIndex--;
          const j = Math.floor(this.rng.next() * (i + 1));
          const target = this.indices[i]; this.indices[i] = this.indices[j]; this.indices[j] = target;
        } else if (this.phase === 'audit') {
          if (this.source === graph.n) { this.phase = 'verify'; continue; }
          const rowEnd = graph.indptr[this.source + 1];
          if (rowEnd < graph.indptr[this.source] || rowEnd > this.indices.length)
            throw new Error('Invalid CSR row offsets in the rewiring input.');
          if (this.edge === rowEnd) {
            this.source++;
          } else {
            const oldTarget = graph.indices[this.edge], target = this.indices[this.edge];
            if (oldTarget >= graph.n || target >= graph.n)
              throw new Error('Invalid target neuron in the rewiring input.');
            this.degreeDelta[oldTarget]--;
            this.degreeDelta[target]++;
            if (oldTarget === this.source) this.originalSelfLoops++;
            if (target === this.source) this.selfLoops++;
            if (this.lastSource[target] === this.source) {
              this.parallelEdges++;
              if (this.lastDuplicateSource !== this.source) {
                this.parallelRows++; this.lastDuplicateSource = this.source;
              }
            }
            this.lastSource[target] = this.source;
            this.edge++;
          }
        } else if (this.phase === 'verify') {
          if (this.verifyIndex === graph.n) { this.finish(); continue; }
          if (this.degreeDelta[this.verifyIndex++] !== 0)
            throw new Error('Rewiring did not preserve every incoming stub degree.');
        }
        operations++;
        this.completedOperations++;
      }
      // Finish without requiring an extra empty call after the last audit check.
      if (this.phase === 'verify' && this.verifyIndex === graph.n) this.finish();
      return this.progress;
    }

    finish() {
      if (this.done) return;
      const graph = this.original;
      const metadata = Object.freeze({algorithm: ALGORITHM, seed: this.seed, n: graph.n,
        edgeSlots: this.indices.length, distinctDirectedPairs: this.indices.length - this.parallelEdges,
        selfLoopSlots: this.selfLoops, originalSelfLoopSlots: this.originalSelfLoops,
        parallelEdgeSlots: this.parallelEdges, parallelSourceRows: this.parallelRows,
        parallelEdgeDefinition: 'Additional edge slots beyond the first for each source-target pair.',
        degreePreservationVerified: true, outDegreePreserved: true, inDegreePreserved: true,
        sourceWeightsPreserved: true, globalSignedWeightMultisetPreserved: true,
        incomingWeightedStrengthPreserved: false, distinctNeighborDegreesPreserved: false,
        graphType: 'Directed configuration-model multigraph; self loops and parallel edges allowed.',
        degreeDefinition: 'Incoming and outgoing edge-stub counts, counting parallel edges separately.',
        weightDefinition: 'Every original source edge retains its signed weight; only its target is permuted.',
        limitations: 'Incoming weighted strengths, per-target excitatory/inhibitory balance, spatial geometry and distinct-neighbor degrees are not matched. One fixed null graph is not a population of rewired networks.',
        rng: 'xoshiro128**; graph seed independent of neural-noise and trial seeds'});
      this.output = Object.freeze({graph: Object.freeze({n: graph.n, indptr: graph.indptr,
        indices: this.indices, weights: graph.weights}), metadata});
      this.degreeDelta = null;
      this.lastSource = null;
      this.phase = 'complete';
    }

    result() {
      if (!this.done) throw new Error('Finish preparing the rewired graph before using it.');
      return this.output;
    }
  }

  async function prepareRewire(graph, {seed = DEFAULT_SEED, progress = () => {},
    shouldCancel = () => false, yieldControl = () => new Promise(resolve => setTimeout(resolve, 0)),
    batchOperations = 50000} = {}) {
    const checkCancellation = () => {
      if (shouldCancel()) {
        const error = new Error('Rewired graph preparation was cancelled.');
        error.name = 'AbortError';
        throw error;
      }
    };
    checkCancellation();
    const builder = new TargetShuffleRewire(graph, {seed});
    progress(builder.progress);
    while (!builder.done) {
      checkCancellation();
      progress(builder.advance(batchOperations));
      if (!builder.done) { await yieldControl(); checkCancellation(); }
    }
    checkCancellation();
    return builder.result();
  }

  const api = {TargetShuffleRewire, prepareRewire, DEFAULT_SEED, ALGORITHM};
  root.FlyHamletMemoryRewire = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
