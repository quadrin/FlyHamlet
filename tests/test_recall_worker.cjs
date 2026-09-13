/* Run: node --test tests/test_recall_worker.cjs. No browser or dependencies.
 * A controlled event loop exercises loading races and pause/new/completion;
 * loader integrity checks are covered by test_learning_worker.cjs.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');

const workerSource = fs.readFileSync(path.join(__dirname, '../site/recall-worker.js'), 'utf8');
const graph = {n: 2, indptr: new Uint32Array([0, 1, 2]),
  indices: new Uint32Array([1, 0]), weights: new Float32Array([0.5, -0.25])};
const manifest = {format: 'flyhamlet-csr-v1', n: 2, edgeCount: 2};
const loadedGraph = {connectome: graph, manifest};

function runner(options = {}) {
  let clock = 0;
  let nextTimer = 0;
  let loads = 0;
  const timers = new Map();
  const sent = [];
  const instances = [];
  class FakeExperiment {
    constructor(conn, data, seed, config = {}) {
      this.conn = conn;
      this.seed = seed;
      this.dtMs = 0.1;
      this.steps = 0;
      this.limit = config.steps || 1000;
      this.records = [];
      this.metadata = {n: data.n, seed};
      this.protocol = {fixture: true};
      instances.push(this);
    }
    get simTimeS() { return this.steps * this.dtMs / 1000; }
    get done() { return this.steps >= this.limit; }
    get progress() { return {steps: this.steps, simTimeS: this.simTimeS, done: this.done}; }
    step() {
      clock += 0.08; // Each atomic operation has a measurable compute cost.
      this.steps++;
      if (this.steps % 10 !== 0) return null;
      const record = {trial: this.steps / 10, seed: this.seed};
      this.records.push(record);
      return record;
    }
    result() { return {seed: this.seed, complete: this.done, records: this.records.slice()}; }
  }
  const context = vm.createContext({
    performance: {now: () => clock}, crypto: webcrypto, Uint32Array,
    importScripts: () => {},
    postMessage: message => sent.push(structuredClone(message)),
    setTimeout: (callback, delay) => {
      const id = ++nextTimer;
      timers.set(id, {callback, at: clock + delay});
      return id;
    },
    clearTimeout: id => timers.delete(id),
    FlyHamletRecall: {RecallExperiment: options.Experiment || FakeExperiment},
    FlyHamletConnectome: {load: async (_url, config) => {
      loads++;
      config.progress('fixture loading', 0, 2);
      return options.load ? await options.load() : loadedGraph;
    }}
  });
  context.self = context;
  vm.runInContext(workerSource, context);
  return {
    sent, timers, instances, get loads() { return loads; },
    dispatch: message => context.onmessage({data: message}),
    advance(ms = 2) {
      clock += ms;
      const due = [...timers].find(([, timer]) => timer.at <= clock);
      if (!due) return false;
      timers.delete(due[0]);
      due[1].callback();
      return true;
    },
    drain(limit = 10000) {
      for (let i = 0; timers.size && i < limit; i++) this.advance();
      assert.equal(timers.size, 0, 'runner should terminate without scheduling forever');
    }
  };
}

test('pause during load survives initialization; superseded loads start only newest seed', async () => {
  let resolve;
  const pending = new Promise(done => { resolve = done; });
  const worker = runner({load: () => pending});
  const oldRequest = worker.dispatch({type: 'init', seed: 17});
  const newRequest = worker.dispatch({type: 'new', seed: 91});
  await worker.dispatch({type: 'pause'});
  resolve(loadedGraph);
  await Promise.all([oldRequest, newRequest]);
  assert.equal(worker.loads, 1, 'concurrent requests share the immutable graph download');
  assert.equal(worker.instances.length, 1);
  assert.equal(worker.instances[0].seed, 91);
  assert.equal(worker.timers.size, 0, 'pause applies even before the graph is available');
  const ready = worker.sent.filter(message => message.type === 'ready');
  assert.deepEqual(ready.map(message => [message.seed, message.generation]), [[91, 2]]);
  await worker.dispatch({type: 'resume'});
  worker.advance();
  assert(worker.instances[0].steps > 0);
});

test('pause and export do not advance neural state; resume preserves progress', async () => {
  const worker = runner();
  await worker.dispatch({type: 'init', seed: 71});
  worker.advance();
  await worker.dispatch({type: 'pause'});
  const steps = worker.instances[0].steps;
  worker.advance(10000);
  await worker.dispatch({type: 'export'});
  assert.equal(worker.instances[0].steps, steps);
  assert.equal(worker.timers.size, 0);
  const exported = worker.sent.find(message => message.type === 'export');
  assert.equal(exported.result.complete, false);
  assert.equal(exported.seed, 71);
  assert.equal(exported.generation, 1);
  await worker.dispatch({type: 'resume'});
  worker.advance(2);
  assert(worker.instances[0].steps > steps);
  assert(worker.instances[0].steps < steps + 100, 'pause wall time is excluded from catch-up');
});

test('completion sends every trial once, flushes the final tick, and cannot resume', async () => {
  const worker = runner();
  await worker.dispatch({type: 'init', seed: 123, speed: 100, options: {steps: 60}});
  worker.drain();
  const ticks = worker.sent.filter(message => message.type === 'tick');
  const records = ticks.flatMap(message => message.records);
  assert.deepEqual(records.map(record => record.trial), [1, 2, 3, 4, 5, 6]);
  assert(ticks.at(-1).progress.done);
  assert(ticks.at(-1).realTimeRatio > 0);
  assert.equal(worker.sent.filter(message => message.type === 'complete').length, 1);
  assert.equal(worker.sent.at(-1).type, 'complete');
  assert.equal(worker.sent.at(-1).result.records.length, 6);
  await worker.dispatch({type: 'resume'});
  assert.equal(worker.timers.size, 0);
  assert.equal(worker.instances[0].steps, 60);
});

test('new discards old timers and state, caches graph, and defaults back to 1x', async () => {
  const worker = runner();
  await worker.dispatch({type: 'init', seed: 1, speed: 100});
  const oldCallback = [...worker.timers.values()][0].callback;
  worker.advance();
  await worker.dispatch({type: 'new', seed: 2});
  oldCallback();
  assert.equal(worker.instances[1].steps, 0, 'a stale callback cannot execute a new session');
  worker.advance(2);
  assert(worker.instances[1].steps <= 20, 'new defaults to 1x regardless of previous speed');
  assert.equal(worker.loads, 1);
  assert.equal(worker.instances[0].conn, worker.instances[1].conn);
  const ready = worker.sent.filter(message => message.type === 'ready');
  assert.deepEqual(ready.map(message => [message.seed, message.generation]), [[1, 1], [2, 2]]);
});

test('a failed shared load emits one current-generation error and remains retryable', async () => {
  let reject;
  let attempt = 0;
  const pending = new Promise((_resolve, fail) => { reject = fail; });
  const worker = runner({load: () => ++attempt === 1 ? pending : loadedGraph});
  const oldRequest = worker.dispatch({type: 'init', seed: 4});
  const newRequest = worker.dispatch({type: 'new', seed: 5});
  reject(new Error('network fixture failure'));
  await Promise.all([oldRequest, newRequest]);
  const errors = worker.sent.filter(message => message.type === 'error');
  assert.equal(errors.length, 1);
  assert.equal(errors[0].generation, 2);
  assert.equal(errors[0].seed, 5);
  await worker.dispatch({type: 'new', seed: 6, autoplay: false});
  assert.equal(worker.sent.findLast(message => message.type === 'ready').seed, 6);
  assert.equal(worker.loads, 2);
});

test('actual recall engine agrees with direct execution across fit and autonomous phases', async () => {
  const {RecallExperiment} = require('../site/recall-model.js');
  const connectome = {n: 16, indptr: new Uint32Array([...Array.from({length: 9}, (_, index) => index), ...Array(8).fill(8)]),
    indices: new Uint32Array(Array.from({length: 8}, (_, index) => index + 8)),
    weights: new Float32Array(8).fill(80)};
  const data = {config: {sim: {dt_ms: 1, v_rest_mV: -52, v_reset_mV: -52, v_thresh_mV: -45,
    tau_mem_ms: 20, tau_syn_ms: 5, t_refractory_ms: 2.2, delay_ms: 2,
    poisson_weight_mV: 68.75, dtype: 'float32', background: {enabled: false}}},
    targets: {eye_L: [0, 1, 2, 3], eye_R: [4, 5, 6, 7]}, layout: Array.from('abcdefghijklmnopqrstuvwxyz ')};
  const options = {phrase: 'to', trainEpisodes: 2, recallEpisodes: 1, ablatedEpisodes: 1,
    poolCount: 8, maxDecisions: 4, trialMs: 20, warmupMs: 5};
  const worker = runner({Experiment: RecallExperiment, load: () => ({connectome, manifest: data})});
  await worker.dispatch({type: 'init', seed: 451, speed: 100, options});
  worker.drain();
  const direct = new RecallExperiment(connectome, data, 451, options);
  while (!direct.done) direct.step();
  const complete = worker.sent.find(message => message.type === 'complete');
  assert(complete, 'real engine must complete after dynamic END/cap stopping');
  assert.deepEqual(complete.result, direct.result());
  assert.equal(complete.result.fitCount, 1);
  assert.equal(worker.sent.flatMap(message => message.records || []).length, complete.result.trials.length);
  assert.equal(worker.sent.find(message => message.type === 'ready').metadata.protocol.phrase, 'to');
});

test('benchmark scoring requires an actual END and counts character errors independently', () => {
  const {editDistance, scoreEpisodes} = require('../scripts/benchmark_recall.cjs');
  const scores = scoreEpisodes([
    {output: 'to be', stoppedBy: 'END'},
    {output: 'to be', stoppedBy: 'cap'},
    {output: 'to bee', stoppedBy: 'END'},
    {output: '', stoppedBy: 'END'}
  ], 'to be');
  assert.equal(scores.exact, 1, 'a correct character string without END is not successful recall');
  assert.equal(scores.ended, 3);
  assert.equal(scores.capped, 1);
  assert.equal(scores.meanEditDistance, 1.5);
  assert.equal(editDistance('to be', 'to be'), 0);
  assert.equal(editDistance('to be', 'to bee'), 1);
  assert.equal(editDistance('to be', 'to ba'), 1);
  assert.equal(editDistance('', 'to be'), 5);
});
