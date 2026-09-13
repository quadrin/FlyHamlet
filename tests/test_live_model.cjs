/* Run: node --test tests/test_live_model.cjs (no npm dependencies).
 * These fixtures exercise the same Brian2 reference spike trains as test_sim.py.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {LIFNetwork, FlyArena, WindowedRate, SeededRandom, rayToWall} = require('../site/live-model.js');

const SIM = {
  dt_ms: 0.1, v_rest_mV: -52, v_reset_mV: -52, v_thresh_mV: -45,
  tau_mem_ms: 20, tau_syn_ms: 5, t_refractory_ms: 2.2, delay_ms: 1.8,
  w_syn_mV: 0.275, poisson_weight_mV: 68.75, dtype: 'float32',
  rest_eps_mV: 1e-5, background: {enabled: false}
};

function graph(n = 4, edges = [], signs = new Array(n).fill(1), dtype = 'float32') {
  const offsets = [0], indices = [], weights = [];
  for (let pre = 0; pre < n; pre++) {
    for (const [source, target, count] of edges.filter(e => e[0] === pre).sort((a, b) => a[1] - b[1])) {
      indices.push(target); weights.push(count * signs[source] * SIM.w_syn_mV);
    }
    offsets.push(indices.length);
  }
  return {n, indptr: new Uint32Array(offsets), indices: new Uint32Array(indices),
    weights: dtype === 'float64' ? new Float64Array(weights) : new Float32Array(weights)};
}
function run(net, steps) {
  const events = Array.from({length: net.n}, () => []);
  for (let step = 0; step < steps; step++) for (const neuron of net.step()) events[neuron].push(step);
  return events;
}

test('all neurons remain silent and at rest without input', () => {
  const net = new LIFNetwork(graph(6, [[0, 1, 40], [1, 2, 200]]), SIM, 42);
  assert.deepEqual(run(net, 1000), Array.from({length: 6}, () => []));
  assert.deepEqual([...net.v], new Array(6).fill(-52));
  assert.deepEqual([...net.g], new Array(6).fill(0));
  assert.equal(net.activeCount, 0);
});

const chains = {
  weak: {edges: [[0, 1, 40], [1, 2, 25], [1, 3, 8], [2, 3, 12]], signs: [1, 1, -1, 1]},
  strong: {edges: [[0, 1, 200], [0, 2, 30], [1, 2, 60], [1, 3, 300], [2, 3, 50]], signs: [1, 1, -1, 1]}
};
for (const [name, chain] of Object.entries(chains)) {
  test(`${name} chain matches every Brian2 reference spike`, () => {
    const net = new LIFNetwork(graph(4, chain.edges, chain.signs, 'float64'), {...SIM, dtype: 'float64'}, 0);
    net.injectCurrent([0], 100);
    const events = run(net, 3000);
    const reference = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', `brian2_chain_${name}.json`)));
    for (let neuron = 0; neuron < 4; neuron++) assert.deepEqual(events[neuron], reference[neuron] || []);
  });
}

test('synaptic delay is 18 timesteps and inhibitory sign is preserved', () => {
  const net = new LIFNetwork(graph(2, [[0, 1, 10]], [-1, 1]), SIM, 0);
  net.setVoltage(0, 16.75);
  assert.deepEqual(net.step(), [0]);
  for (let step = 1; step < 18; step++) { net.step(); assert.equal(net.g[1], 0); }
  net.step();
  assert.equal(net.g[1], -2.75);
  assert.equal(net.v[1], -52, 'delivery occurs after state integration');
  net.step();
  assert(net.v[1] < -52);
});

test('inputs on refractory cells are dropped until the exact resume timestep', () => {
  const cfg = {...SIM, dt_ms: 1, delay_ms: 1, t_refractory_ms: 3};
  const net = new LIFNetwork(graph(2, [[0, 1, 40]]), cfg, 0);
  net.injectPoisson([0], 0); // sensory-style presynaptic neuron has no refractory interval
  net.setVoltage(0, 16.75); net.setVoltage(1, 16.75);
  assert.deepEqual(net.step(), [0, 1]);
  assert.equal(net.resumeAt[1], 3);
  for (let step = 1; step <= 2; step++) {
    net.setVoltage(0, 16.75); net.step();
    assert.equal(net.g[1], 0, `input on refractory step ${step} must be discarded`);
  }
  net.step();
  assert.equal(net.g[1], 11, 'input is accepted on the first nonrefractory step');
});

test('input on a cell that spikes this step is discarded by the final reset', () => {
  const net = new LIFNetwork(graph(2, [[0, 1, 40]]), {...SIM, delay_ms: 0.1}, 0);
  net.setVoltage(0, 16.75); net.step();
  net.setVoltage(1, 16.75);
  assert.deepEqual(net.step(), [1]);
  assert.equal(net.v[1], SIM.v_reset_mV);
  assert.equal(net.g[1], 0);
});

test('browser seeds reproduce neural input streams, and a new seed changes them', () => {
  function seededRun(seed) {
    const net = new LIFNetwork(graph(3, [[0, 1, 80], [1, 2, 40]]), SIM, seed);
    net.injectPoisson([0], 300);
    return run(net, 2000);
  }
  const first = seededRun(123);
  assert(first[0].length > 10);
  assert.deepEqual(first, seededRun(123));
  assert.notDeepEqual(first, seededRun(124));
});

test('sparse sensory input follows Bernoulli trials without duplicate targets', () => {
  const rng = new SeededRandom(7);
  let count = 0;
  for (let draw = 0; draw < 10000; draw++) {
    const hits = [];
    rng.bernoulliIndices(162, 0.015, index => hits.push(index));
    assert.equal(new Set(hits).size, hits.length);
    assert(hits.every(index => index >= 0 && index < 162));
    count += hits.length;
  }
  assert(Math.abs(count / 10000 - 162 * 0.015) < 0.08);
});

test('float32 fixed-point retirement preserves voltage and later input reactivates it', () => {
  const net = new LIFNetwork(graph(1), SIM, 0);
  net.setVoltage(0, -51.9998);
  const plateau = net.v[0];
  net.step();
  assert.equal(net.activeCount, 0);
  assert.equal(net.v[0], plateau);
  assert(Math.abs(net.v[0] - SIM.v_rest_mV) > SIM.rest_eps_mV);
  net.addInput(0, 1);
  assert.equal(net.activeCount, 1);
  net.step();
  assert.notEqual(net.v[0], plateau);
});

test('motor rates use the full sliding window, including silent steps', () => {
  const rate = new WindowedRate(500, 0.1, 2);
  rate.push(1);
  assert.equal(rate.hz, 10);
  for (let step = 0; step < 499; step++) rate.push(0);
  assert.equal(rate.hz, 10);
  rate.push(0);
  assert.equal(rate.hz, 0);
});

function arenaManifest(start = 'center') {
  return {config: {sim: SIM, arena: {
    width_mm: 9, height_mm: 3, control_interval_ms: 1, log_interval_ms: 10, start,
    motor: {window_ms: 50, base_speed_mm_s: 8, speed_gain: 1, backward_gain: 1,
      turn_gain: 12, max_speed_mm_s: 25, max_turn_deg_s: 720, turn_sign: 1},
    looming: {wall_distance_mm: 1, max_rate_hz: 150, exponent: 2, eye_ray_angles_deg: [0, 20, 45, 70, 95]},
    typewriter: {rows: 3, cols: 9}
  }}, targets: {turn_L: [0], turn_R: [1], fwd_L: [2], fwd_R: [3],
    bwd_L: [4], bwd_R: [5], GF: [6], eye_L: [7], eye_R: [8]},
    layout: [...'abcdefghijklmnopqrstuvwxyz ']};
}

test('arena types only key entries, clamps walls, and starts a fresh seeded session', () => {
  const data = graph(9), manifest = arenaManifest();
  const arena = new FlyArena(data, manifest, 42);
  assert.equal(arena.lastKey, 13);
  assert.equal(arena.keyCount, 0, 'starting region is never typed');
  assert.deepEqual(arena.eyeRates, [0, 0], 'first control interval starts without looming input');
  const keys = [];
  for (let step = 0; step < 1000; step++) {
    const result = arena.stepControl();
    if (result.key) keys.push(result.key[1]);
    assert(arena.fly.x >= 0 && arena.fly.x <= 9 && arena.fly.y >= 0 && arena.fly.y <= 3);
  }
  assert.deepEqual(keys, [14, 15, 16, 17]);
  assert.equal(arena.fly.x, 9);
  assert(arena.wallSteps > 0);
  const randomManifest = arenaManifest('random');
  const first = new FlyArena(data, randomManifest, 12);
  const repeated = new FlyArena(data, randomManifest, 12);
  const fresh = new FlyArena(data, randomManifest, 13);
  assert.deepEqual(first.sample(), repeated.sample());
  assert.notDeepEqual(first.sample(), fresh.sample());
  assert.equal(fresh.net.timeS, 0);
  assert.equal(fresh.net.totalSpikes, 0);
});

test('motor activity feeds back through turning, and rays see the arena boundary', () => {
  const arena = new FlyArena(graph(9), arenaManifest(), 1);
  arena.net.injectCurrent([0], 100);
  for (let step = 0; step < 100; step++) arena.stepControl();
  assert(arena.groups.turn_L.hz > 0);
  assert.equal(arena.groups.turn_R.hz, 0);
  assert(arena.fly.heading > 0);
  assert(arena.sample().omega_deg_s > 0);
  assert.equal(rayToWall(4, 2, 0, 9, 3), 5);
  assert.equal(rayToWall(4, 2, Math.PI, 9, 3), 4);
});
