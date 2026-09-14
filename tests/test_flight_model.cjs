/* Run: node --test tests/test_flight_model.cjs (no npm dependencies). */
const test = require('node:test');
const assert = require('node:assert/strict');

require('../site/live-model.js');
const {FlyArena} = require('../site/flight-model.js');

const SIM = {
  dt_ms: 0.1, v_rest_mV: -52, v_reset_mV: -52, v_thresh_mV: -45,
  tau_mem_ms: 20, tau_syn_ms: 5, t_refractory_ms: 2.2, delay_ms: 1.8,
  w_syn_mV: 0.275, poisson_weight_mV: 68.75, dtype: 'float32',
  rest_eps_mV: 1e-5, background: {enabled: false}
};

function graph(n = 9) {
  return {n, indptr: new Uint32Array(n + 1), indices: new Uint32Array(),
    weights: new Float32Array()};
}

function manifest() {
  return {config: {sim: SIM, arena: {
    width_mm: 9, height_mm: 3, control_interval_ms: 1, log_interval_ms: 10, start: 'center',
    motor: {window_ms: 50, base_speed_mm_s: 8, speed_gain: 1, backward_gain: 1,
      turn_gain: 12, max_speed_mm_s: 25, max_turn_deg_s: 720, turn_sign: 1},
    looming: {wall_distance_mm: 1, max_rate_hz: 150, exponent: 2,
      eye_ray_angles_deg: [0, 20, 45, 70, 95]},
    typewriter: {rows: 3, cols: 9},
    flight: {ceiling_mm: 1.2}
  }}, targets: {turn_L: [0], turn_R: [1], fwd_L: [2], fwd_R: [3],
    bwd_L: [4], bwd_R: [5], GF: [6], eye_L: [7], eye_R: [8]},
    layout: [...'abcdefghijklmnopqrstuvwxyz ']};
}

test('giant-fiber activity can launch the fly and the downstream plant lands it', () => {
  const arena = new FlyArena(graph(), manifest(), 1);
  for (let i = 0; i < 500; i++) arena.groups.GF.push(1);
  const takeoff = arena.stepControl();
  assert.equal(takeoff.key, null);
  assert.equal(arena.fly.airborne, true);
  assert(arena.fly.z > 0);
  assert(arena.fly.vz > 0);
  const sample = arena.sample();
  assert.equal(sample.airborne, true);
  assert(sample.z_mm > 0);
  assert(Number.isFinite(sample.vx_mm_s));
  assert(Number.isFinite(sample.roll_deg));

  for (let i = 0; i < 4000 && arena.fly.airborne; i++) arena.stepControl();
  assert.equal(arena.fly.airborne, false);
  assert.equal(arena.fly.z, 0);
  assert.equal(arena.fly.vz, 0);
});

test('flying across a key region does not type until keyboard contact', () => {
  const arena = new FlyArena(graph(), manifest(), 2);
  const startingKey = arena.lastKey;
  arena.fly.airborne = true;
  arena.fly.z = 0.5;
  arena.fly.vz = 0;
  arena.fly.wingPower = 1;
  arena.fly.x = 7.5;
  arena.fly.y = 1.5;
  const airborne = arena.stepControl();
  assert.equal(airborne.key, null);
  assert.equal(arena.lastKey, startingKey);
  assert.equal(arena.keyCount, 0);

  arena.fly.airborne = false;
  arena.fly.z = 0;
  arena.fly.vz = 0;
  arena.fly.wingPower = 0;
  const contact = arena.stepControl();
  assert(contact.key, 'contact with a new key region should type');
  assert.notEqual(contact.key[1], startingKey);
  assert.equal(arena.keyCount, 1);
});
