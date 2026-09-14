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
  return {n, indptr: new Uint32Array(n + 1), indices: new Uint32Array(), weights: new Float32Array()};
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

test('giant-fiber takeoff spins a physical wing oscillator whose strokes create lift', () => {
  const arena = new FlyArena(graph(), manifest(), 1);
  for (let i = 0; i < 500; i++) arena.groups.GF.push(1);
  let maxZ = 0;
  const initialPhase = arena.fly.wingPhase;
  for (let i = 0; i < 40; i++) {
    const result = arena.stepControl();
    assert.equal(result.key, null);
    maxZ = Math.max(maxZ, arena.fly.z);
  }
  assert.equal(arena.fly.airborne, true);
  assert(arena.fly.wingFrequency > 150);
  assert.notEqual(arena.fly.wingPhase, initialPhase);
  assert(arena.fly.leftWingAmplitude > 0);
  assert(arena.fly.rightWingAmplitude > 0);
  assert(maxZ > 0, 'wing strokes must produce positive altitude');
  const sample = arena.sample();
  assert(sample.wing_frequency_hz > 150);
  assert(Number.isFinite(sample.wing_phase_rad));

  for (let i = 0; i < 5000 && arena.fly.airborne; i++) arena.stepControl();
  assert.equal(arena.fly.airborne, false);
  assert.equal(arena.fly.z, 0);
  assert.equal(arena.fly.wingFrequency, 0);
});

test('a silent brain leaves the fly standing still', () => {
  const arena = new FlyArena(graph(), manifest(), 3);
  const x0 = arena.fly.x, y0 = arena.fly.y;
  for (let i = 0; i < 80; i++) arena.stepControl();
  assert.equal(arena.fly.airborne, false);
  assert.equal(arena.fly.gaitFrequency, 0, 'no descending drive means no stepping');
  assert(Math.hypot(arena.fly.x - x0, arena.fly.y - y0) < 1e-6,
    'the fly must not creep when the descending neurons are silent');
});

test('walking advances an alternating gait oscillator and ground traction moves the body', () => {
  const arena = new FlyArena(graph(), manifest(), 3);
  const drive = () => { for (let i = 0; i < 500; i++) { arena.groups.fwd_L.push(1); arena.groups.fwd_R.push(1); } };
  drive();
  const x0 = arena.fly.x;
  const phase0 = arena.fly.gaitPhase;
  for (let i = 0; i < 80; i++) { arena.stepControl(); drive(); }
  assert.equal(arena.fly.airborne, false);
  assert(arena.fly.gaitFrequency > 0);
  assert.notEqual(arena.fly.gaitPhase, phase0);
  assert(arena.fly.x > x0, 'stance traction should move the body forward');
  assert.equal(arena.fly.wingFrequency, 0);
  const sample = arena.sample();
  assert(sample.gait_frequency_hz > 0);
  assert(sample.gait_duty_factor > 0.5 && sample.gait_duty_factor < 0.7);
});

test('a flight bout ends once the giant fibre goes quiet', () => {
  const arena = new FlyArena(graph(), manifest(), 1);
  for (let i = 0; i < 500; i++) arena.groups.GF.push(1);
  for (let i = 0; i < 40; i++) arena.stepControl();
  assert.equal(arena.fly.airborne, true, 'the giant fibre should launch the fly');

  // Hold the locomotor drive high but let the giant fibre fall silent. Walking
  // commands alone must not keep the fly airborne.
  let steps = 0;
  while (arena.fly.airborne && steps < 20000) {
    arena.stepControl();
    for (let i = 0; i < 500; i++) { arena.groups.fwd_L.push(1); arena.groups.fwd_R.push(1); }
    steps++;
  }
  assert.equal(arena.fly.airborne, false, 'the bout must end without the giant fibre');
  assert(steps * 0.001 < 12, `bout ran ${(steps * 0.001).toFixed(2)}s after the giant fibre stopped`);
});

test('steering in flight fires saccades rather than a steady turn', () => {
  const arena = new FlyArena(graph(), manifest(), 1);
  for (let i = 0; i < 500; i++) arena.groups.GF.push(1);
  for (let i = 0; i < 40; i++) arena.stepControl();
  assert.equal(arena.fly.airborne, true);

  const rates = [];
  for (let i = 0; i < 600 && arena.fly.airborne; i++) {
    for (let k = 0; k < 500; k++) { arena.groups.GF.push(1); arena.groups.turn_L.push(1); }
    arena.stepControl();
    rates.push(Math.abs(arena.fly.omega));
  }
  const peak = Math.max(...rates);
  const median = [...rates].sort((a, b) => a - b)[Math.floor(rates.length / 2)];
  assert(peak > 0, 'the fly should turn at all');
  assert(peak > median * 3,
    `turning should burst, not hold steady (peak ${peak.toFixed(2)} vs median ${median.toFixed(2)} rad/s)`);
});

test('flying across a key region does not type until keyboard contact', () => {
  const arena = new FlyArena(graph(), manifest(), 2);
  const startingKey = arena.lastKey;
  arena.fly.airborne = true;
  arena.fly.z = 0.5;
  arena.fly.vz = 0;
  arena.fly.wingDrive = 1;
  arena.fly.x = 7.5;
  arena.fly.y = 1.5;
  const airborne = arena.stepControl();
  assert.equal(airborne.key, null);
  assert.equal(arena.lastKey, startingKey);
  assert.equal(arena.keyCount, 0);

  arena.fly.airborne = false;
  arena.fly.z = 0;
  arena.fly.vz = 0;
  arena.fly.wingDrive = 0;
  const contact = arena.stepControl();
  assert(contact.key, 'contact with a new key region should type');
  assert.notEqual(contact.key[1], startingKey);
  assert.equal(arena.keyCount, 1);
});
