/* Connectome-driven 3D locomotion layered onto FlyHamlet's brain arena.
 * Descending-neuron activity supplies motor commands. Deterministic downstream
 * wing and leg oscillators stand in for the VNC, muscles and mechanics outside
 * the FAFB brain connectome. No key, destination or path is selected here.
 */
(function (root) {
  'use strict';

  const api = root.FlyHamletLive;
  if (!api?.FlyArena) throw new Error('flight-model.js must load after live-model.js');
  const BaseFlyArena = api.FlyArena;
  const TAU = 2 * Math.PI;
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const modulo = (value, modulus) => ((value % modulus) + modulus) % modulus;

  const DEFAULT_FLIGHT = Object.freeze({
    ceiling_mm: 12,
    gf_takeoff_hz: 6,
    gf_reset_hz: 1.5,
    gf_full_scale_hz: 35,
    motor_full_scale_hz: 30,
    wing_frequency_min_hz: 175,
    wing_frequency_max_hz: 225,
    wing_stroke_amplitude_deg: 72,
    wing_rise_s: 0.020,
    wing_decay_s: 0.45,
    wing_turn_asymmetry: 0.22,
    max_wing_accel_mm_s2: 21000,
    vertical_drag_s: 4.5,
    air_drag_s: 3.2,
    yaw_accel_rad_s2: 180,
    yaw_drag_s: 7,
    max_bank_deg: 38,
    max_pitch_deg: 25,
    attitude_response_s: 11,
    wall_restitution: 0.32,
    ceiling_restitution: 0.12,
    gait_frequency_min_hz: 5,
    gait_frequency_max_hz: 18,
    gait_duty_slow: 0.68,
    gait_duty_fast: 0.54,
    ground_accel_mm_s2: 600,
    ground_drag_s: 18,
    ground_yaw_accel_rad_s2: 150,
    ground_yaw_drag_s: 12,
    gravity_mm_s2: 9810,
  });

  class FlightArena extends BaseFlyArena {
    constructor(connectome, manifest, seed) {
      super(connectome, manifest, seed);
      this.flightCfg = {...DEFAULT_FLIGHT, ...(this.cfg.arena.flight || {})};
      const f = this.fly;
      f.z = 0; f.vz = 0;
      f.vx = f.v * Math.cos(f.heading); f.vy = f.v * Math.sin(f.heading);
      f.pitch = 0; f.roll = 0;
      f.wingDrive = 0; f.wingPhase = 0; f.wingFrequency = 0;
      f.leftWingAmplitude = 0; f.rightWingAmplitude = 0;
      f.gaitPhase = 0; f.gaitFrequency = 0; f.gaitDuty = this.flightCfg.gait_duty_slow;
      f.airborne = false; f.takeoffArmed = true;
    }

    sample() {
      const sample = super.sample();
      const f = this.fly;
      return {...sample,
        z_mm: f.z, vz_mm_s: f.vz, vx_mm_s: f.vx, vy_mm_s: f.vy,
        pitch_deg: f.pitch * 180 / Math.PI, roll_deg: f.roll * 180 / Math.PI,
        wing_power: f.wingDrive, wing_phase_rad: f.wingPhase,
        wing_frequency_hz: f.wingFrequency,
        wing_left_amplitude: f.leftWingAmplitude,
        wing_right_amplitude: f.rightWingAmplitude,
        gait_phase_rad: f.gaitPhase, gait_frequency_hz: f.gaitFrequency,
        gait_duty_factor: f.gaitDuty, airborne: f.airborne};
    }

    updateWingPlant(dt, gf, fwd, bwd, turnNorm) {
      const c = this.flightCfg, f = this.fly;
      const gfDrive = clamp(gf / c.gf_full_scale_hz, 0, 1);
      const locomotorDrive = clamp((fwd + bwd) / c.motor_full_scale_hz, 0, 1);
      const targetDrive = f.airborne ? clamp(Math.max(gfDrive, 0.78 * locomotorDrive), 0, 1) : 0;
      const tau = targetDrive > f.wingDrive ? c.wing_rise_s : c.wing_decay_s;
      f.wingDrive += (targetDrive - f.wingDrive) * (1 - Math.exp(-dt / Math.max(1e-4, tau)));
      if (f.wingDrive < 1e-4) f.wingDrive = 0;

      f.wingFrequency = f.wingDrive > 0
        ? c.wing_frequency_min_hz + (c.wing_frequency_max_hz - c.wing_frequency_min_hz) * f.wingDrive
        : 0;
      f.wingPhase = modulo(f.wingPhase + TAU * f.wingFrequency * dt, TAU);
      f.leftWingAmplitude = clamp(f.wingDrive * (1 - c.wing_turn_asymmetry * turnNorm), 0, 1);
      f.rightWingAmplitude = clamp(f.wingDrive * (1 + c.wing_turn_asymmetry * turnNorm), 0, 1);

      const strokeVelocity = Math.abs(Math.cos(f.wingPhase));
      const frequencyScale = f.wingFrequency / c.wing_frequency_max_hz;
      const leftForce = c.max_wing_accel_mm_s2 * f.leftWingAmplitude ** 2
        * frequencyScale ** 2 * (0.30 + 0.70 * strokeVelocity ** 2);
      const rightForce = c.max_wing_accel_mm_s2 * f.rightWingAmplitude ** 2
        * frequencyScale ** 2 * (0.30 + 0.70 * strokeVelocity ** 2);
      return {leftForce, rightForce, totalForce: 0.5 * (leftForce + rightForce)};
    }

    updateFlight(dt, fwd, bwd, turnNorm, wing) {
      const c = this.flightCfg, f = this.fly;
      const signedDrive = clamp((fwd - bwd) / c.motor_full_scale_hz, -1, 1);
      const pitchTarget = -signedDrive * c.max_pitch_deg * Math.PI / 180;
      const rollTarget = -turnNorm * c.max_bank_deg * Math.PI / 180;
      const attitudeAlpha = 1 - Math.exp(-c.attitude_response_s * dt);
      f.pitch += (pitchTarget - f.pitch) * attitudeAlpha;
      f.roll += (rollTarget - f.roll) * attitudeAlpha;

      const verticalWing = wing.totalForce * Math.cos(f.pitch) * Math.cos(f.roll);
      const forwardWing = -wing.totalForce * Math.sin(f.pitch);
      const lateralWing = wing.totalForce * Math.sin(f.roll) * 0.35;
      const forwardX = Math.cos(f.heading), forwardY = Math.sin(f.heading);
      const lateralX = -forwardY, lateralY = forwardX;
      f.vx += (forwardWing * forwardX + lateralWing * lateralX - c.air_drag_s * f.vx) * dt;
      f.vy += (forwardWing * forwardY + lateralWing * lateralY - c.air_drag_s * f.vy) * dt;
      f.vz += (verticalWing - c.gravity_mm_s2 - c.vertical_drag_s * f.vz) * dt;

      const wingImbalance = (wing.rightForce - wing.leftForce) /
        Math.max(1, c.max_wing_accel_mm_s2);
      f.omega += (c.yaw_accel_rad_s2 * wingImbalance - c.yaw_drag_s * f.omega) * dt;
      f.heading = modulo(f.heading + f.omega * dt + Math.PI, TAU) - Math.PI;

      const nx = f.x + f.vx * dt, ny = f.y + f.vy * dt;
      if (nx < 0 || nx > this.width) {
        f.vx = -f.vx * c.wall_restitution; f.x = clamp(nx, 0, this.width); this.wallSteps++;
      } else f.x = nx;
      if (ny < 0 || ny > this.height) {
        f.vy = -f.vy * c.wall_restitution; f.y = clamp(ny, 0, this.height); this.wallSteps++;
      } else f.y = ny;
      f.z += f.vz * dt;
      if (f.z >= c.ceiling_mm) {
        f.z = c.ceiling_mm;
        if (f.vz > 0) f.vz = -f.vz * c.ceiling_restitution;
      }
      if (f.z <= 0 && f.vz <= 0) {
        f.z = 0; f.vz = 0;
        if (f.wingDrive < 0.32 && wing.totalForce < c.gravity_mm_s2 * 0.75) {
          f.airborne = false;
          f.wingDrive = 0; f.wingFrequency = 0;
          f.leftWingAmplitude = 0; f.rightWingAmplitude = 0;
          f.roll *= 0.35; f.pitch *= 0.35;
        }
      }
      f.v = f.vx * Math.cos(f.heading) + f.vy * Math.sin(f.heading);
    }

    updateWalking(dt, fwd, bwd, turnNorm) {
      const c = this.flightCfg, m = this.cfg.arena.motor, f = this.fly;
      const desiredSpeed = clamp(m.base_speed_mm_s + m.speed_gain * fwd - m.backward_gain * bwd,
        -m.max_speed_mm_s, m.max_speed_mm_s);
      const walkDrive = clamp(desiredSpeed / Math.max(1e-6, m.max_speed_mm_s), -1, 1);
      const gaitActivity = clamp(Math.max(Math.abs(walkDrive), 0.35 * Math.abs(turnNorm)), 0, 1);
      f.gaitFrequency = gaitActivity > 0.02
        ? c.gait_frequency_min_hz + (c.gait_frequency_max_hz - c.gait_frequency_min_hz) * gaitActivity
        : 0;
      f.gaitDuty = c.gait_duty_slow + (c.gait_duty_fast - c.gait_duty_slow) * gaitActivity;
      const direction = walkDrive < -0.02 ? -1 : 1;
      f.gaitPhase = modulo(f.gaitPhase + direction * TAU * f.gaitFrequency * dt, TAU);

      const phases = [0, Math.PI, 0, Math.PI, 0, Math.PI];
      let traction = 0;
      for (const offset of phases) {
        const p = modulo(f.gaitPhase + offset, TAU) / TAU;
        if (p < f.gaitDuty) traction += Math.sin(Math.PI * p / f.gaitDuty);
      }
      traction = Math.max(0.18, traction / 3);
      const fx = Math.cos(f.heading), fy = Math.sin(f.heading);
      f.vx += (c.ground_accel_mm_s2 * walkDrive * traction * fx - c.ground_drag_s * f.vx) * dt;
      f.vy += (c.ground_accel_mm_s2 * walkDrive * traction * fy - c.ground_drag_s * f.vy) * dt;
      f.omega += (c.ground_yaw_accel_rad_s2 * turnNorm * traction - c.ground_yaw_drag_s * f.omega) * dt;
      f.heading = modulo(f.heading + f.omega * dt + Math.PI, TAU) - Math.PI;

      const nx = f.x + f.vx * dt, ny = f.y + f.vy * dt;
      if (!(0 <= nx && nx <= this.width && 0 <= ny && ny <= this.height)) this.wallSteps++;
      f.x = clamp(nx, 0, this.width); f.y = clamp(ny, 0, this.height);
      if (f.x !== nx) f.vx = 0;
      if (f.y !== ny) f.vy = 0;
      f.z = 0; f.vz = 0;
      f.wingDrive = 0; f.wingFrequency = 0;
      f.leftWingAmplitude = 0; f.rightWingAmplitude = 0;
      const settle = 1 - Math.exp(-c.attitude_response_s * dt);
      f.roll += (0 - f.roll) * settle; f.pitch += (0 - f.pitch) * settle;
      f.v = f.vx * Math.cos(f.heading) + f.vy * Math.sin(f.heading);
    }

    stepControl() {
      const counts = this.groupCounts;
      for (let step = 0; step < this.controlSteps; ++step) {
        counts.fill(0);
        for (const spike of this.net.step()) {
          const mask = this.groupMask[spike];
          if (mask) for (let bit = 0; bit < counts.length; ++bit) if (mask & (1 << bit)) counts[bit]++;
        }
        for (let bit = 0; bit < counts.length; ++bit) this.groups[this.groupNames[bit]].push(counts[bit]);
      }

      const c = this.flightCfg, m = this.cfg.arena.motor, g = this.groups, f = this.fly;
      const dt = this.controlSteps * this.net.dt * 1e-3;
      const fwd = 0.5 * (g.fwd_L.hz + g.fwd_R.hz);
      const bwd = 0.5 * (g.bwd_L.hz + g.bwd_R.hz);
      const turn = g.turn_L.hz - g.turn_R.hz;
      const maxTurn = Math.max(1e-6, m.max_turn_deg_s);
      const turnNorm = clamp(m.turn_sign * m.turn_gain * turn / maxTurn, -1, 1);
      const gf = g.GF.hz;

      if (!f.takeoffArmed && gf <= c.gf_reset_hz) f.takeoffArmed = true;
      if (!f.airborne && f.takeoffArmed && gf >= c.gf_takeoff_hz) {
        f.airborne = true; f.takeoffArmed = false;
        f.wingDrive = Math.max(f.wingDrive, clamp(0.93 + 0.07 * gf / c.gf_full_scale_hz, 0, 1));
        f.wingPhase = 0;
      }

      const wasAirborne = f.airborne;
      if (f.airborne) {
        const wing = this.updateWingPlant(dt, gf, fwd, bwd, turnNorm);
        this.updateFlight(dt, fwd, bwd, turnNorm, wing);
      } else {
        this.updateWalking(dt, fwd, bwd, turnNorm);
      }

      this.sense();
      this.controlIndex++;
      let key = null;
      if (!f.airborne) {
        const keyIndex = this.keyAt(f.x, f.y);
        if (keyIndex !== this.lastKey) {
          key = [this.net.timeS, keyIndex, this.layout[keyIndex]];
          this.lastKey = keyIndex; this.keyCount++;
        } else if (wasAirborne) this.lastKey = keyIndex;
      }
      return {sample: this.controlIndex % this.logEveryControl === 0 ? this.sample() : null, key};
    }

    get metadata() {
      return {...super.metadata, flight: {...this.flightCfg}};
    }
  }

  root.FlyHamletLive = {...api, BaseFlyArena, FlyArena: FlightArena};
  if (typeof module !== 'undefined' && module.exports) module.exports = root.FlyHamletLive;
})(globalThis);
