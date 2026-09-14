/* Connectome-driven 3D locomotion layered onto FlyHamlet's brain arena.
 * Descending-neuron activity supplies motor commands. A deterministic downstream
 * plant stands in for the VNC, muscles and aerodynamics that are outside FAFB.
 * No key, destination, or flight path is selected by this code.
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
    takeoff_impulse_mm_s: 260,
    gravity_mm_s2: 9810,
    max_lift_mm_s2: 15000,
    wing_rise_s: 0.025,
    wing_decay_s: 0.50,
    vertical_drag_s: 4.5,
    air_speed_gain: 1.4,
    air_backward_gain: 1.0,
    max_air_speed_mm_s: 45,
    air_velocity_response_s: 7,
    yaw_response_s: 10,
    max_bank_deg: 38,
    max_pitch_deg: 24,
    attitude_response_s: 10,
    wall_restitution: 0.35,
    ceiling_restitution: 0.15,
  });

  class FlightArena extends BaseFlyArena {
    constructor(connectome, manifest, seed) {
      super(connectome, manifest, seed);
      this.flightCfg = {...DEFAULT_FLIGHT, ...(this.cfg.arena.flight || {})};
      const f = this.fly;
      f.z = 0;
      f.vz = 0;
      f.vx = f.v * Math.cos(f.heading);
      f.vy = f.v * Math.sin(f.heading);
      f.pitch = 0;
      f.roll = 0;
      f.wingPower = 0;
      f.airborne = false;
      f.takeoffArmed = true;
    }

    sample() {
      const sample = super.sample();
      const f = this.fly;
      return {...sample,
        z_mm: f.z,
        vz_mm_s: f.vz,
        vx_mm_s: f.vx,
        vy_mm_s: f.vy,
        pitch_deg: f.pitch * 180 / Math.PI,
        roll_deg: f.roll * 180 / Math.PI,
        wing_power: f.wingPower,
        airborne: f.airborne};
    }

    stepControl() {
      const counts = this.groupCounts;
      for (let step = 0; step < this.controlSteps; ++step) {
        counts.fill(0);
        for (const spike of this.net.step()) {
          const mask = this.groupMask[spike];
          if (mask) {
            for (let bit = 0; bit < counts.length; ++bit) {
              if (mask & (1 << bit)) counts[bit]++;
            }
          }
        }
        for (let bit = 0; bit < counts.length; ++bit) {
          this.groups[this.groupNames[bit]].push(counts[bit]);
        }
      }

      const m = this.cfg.arena.motor;
      const c = this.flightCfg;
      const g = this.groups;
      const f = this.fly;
      const dt = this.controlSteps * this.net.dt * 1e-3;
      const turn = g.turn_L.hz - g.turn_R.hz;
      const fwd = 0.5 * (g.fwd_L.hz + g.fwd_R.hz);
      const bwd = 0.5 * (g.bwd_L.hz + g.bwd_R.hz);
      const gf = g.GF.hz;
      const targetOmega = clamp(m.turn_sign * m.turn_gain * Math.PI / 180 * turn,
        -m.max_turn_deg_s * Math.PI / 180, m.max_turn_deg_s * Math.PI / 180);

      if (!f.takeoffArmed && gf <= c.gf_reset_hz) f.takeoffArmed = true;
      if (!f.airborne && f.takeoffArmed && gf >= c.gf_takeoff_hz) {
        f.airborne = true;
        f.takeoffArmed = false;
        f.vz = Math.max(f.vz, c.takeoff_impulse_mm_s * clamp(gf / c.gf_full_scale_hz, 0.65, 1));
        f.wingPower = Math.max(f.wingPower, clamp(0.55 + 0.25 * gf / c.gf_full_scale_hz, 0, 1));
        f.vx = f.v * Math.cos(f.heading);
        f.vy = f.v * Math.sin(f.heading);
      }

      let key = null;
      const wasAirborne = f.airborne;
      if (f.airborne) {
        const yawAlpha = 1 - Math.exp(-c.yaw_response_s * dt);
        f.omega += (targetOmega - f.omega) * yawAlpha;
        f.heading = modulo(f.heading + f.omega * dt + Math.PI, TAU) - Math.PI;

        const signedAirSpeed = clamp(c.air_speed_gain * fwd - c.air_backward_gain * bwd,
          -c.max_air_speed_mm_s, c.max_air_speed_mm_s);
        const targetVx = signedAirSpeed * Math.cos(f.heading);
        const targetVy = signedAirSpeed * Math.sin(f.heading);
        const velocityAlpha = 1 - Math.exp(-c.air_velocity_response_s * dt);
        const oldVx = f.vx;
        const oldVy = f.vy;
        f.vx += (targetVx - f.vx) * velocityAlpha;
        f.vy += (targetVy - f.vy) * velocityAlpha;

        const gfDrive = clamp(gf / c.gf_full_scale_hz, 0, 1);
        const locomotorDrive = clamp((fwd + bwd + 0.25 * (g.turn_L.hz + g.turn_R.hz)) /
          c.motor_full_scale_hz, 0, 1);
        const wingTarget = clamp(Math.max(gfDrive, 0.72 * locomotorDrive), 0, 1);
        const wingTau = wingTarget > f.wingPower ? c.wing_rise_s : c.wing_decay_s;
        const wingAlpha = 1 - Math.exp(-dt / Math.max(1e-4, wingTau));
        f.wingPower += (wingTarget - f.wingPower) * wingAlpha;

        const verticalAcceleration = c.max_lift_mm_s2 * f.wingPower - c.gravity_mm_s2
          - c.vertical_drag_s * f.vz;
        f.vz += verticalAcceleration * dt;
        f.z += f.vz * dt;

        const nx = f.x + f.vx * dt;
        const ny = f.y + f.vy * dt;
        if (nx < 0 || nx > this.width) {
          f.vx = -f.vx * c.wall_restitution;
          f.x = clamp(nx, 0, this.width);
          f.heading = modulo(Math.atan2(f.vy, f.vx) + Math.PI, TAU) - Math.PI;
          this.wallSteps++;
        } else f.x = nx;
        if (ny < 0 || ny > this.height) {
          f.vy = -f.vy * c.wall_restitution;
          f.y = clamp(ny, 0, this.height);
          f.heading = modulo(Math.atan2(f.vy, f.vx) + Math.PI, TAU) - Math.PI;
          this.wallSteps++;
        } else f.y = ny;

        if (f.z >= c.ceiling_mm) {
          f.z = c.ceiling_mm;
          if (f.vz > 0) f.vz = -f.vz * c.ceiling_restitution;
        }

        const forwardAcceleration = ((f.vx - oldVx) * Math.cos(f.heading)
          + (f.vy - oldVy) * Math.sin(f.heading)) / Math.max(dt, 1e-6);
        const bankFraction = clamp(f.omega / (m.max_turn_deg_s * Math.PI / 180), -1, 1);
        const pitchFraction = clamp(-forwardAcceleration / 1200, -1, 1);
        const rollTarget = -bankFraction * c.max_bank_deg * Math.PI / 180;
        const pitchTarget = pitchFraction * c.max_pitch_deg * Math.PI / 180;
        const attitudeAlpha = 1 - Math.exp(-c.attitude_response_s * dt);
        f.roll += (rollTarget - f.roll) * attitudeAlpha;
        f.pitch += (pitchTarget - f.pitch) * attitudeAlpha;

        if (f.z <= 0 && f.vz <= 0) {
          f.z = 0;
          f.vz = 0;
          f.airborne = false;
          f.wingPower = 0;
          f.roll *= 0.35;
          f.pitch *= 0.35;
          f.v = f.vx * Math.cos(f.heading) + f.vy * Math.sin(f.heading);
        } else {
          f.v = f.vx * Math.cos(f.heading) + f.vy * Math.sin(f.heading);
        }
      } else {
        f.omega = targetOmega;
        f.v = clamp(m.base_speed_mm_s + m.speed_gain * fwd - m.backward_gain * bwd,
          -m.max_speed_mm_s, m.max_speed_mm_s);
        f.heading = modulo(f.heading + f.omega * dt + Math.PI, TAU) - Math.PI;
        const nx = f.x + f.v * Math.cos(f.heading) * dt;
        const ny = f.y + f.v * Math.sin(f.heading) * dt;
        if (!(0 <= nx && nx <= this.width && 0 <= ny && ny <= this.height)) this.wallSteps++;
        f.x = clamp(nx, 0, this.width);
        f.y = clamp(ny, 0, this.height);
        f.vx = f.v * Math.cos(f.heading);
        f.vy = f.v * Math.sin(f.heading);
        f.z = 0;
        f.vz = 0;
        f.wingPower = 0;
        const settle = 1 - Math.exp(-c.attitude_response_s * dt);
        f.roll += (0 - f.roll) * settle;
        f.pitch += (0 - f.pitch) * settle;
      }

      this.sense();
      this.controlIndex++;

      // Keys are physical contacts. Crossing a region in flight does not type it;
      // walking into it or touching down on it does.
      if (!f.airborne) {
        const keyIndex = this.keyAt(f.x, f.y);
        if (keyIndex !== this.lastKey) {
          key = [this.net.timeS, keyIndex, this.layout[keyIndex]];
          this.lastKey = keyIndex;
          this.keyCount++;
        } else if (wasAirborne) {
          // Landing on the same key is a contact but not a new character, matching
          // the existing "enter a new key region" semantics.
          this.lastKey = keyIndex;
        }
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
