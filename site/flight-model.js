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
    // A flight bout must end. Walking commands alone must not hold the fly up,
    // or a takeoff becomes a cruise it never leaves. The locomotor share of the
    // wing drive fades over the bout; only the giant fibre renews it.
    flight_locomotor_gain: 0.78,
    flight_sustain_s: 0.8,
    wing_turn_asymmetry: 0.22,
    wing_force_response_s: 0.030,
    wing_torque_response_s: 0.040,
    // The old per-stroke term (0.30 + 0.70 cos^2) averaged 0.65 over a cycle.
    // The cycle-averaged force has no such term, so carry the 0.65 here and
    // keep the lift the model was tuned around: 21000 * 0.65 = 13650.
    max_wing_accel_mm_s2: 13650,
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
    // A quiet brain should leave the fly standing, not creeping. This overrides
    // arena.motor.base_speed_mm_s for the live view only; the recorded
    // experiments in flyhamlet/ keep their own value.
    ground_base_speed_mm_s: 0,
    // Real flies fly in straight runs broken by saccades: fast body turns of
    // tens of degrees in tens of milliseconds. Part of the turn demand still
    // steers continuously; the rest charges the next saccade.
    yaw_smooth_share: 0.25,
    saccade_charge_rate_deg_s: 400,
    saccade_trigger_deg: 22,
    saccade_max_deg: 90,
    saccade_duration_s: 0.055,
    saccade_refractory_s: 0.09,
    saccade_stop_drag_s: 60,
    saccade_charge_decay_s: 0.5,
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
      f.wingForce = 0;
      f.wingTorque = 0;
      f.gaitPhase = 0; f.gaitFrequency = 0; f.gaitDuty = this.flightCfg.gait_duty_slow;
      f.airborne = false; f.takeoffArmed = true;
      f.airborneTime = 0;
      f.turnCharge = 0; f.saccadeLeft = 0; f.saccadeOmega = 0; f.saccadeCooldown = 0;
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
      const sustain = Math.exp(-f.airborneTime / Math.max(1e-3, c.flight_sustain_s));
      const targetDrive = f.airborne
        ? clamp(Math.max(gfDrive, c.flight_locomotor_gain * locomotorDrive * sustain), 0, 1)
        : 0;

      const tau = targetDrive > f.wingDrive ? c.wing_rise_s : c.wing_decay_s;
      f.wingDrive += (targetDrive - f.wingDrive) * (1 - Math.exp(-dt / Math.max(1e-4, tau)));
      if (f.wingDrive < 1e-4) f.wingDrive = 0;

      f.wingFrequency = f.wingDrive > 0
        ? c.wing_frequency_min_hz + (c.wing_frequency_max_hz - c.wing_frequency_min_hz) * f.wingDrive
        : 0;
      f.wingPhase = modulo(f.wingPhase + TAU * f.wingFrequency * dt, TAU);

      f.leftWingAmplitude = clamp(f.wingDrive * (1 - c.wing_turn_asymmetry * turnNorm), 0, 1);
      f.rightWingAmplitude = clamp(f.wingDrive * (1 + c.wing_turn_asymmetry * turnNorm), 0, 1);

      const frequencyScale = f.wingFrequency / c.wing_frequency_max_hz;
      const meanAmp = 0.5 * (f.leftWingAmplitude + f.rightWingAmplitude);
      const targetForce = c.max_wing_accel_mm_s2 * meanAmp * meanAmp * frequencyScale * frequencyScale;
      const targetTorque = (f.rightWingAmplitude - f.leftWingAmplitude) * frequencyScale;

      f.wingForce += (targetForce - f.wingForce)
        * (1 - Math.exp(-dt / c.wing_force_response_s));
      f.wingTorque += (targetTorque - f.wingTorque)
        * (1 - Math.exp(-dt / c.wing_torque_response_s));

      return {totalForce: f.wingForce, turnTorque: f.wingTorque};
    }

    /* Steering in flight. A quarter of the turn demand steers continuously; the
     * rest charges a saccade, which fires as a short burst of high yaw rate.
     * A steady torque alone draws smooth arcs, and smooth arcs are the clearest
     * sign that this is not a fly. */
    updateSaccade(dt, turnNorm, turnTorque) {
      const c = this.flightCfg, f = this.fly;
      const charge = () => {
        f.turnCharge += turnNorm * c.saccade_charge_rate_deg_s * dt;
        f.turnCharge *= Math.exp(-dt / Math.max(1e-3, c.saccade_charge_decay_s));
      };

      if (f.saccadeLeft > 0) {
        f.saccadeLeft = Math.max(0, f.saccadeLeft - dt);
        f.omega = f.saccadeOmega;
        if (f.saccadeLeft === 0) f.saccadeCooldown = c.saccade_refractory_s;
        return;
      }

      if (f.saccadeCooldown > 0) {
        // A saccade is stopped by counter-torque. Left to the ordinary yaw drag
        // the body coasts for 140 ms and the turn never reads as a burst.
        f.saccadeCooldown = Math.max(0, f.saccadeCooldown - dt);
        f.omega -= c.saccade_stop_drag_s * f.omega * dt;
        charge();
        return;
      }

      charge();

      if (Math.abs(f.turnCharge) >= c.saccade_trigger_deg) {
        const amplitude = clamp(f.turnCharge, -c.saccade_max_deg, c.saccade_max_deg);
        f.saccadeLeft = c.saccade_duration_s;
        f.saccadeOmega = amplitude * Math.PI / 180 / c.saccade_duration_s;
        f.turnCharge = 0;
        f.omega = f.saccadeOmega;
        return;
      }

      f.omega += (c.yaw_accel_rad_s2 * turnTorque * c.yaw_smooth_share
        - c.yaw_drag_s * f.omega) * dt;
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

      f.airborneTime += dt;
      this.updateSaccade(dt, turnNorm, wing.turnTorque);
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
          f.airborneTime = 0;
          f.turnCharge = 0; f.saccadeLeft = 0; f.saccadeCooldown = 0;
          f.wingDrive = 0; f.wingFrequency = 0;
          f.leftWingAmplitude = 0; f.rightWingAmplitude = 0;
          f.wingForce = 0; f.wingTorque = 0;
          f.roll *= 0.35; f.pitch *= 0.35;
        }
      }
      f.v = f.vx * Math.cos(f.heading) + f.vy * Math.sin(f.heading);
    }

    updateWalking(dt, fwd, bwd, turnNorm) {
      const c = this.flightCfg, m = this.cfg.arena.motor, f = this.fly;
      const baseSpeed = c.ground_base_speed_mm_s ?? m.base_speed_mm_s;
      const desiredSpeed = clamp(baseSpeed + m.speed_gain * fwd - m.backward_gain * bwd,
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
      f.wingForce = 0; f.wingTorque = 0;
      f.airborneTime = 0; f.saccadeLeft = 0;
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
