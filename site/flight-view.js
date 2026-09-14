/* Live 3D rendering for the connectome-driven flight and walking model.
 * Position, wingbeat and gait phases come from worker samples; this file projects them.
 */
(() => {
  'use strict';

  const arena = document.getElementById('arena');
  const stage = arena?.closest('.arena-stage');
  const keyGrid = document.getElementById('key-grid');
  const sessionSelect = document.getElementById('fly');
  const play = document.getElementById('play');
  const speedSelect = document.getElementById('speed');
  const currentKey = document.getElementById('current-key');
  if (!arena || !stage || !keyGrid || !sessionSelect) return;

  const TAU = 2 * Math.PI;
  const overscan = 64;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scriptURL = document.currentScript?.src || new URL('site/flight-view.js', document.baseURI).href;
  const sprite = new Image();
  sprite.src = new URL('assets/fruit-fly.png', scriptURL).href;
  let metadata = null, latestSample = null, sampleReceivedAt = 0;

  const NativeWorker = window.Worker;
  if (NativeWorker) {
    window.Worker = class FlyHamletObservedWorker extends NativeWorker {
      constructor(url, options) {
        super(url, options);
        if (String(url).includes('live-worker.js')) {
          this.addEventListener('message', (event) => {
            const message = event.data || {};
            if (message.type === 'ready') {
              metadata = message.metadata || null;
              latestSample = message.initial || null;
              sampleReceivedAt = performance.now();
            } else if (message.type === 'batch' && message.samples?.length) {
              latestSample = message.samples[message.samples.length - 1];
              sampleReceivedAt = performance.now();
            }
          });
        }
      }
    };
  }

  const overlay = document.createElement('canvas');
  overlay.className = 'flight-overlay';
  overlay.setAttribute('aria-hidden', 'true');
  stage.appendChild(overlay);
  const ctx = overlay.getContext('2d');
  const style = document.createElement('style');
  style.textContent = `.flight-overlay{position:absolute;inset:-${overscan}px;width:calc(100% + ${overscan * 2}px);height:calc(100% + ${overscan * 2}px);display:block;pointer-events:none;z-index:4}`;
  document.head.appendChild(style);

  const isLive = () => sessionSelect.value === 'live';
  const arenaContext = arena.getContext('2d');
  if (arenaContext) {
    const originalDrawImage = arenaContext.drawImage.bind(arenaContext);
    arenaContext.drawImage = (image, ...args) => {
      const source = image?.currentSrc || image?.src || '';
      const suppress = isLive() && latestSample && sprite.complete && sprite.naturalWidth > 0;
      if (suppress && source.includes('fruit-fly.png')) return;
      originalDrawImage(image, ...args);
    };
    const originalFill = arenaContext.fill.bind(arenaContext);
    arenaContext.fill = (...args) => {
      const suppress = isLive() && latestSample && sprite.complete && sprite.naturalWidth > 0;
      if (suppress && typeof arenaContext.fillStyle !== 'string') return;
      originalFill(...args);
    };
  }

  function fitCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = overlay.clientWidth, height = overlay.clientHeight;
    const pw = Math.round(width * ratio), ph = Math.round(height * ratio);
    if (overlay.width !== pw || overlay.height !== ph) { overlay.width = pw; overlay.height = ph; }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return {width, height};
  }

  function projectedSample(timestamp) {
    if (!latestSample) return null;
    const sample = {...latestSample};
    if (play?.getAttribute('aria-pressed') !== 'true' || reducedMotion.matches) return sample;
    const rate = Math.max(0.1, Number(speedSelect?.value) || 1);
    const dt = Math.min(0.05, Math.max(0, timestamp - sampleReceivedAt) / 1000 * rate);
    sample.x += (sample.vx_mm_s || 0) * dt;
    sample.y += (sample.vy_mm_s || 0) * dt;
    sample.z_mm = Math.max(0, (sample.z_mm || 0) + (sample.vz_mm_s || 0) * dt);
    sample.heading_deg += (sample.omega_deg_s || 0) * dt;
    sample.wing_phase_rad = ((sample.wing_phase_rad || 0) + TAU * (sample.wing_frequency_hz || 0) * dt) % TAU;
    sample.gait_phase_rad = ((sample.gait_phase_rad || 0) + TAU * (sample.gait_frequency_hz || 0) * dt) % TAU;
    return sample;
  }

  function syncContact(sample) {
    if (!metadata || !sample || !isLive()) return;
    const airborne = Boolean(sample.airborne) || (sample.z_mm || 0) > 0.03;
    if (airborne) {
      keyGrid.querySelectorAll('.keycap.is-active').forEach(key => key.classList.remove('is-active'));
      if (currentKey) currentKey.textContent = '—';
      return;
    }
    const col = Math.max(0, Math.min(metadata.cols - 1, Math.trunc(sample.x / metadata.W * metadata.cols)));
    const row = Math.max(0, Math.min(metadata.rows - 1, Math.trunc(sample.y / metadata.H * metadata.rows)));
    const index = row * metadata.cols + col;
    keyGrid.querySelectorAll('.keycap.is-active').forEach(key => {
      if (Number(key.dataset.index) !== index) key.classList.remove('is-active');
    });
    keyGrid.querySelector(`.keycap[data-index="${index}"]`)?.classList.add('is-active');
    const character = metadata.layout?.[index];
    if (currentKey && character != null) currentKey.textContent = character === ' ' ? 'Space' : character.toUpperCase();
  }

  function drawShadow(x, y, size, altitudeFraction, heading) {
    const alpha = 0.28 * (1 - 0.78 * altitudeFraction);
    const radius = size * (0.28 + 0.18 * altitudeFraction);
    ctx.save(); ctx.translate(x + size * 0.09 * altitudeFraction, y + size * 0.12 * altitudeFraction);
    ctx.rotate(Math.PI / 2 - heading); ctx.scale(1.35 + altitudeFraction * 0.45, 0.58 + altitudeFraction * 0.12);
    const shadow = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
    shadow.addColorStop(0, `rgba(35,25,12,${alpha})`); shadow.addColorStop(0.48, `rgba(35,25,12,${alpha * 0.55})`); shadow.addColorStop(1, 'rgba(35,25,12,0)');
    ctx.fillStyle = shadow; ctx.beginPath(); ctx.arc(0, 0, radius, 0, TAU); ctx.fill(); ctx.restore();
  }

  function drawWing(side, size, phase, amplitude, alpha) {
    if (amplitude <= 0.01) return;
    const maxStroke = (metadata?.flight?.wing_stroke_amplitude_deg || 72) * Math.PI / 180;
    const stroke = Math.sin(phase) * maxStroke * amplitude;
    const baseX = side * size * 0.065, baseY = -size * 0.015;
    ctx.save(); ctx.translate(baseX, baseY); ctx.rotate(side * (0.72 + stroke));
    ctx.globalAlpha = alpha;
    ctx.fillStyle = 'rgba(225,232,224,.62)'; ctx.strokeStyle = 'rgba(88,87,73,.50)'; ctx.lineWidth = Math.max(0.7, size * 0.008);
    ctx.beginPath(); ctx.moveTo(0, 0);
    ctx.bezierCurveTo(side * size * 0.06, -size * 0.08, side * size * 0.22, -size * 0.19, side * size * 0.29, -size * 0.09);
    ctx.bezierCurveTo(side * size * 0.27, size * 0.02, side * size * 0.11, size * 0.09, 0, 0);
    ctx.fill(); ctx.stroke(); ctx.restore();
  }

  function drawWings(sample, size) {
    const airborne = Boolean(sample.airborne) || (sample.z_mm || 0) > 0.03;
    if (!airborne) return;
    const phase = sample.wing_phase_rad || 0;
    const frequency = sample.wing_frequency_hz || 0;
    const leftAmp = sample.wing_left_amplitude ?? sample.wing_power ?? 0;
    const rightAmp = sample.wing_right_amplitude ?? sample.wing_power ?? 0;
    if (reducedMotion.matches || frequency <= 0) {
      drawWing(-1, size, phase, leftAmp, 0.5); drawWing(1, size, phase, rightAmp, 0.5); return;
    }
    // A display cannot resolve ~200 Hz directly. Integrate several physical wing
    // poses across one exposure; this is motion blur from the simulated oscillator,
    // while the newest pose still shows its exact phase.
    const exposure = 1 / 120;
    const ghosts = 5;
    for (let i = ghosts - 1; i >= 0; --i) {
      const pastPhase = phase - TAU * frequency * exposure * i / (ghosts - 1);
      const alpha = i === 0 ? 0.52 : 0.06 + 0.12 * (1 - i / ghosts);
      drawWing(-1, size, pastPhase, leftAmp, alpha); drawWing(1, size, pastPhase, rightAmp, alpha);
    }
  }

  const LEG_LAYOUT = [
    {side:-1, y:-0.13, phase:0}, {side:-1, y:0.00, phase:Math.PI}, {side:-1, y:0.14, phase:0},
    {side:1, y:-0.13, phase:Math.PI}, {side:1, y:0.00, phase:0}, {side:1, y:0.14, phase:Math.PI},
  ];

  function drawLegs(sample, size) {
    const airborne = Boolean(sample.airborne) || (sample.z_mm || 0) > 0.03;
    const gaitPhase = sample.gait_phase_rad || 0;
    const duty = Math.max(0.45, Math.min(0.8, sample.gait_duty_factor || 0.6));
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (const leg of LEG_LAYOUT) {
      const phase = ((gaitPhase + leg.phase) % TAU + TAU) % TAU;
      const cycle = phase / TAU;
      const stance = !airborne && cycle < duty;
      let sweep;
      if (airborne) sweep = 0.12;
      else if (stance) sweep = 0.16 - 0.32 * cycle / duty;
      else sweep = -0.16 + 0.32 * (cycle - duty) / (1 - duty);
      const lift = airborne ? 0.11 : (stance ? 0 : Math.sin(Math.PI * (cycle - duty) / (1 - duty)) * 0.10);
      const hipX = leg.side * size * 0.055, hipY = leg.y * size;
      const footX = leg.side * size * (airborne ? 0.19 : 0.31);
      const footY = hipY + sweep * size;
      const kneeX = leg.side * size * (airborne ? 0.13 : 0.18);
      const kneeY = (hipY + footY) * 0.5 - lift * size;
      ctx.strokeStyle = stance ? 'rgba(63,48,34,.90)' : 'rgba(76,57,39,.72)';
      ctx.lineWidth = Math.max(1, size * 0.013);
      ctx.beginPath(); ctx.moveTo(hipX, hipY); ctx.lineTo(kneeX, kneeY); ctx.lineTo(footX, footY); ctx.stroke();
    }
  }

  function drawFallback(size) {
    ctx.fillStyle = '#795633'; ctx.beginPath(); ctx.ellipse(0, size * 0.06, size * 0.085, size * 0.23, 0, 0, TAU); ctx.fill();
    ctx.fillStyle = '#853e26'; for (const side of [-1,1]) { ctx.beginPath(); ctx.ellipse(side * size * .057, -size * .17, size * .053, size * .069, side * .2, 0, TAU); ctx.fill(); }
  }

  function drawFly(sample, dimensions) {
    const {width, height} = dimensions;
    const innerWidth = Math.max(1, width - overscan * 2), innerHeight = Math.max(1, height - overscan * 2);
    const heading = (sample.heading_deg || 0) * Math.PI / 180;
    const pitch = (sample.pitch_deg || 0) * Math.PI / 180, roll = (sample.roll_deg || 0) * Math.PI / 180;
    const ceiling = Math.max(1, metadata?.flight?.ceiling_mm || 12);
    const z = Math.max(0, Math.min(ceiling, sample.z_mm || 0)), altitudeFraction = z / ceiling;
    const groundX = overscan + sample.x / metadata.W * innerWidth;
    const groundY = height - overscan - sample.y / metadata.H * innerHeight;
    const bodyX = groundX, bodyY = groundY - z / metadata.H * innerHeight * 0.62;
    const size = Math.max(54, Math.min(112, innerWidth * 0.14)), scale = 1 + altitudeFraction * 0.16;
    drawShadow(groundX, groundY, size, altitudeFraction, heading);

    ctx.save(); ctx.translate(bodyX, bodyY); ctx.rotate(Math.PI / 2 - heading);
    ctx.scale(Math.max(0.56, Math.cos(roll)) * scale, Math.max(0.62, Math.cos(pitch)) * scale);
    drawWings(sample, size); drawLegs(sample, size);
    if (sprite.complete && sprite.naturalWidth > 0) {
      const sh = size * sprite.naturalHeight / sprite.naturalWidth;
      ctx.drawImage(sprite, -size / 2, -sh / 2, size, sh);
    } else drawFallback(size);
    ctx.restore();
  }

  function frame(timestamp) {
    requestAnimationFrame(frame);
    const dimensions = fitCanvas(); ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    if (!isLive() || !metadata || !latestSample) { overlay.hidden = true; return; }
    overlay.hidden = false;
    const sample = projectedSample(timestamp); syncContact(sample); drawFly(sample, dimensions);
  }

  sessionSelect.addEventListener('change', () => {
    if (isLive()) { metadata = null; latestSample = null; } else overlay.hidden = true;
  });
  if ('ResizeObserver' in window) new ResizeObserver(fitCanvas).observe(stage);
  else window.addEventListener('resize', fitCanvas);
  requestAnimationFrame(frame);
})();
