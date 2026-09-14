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
  const modulo = (value, modulus) => ((value % modulus) + modulus) % modulus;
  const overscan = 64;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scriptURL = document.currentScript?.src || new URL('site/flight-view.js', document.baseURI).href;
  const sprite = new Image();
  sprite.src = new URL('assets/fruit-fly.png', scriptURL).href;
  const shadowSprite = new Image();
  shadowSprite.src = new URL('assets/fly-shadow.png', scriptURL).href;
  const wingSheet = new Image();
  wingSheet.src = new URL('assets/fly-wings.png', scriptURL).href;
  const ready = (image) => image.complete && image.naturalWidth > 0;

  /* fly-wings.png is a 4 x 3 grid of wing pairs. The 12 cells are one wingbeat,
   * read left to right along each row. Cell 0 is the top of the upstroke.
   * Each cell holds both wings, head up, so a cell is clipped to one half to
   * give the left and right wing their own stroke phase. */
  const WING_SHEET_COLS = 4;
  const WING_SHEET_ROWS = 3;
  const WING_SHEET_FRAMES = WING_SHEET_COLS * WING_SHEET_ROWS;
  const WING_SHEET_SPAN = 0.86;      // pair width, as a fraction of the body size
  const WING_SHEET_ANCHOR_Y = -0.05; // hinge height on the thorax, fraction of size
  const SHADOW_SPRITE_GAIN = 1.7;    // the sprite is softer than the old gradient
  let metadata = null, latestSample = null, sampleReceivedAt = 0;
  let renderSample = null;
  let lastFrameAt = 0;

  function lerpAngle(a, b, t) {
    let d = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI;
    if (d < -Math.PI) d += Math.PI * 2;
    return a + d * t;
  }

  function smoothSample(target, dt) {
    if (!target) return null;
    if (!renderSample) {
      renderSample = {...target};
      return renderSample;
    }
    const a = 1 - Math.exp(-dt / 0.045);
    renderSample.x += (target.x - renderSample.x) * a;
    renderSample.y += (target.y - renderSample.y) * a;
    renderSample.z_mm += ((target.z_mm || 0) - (renderSample.z_mm || 0)) * a;
    renderSample.vx_mm_s += ((target.vx_mm_s || 0) - (renderSample.vx_mm_s || 0)) * a;
    renderSample.vy_mm_s += ((target.vy_mm_s || 0) - (renderSample.vy_mm_s || 0)) * a;
    renderSample.vz_mm_s += ((target.vz_mm_s || 0) - (renderSample.vz_mm_s || 0)) * a;
    renderSample.pitch_deg += ((target.pitch_deg || 0) - (renderSample.pitch_deg || 0)) * a;
    renderSample.roll_deg += ((target.roll_deg || 0) - (renderSample.roll_deg || 0)) * a;
    renderSample.heading_deg = lerpAngle(
      (renderSample.heading_deg || 0) * Math.PI / 180,
      (target.heading_deg || 0) * Math.PI / 180,
      a
    ) * 180 / Math.PI;

    renderSample.wing_phase_rad = target.wing_phase_rad;
    renderSample.wing_frequency_hz = target.wing_frequency_hz;
    renderSample.wing_left_amplitude = target.wing_left_amplitude;
    renderSample.wing_right_amplitude = target.wing_right_amplitude;
    renderSample.gait_phase_rad = target.gait_phase_rad;
    renderSample.gait_frequency_hz = target.gait_frequency_hz;
    renderSample.gait_duty_factor = target.gait_duty_factor;
    renderSample.airborne = target.airborne;
    return renderSample;
  }

  const NativeWorker = window.Worker;
  if (NativeWorker) {
    window.Worker = class FlyHamletObservedWorker extends NativeWorker {
      constructor(url, options) {
        const liveWorker = String(url).includes('live-worker.js');
        let workerURL = url;
        if (liveWorker) {
          const versioned = new URL(url, document.baseURI);
          versioned.searchParams.set('v', '6');
          workerURL = versioned;
        }
        super(workerURL, options);
        if (liveWorker) {
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
    if (ready(shadowSprite)) {
      ctx.globalAlpha = Math.min(1, alpha * SHADOW_SPRITE_GAIN);
      ctx.drawImage(shadowSprite, -radius, -radius, radius * 2, radius * 2);
    } else {
      const shadow = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
      shadow.addColorStop(0, `rgba(35,25,12,${alpha})`); shadow.addColorStop(0.48, `rgba(35,25,12,${alpha * 0.55})`); shadow.addColorStop(1, 'rgba(35,25,12,0)');
      ctx.fillStyle = shadow; ctx.beginPath(); ctx.arc(0, 0, radius, 0, TAU); ctx.fill();
    }
    ctx.restore();
  }

  function drawWingSprite(side, size, phase, amplitude, alpha) {
    if (amplitude <= 0.01) return;
    const cellW = wingSheet.naturalWidth / WING_SHEET_COLS;
    const cellH = wingSheet.naturalHeight / WING_SHEET_ROWS;
    const frame = Math.min(WING_SHEET_FRAMES - 1,
      Math.floor(modulo(phase, TAU) / TAU * WING_SHEET_FRAMES));
    const sx = (frame % WING_SHEET_COLS) * cellW;
    const sy = Math.floor(frame / WING_SHEET_COLS) * cellH;

    const drawW = size * WING_SHEET_SPAN * (0.92 + 0.08 * amplitude);
    const drawH = drawW * cellH / cellW;
    const top = WING_SHEET_ANCHOR_Y * size - drawH / 2;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.rect(side < 0 ? -drawW : 0, top - drawH, drawW, drawH * 3);
    ctx.clip();
    ctx.drawImage(wingSheet, sx, sy, cellW, cellH, -drawW / 2, top, drawW, drawH);
    ctx.restore();
  }

  function drawWings(sample, size) {
    // The body sprite already carries a pair of wings. Animate a second pair
    // only when fly-wings.png is available, so the fly never grows four wings.
    if (!ready(wingSheet)) return;
    const airborne = Boolean(sample.airborne) || (sample.z_mm || 0) > 0.03;
    if (!airborne) return;

    const phase = sample.wing_phase_rad || 0;
    const frequency = sample.wing_frequency_hz || 0;
    const leftAmp = sample.wing_left_amplitude ?? sample.wing_power ?? 0;
    const rightAmp = sample.wing_right_amplitude ?? sample.wing_power ?? 0;

    if (reducedMotion.matches || frequency <= 0) {
      drawWingSprite(-1, size, phase, leftAmp, 0.42);
      drawWingSprite(1, size, phase, rightAmp, 0.42);
      return;
    }

    const exposure = 1 / 90;
    const ghosts = 6;
    for (let i = ghosts - 1; i >= 0; --i) {
      const p = phase - TAU * frequency * exposure * i / (ghosts - 1);
      const a = i === 0 ? 0.34 : 0.05 + 0.07 * (1 - i / ghosts);
      drawWingSprite(-1, size, p, leftAmp, a);
      drawWingSprite(1, size, p, rightAmp, a);
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
    drawWings(sample, size);
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
    const dt = lastFrameAt ? Math.min(0.05, (timestamp - lastFrameAt) / 1000) : 0;
    lastFrameAt = timestamp;
    const projected = projectedSample(timestamp);
    const sample = smoothSample(projected, dt);
    syncContact(sample);
    drawFly(sample, dimensions);
  }

  sessionSelect.addEventListener('change', () => {
    renderSample = null;
    if (isLive()) { metadata = null; latestSample = null; } else overlay.hidden = true;
  });
  if ('ResizeObserver' in window) new ResizeObserver(fitCanvas).observe(stage);
  else window.addEventListener('resize', fitCanvas);
  requestAnimationFrame(frame);
})();
