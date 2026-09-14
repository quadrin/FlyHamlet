/* Live 3D rendering for the connectome-driven flight model.
 * Position and attitude come from worker samples; this file only projects them.
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

  const overscan = 64;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scriptURL = document.currentScript?.src || new URL('site/flight-view.js', document.baseURI).href;
  const sprite = new Image();
  sprite.src = new URL('assets/fruit-fly.png', scriptURL).href;

  let metadata = null;
  let latestSample = null;
  let sampleReceivedAt = 0;

  // Observe only FlyHamlet's live worker. replay.js keeps ownership of the worker;
  // this listener receives the same authoritative samples for rendering.
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
  style.textContent = `
    .flight-overlay {
      position: absolute;
      inset: -${overscan}px;
      width: calc(100% + ${overscan * 2}px);
      height: calc(100% + ${overscan * 2}px);
      display: block;
      pointer-events: none;
      z-index: 4;
    }
  `;
  document.head.appendChild(style);

  function isLive() {
    return sessionSelect.value === 'live';
  }

  // replay.js still draws the model trail on #arena. In live mode suppress only
  // its old 2D fly sprite/shadow after the 3D renderer is ready.
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
      const gradientFill = typeof arenaContext.fillStyle !== 'string';
      if (suppress && gradientFill) return;
      originalFill(...args);
    };
  }

  function fitCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = overlay.clientWidth;
    const height = overlay.clientHeight;
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (overlay.width !== pixelWidth || overlay.height !== pixelHeight) {
      overlay.width = pixelWidth;
      overlay.height = pixelHeight;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return {width, height};
  }

  function projectedSample(timestamp) {
    if (!latestSample) return null;
    const sample = {...latestSample};
    const running = play?.getAttribute('aria-pressed') === 'true';
    if (!running || reducedMotion.matches) return sample;
    const requestedRate = Math.max(0.1, Number(speedSelect?.value) || 1);
    const wallSeconds = Math.max(0, timestamp - sampleReceivedAt) / 1000;
    const dt = Math.min(0.05, wallSeconds * requestedRate);
    sample.x += (sample.vx_mm_s || 0) * dt;
    sample.y += (sample.vy_mm_s || 0) * dt;
    sample.z_mm = Math.max(0, (sample.z_mm || 0) + (sample.vz_mm_s || 0) * dt);
    sample.heading_deg += (sample.omega_deg_s || 0) * dt;
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
    const col = Math.max(0, Math.min(metadata.cols - 1,
      Math.trunc(sample.x / metadata.W * metadata.cols)));
    const row = Math.max(0, Math.min(metadata.rows - 1,
      Math.trunc(sample.y / metadata.H * metadata.rows)));
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
    ctx.save();
    ctx.translate(x + size * 0.09 * altitudeFraction, y + size * 0.12 * altitudeFraction);
    ctx.rotate(Math.PI / 2 - heading);
    ctx.scale(1.35 + altitudeFraction * 0.45, 0.58 + altitudeFraction * 0.12);
    const shadow = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
    shadow.addColorStop(0, `rgba(35,25,12,${alpha})`);
    shadow.addColorStop(0.48, `rgba(35,25,12,${alpha * 0.55})`);
    shadow.addColorStop(1, 'rgba(35,25,12,0)');
    ctx.fillStyle = shadow;
    ctx.beginPath();
    ctx.arc(0, 0, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawFallback(size) {
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = '#544335';
    for (const side of [-1, 1]) {
      for (let leg = 0; leg < 3; leg++) {
        ctx.beginPath();
        ctx.moveTo(side * size * 0.055, (leg - 1) * size * 0.07);
        ctx.lineTo(side * size * 0.18, (leg - 1) * size * 0.13);
        ctx.lineTo(side * size * 0.25, (leg - 1) * size * 0.19);
        ctx.stroke();
      }
      ctx.fillStyle = 'rgba(232,230,211,.72)';
      ctx.beginPath();
      ctx.ellipse(side * size * 0.16, size * 0.10, size * 0.1, size * 0.26,
        side * -0.48, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.fillStyle = '#795633';
    ctx.beginPath();
    ctx.ellipse(0, size * 0.06, size * 0.085, size * 0.23, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawFly(sample, timestamp, dimensions) {
    const {width, height} = dimensions;
    const innerWidth = Math.max(1, width - overscan * 2);
    const innerHeight = Math.max(1, height - overscan * 2);
    const heading = (sample.heading_deg || 0) * Math.PI / 180;
    const pitch = (sample.pitch_deg || 0) * Math.PI / 180;
    const roll = (sample.roll_deg || 0) * Math.PI / 180;
    const ceiling = Math.max(1, metadata?.flight?.ceiling_mm || 12);
    const z = Math.max(0, Math.min(ceiling, sample.z_mm || 0));
    const altitudeFraction = z / ceiling;
    const groundX = overscan + sample.x / metadata.W * innerWidth;
    const groundY = height - overscan - sample.y / metadata.H * innerHeight;
    const altitudePixels = z / metadata.H * innerHeight * 0.62;
    const bodyX = groundX;
    const bodyY = groundY - altitudePixels;
    const baseSize = Math.max(54, Math.min(112, innerWidth * 0.14));
    const perspectiveScale = 1 + altitudeFraction * 0.16;
    const airborne = Boolean(sample.airborne) || z > 0.03;
    const wingPower = Math.max(0, Math.min(1, sample.wing_power || 0));

    drawShadow(groundX, groundY, baseSize, altitudeFraction, heading);

    ctx.save();
    ctx.translate(bodyX, bodyY);
    ctx.rotate(Math.PI / 2 - heading);
    ctx.scale(Math.max(0.56, Math.cos(roll)) * perspectiveScale,
      Math.max(0.62, Math.cos(pitch)) * perspectiveScale);

    if (sprite.complete && sprite.naturalWidth > 0) {
      const spriteHeight = baseSize * sprite.naturalHeight / sprite.naturalWidth;
      // The real wingbeat is faster than a display can resolve. Ghost only the
      // wing-bearing image regions; the body position remains entirely physical.
      if (airborne && wingPower > 0.03 && !reducedMotion.matches) {
        const beat = Math.sin(timestamp * 0.001 * Math.PI * 2 * 37);
        for (const side of [-1, 1]) {
          ctx.save();
          ctx.beginPath();
          ctx.rect(side < 0 ? -baseSize / 2 : baseSize * 0.10,
            -spriteHeight / 2, baseSize * 0.40, spriteHeight);
          ctx.clip();
          ctx.globalAlpha = (0.07 + Math.abs(beat) * 0.08) * wingPower;
          ctx.rotate(side * beat * 0.10 * wingPower);
          ctx.drawImage(sprite, -baseSize / 2, -spriteHeight / 2, baseSize, spriteHeight);
          ctx.restore();
        }
      }
      ctx.drawImage(sprite, -baseSize / 2, -spriteHeight / 2, baseSize, spriteHeight);
    } else {
      drawFallback(baseSize);
    }
    ctx.restore();
  }

  function frame(timestamp) {
    requestAnimationFrame(frame);
    const dimensions = fitCanvas();
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    if (!isLive() || !metadata || !latestSample) {
      overlay.hidden = true;
      return;
    }
    overlay.hidden = false;
    const sample = projectedSample(timestamp);
    syncContact(sample);
    drawFly(sample, timestamp, dimensions);
  }

  sessionSelect.addEventListener('change', () => {
    if (isLive()) {
      metadata = null;
      latestSample = null;
    } else {
      overlay.hidden = true;
    }
  });

  if ('ResizeObserver' in window) new ResizeObserver(fitCanvas).observe(stage);
  else window.addEventListener('resize', fitCanvas);
  requestAnimationFrame(frame);
})();
