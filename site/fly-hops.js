/* Decorative fly motion layered over the simulator's model-derived key entries. */
(() => {
  'use strict';

  const arena = document.getElementById('arena');
  const stage = arena?.closest('.arena-stage');
  const keyGrid = document.getElementById('key-grid');
  const play = document.getElementById('play');
  if (!arena || !stage || !keyGrid) return;

  // replay.js keeps drawing the model path and trail on this canvas. Suppress
  // only its fly artwork and fly shadow so the trail remains available.
  const context = arena.getContext('2d');
  if (context) {
    const originalDrawImage = context.drawImage.bind(context);
    context.drawImage = (image, ...args) => {
      const source = image?.currentSrc || image?.src || '';
      if (source.includes('fruit-fly.png')) return;
      originalDrawImage(image, ...args);
    };

    const originalFill = context.fill.bind(context);
    context.fill = (...args) => {
      const isGradient = typeof CanvasGradient !== 'undefined'
        && context.fillStyle instanceof CanvasGradient;
      if (isGradient) return;
      originalFill(...args);
    };
  }

  const scriptURL = document.currentScript?.src
    || new URL('site/fly-hops.js', document.baseURI).href;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const fly = new Image();
  fly.className = 'key-hopping-fly';
  fly.alt = '';
  fly.setAttribute('aria-hidden', 'true');
  fly.draggable = false;
  fly.src = new URL('assets/fruit-fly.png', scriptURL).href;

  const style = document.createElement('style');
  style.textContent = `
    .key-hopping-fly {
      position: absolute;
      left: 50%;
      top: 50%;
      z-index: 4;
      height: auto;
      pointer-events: none;
      user-select: none;
      transform-origin: 50% 50%;
      will-change: left, top, transform, filter;
    }
  `;
  document.head.appendChild(style);
  stage.appendChild(fly);

  let position = null;
  let flight = null;
  let targetIndex = null;
  let lastAngle = 0;
  let lastFrame = null;

  function flySize() {
    const width = Math.max(54, Math.min(118, stage.clientWidth * 0.14));
    fly.style.width = `${width}px`;
  }

  function keyCenter(key) {
    return {
      x: key.offsetLeft + key.offsetWidth / 2,
      y: key.offsetTop + key.offsetHeight / 2,
    };
  }

  function renderFly(x, y, angle, landed, timestamp = 0) {
    const bob = landed ? 0 : Math.sin(timestamp * 0.018) * 2.2;
    const pulse = landed ? 0.92 : 1.02 + Math.sin(timestamp * 0.055) * 0.025;
    const z = landed ? 6 : 26;
    fly.style.left = `${x}px`;
    fly.style.top = `${y + bob}px`;
    fly.style.transform = `translate3d(-50%, -50%, ${z}px) rotate(${angle}deg) scale(${pulse})`;
    fly.style.filter = landed
      ? 'drop-shadow(2px 5px 2px rgba(35, 25, 12, 0.30))'
      : 'drop-shadow(6px 14px 7px rgba(35, 25, 12, 0.22))';
  }

  function snapTo(key) {
    position = keyCenter(key);
    flight = null;
    renderFly(position.x, position.y, lastAngle, true);
  }

  function beginFlight(key) {
    const destination = keyCenter(key);
    if (!position) {
      position = {
        x: stage.clientWidth * (0.35 + Math.random() * 0.30),
        y: -Math.max(28, stage.clientHeight * 0.18),
      };
    }

    const dx = destination.x - position.x;
    const dy = destination.y - position.y;
    const distance = Math.hypot(dx, dy);
    const duration = Math.max(520, Math.min(1100, 500 + distance * 1.25))
      * (0.86 + Math.random() * 0.28);
    lastAngle = Math.atan2(dy, dx) * 180 / Math.PI + 90;
    flight = {
      from: { ...position },
      to: destination,
      start: null,
      duration,
      arc: stage.clientHeight * (0.18 + Math.random() * 0.10),
      wobble: (Math.random() - 0.5) * stage.clientWidth * 0.045,
      angle: lastAngle,
    };
  }

  function easeInOutCubic(value) {
    return value < 0.5
      ? 4 * value * value * value
      : 1 - Math.pow(-2 * value + 2, 3) / 2;
  }

  function syncTarget(reset = false) {
    if (reset) targetIndex = null;
    const key = keyGrid.querySelector('.keycap.is-active');
    if (!key) return;
    const nextIndex = key.dataset.index;
    if (nextIndex === targetIndex) return;
    targetIndex = nextIndex;
    if (reducedMotion.matches) snapTo(key);
    else beginFlight(key);
  }

  const observer = new MutationObserver((mutations) => {
    const rebuilt = mutations.some((mutation) => mutation.type === 'childList');
    syncTarget(rebuilt);
  });
  observer.observe(keyGrid, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['class'],
  });

  function frame(timestamp) {
    requestAnimationFrame(frame);
    syncTarget();

    if (!position) {
      lastFrame = timestamp;
      return;
    }

    const running = play?.getAttribute('aria-pressed') === 'true';
    if (!running) {
      if (flight?.start != null && lastFrame != null) flight.start += timestamp - lastFrame;
      lastFrame = timestamp;
      return;
    }

    if (reducedMotion.matches) {
      const key = targetIndex == null
        ? null
        : keyGrid.querySelector(`.keycap[data-index="${targetIndex}"]`);
      if (key) snapTo(key);
      lastFrame = timestamp;
      return;
    }

    if (flight) {
      if (flight.start == null) flight.start = timestamp;
      const raw = Math.min(1, Math.max(0, (timestamp - flight.start) / flight.duration));
      const eased = easeInOutCubic(raw);
      const x = flight.from.x + (flight.to.x - flight.from.x) * eased
        + Math.sin(eased * Math.PI * 2) * flight.wobble;
      const y = flight.from.y + (flight.to.y - flight.from.y) * eased
        - Math.sin(eased * Math.PI) * flight.arc;
      position = { x, y };
      renderFly(x, y, flight.angle, false, timestamp);

      if (raw >= 1) {
        position = { ...flight.to };
        lastAngle = flight.angle;
        flight = null;
        renderFly(position.x, position.y, lastAngle, true, timestamp);
      }
    } else {
      renderFly(position.x, position.y, lastAngle, true, timestamp);
    }

    lastFrame = timestamp;
  }

  function handleResize() {
    flySize();
    if (flight || targetIndex == null) return;
    const key = keyGrid.querySelector(`.keycap[data-index="${targetIndex}"]`);
    if (key) snapTo(key);
  }

  flySize();
  if ('ResizeObserver' in window) new ResizeObserver(handleResize).observe(stage);
  else window.addEventListener('resize', handleResize);
  reducedMotion.addEventListener('change', () => syncTarget(true));
  requestAnimationFrame(frame);
})();
