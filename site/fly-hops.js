/* Decorative fly motion layered over the simulator. */
(() => {
  'use strict';

  const arena = document.getElementById('arena');
  const stage = arena?.closest('.arena-stage');
  const keyGrid = document.getElementById('key-grid');
  const play = document.getElementById('play');
  if (!arena || !stage || !keyGrid) return;

  // replay.js still draws the model path/trail. Hide only its fly sprite/shadow.
  const context = arena.getContext('2d');
  if (context) {
    const originalDrawImage = context.drawImage.bind(context);
    context.drawImage = (image, ...args) => {
      const source = image?.currentSrc || image?.src || '';
      if (!source.includes('fruit-fly.png')) originalDrawImage(image, ...args);
    };
    const originalFill = context.fill.bind(context);
    context.fill = (...args) => {
      const isGradient = typeof CanvasGradient !== 'undefined' && context.fillStyle instanceof CanvasGradient;
      if (!isGradient) originalFill(...args);
    };
  }

  const scriptURL = document.currentScript?.src || new URL('site/fly-hops.js', document.baseURI).href;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const fly = new Image();
  fly.className = 'key-hopping-fly';
  fly.alt = '';
  fly.setAttribute('aria-hidden', 'true');
  fly.draggable = false;
  fly.src = new URL('assets/fruit-fly.png', scriptURL).href;

  const style = document.createElement('style');
  style.textContent = `.key-hopping-fly{position:absolute;left:50%;top:50%;z-index:4;height:auto;pointer-events:none;user-select:none;transform-origin:50% 50%;will-change:left,top,transform,filter}`;
  document.head.appendChild(style);
  stage.appendChild(fly);

  let position = null;
  let flight = null;
  let currentKey = null;
  let lastAngle = 0;
  let landedAt = 0;
  let nextTakeoffDelay = 900;
  let lastFrame = null;

  function flySize() {
    fly.style.width = `${Math.max(54, Math.min(118, stage.clientWidth * 0.14))}px`;
  }

  function keys() {
    return [...keyGrid.querySelectorAll('.keycap')];
  }

  function keyCenter(key) {
    return { x: key.offsetLeft + key.offsetWidth / 2, y: key.offsetTop + key.offsetHeight / 2 };
  }

  function randomKey() {
    const all = keys();
    if (!all.length) return null;
    const candidates = all.length > 1 ? all.filter((key) => key !== currentKey) : all;
    return candidates[Math.floor(Math.random() * candidates.length)];
  }

  function renderFly(x, y, angle, landed, timestamp = 0) {
    const bob = landed ? 0 : Math.sin(timestamp * 0.018) * 2.2;
    const pulse = landed ? 0.92 : 1.02 + Math.sin(timestamp * 0.055) * 0.025;
    fly.style.left = `${x}px`;
    fly.style.top = `${y + bob}px`;
    fly.style.transform = `translate3d(-50%,-50%,${landed ? 6 : 26}px) rotate(${angle}deg) scale(${pulse})`;
    fly.style.filter = landed
      ? 'drop-shadow(2px 5px 2px rgba(35,25,12,.30))'
      : 'drop-shadow(6px 14px 7px rgba(35,25,12,.22))';
  }

  function snapTo(key, timestamp = performance.now()) {
    currentKey = key;
    position = keyCenter(key);
    flight = null;
    landedAt = timestamp;
    nextTakeoffDelay = 700 + Math.random() * 1400;
    renderFly(position.x, position.y, lastAngle, true, timestamp);
  }

  function beginFlight(key, timestamp) {
    const destination = keyCenter(key);
    if (!position) {
      position = {
        x: stage.clientWidth * (0.25 + Math.random() * 0.5),
        y: -Math.max(28, stage.clientHeight * 0.18),
      };
    }
    const dx = destination.x - position.x;
    const dy = destination.y - position.y;
    const distance = Math.hypot(dx, dy);
    lastAngle = Math.atan2(dy, dx) * 180 / Math.PI + 90;
    flight = {
      key,
      from: { ...position },
      to: destination,
      start: timestamp,
      duration: Math.max(520, Math.min(1100, 500 + distance * 1.25)) * (0.86 + Math.random() * 0.28),
      arc: stage.clientHeight * (0.18 + Math.random() * 0.10),
      wobble: (Math.random() - 0.5) * stage.clientWidth * 0.045,
      angle: lastAngle,
    };
  }

  function ease(value) {
    return value < 0.5 ? 4 * value ** 3 : 1 - ((-2 * value + 2) ** 3) / 2;
  }

  function startIfReady(timestamp) {
    if (flight) return;
    const target = randomKey();
    if (!target) return;
    if (!position) beginFlight(target, timestamp);
    else if (timestamp - landedAt >= nextTakeoffDelay) beginFlight(target, timestamp);
  }

  function frame(timestamp) {
    requestAnimationFrame(frame);
    const running = play?.getAttribute('aria-pressed') === 'true';
    if (!running) {
      if (flight && lastFrame != null) flight.start += timestamp - lastFrame;
      lastFrame = timestamp;
      return;
    }

    if (reducedMotion.matches) {
      if (!currentKey || timestamp - landedAt >= nextTakeoffDelay) {
        const target = randomKey();
        if (target) snapTo(target, timestamp);
      }
      lastFrame = timestamp;
      return;
    }

    startIfReady(timestamp);
    if (flight) {
      const raw = Math.min(1, Math.max(0, (timestamp - flight.start) / flight.duration));
      const t = ease(raw);
      const x = flight.from.x + (flight.to.x - flight.from.x) * t + Math.sin(t * Math.PI * 2) * flight.wobble;
      const y = flight.from.y + (flight.to.y - flight.from.y) * t - Math.sin(t * Math.PI) * flight.arc;
      position = { x, y };
      renderFly(x, y, flight.angle, false, timestamp);
      if (raw >= 1) snapTo(flight.key, timestamp);
    } else if (position) {
      renderFly(position.x, position.y, lastAngle, true, timestamp);
    }
    lastFrame = timestamp;
  }

  function handleResize() {
    flySize();
    if (currentKey && !flight) snapTo(currentKey);
  }

  new MutationObserver(() => {
    if (!currentKey || !currentKey.isConnected) {
      currentKey = null;
      position = null;
      flight = null;
    }
  }).observe(keyGrid, { childList: true });

  flySize();
  if ('ResizeObserver' in window) new ResizeObserver(handleResize).observe(stage);
  else window.addEventListener('resize', handleResize);
  reducedMotion.addEventListener('change', () => { flight = null; landedAt = 0; });
  requestAnimationFrame(frame);
})();
