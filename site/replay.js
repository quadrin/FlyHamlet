/* Live full-connectome simulation and saved-run viewer. Visual altitude is decorative. */
(() => {
  'use strict';

  const byId = (id) => document.getElementById(id);
  const arena = byId('arena');
  const strip = byId('strip');
  const actx = arena.getContext('2d');
  const sctx = strip.getContext('2d');
  const keyGrid = byId('key-grid');
  const typed = byId('typed');
  const els = Object.fromEntries([
    'fly', 'speed', 'play', 'play-icon', 'play-label', 'restart', 'scrub',
    'time', 'status', 'loading', 'loading-message', 'retry', 'trail',
    'session-kind', 'compute-rate', 'recording-total', 'session-note',
    'f-run', 'f-keys', 'f-spikes', 'typed-count', 'current-key', 'manuscript-state',
    'r-loomL', 'r-loomR', 'r-dnL', 'r-dnR', 'r-gf', 'r-motion',
  ].map((id) => [id, byId(id)]));
  const controls = ['fly', 'speed', 'play', 'restart', 'scrub', 'trail'];
  const media = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scriptURL = document.currentScript?.src || new URL('site/replay.js', document.baseURI).href;
  const sprite = new Image();
  // Matches the canvas overscan in style.css; positions still map to the key grid.
  const arenaInset = 64;
  let liveMode = true;
  let liveWorker = null;
  let liveWorkerReady = false;
  let liveSeed = null;
  let liveKeyOffset = 0;
  let D = null;
  let T = null;
  let i = 0;
  let playing = false;
  let loading = true;
  let reduced = media.matches;
  let trailVisible = false;
  let rafId = null;
  let lastTimestamp = null;
  let playbackTime = 0;
  let animationTime = 0;
  let typedCount = -1;
  let activeKey = -1;
  let requestId = 0;
  let controller = null;
  let keyElements = [];
  let colors = {};
  let arenaSize = { width: 0, height: 0 };
  let stripSize = { width: 0, height: 0 };

  function setText(id, value) {
    if (els[id]) els[id].textContent = value;
  }

  function readColors() {
    const style = getComputedStyle(document.documentElement);
    const fallback = {
      left: '#9c5738', right: '#5b7b79', gf: '#9c8151',
      grid: '#d6d2c6', muted: '#746f63', trail: '#b98545', ink: '#302d27',
    };
    colors = Object.fromEntries(Object.entries(fallback).map(([name, value]) => [
      name, style.getPropertyValue(`--${name}`).trim() || value,
    ]));
  }

  function setStatus(message, state) {
    setText('status', message);
    if (els.status) els.status.dataset.state = state;
  }

  function syncPlayButton() {
    setText('play-label', playing ? 'Pause' : (liveMode ? 'Resume' : (D && i === T.t_s.length - 1 ? 'Replay' : 'Play')));
    setText('play-icon', playing ? 'Ⅱ' : '▶');
    els.play.setAttribute('aria-label', liveMode ? (playing ? 'Pause live session' : 'Resume live session') : (playing ? 'Pause recording' : 'Play recording'));
    els.play.setAttribute('aria-pressed', String(playing));
  }

  function setControlsDisabled(disabled) {
    controls.forEach((id) => { if (els[id]) els[id].disabled = disabled; });
  }

  function fit(canvas, context) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width, height };
  }

  // The recordings use a bottom-left origin; CSS grid starts at the top left.
  function buildKeyboard() {
    const fragment = document.createDocumentFragment();
    keyElements = Array(D.layout.length);
    for (let row = D.rows - 1; row >= 0; row -= 1) {
      for (let col = 0; col < D.cols; col += 1) {
        const index = row * D.cols + col;
        const character = D.layout[index];
        const key = document.createElement('div');
        key.className = 'keycap';
        key.dataset.index = index;
        key.setAttribute('aria-label', character === ' ' ? 'Space' : character.toUpperCase());
        const face = document.createElement('span');
        face.className = 'key-face';
        const label = document.createElement('span');
        label.className = 'key-label';
        label.textContent = character === ' ' ? '␣' : character.toUpperCase();
        face.appendChild(label);
        key.appendChild(face);
        fragment.appendChild(key);
        keyElements[index] = key;
      }
    }
    keyGrid.replaceChildren(fragment);
    keyGrid.style.gridTemplateColumns = `repeat(${D.cols}, minmax(0, 1fr))`;
    keyGrid.style.gridTemplateRows = `repeat(${D.rows}, minmax(0, 1fr))`;
    activeKey = -1;
  }

  function sampleAt(time) {
    let low = 0;
    let high = T.t_s.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (T.t_s[middle] <= time) low = middle + 1;
      else high = middle;
    }
    return Math.max(0, low - 1);
  }

  function enteredCount(time) {
    let low = 0;
    let high = D.keys.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (D.keys[middle][0] <= time) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  function drawFly(x, y, heading) {
    // Keep the center exactly on the recorded position. Hover is a small change
    // in scale and shadow softness, never a displacement of the recorded path.
    const size = Math.max(54, Math.min(118, (arenaSize.width - 2 * arenaInset) * 0.14));
    const moving = playing && !reduced;
    const hover = moving ? Math.sin(animationTime * 2.7) : 0;
    const wingbeat = moving ? Math.sin(animationTime * 67) : 0;
    const scale = 1 + hover * 0.018;
    const rotation = Math.PI / 2 - heading * Math.PI / 180;

    actx.save();
    actx.translate(x + size * 0.12, y + size * (0.19 + hover * 0.013));
    actx.rotate(rotation);
    const shadow = actx.createRadialGradient(0, 0, 0, 0, 0, size * 0.37);
    shadow.addColorStop(0, 'rgba(35, 25, 12, 0.30)');
    shadow.addColorStop(0.4, 'rgba(35, 25, 12, 0.17)');
    shadow.addColorStop(1, 'rgba(35, 25, 12, 0)');
    actx.scale(0.85 + hover * 0.035, 1.08 + hover * 0.025);
    actx.fillStyle = shadow;
    actx.beginPath();
    actx.arc(0, 0, size * 0.37, 0, Math.PI * 2);
    actx.fill();
    actx.restore();

    actx.save();
    actx.translate(x, y);
    actx.rotate(rotation);
    actx.scale(scale, scale);
    if (sprite.complete && sprite.naturalWidth > 0) {
      const height = size * sprite.naturalHeight / sprite.naturalWidth;
      // Very faint wing-edge exposures convey a fast wingbeat without moving
      // the abdomen or covering up the anatomical detail of the source image.
      if (moving) {
        for (const side of [-1, 1]) {
          actx.save();
          actx.beginPath();
          actx.rect(side < 0 ? -size / 2 : size * 0.12, -height / 2, size * 0.38, height);
          actx.clip();
          actx.globalAlpha = 0.075 + Math.abs(wingbeat) * 0.055;
          actx.rotate(side * wingbeat * 0.05);
          actx.drawImage(sprite, -size / 2, -height / 2, size, height);
          actx.restore();
        }
      }
      actx.drawImage(sprite, -size / 2, -height / 2, size, height);
    } else {
      // An anatomical fallback also keeps the replay usable if the image fails.
      actx.lineWidth = 0.8;
      actx.strokeStyle = '#544335';
      for (const side of [-1, 1]) {
        for (let leg = 0; leg < 3; leg += 1) {
          actx.beginPath();
          actx.moveTo(side * size * 0.055, (leg - 1) * size * 0.07);
          actx.lineTo(side * size * 0.18, (leg - 1) * size * 0.13);
          actx.lineTo(side * size * 0.25, (leg - 1) * size * 0.19);
          actx.stroke();
        }
        actx.fillStyle = 'rgba(232, 230, 211, 0.72)';
        actx.beginPath();
        actx.ellipse(side * size * 0.16, size * 0.10, size * 0.1, size * 0.26, side * -0.48, 0, Math.PI * 2);
        actx.fill();
        actx.stroke();
      }
      const body = actx.createLinearGradient(-size * 0.1, 0, size * 0.1, 0);
      body.addColorStop(0, '#523b24');
      body.addColorStop(0.5, '#c9a46b');
      body.addColorStop(1, '#493420');
      actx.fillStyle = body;
      actx.beginPath();
      actx.ellipse(0, size * 0.06, size * 0.085, size * 0.23, 0, 0, Math.PI * 2);
      actx.fill();
      actx.fillStyle = '#853e26';
      for (const side of [-1, 1]) {
        actx.beginPath();
        actx.ellipse(side * size * 0.057, -size * 0.17, size * 0.053, size * 0.069, side * 0.2, 0, Math.PI * 2);
        actx.fill();
      }
    }
    actx.restore();
  }

  function drawArena() {
    const { width, height } = arenaSize;
    actx.clearRect(0, 0, width, height);
    if (!D || !width || !height) return;
    const X = (x) => arenaInset + x / D.W * (width - 2 * arenaInset);
    const Y = (y) => height - arenaInset - y / D.H * (height - 2 * arenaInset);
    if (trailVisible) {
      const start = sampleAt(T.t_s[i] - 15);
      const length = Math.max(1, i - start);
      actx.strokeStyle = colors.trail;
      actx.lineWidth = 1.5;
      actx.lineCap = 'round';
      // Eight fading segments keep the trail economical even at high speed.
      for (let segment = 0; segment < 8; segment += 1) {
        const from = start + Math.floor(length * segment / 8);
        const to = Math.min(i, start + Math.ceil(length * (segment + 1) / 8));
        actx.globalAlpha = 0.08 + segment / 7 * 0.5;
        actx.beginPath();
        actx.moveTo(X(T.x[from]), Y(T.y[from]));
        for (let j = from + 1; j <= to; j += 1) actx.lineTo(X(T.x[j]), Y(T.y[j]));
        actx.stroke();
      }
      actx.globalAlpha = 1;
    }
    drawFly(X(T.x[i]), Y(T.y[i]), T.heading_deg[i]);
  }

  function drawStrip() {
    const { width, height } = stripSize;
    sctx.clearRect(0, 0, width, height);
    if (!D || !width || !height) return;
    const left = 34;
    const bottom = 23;
    const top = 15;
    const right = 18;
    const maxHz = 160;
    const end = T.t_s[i];
    const start = end - 20;
    const first = sampleAt(Math.max(0, start));
    const X = (time) => left + (time - start) / 20 * (width - left - right);
    const Y = (value) => top + (1 - Math.min(maxHz, Math.max(0, value)) / maxHz) * (height - top - bottom);
    sctx.strokeStyle = colors.grid;
    sctx.lineWidth = 1;
    sctx.font = '10px "IBM Plex Mono", monospace';
    sctx.fillStyle = colors.muted;
    sctx.textAlign = 'right';
    sctx.textBaseline = 'middle';
    for (const value of [0, 50, 100, 150]) {
      sctx.beginPath();
      sctx.moveTo(left, Y(value));
      sctx.lineTo(width - right, Y(value));
      sctx.stroke();
      sctx.fillText(String(value), left - 7, Y(value));
    }
    sctx.textAlign = 'center';
    sctx.textBaseline = 'top';
    for (let second = Math.ceil(Math.max(0, start) / 5) * 5; second <= end; second += 5) {
      sctx.fillText(`${second} s`, X(second), height - bottom + 8);
    }
    sctx.save();
    sctx.beginPath();
    sctx.rect(left, top, width - left - right, height - top - bottom);
    sctx.clip();
    const line = (values, color, dashed) => {
      sctx.strokeStyle = color;
      sctx.lineWidth = 1.55;
      sctx.setLineDash(dashed ? [4, 3] : []);
      sctx.beginPath();
      for (let j = first; j <= i; j += 1) {
        if (j === first) sctx.moveTo(X(T.t_s[j]), Y(values[j]));
        else sctx.lineTo(X(T.t_s[j]), Y(values[j]));
      }
      sctx.stroke();
    };
    line(T.loom_L_hz, colors.left, false);
    line(T.loom_R_hz, colors.right, false);
    line(T.turnDN_L_hz, colors.left, true);
    line(T.turnDN_R_hz, colors.right, true);
    line(T.GF_hz, colors.gf, false);
    sctx.restore();
    sctx.fillStyle = colors.muted;
    sctx.textAlign = 'left';
    sctx.textBaseline = 'top';
    sctx.fillText('Hz', 7, 0);
  }

  function drawTyped() {
    const count = enteredCount(T.t_s[i]);
    setText('manuscript-state', liveMode ? 'Live session' : (i === T.t_s.length - 1 ? 'Complete' : 'In progress'));
    if (count !== typedCount) {
      typedCount = count;
      typed.textContent = (liveMode && liveKeyOffset ? '…' : '') + D.keys.slice(0, count).map((key) => key[2]).join('');
      const cursor = document.createElement('span');
      cursor.className = 'cursor';
      cursor.setAttribute('aria-hidden', 'true');
      typed.appendChild(cursor);
      typed.scrollTop = typed.scrollHeight;
      setText('typed-count', (count + (liveMode ? liveKeyOffset : 0)).toLocaleString());
    }
    const current = count ? D.keys[count - 1][1] : (liveMode ? Math.min(D.rows - 1, Math.floor(T.y[i] / D.H * D.rows)) * D.cols + Math.min(D.cols - 1, Math.floor(T.x[i] / D.W * D.cols)) : -1);
    if (current !== activeKey) {
      keyElements[activeKey]?.classList.remove('is-active');
      activeKey = current;
      keyElements[activeKey]?.classList.add('is-active');
    }
    setText('current-key', current < 0 ? '—' : (D.layout[current] === ' ' ? 'Space' : D.layout[current].toUpperCase()));
  }

  function draw() {
    if (!D) return;
    drawArena();
    drawStrip();
    drawTyped();
    setText('r-loomL', `${T.loom_L_hz[i].toFixed(1)} Hz`);
    setText('r-loomR', `${T.loom_R_hz[i].toFixed(1)} Hz`);
    setText('r-dnL', `${T.turnDN_L_hz[i].toFixed(1)} Hz`);
    setText('r-dnR', `${T.turnDN_R_hz[i].toFixed(1)} Hz`);
    setText('r-gf', `${T.GF_hz[i].toFixed(1)} Hz`);
    setText('r-motion', `${T.v_mm_s[i].toFixed(1)} mm/s · ${T.omega_deg_s[i].toFixed(0)}°/s`);
    setText('time', liveMode ? `${T.t_s[i].toFixed(2)} s simulated` : `${T.t_s[i].toFixed(2)} s / ${D.duration.toFixed(2)} s`);
    els.scrub.value = i;
    els.scrub.setAttribute('aria-valuetext', `${T.t_s[i].toFixed(2)} of ${D.duration.toFixed(0)} seconds`);
    els.scrub.style.setProperty('--progress', `${i / Math.max(1, T.t_s.length - 1) * 100}%`);
  }

  function resize() {
    arenaSize = fit(arena, actx);
    stripSize = fit(strip, sctx);
    draw();
  }

  function setPlaying(next) {
    if (rafId !== null) cancelAnimationFrame(rafId);
    rafId = null;
    playing = Boolean(next && D && !loading);
    lastTimestamp = null;
    if (liveMode) {
      liveWorker?.postMessage({ type: playing ? 'resume' : 'pause' });
      syncPlayButton();
      if (D && !loading) {
        setStatus(playing ? 'Running live' : 'Live session paused', playing ? 'playing' : 'paused');
        setText('compute-rate', playing ? 'Computing in this browser' : 'Paused · brain state retained');
        draw();
      }
      if (playing) rafId = requestAnimationFrame(frame);
      return;
    }
    if (playing && i >= T.t_s.length - 1) {
      i = 0;
      playbackTime = T.t_s[0];
    }
    syncPlayButton();
    if (D && !loading) {
      setStatus(playing ? 'Playing' : (i === T.t_s.length - 1 ? 'Recording complete' : 'Paused'),
        playing ? 'playing' : (i === T.t_s.length - 1 ? 'complete' : 'paused'));
      draw();
    }
    if (playing) rafId = requestAnimationFrame(frame);
  }

  function frame(timestamp) {
    rafId = null;
    if (!playing || !D || loading) return;
    if (lastTimestamp !== null) {
      const elapsed = Math.max(0, timestamp - lastTimestamp) / 1000;
      if (!liveMode) playbackTime += elapsed * Number(els.speed.value);
      if (!reduced) animationTime += elapsed;
    }
    lastTimestamp = timestamp;
    if (liveMode) {
      if (!reduced) drawArena();
      rafId = requestAnimationFrame(frame);
      return;
    }
    const nextIndex = sampleAt(playbackTime);
    if (nextIndex !== i) {
      i = nextIndex;
      draw();
    } else if (!reduced) {
      drawArena();
    }
    if (playbackTime >= T.t_s[T.t_s.length - 1]) {
      playbackTime = T.t_s[T.t_s.length - 1];
      setPlaying(false);
      return;
    }
    rafId = requestAnimationFrame(frame);
  }

  function validateData(data) {
    const fields = ['t_s', 'x', 'y', 'heading_deg', 'v_mm_s', 'omega_deg_s',
      'loom_L_hz', 'loom_R_hz', 'turnDN_L_hz', 'turnDN_R_hz', 'GF_hz'];
    const length = data?.traj?.t_s?.length;
    if (!length || !Array.isArray(data.keys) || !Array.isArray(data.layout)
      || data.layout.length !== data.rows * data.cols || !(data.W > 0) || !(data.H > 0)
      || !Number.isFinite(data.duration) || !Number.isFinite(data.spikes)
      || fields.some((field) => !Array.isArray(data.traj[field]) || data.traj[field].length !== length)) {
      throw new Error('This recording is incomplete.');
    }
    return data;
  }

  function showMode(live) {
    liveMode = live;
    document.body.dataset.mode = live ? 'live' : 'recording';
    els.scrub.hidden = live;
    els['compute-rate'].hidden = !live;
    els['recording-total'].hidden = live;
    setText('session-kind', live ? 'Live simulation' : 'Recorded experiment');
    setText('session-note', live
      ? 'New session resets the brain and starts with a fresh random seed. Speed depends on your device.'
      : 'A saved 300-second run. Choose Live session to start a new simulation.');
    els.restart.textContent = live ? 'New session' : '↺';
    els.restart.setAttribute('aria-label', live ? 'Start a new live session' : 'Restart recording');
    els.restart.title = live ? 'Start a new live session with a fresh seed' : 'Restart recording';
    els.speed.setAttribute('aria-label', live ? 'Target simulation speed' : 'Playback speed');
    els.speed.title = live ? 'Target speed; actual speed depends on your device' : 'Playback speed';
  }

  function liveError(message) {
    if (!liveMode) return;
    setPlaying(false);
    loading = false;
    D = null;
    T = null;
    setControlsDisabled(true);
    els.fly.disabled = false;
    els.loading.hidden = false;
    els.loading.classList.add('is-error');
    els.retry.hidden = false;
    setText('loading-message', message || 'The live simulator could not start. Please try again.');
    setStatus('Live session unavailable', 'error');
    arena.setAttribute('aria-busy', 'false');
    liveWorker?.terminate();
    liveWorker = null;
    liveWorkerReady = false;
  }

  function handleLiveMessage(event) {
    const message = event.data;
    if (!liveMode) return;
    const messageSeed = message.seed ?? message.metadata?.seed;
    if (messageSeed != null && messageSeed !== liveSeed) return;
    if (message.type === 'progress') {
      const stage = message.message || '';
      const label = stage.startsWith('Expanding') || stage.includes('ready') ? 'Preparing the brain wiring…' : 'Loading the full connectome (49 MB on first visit)…';
      setText('loading-message', label);
      return;
    }
    if (message.type === 'error') {
      liveError(message.message);
      return;
    }
    if (message.type === 'ready') {
      const metadata = message.metadata;
      if (!metadata || metadata.seed !== liveSeed) return;
      liveWorkerReady = true;
      liveKeyOffset = 0;
      const initial = message.initial;
      const fields = ['t_s', 'x', 'y', 'heading_deg', 'v_mm_s', 'omega_deg_s',
        'loom_L_hz', 'loom_R_hz', 'turnDN_L_hz', 'turnDN_R_hz', 'fwdDN_hz', 'bwdDN_hz', 'GF_hz'];
      T = Object.fromEntries(fields.map(field => [field, [initial[field] ?? 0]]));
      D = { ...metadata, traj: T, keys: [], spikes: 0, duration: 0 };
      i = 0;
      animationTime = 0;
      typedCount = -1;
      setText('f-run', `live · seed ${liveSeed}`);
      setText('f-spikes', '0');
      setText('f-keys', '0');
      setText('compute-rate', `${metadata.n.toLocaleString()} neurons · starting`);
      buildKeyboard();
      loading = false;
      setControlsDisabled(false);
      els.loading.hidden = true;
      arena.setAttribute('aria-busy', 'false');
      resize();
      setPlaying(!reduced);
      return;
    }
    if (message.type === 'batch' && D && !loading) {
      for (const sample of message.samples) {
        for (const field of Object.keys(T)) T[field].push(sample[field] ?? 0);
      }
      if (T.t_s.length > 6000) {
        for (const field of Object.keys(T)) T[field] = T[field].slice(-4000);
      }
      D.keys.push(...message.keys);
      if (D.keys.length > 2000) {
        const remove = D.keys.length - 1600;
        D.keys.splice(0, remove);
        liveKeyOffset += remove;
        typedCount = -1;
      }
      i = T.t_s.length - 1;
      D.spikes = message.spikes;
      D.duration = message.simTime;
      setText('f-spikes', message.spikes.toLocaleString());
      const ratio = message.realTimeRatio;
      setText('compute-rate', !playing ? 'Paused · brain state retained' : (Number.isFinite(ratio) ? `${ratio.toFixed(2)}× real time · computing locally` : 'Computing in this browser'));
      draw();
    }
  }

  function startLive() {
    ++requestId;
    controller?.abort();
    setPlaying(false);
    showMode(true);
    loading = true;
    D = null;
    T = null;
    liveSeed = crypto.getRandomValues(new Uint32Array(1))[0];
    typedCount = -1;
    setControlsDisabled(true);
    // Users can select a saved run while the larger brain dataset loads.
    els.fly.disabled = false;
    els.loading.hidden = false;
    els.loading.classList.remove('is-error');
    els.retry.hidden = true;
    arena.setAttribute('aria-busy', 'true');
    setStatus(liveWorkerReady ? 'Starting new session' : 'Loading brain wiring', 'loading');
    setText('loading-message', liveWorkerReady ? 'Starting a fresh brain state…' : 'Loading the full connectome. The first visit downloads the brain wiring.');
    setText('compute-rate', 'Preparing live simulation');
    setText('f-run', `live · seed ${liveSeed}`);
    setText('f-spikes', '—');
    setText('time', '0.00 s simulated');
    setText('current-key', '—');
    for (const field of ['r-loomL', 'r-loomR', 'r-dnL', 'r-dnR', 'r-gf', 'r-motion']) setText(field, '—');
    sctx.clearRect(0, 0, stripSize.width, stripSize.height);
    typed.textContent = '';
    setText('typed-count', '0');
    try {
      if (!liveWorker) {
        liveWorker = new Worker(new URL('live-worker.js', scriptURL));
        liveWorker.addEventListener('message', handleLiveMessage);
        liveWorker.addEventListener('error', () => liveError('The simulator stopped unexpectedly. Try again, or choose a saved recording.'));
      }
      liveWorker.postMessage({ type: liveWorkerReady ? 'new' : 'init', seed: liveSeed,
        speed: Number(els.speed.value), autoplay: false,
        manifestURL: new URL('model/manifest.json', scriptURL).href });
    } catch (error) {
      liveError('This browser could not start the live simulator. Try a current browser with Web Worker support.');
    }
  }

  async function loadFly(id) {
    if (id === 'live') { startLive(); return; }
    setPlaying(false);
    if (liveWorker && !liveWorkerReady) { liveWorker.terminate(); liveWorker = null; }
    showMode(false);
    const thisRequest = ++requestId;
    controller?.abort();
    controller = new AbortController();
    setPlaying(false);
    loading = true;
    setControlsDisabled(true);
    setStatus('Loading recording', 'loading');
    if (els.loading) {
      els.loading.hidden = false;
      els.loading.classList.remove('is-error');
    }
    setText('loading-message', `Loading fly ${id}…`);
    if (els.retry) els.retry.hidden = true;
    arena.setAttribute('aria-busy', 'true');
    try {
      const response = await fetch(new URL(`data/fly${id}.json`, scriptURL), { signal: controller.signal });
      if (!response.ok) throw new Error(`The recording could not be loaded (${response.status}).`);
      const data = validateData(await response.json());
      if (thisRequest !== requestId) return;
      D = data;
      T = D.traj;
      i = 0;
      playbackTime = T.t_s[0];
      typedCount = -1;
      animationTime = 0;
      setText('f-run', `fly ${D.fly} · seed ${D.seed}`);
      setText('f-keys', D.keys.length.toLocaleString());
      setText('f-spikes', D.spikes.toLocaleString());
      els.scrub.max = T.t_s.length - 1;
      buildKeyboard();
      loading = false;
      setControlsDisabled(false);
      if (els.loading) els.loading.hidden = true;
      arena.setAttribute('aria-busy', 'false');
      resize();
      setPlaying(!reduced);
    } catch (error) {
      if (error.name === 'AbortError' || thisRequest !== requestId) return;
      loading = false;
      D = null;
      T = null;
      els.fly.disabled = false;
      setStatus('Unable to load recording. Please retry.', 'error');
      setText('loading-message', 'The recording could not be loaded. Check your connection and try again.');
      if (els.loading) els.loading.classList.add('is-error');
      if (els.retry) els.retry.hidden = false;
      arena.setAttribute('aria-busy', 'false');
      actx.clearRect(0, 0, arenaSize.width, arenaSize.height);
      syncPlayButton();
    }
  }

  els.play.addEventListener('click', () => setPlaying(!playing));
  els.restart?.addEventListener('click', () => {
    if (liveMode) { startLive(); return; }
    if (!D || loading) return;
    i = 0;
    playbackTime = T.t_s[0];
    typedCount = -1;
    setPlaying(!reduced);
  });
  els.scrub.addEventListener('input', () => {
    if (liveMode || !D || loading) return;
    i = Math.max(0, Math.min(T.t_s.length - 1, Number(els.scrub.value)));
    playbackTime = T.t_s[i];
    lastTimestamp = null;
    draw();
    if (i === T.t_s.length - 1) setPlaying(false);
    else if (!playing) {
      syncPlayButton();
      setStatus('Paused', 'paused');
    }
  });
  els.speed.addEventListener('change', () => {
    lastTimestamp = null;
    if (liveMode) liveWorker?.postMessage({ type: 'speed', value: Number(els.speed.value) });
  });
  els.fly.addEventListener('change', () => loadFly(els.fly.value));
  els.retry?.addEventListener('click', () => loadFly(els.fly.value));
  els.trail?.setAttribute('aria-pressed', 'false');
  els.trail?.addEventListener('click', () => {
    trailVisible = !trailVisible;
    els.trail.setAttribute('aria-pressed', String(trailVisible));
    drawArena();
  });
  document.addEventListener('keydown', (event) => {
    const interactive = event.target instanceof Element
      && event.target.closest('input, select, button, textarea, a, summary, #typed, [contenteditable], [role="button"], [role="slider"]');
    if (event.code === 'Space' && !event.repeat && !interactive && D && !loading) {
      event.preventDefault();
      setPlaying(!playing);
    }
  });
  document.addEventListener('visibilitychange', () => {
    lastTimestamp = null;
    if (document.hidden && liveMode && playing) setPlaying(false);
  });
  media.addEventListener('change', (event) => {
    reduced = event.matches;
    if (reduced) setPlaying(false);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); draw(); });
  new MutationObserver(() => { readColors(); draw(); }).observe(document.documentElement, {
    attributes: true, attributeFilter: ['data-theme'],
  });
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(resize);
    observer.observe(arena);
    observer.observe(strip);
  } else {
    window.addEventListener('resize', resize);
  }
  sprite.addEventListener('load', drawArena);
  sprite.src = new URL('assets/fruit-fly.png', scriptURL).href;
  readColors();
  resize();
  loadFly(els.fly.value);
})();
