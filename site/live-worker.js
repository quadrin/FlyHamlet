/* Independent, bounded worker execution. Protocol: init / new / pause / resume / speed.
 * The immutable full connectome is loaded once; each new session gets fresh state.
 */
'use strict';
importScripts('live-model.js', 'flight-model.js?v=2');

let connectome = null;
let manifest = null;
let arena = null;
let running = false;
let desiredRunning = false;
let speed = 1;
let timer = null;
let scheduleTicket = 0;
let revision = 0;
let requestedSeed = 0;
let loading = null;
let sessionStarted = 0;
let elapsedActiveMs = 0;
let activeStarted = 0;
let paceWall = 0;
let paceSim = 0;
let lastSent = 0;
let pendingSamples = [];
let pendingKeys = [];

const now = () => performance.now();
const newSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
const emit = message => postMessage({seed: requestedSeed, generation: revision, ...message});
const progress = (message, loaded = 0, total = 0) => emit({type: 'progress', message, loaded, total});
// MessageChannel yields to incoming worker messages without the nested-timer
// minimum delay, so the neural solver can use the available CPU when behind.
const tickChannel = typeof MessageChannel === 'undefined' ? null : new MessageChannel();
if (tickChannel) tickChannel.port1.onmessage = event => {
  if (event.data === scheduleTicket) tick();
};

async function fetchArray(url, name, integrity) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}.`);
  const total = Number(response.headers.get('content-length')) || integrity?.bytes || 0;
  let loaded = 0;
  let bytes;
  if (response.body) {
    const chunks = [];
    const reader = response.body.getReader();
    let lastProgress = 0;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      chunks.push(value); loaded += value.byteLength;
      if (now() - lastProgress > 150) {
        progress(`Loading ${name}`, loaded, total); lastProgress = now();
      }
    }
    bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  } else { bytes = new Uint8Array(await response.arrayBuffer()); loaded = bytes.byteLength; }
  progress(`Expanding ${name}`, loaded, total);
  let buffer = bytes.buffer;
  // Detect gzip bytes: this also works when an HTTP server already decompressed.
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (typeof DecompressionStream === 'undefined')
      throw new Error('This browser cannot expand the connectome. Please use a current Chrome, Edge, Firefox, or Safari.');
    buffer = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  if (integrity?.rawBytes && buffer.byteLength !== integrity.rawBytes)
    throw new Error(`The downloaded ${name} has the wrong size. Please reload.`);
  if (integrity?.rawSha256 && crypto.subtle) {
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    const hex = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
    if (hex !== integrity.rawSha256) throw new Error(`The downloaded ${name} failed its integrity check. Please reload.`);
  }
  return buffer;
}

async function loadConnectome(manifestURL) {
  progress('Loading the full connectome manifest');
  const url = new URL(manifestURL || 'model/manifest.json', self.location.href);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load the live model: HTTP ${response.status}.`);
  const data = await response.json();
  if (data.format !== 'flyhamlet-csr-v1') throw new Error('Unsupported live connectome format.');
  const arrays = {};
  // Sequential expansion bounds peak temporary memory on phones and laptops.
  for (const name of ['indptr', 'indices', 'weights']) {
    const file = data.files[name];
    const path = typeof file === 'string' ? file : file.url;
    const buffer = await fetchArray(new URL(path, url), name, data.integrity?.[name]);
    arrays[name] = name === 'weights' ? new Float32Array(buffer) : new Uint32Array(buffer);
  }
  if (arrays.indptr.length !== data.n + 1 || arrays.indices.length !== data.edgeCount ||
      arrays.weights.length !== data.edgeCount || arrays.indptr[data.n] !== data.edgeCount)
    throw new Error('The full connectome data is incomplete.');
  // Reject bad indices before entering the integration loop.
  for (let i = 0; i < data.n; ++i) if (arrays.indptr[i] > arrays.indptr[i + 1])
    throw new Error('Invalid connectome row offsets.');
  for (const index of arrays.indices) if (index >= data.n) throw new Error('Invalid connectome target index.');
  manifest = data;
  connectome = {n: data.n, ...arrays};
  progress('Full connectome ready', data.edgeCount, data.edgeCount);
}

function clearTimer() {
  scheduleTicket++;
  if (timer !== null) clearTimeout(timer);
  timer = null;
}
function activeMilliseconds() { return elapsedActiveMs + (running ? now() - activeStarted : 0); }
function sendBatch(force = false) {
  const time = now();
  if (!arena || (!force && time - lastSent < 45)) return;
  if (!pendingSamples.length && !pendingKeys.length && !force) return;
  const activeMs = activeMilliseconds();
  emit({type: 'batch', seed: arena.seed, samples: pendingSamples, keys: pendingKeys,
    spikes: arena.net.totalSpikes, simTime: arena.net.timeS,
    realTimeRatio: activeMs > 0 ? arena.net.timeS * 1000 / activeMs : 0,
    wallTime: activeMs / 1000, activeNeurons: arena.net.activeCount,
    keyCount: arena.keyCount, sessionAge: (time - sessionStarted) / 1000});
  pendingSamples = []; pendingKeys = []; lastSent = time;
}

function pause() {
  desiredRunning = false;
  if (running) elapsedActiveMs += now() - activeStarted;
  running = false; clearTimer(); sendBatch(true);
  emit({type: 'state', running: false});
}
function resume() {
  desiredRunning = true;
  if (!arena || running) return;
  running = true; activeStarted = now(); paceWall = activeStarted; paceSim = arena.net.timeS;
  emit({type: 'state', running: true});
  schedule(0);
}
function schedule(delay) {
  clearTimer();
  if (!running) return;
  if (delay === 0 && tickChannel) tickChannel.port2.postMessage(scheduleTicket);
  else timer = setTimeout(tick, delay);
}
function tick() {
  timer = null;
  if (!running || !arena) return;
  try {
    const started = now();
    const target = paceSim + (started - paceWall) * speed / 1000;
    const controlSeconds = arena.controlSteps * arena.net.dt * 1e-3;
    // Each slice is bounded in wall time; a control step is the smallest atomic
    // interval. The worker yields between slices so Pause and New stay responsive.
    while (arena.net.timeS + controlSeconds <= target + 1e-12 && now() - started < 12) {
      const result = arena.stepControl();
      if (result.sample) pendingSamples.push(result.sample);
      if (result.key) pendingKeys.push(result.key);
    }
    sendBatch();
    const lag = target - arena.net.timeS;
    schedule(lag >= controlSeconds ? 0 : Math.max(1, Math.min(12, controlSeconds * 1000 / speed)));
  } catch (error) { fail(error); }
}

function beginSession(seed, autoplay) {
  clearTimer(); running = false;
  arena = new FlyHamletLive.FlyArena(connectome, manifest, seed === undefined ? newSeed() : seed);
  pendingSamples = []; pendingKeys = [];
  elapsedActiveMs = 0; sessionStarted = now(); lastSent = 0;
  emit({type: 'ready', metadata: arena.metadata, initial: arena.sample()});
  if (autoplay) resume();
  else { desiredRunning = false; emit({type: 'state', running: false}); }
}
function fail(error) {
  running = false; desiredRunning = false; clearTimer();
  emit({type: 'error', message: error?.message || String(error)});
}

self.onmessage = async event => {
  const message = event.data || {};
  try {
    if (message.type === 'init' || message.type === 'new') {
      const token = ++revision;
      requestedSeed = message.seed === undefined ? newSeed() : Number(message.seed) >>> 0;
      desiredRunning = message.autoplay ?? true;
      clearTimer(); running = false;
      if (message.speed !== undefined) speed = Math.max(0.1, Math.min(100, Number(message.speed) || 1));
      if (!connectome) {
        if (!loading) loading = loadConnectome(message.manifestURL).finally(() => { loading = null; });
        await loading;
      }
      if (token !== revision) return;
      beginSession(requestedSeed, desiredRunning);
    } else if (message.type === 'pause') pause();
    else if (message.type === 'resume') resume();
    else if (message.type === 'speed') {
      speed = Math.max(0.1, Math.min(100, Number(message.value) || 1));
      if (arena) { paceWall = now(); paceSim = arena.net.timeS; }
    }
  } catch (error) { fail(error); }
};
