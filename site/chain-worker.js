/* A fresh self-driven phrase-recall experiment on the complete FlyWire network. The graph
 * and the rescaled slow-dynamics graph are cached across runs; neural state, decoders
 * and the trial schedule are recreated. Every message carries the requested seed and
 * generation for stale-run guards.
 */
'use strict';
importScripts('live-model.js', 'recall-model.js', 'sequence-model.js', 'chain-model.js', 'connectome-loader.js');

let connectome = null;
let manifest = null;
let loading = null;
let prepared = null;
let preparing = null;
let experiment = null;
let requestedSeed = 0;
let generation = 0;
let speed = 1;
let desiredRunning = false;
let running = false;
let timer = null;
let scheduleTicket = 0;
let elapsedActiveMs = 0;
let activeStarted = 0;
let paceWall = 0;
let paceSim = 0;
let lastSent = 0;
let pendingRecords = [];
let completed = false;

const now = () => performance.now();
const newSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
const emit = message => postMessage({seed: requestedSeed, generation, ...message});
const reportProgress = (message, loaded = 0, total = 0) => emit({type: 'progress', message, loaded, total});
const normalizeSpeed = value => Math.max(0.1, Math.min(100, Number(value) || 1));
// Yield through the event queue without nested timeout clamping when CPU-bound.
const tickChannel = typeof MessageChannel === 'undefined' ? null : new MessageChannel();
if (tickChannel) tickChannel.port1.onmessage = event => {
  if (event.data === scheduleTicket) tick();
};

function cancelScheduled() {
  ++scheduleTicket;
  if (timer !== null) clearTimeout(timer);
  timer = null;
}

function activeMilliseconds() {
  return elapsedActiveMs + (running ? now() - activeStarted : 0);
}

function sendTick(force = false) {
  const time = now();
  if (!experiment || (!force && time - lastSent < 45)) return;
  const activeMs = activeMilliseconds();
  emit({type: 'tick', progress: experiment.progress, records: pendingRecords,
    realTimeRatio: activeMs > 0 ? experiment.simTimeS * 1000 / activeMs : 0,
    wallTime: activeMs / 1000});
  pendingRecords = [];
  lastSent = time;
}

function stop() {
  if (running) elapsedActiveMs += now() - activeStarted;
  running = false;
  cancelScheduled();
}

function pause() {
  desiredRunning = false;
  stop();
  sendTick(true);
  emit({type: 'state', running: false});
}

function resume() {
  desiredRunning = true;
  if (!experiment || running || experiment.done || completed) return;
  running = true;
  activeStarted = now();
  paceWall = activeStarted;
  paceSim = experiment.simTimeS;
  emit({type: 'state', running: true});
  schedule(0);
}

function schedule(delay) {
  cancelScheduled();
  if (!running) return;
  const ticket = scheduleTicket;
  if (delay === 0 && tickChannel) tickChannel.port2.postMessage(ticket);
  else timer = setTimeout(() => { if (ticket === scheduleTicket) tick(); }, delay);
}

function finish() {
  if (completed) return;
  completed = true;
  desiredRunning = false;
  stop();
  sendTick(true);
  emit({type: 'state', running: false});
  emit({type: 'complete', result: experiment.result()});
}

function tick() {
  timer = null;
  if (!running || !experiment) return;
  try {
    const started = now();
    const target = paceSim + (started - paceWall) * speed / 1000;
    const stepSeconds = experiment.dtMs / 1000;
    // One neural timestep is the atomic operation; pause/new can be processed
    // after each <=12 ms slice plus that single timestep's compute duration.
    while (!experiment.done && experiment.simTimeS + stepSeconds <= target + 1e-12 &&
        now() - started < 12) {
      const record = experiment.step();
      if (record) pendingRecords.push(record);
    }
    if (experiment.done) { finish(); return; }
    sendTick();
    const lag = target - experiment.simTimeS;
    schedule(lag >= stepSeconds ? 0 : Math.max(1, Math.min(12, stepSeconds * 1000 / speed)));
  } catch (error) { fail(error); }
}

function fail(error) {
  stop();
  desiredRunning = false;
  emit({type: 'error', message: error?.message || String(error)});
}

function begin(options) {
  experiment = new FlyHamletChain.ChainExperiment(connectome, manifest, requestedSeed, options, prepared);
  if (!(experiment.dtMs > 0) || !Number.isFinite(experiment.dtMs) ||
      !Number.isFinite(experiment.simTimeS))
    throw new Error('The experiment has an invalid neural timestep or clock.');
  pendingRecords = [];
  completed = false;
  elapsedActiveMs = 0;
  lastSent = 0;
  emit({type: 'ready', metadata: {...experiment.metadata, protocol: experiment.protocol},
    progress: experiment.progress});
  if (experiment.done) finish();
  else if (desiredRunning) resume();
  else emit({type: 'state', running: false});
}

async function prepareSlow(scale) {
  reportProgress('Preparing the slower-dynamics comparison network', 0, connectome.weights.length);
  const slow = await FlyHamletSequence.prepareSlowGraph(connectome, scale);
  reportProgress('Verifying the comparison network', 0, slow.graph.weights.byteLength);
  const digest = await crypto.subtle.digest('SHA-256', slow.graph.weights);
  const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  return {graph: slow.graph, metadata: {...slow.metadata, scaledWeightsSha256: hash}};
}

self.onmessage = async event => {
  const message = event.data || {};
  // A captured generation prevents a superseded asynchronous request from
  // reporting an error or launching its experiment after a newer request.
  let token = generation;
  try {
    if (message.type === 'init' || message.type === 'new') {
      token = ++generation;
      requestedSeed = message.seed === undefined ? newSeed() : Number(message.seed) >>> 0;
      stop();
      experiment = null;
      pendingRecords = [];
      completed = false;
      desiredRunning = message.autoplay ?? true;
      speed = normalizeSpeed(message.speed ?? 1);
      if (!connectome) {
        if (!loading) {
          loading = FlyHamletConnectome.load(message.manifestURL, {progress: reportProgress})
            .then(loaded => { connectome = loaded.connectome; manifest = loaded.manifest; })
            .finally(() => { loading = null; });
        }
        await loading;
      }
      if (token !== generation) return;
      const scale = message.options?.slowWeightScale ?? FlyHamletChain.DEFAULTS.slowWeightScale;
      if (!prepared || prepared.metadata.weightScale !== scale) {
        // The rescaled graph is independent of experiment seeds. Concurrent New
        // requests share this reusable preparation, then only the newest
        // generation can instantiate its mutable neural/decoder state.
        if (!preparing) preparing = prepareSlow(scale).then(slow => { prepared = slow; }).finally(() => { preparing = null; });
        await preparing;
      }
      if (token !== generation) return;
      begin(message.options);
    } else if (message.type === 'pause') pause();
    else if (message.type === 'resume') resume();
    else if (message.type === 'speed') {
      speed = normalizeSpeed(message.value);
      if (experiment) { paceWall = now(); paceSim = experiment.simTimeS; }
    } else if (message.type === 'export') {
      sendTick(true);
      emit({type: 'export', result: experiment ? experiment.result() : null});
    }
  } catch (error) {
    if (token === generation) fail(error);
  }
};
