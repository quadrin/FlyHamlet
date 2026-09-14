/* Observer UI for the two-letter state-decoding assay. Stimulus and prediction
 * labels are for inspection only. No UI choice or display history reaches a readout. */
(() => {
  'use strict';
  const $ = id => document.getElementById(`lab-${id}`);
  const {summarizeSequenceTrials, CONDITIONS, POSITIONS, DEFAULTS} = window.FlyHamletSequence;
  const labels = {original: 'Original', slow: 'Slower dynamics', reset: 'Reset'};
  const colors = {original: '#78e5ed', slow: '#ffba79', reset: '#d0aff6'};
  const scriptURL = document.currentScript.src;
  const delays = DEFAULTS.delaysMs;
  const trialMs = 2 * DEFAULTS.letterMs + DEFAULTS.gapMs + delays[delays.length - 1];
  const totalTrials = 3 * 4 * (DEFAULTS.trainPerPair + DEFAULTS.evaluationPerPair);
  const defaultConclusion = 'No result yet. This experiment tests recoverable information in neural state. It does not demonstrate autonomous typing, learned motor control, language, or memory for the full play.';
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = value; };
  const symbol = value => value === ' ' ? '␣' : (value ?? '—');
  const pairText = pair => pair ? `${pair[0]} ${pair[1]}` : '— —';
  const percent = value => `${(100 * value).toFixed(1)}%`;
  const freshSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
  const snapshotOf = (record, delay) => delay === 0 ? record.cueEnd : record.snapshots.find(item => item.delayMs === delay);
  const metricOf = (condition, delay) => delay === 0 ? metrics[condition].cueEnd : metrics[condition].delays.find(item => item.delayMs === delay);
  let worker = null, generation = 0, seed = null, active = false, started = false;
  let running = false, loading = false, result = null, progress = null, records = [];
  let metrics = summarizeSequenceTrials([]), keys = [], layout = [], selected = [null, null];

  function status(message, state = 'paused') { text('status', message); $('status').dataset.state = state; }
  function syncControls() {
    $('run').disabled = loading || Boolean(result);
    text('run', result ? 'Experiment complete' : running ? 'Pause' : started ? 'Resume' : 'Run experiment');
    $('run').setAttribute('aria-pressed', String(running)); $('seed').disabled = loading || started;
    $('export').disabled = !records.length || (!worker && !result);
  }
  function buildKeyboard(nextLayout) {
    layout = nextLayout; keys = Array(layout.length); const fragment = document.createDocumentFragment();
    for (let row = 2; row >= 0; --row) for (let col = 0; col < 9; ++col) {
      const index = row * 9 + col, letter = layout[index];
      const key = document.createElement('div'); key.className = `keycap${'to'.includes(letter) ? ' task-key' : ''}`;
      const face = document.createElement('span'); face.className = 'key-face';
      const label = document.createElement('span'); label.className = 'key-label'; label.textContent = symbol(letter).toUpperCase();
      face.append(label); key.append(face); fragment.append(key); keys[index] = key;
    }
    $('keyboard').replaceChildren(fragment); placeFly();
  }
  function placeFly() {
    const [first, second] = selected;
    const firstIndex = layout.indexOf(first), index = layout.indexOf(second), grid = $('keyboard');
    const x = index < 0 ? .5 : (index % 9 + .5) / 9;
    const y = index < 0 ? .5 : (2 - Math.floor(index / 9) + .5) / 3;
    $('fly').style.left = `${grid.offsetLeft + x * grid.clientWidth}px`;
    $('fly').style.top = `${grid.offsetTop + y * grid.clientHeight}px`;
    keys.forEach((key, i) => { key.classList.toggle('is-pressed', i === index || i === firstIndex); key.classList.toggle('is-first', i === firstIndex && first !== null); });
  }
  function inspectLatest() {
    const delay = Number($('inspect').value);
    const record = [...records].reverse().find(row => row.phase === 'evaluation' && (!progress?.condition || row.condition === progress.condition));
    if (!record) {
      selected = [null, null]; text('choice', '— —'); text('action', 'Waiting for held-out decisions');
      text('inspect-note', 'Only changes this display'); placeFly(); return;
    }
    const snapshot = snapshotOf(record, delay);
    selected = [snapshot.predictions.first.prediction, snapshot.predictions.second.prediction]; placeFly();
    text('choice', pairText(snapshot.pairPrediction));
    text('inspect-note', `${snapshot.nonrestVoltageNeurons} neurons off rest · ${snapshot.pairCorrect ? 'exact pair' : snapshot.predictions.second.correct ? 'second letter only' : snapshot.predictions.first.correct ? 'first letter only' : 'both wrong'}`);
    text('action', `Last test: ${pairText(record.pair)} → ${pairText(snapshot.pairPrediction)} at ${delay === 0 ? 'final cue step' : `${delay} ms`} · ${labels[record.condition]}`);
  }
  function appendLog(record) {
    if (record.phase !== 'evaluation') return;
    const entry = document.createElement('span'); entry.className = 'memory-log-entry';
    const label = document.createElement('b'); label.textContent = `${labels[record.condition]} / test ${record.phaseTrial} / pair ${record.pair}`;
    const choices = document.createElement('span'); choices.textContent = [0, ...delays].map(delay => `${delay}:${snapshotOf(record, delay).pairPrediction}`).join('  ');
    const note = document.createElement('small');
    const primary = snapshotOf(record, delays.includes(100) ? 100 : delays[0]);
    note.textContent = `At ${primary.delayMs} ms: first ${primary.predictions.first.correct ? 'correct' : 'wrong'} · second ${primary.predictions.second.correct ? 'correct' : 'wrong'}`;
    entry.append(label, choices, note); $('output').append(entry); $('output').scrollTop = $('output').scrollHeight;
    text('output-phase', labels[record.condition]);
    text('observations', `${records.filter(row => row.phase === 'evaluation').length} evaluated trials`);
  }
  function drawChart(canvas) {
    const ctx = canvas.getContext('2d'), width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
    const left = 42, right = 17, top = 17, bottom = 37, span = delays[delays.length - 1];
    const X = delay => left + delay / span * (width - left - right);
    const Y = value => top + (1 - value) * (height - top - bottom);
    ctx.font = '10px monospace'; ctx.textAlign = 'right'; ctx.lineWidth = 1;
    for (const value of [0, .5, 1]) {
      ctx.strokeStyle = '#345566'; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(X(0), Y(value)); ctx.lineTo(X(span), Y(value)); ctx.stroke();
      ctx.fillStyle = '#b8cbd7'; ctx.fillText(`${Math.round(value * 100)}%`, left - 7, Y(value) + 3);
    }
    ctx.strokeStyle = '#91a6b3'; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(X(0), Y(.25)); ctx.lineTo(X(span), Y(.25)); ctx.stroke();
    ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(X(0), Y(.5)); ctx.lineTo(X(span), Y(.5)); ctx.stroke();
    ctx.setLineDash([]); ctx.textAlign = 'center';
    for (const delay of [0, ...delays]) ctx.fillText(delay, X(delay), height - 23);
    ctx.fillStyle = '#91aebb'; ctx.font = '9px monospace'; ctx.fillText('Time after input off (ms); 0 = final cue step, input on', (left + width - right) / 2, height - 7);
    for (const condition of CONDITIONS) {
      const rows = [0, ...delays].map(delay => metricOf(condition, delay)).filter(row => row.total);
      ctx.strokeStyle = colors[condition]; ctx.fillStyle = colors[condition];
      ctx.globalAlpha = .55; ctx.lineWidth = .8; ctx.setLineDash([]);
      for (const row of rows) {
        const x = X(row.delayMs), low = Y(row.wilson95[0]), high = Y(row.wilson95[1]);
        ctx.beginPath(); ctx.moveTo(x, low); ctx.lineTo(x, high);
        ctx.moveTo(x - 2, low); ctx.lineTo(x + 2, low); ctx.moveTo(x - 2, high); ctx.lineTo(x + 2, high); ctx.stroke();
      }
      ctx.globalAlpha = 1; ctx.lineWidth = 1.7;
      ctx.setLineDash(condition === 'slow' ? [5, 3] : condition === 'reset' ? [2, 3] : []); ctx.beginPath();
      rows.forEach((row, i) => { const x = X(row.delayMs), y = Y(row.accuracy); if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke(); ctx.setLineDash([]);
      for (const row of rows) {
        ctx.beginPath(); ctx.arc(X(row.delayMs), Y(row.accuracy), 2.7, 0, Math.PI * 2);
        if (row.inputActive) { ctx.lineWidth = 1.4; ctx.stroke(); } else ctx.fill();
      }
    }
  }
  function updateResults() {
    metrics = result ? result.metrics : summarizeSequenceTrials(records);
    const focus = delays.includes(100) ? 100 : delays[0];
    for (const condition of CONDITIONS) {
      const metric = metricOf(condition, focus);
      text(`${condition}-score`, metric.total ? percent(metric.accuracy) : '—');
      text(`${condition}-note`, metric.total ? `${metric.correct}/${metric.total} · first ${percent(metric.first.accuracy)} · second ${percent(metric.second.accuracy)}`
        : condition === 'original' ? 'Exact pair, held-out noise' : condition === 'slow' ? 'Explicit model hypothesis' : 'No remaining neural state');
    }
    const fragment = document.createDocumentFragment();
    for (const delay of [0, ...delays]) {
      const row = document.createElement('tr'), label = document.createElement('th'); label.scope = 'row';
      label.textContent = delay === 0 ? 'Final cue step · input on (diagnostic)' : `${delay} ms after off`; row.append(label);
      for (const condition of CONDITIONS) {
        const metric = metricOf(condition, delay), cell = document.createElement('td');
        if (metric.total) {
          const accuracy = document.createElement('b'); accuracy.textContent = `${percent(metric.accuracy)} · ${metric.correct}/${metric.total}`;
          accuracy.title = `95% Wilson interval: ${percent(metric.wilson95[0])}–${percent(metric.wilson95[1])}`;
          const positions = document.createElement('small'); positions.textContent = `first ${percent(metric.first.accuracy)} · second ${percent(metric.second.accuracy)}`;
          const state = document.createElement('small'); state.textContent = `${metric.meanNonrestVoltageNeurons.toFixed(0)} neurons off rest · ${metric.silentTrials}/${metric.total} silent`;
          cell.append(accuracy, positions, state);
        } else cell.textContent = '—';
        row.append(cell);
      }
      fragment.append(row);
    }
    $('delay-results').replaceChildren(fragment); drawChart($('chart')); syncControls();
  }
  function updateProgress(next) {
    progress = next; $('progress').max = next.totalTrials; $('progress').value = next.completed;
    text('progress-text', `${next.completed} / ${next.totalTrials} trials`); text('time', `${next.simTimeS.toFixed(2)} s simulated`);
    text('cue', symbol(next.cue)); text('drive', next.inputActive ? 'ON' : 'OFF');
    text('stage', {first: 'Presenting first letter', gap: 'Gap · no input', second: 'Presenting second letter',
      delay: 'Input off · neural state evolves', complete: 'Experiment complete'}[next.stage] || 'Awaiting trial');
    text('trial-time', `${next.trialTimeMs.toFixed(0)} / ${trialMs} ms`); $('clock').style.left = `${Math.min(100, next.trialTimeMs / trialMs * 100)}%`;
    const conditionIndex = next.phase === 'complete' ? 3 : CONDITIONS.indexOf(next.condition);
    document.querySelectorAll('.phase-track li').forEach((element, index) => {
      element.classList.toggle('is-current', index === conditionIndex); element.classList.toggle('is-complete', index < conditionIndex);
    });
    text('lock', next.phase === 'evaluation' ? 'Current condition: fitted readouts frozen' : next.phase === 'complete' ? 'All conditions complete' : 'Collecting training observations');
    if (running && next.condition) status(`${labels[next.condition]} · ${next.phase === 'train' ? 'training' : 'evaluation'} ${next.phaseTrial}/${next.phaseTotal}`, 'playing');
    inspectLatest();
  }
  function fail(message) {
    running = false; loading = false; started = false; active = false;
    worker?.terminate(); worker = null; generation = 0;
    status('Experiment stopped', 'error'); $('load').hidden = false; $('load').classList.add('is-error'); $('retry').hidden = false;
    text('load-message', message); syncControls();
  }
  function download(data) {
    if (!data) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = `flyhamlet-sequence-${data.seed}${data.complete ? '' : '-partial'}.json`;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function verifyFrozen(data) {
    if (!data.complete) return false;
    return CONDITIONS.every(condition => {
      const model = data.models[condition];
      return model?.fitCount === 1 && model.readouts.length === delays.length + 1 && model.readouts.every(readout =>
        readout.scaler && readout.scaler.fingerprint === readout.scaler.finalFingerprint && POSITIONS.every(position => {
          const fit = readout.positions[position];
          return fit && fit.fingerprint === fit.finalFingerprint && JSON.stringify(fit.weights) === JSON.stringify(fit.frozenWeights);
        }));
    });
  }
  function conclusion(data) {
    const describe = condition => {
      const rows = data.metrics[condition].delays, last = rows[rows.length - 1];
      return `${labels[condition]}: exact pair ${rows.map(row => `${percent(row.accuracy)} at ${row.delayMs} ms`).join(', ')}; first letter ${percent(last.first.accuracy)} and second letter ${percent(last.second.accuracy)} at ${last.delayMs} ms.`;
    };
    return `${describe('original')} ${describe('slow')} ${describe('reset')} Chance is 25% for the pair and 50% per letter; last-letter-only recall gives 50%. Readouts stayed frozen. The slower model is an imposed hypothesis, not measured physiology; these results concern decodable state in this simulation, not typing or memory for a text.`;
  }
  function onMessage(event) {
    const message = event.data;
    if (!active || message.seed !== seed || message.generation !== generation) return;
    if (message.type === 'progress') {
      const percentDone = message.total > 0 ? ` · ${Math.floor(100 * message.loaded / message.total)}%` : '';
      text('load-message', `${message.message}${percentDone}`); return;
    }
    if (message.type === 'error') { fail(message.message); return; }
    if (message.type === 'ready') {
      loading = false; started = true; $('load').hidden = true;
      buildKeyboard(message.metadata.layout); updateProgress(message.progress); syncControls();
    } else if (message.type === 'state') {
      running = message.running; syncControls();
      if (!running && !result && started) status('Paused · experiment state retained');
      if (running && progress) updateProgress(progress);
    } else if (message.type === 'tick') {
      for (const record of message.records) {
        if (record.trial !== records.length + 1) { fail('The trial log arrived out of order. Start a new experiment.'); return; }
        records.push(record); appendLog(record);
      }
      updateProgress(message.progress); text('rate', `${message.realTimeRatio.toFixed(2)}× actual speed`);
      if (message.records.length) updateResults();
    } else if (message.type === 'complete') {
      if (!verifyFrozen(message.result)) { fail('A frozen-readout check failed. This run is invalid.'); return; }
      result = message.result; running = false; status('Experiment complete', 'complete'); text('drive', 'OFF');
      text('lock', 'Verified: all evaluation scalers and weights unchanged'); text('conclusion', conclusion(result)); updateResults();
    } else if (message.type === 'export') download(message.result);
  }
  function clearDisplay() {
    progress = null; selected = [null, null]; placeFly(); $('output').replaceChildren();
    text('output-phase', 'Awaiting evaluation'); text('observations', '0 evaluated trials'); text('cue', '—'); text('choice', '— —'); text('drive', 'OFF');
    text('action', 'Waiting for held-out decisions'); text('inspect-note', 'Only changes this display');
    text('stage', 'Awaiting trial'); text('trial-time', `0 / ${trialMs} ms`); $('clock').style.left = '0%';
    $('progress').value = 0; text('progress-text', `0 / ${totalTrials} trials`); text('time', '0.00 s simulated');
    text('lock', 'Readouts have not been fitted'); text('rate', `Target speed ${$('speed').value}×`);
    text('conclusion', defaultConclusion);
    document.querySelectorAll('.phase-track li').forEach(element => element.classList.remove('is-current', 'is-complete'));
    updateResults();
  }
  function start() {
    const value = Number($('seed').value);
    if (!$('seed').value.trim() || !Number.isInteger(value) || value < 0 || value > 4294967295) {
      $('seed').setCustomValidity('Enter an integer from 0 to 4294967295.'); $('seed').reportValidity(); return;
    }
    $('seed').setCustomValidity(''); seed = value; active = true; loading = true; records = []; result = null; clearDisplay();
    $('load').hidden = false; $('load').classList.remove('is-error'); $('retry').hidden = true;
    text('load-message', 'Loading the full connectome. First use downloads about 49 MB, then prepares the comparison network.');
    status('Preparing experiment', 'loading'); syncControls();
    try {
      if (!worker) {
        worker = new Worker(new URL('sequence-worker.js?v=1', scriptURL)); generation = 0;
        worker.addEventListener('message', onMessage);
        worker.addEventListener('error', () => fail('The experiment worker could not continue. Check your connection and try again.'));
      }
      generation++;
      worker.postMessage({type: 'init', seed, speed: Number($('speed').value), autoplay: !document.hidden, manifestURL: new URL('model/manifest.json', scriptURL).href});
    } catch (error) { fail(error.message || 'Could not start this browser’s experiment worker.'); }
  }
  function reset() {
    active = false; worker?.postMessage({type: 'pause'});
    started = false; running = false; loading = false; result = null; records = [];
    $('seed').value = freshSeed(); $('seed').setCustomValidity(''); $('load').hidden = true;
    clearDisplay(); status('Ready for a new experiment'); syncControls();
  }
  $('run').addEventListener('click', () => { if (!started) start(); else worker?.postMessage({type: running ? 'pause' : 'resume'}); });
  $('new').addEventListener('click', reset); $('retry').addEventListener('click', start);
  $('seed').addEventListener('input', () => $('seed').setCustomValidity(''));
  $('inspect').addEventListener('change', inspectLatest);
  $('speed').addEventListener('change', () => { worker?.postMessage({type: 'speed', value: Number($('speed').value)}); if (!started) text('rate', `Target speed ${$('speed').value}×`); });
  $('export').addEventListener('click', () => result ? download(result) : worker?.postMessage({type: 'export'}));
  document.addEventListener('visibilitychange', () => { if (document.hidden && active && (running || loading)) worker?.postMessage({type: 'pause'}); });
  const resize = () => { placeFly(); drawChart($('chart')); };
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(resize); observer.observe($('chart')); observer.observe($('keyboard'));
  } else window.addEventListener('resize', resize);
  $('speed').value = '1'; $('inspect').value = '100'; buildKeyboard(Array.from('rezndomstyawukihbpgqf xcvjl')); reset();
})();
