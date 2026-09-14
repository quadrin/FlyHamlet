/* Observer UI for the self-driven recall assay. Cue and prediction labels are for
 * inspection only. No UI choice or display history reaches the readout. */
(() => {
  'use strict';
  const $ = id => document.getElementById(`lab-${id}`);
  const {summarizeChainRun, CONDITIONS, DEFAULTS, END, START} = window.FlyHamletChain;
  const labels = {original: 'Original', slow: 'Slower dynamics', reset: 'Reset'};
  const colors = {original: '#78e5ed', slow: '#ffba79', reset: '#d0aff6'};
  const scriptURL = document.currentScript.src;
  const reference = DEFAULTS.phrase, decisionMs = DEFAULTS.cueMs + DEFAULTS.gapMs;
  const defaultBudget = 3 * ((DEFAULTS.trainEpisodes + DEFAULTS.diagnosticEpisodes) * (reference.length + 1) + DEFAULTS.recallEpisodes * DEFAULTS.maxDecisions);
  const defaultConclusion = 'No result yet. This experiment tests whether the network’s own state can carry its position in one trained phrase. It does not demonstrate learning by the connectome, language, or memory for the full play.';
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = value; };
  const symbol = value => value === ' ' ? '␣' : value === START ? 'START' : value === END ? 'END' : (value ?? '—');
  const printable = output => output.replace(/ /g, '␣');
  const percent = value => `${(100 * value).toFixed(1)}%`;
  const freshSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
  let worker = null, generation = 0, seed = null, active = false, started = false, comparatorLength = 6;
  let running = false, loading = false, result = null, progress = null, records = [], episodes = [];
  let metrics = summarizeChainRun([], [], reference), keys = [], layout = [], selectedLetter = null, currentOutput = '', currentEnded = false;

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
      const key = document.createElement('div'); key.className = `keycap${'to bern'.includes(letter) ? ' task-key' : ''}`;
      const face = document.createElement('span'); face.className = 'key-face';
      const label = document.createElement('span'); label.className = 'key-label'; label.textContent = symbol(letter).toUpperCase();
      face.append(label); key.append(face); fragment.append(key); keys[index] = key;
    }
    $('keyboard').replaceChildren(fragment); placeFly();
  }
  function placeFly() {
    const index = layout.indexOf(selectedLetter), grid = $('keyboard');
    const x = index < 0 ? .5 : (index % 9 + .5) / 9;
    const y = index < 0 ? .5 : (2 - Math.floor(index / 9) + .5) / 3;
    $('fly').style.left = `${grid.offsetLeft + x * grid.clientWidth}px`;
    $('fly').style.top = `${grid.offsetTop + y * grid.clientHeight}px`;
    keys.forEach((key, i) => key.classList.toggle('is-pressed', i === index));
  }
  function renderTyped() {
    const fragment = document.createDocumentFragment();
    Array.from(currentOutput).forEach((letter, index) => {
      const span = document.createElement('span'); span.textContent = letter;
      if (reference[index] !== letter) span.className = 'is-wrong';
      fragment.append(span);
    });
    if (currentEnded) { const end = document.createElement('span'); end.className = 'is-end'; end.textContent = '·END'; fragment.append(end); }
    $('typed').replaceChildren(fragment);
  }
  function onRecord(record) {
    if (record.phase === 'recall') {
      if (record.step === 1) { currentOutput = ''; currentEnded = false; }
      if (record.prediction === END) currentEnded = true; else currentOutput += record.prediction;
      renderTyped();
      selectedLetter = record.prediction === END ? null : record.prediction;
      text('choice', symbol(record.prediction));
      text('action', `${labels[record.condition]} · recall ${record.episode}: typed ${symbol(record.prediction)} from its own state`);
      if (currentEnded || record.step >= DEFAULTS.maxDecisions) {
        const finished = {condition: record.condition, episode: record.episode, output: currentOutput, ended: currentEnded};
        appendLog(finished);
      }
    } else {
      currentOutput = ''; currentEnded = false; renderTyped();
      selectedLetter = record.prediction && record.prediction !== END ? record.prediction : record.target && record.target !== END ? record.target : null;
      text('choice', record.prediction ? symbol(record.prediction) : '—');
      text('action', record.phase === 'train' ? `${labels[record.condition]} · training ${record.episode}: cue ${symbol(record.cue)} → target ${symbol(record.target)}`
        : `${labels[record.condition]} · diagnostic ${record.episode}: cue ${symbol(record.cue)} → ${symbol(record.prediction)} (${record.correct ? 'correct' : 'wrong'})`);
    }
    placeFly();
  }
  function appendLog(episode) {
    const entry = document.createElement('span'); entry.className = 'memory-log-entry';
    const label = document.createElement('b'); label.textContent = `${labels[episode.condition]} / recall episode ${episode.episode}`;
    const output = document.createElement('span'); output.textContent = `${printable(episode.output) || '(nothing)'}${episode.ended ? ' ·END' : ' ·cap'}`;
    const note = document.createElement('small');
    let prefix = 0; while (prefix < episode.output.length && episode.output[prefix] === reference[prefix]) prefix++;
    note.textContent = `Chain ${prefix} of ${reference.length}${episode.output === reference && episode.ended ? ' · exact' : ''}`;
    entry.append(label, output, note); $('output').append(entry); $('output').scrollTop = $('output').scrollHeight;
    text('output-phase', labels[episode.condition]);
    text('observations', `${$('output').childElementCount} recall episodes`);
  }
  function drawChart(canvas) {
    const ctx = canvas.getContext('2d'), width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
    const left = 34, right = 14, top = 14, bottom = 34, max = reference.length + 1, groups = DEFAULTS.recallEpisodes;
    const Y = value => top + (1 - value / max) * (height - top - bottom);
    const groupWidth = (width - left - right) / groups, barWidth = Math.max(3, Math.min(14, groupWidth / 4.5));
    ctx.font = '10px monospace'; ctx.textAlign = 'right'; ctx.lineWidth = 1;
    for (const value of [0, 6, 12, 18]) {
      ctx.strokeStyle = '#345566'; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(left, Y(value)); ctx.lineTo(width - right, Y(value)); ctx.stroke();
      ctx.fillStyle = '#b8cbd7'; ctx.fillText(value, left - 6, Y(value) + 3);
    }
    ctx.strokeStyle = '#e4ecf1'; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(left, Y(reference.length)); ctx.lineTo(width - right, Y(reference.length)); ctx.stroke();
    ctx.strokeStyle = '#91a6b3'; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(left, Y(comparatorLength)); ctx.lineTo(width - right, Y(comparatorLength)); ctx.stroke();
    ctx.setLineDash([]); ctx.textAlign = 'center';
    for (let episode = 0; episode < groups; ++episode) {
      const center = left + (episode + .5) * groupWidth;
      ctx.fillStyle = '#b8cbd7'; ctx.fillText(`ep ${episode + 1}`, center, height - 20);
      CONDITIONS.forEach((condition, index) => {
        const value = metrics[condition].recall.chainLengths[episode];
        if (value === undefined) return;
        const x = center + (index - 1) * (barWidth + 3) - barWidth / 2;
        ctx.fillStyle = colors[condition]; ctx.fillRect(x, Y(value), barWidth, Y(0) - Y(value));
      });
    }
    ctx.fillStyle = '#91aebb'; ctx.font = '9px monospace'; ctx.fillText('Autonomous recall episode', (left + width - right) / 2, height - 6);
  }
  function updateResults() {
    metrics = result ? result.metrics : summarizeChainRun(episodes, records, reference);
    for (const condition of CONDITIONS) {
      const recall = metrics[condition].recall;
      text(`${condition}-score`, recall.episodes ? recall.meanChainLength.toFixed(1) : '—');
      text(`${condition}-note`, recall.episodes ? `of ${reference.length} · longest ${recall.maxChainLength} · exact ${recall.exact}/${recall.episodes}`
        : condition === 'original' ? 'Of 18 characters, 6 recall episodes' : condition === 'slow' ? 'Explicit model hypothesis' : 'No state carried between cues');
    }
    const rows = [
      ['Next-letter accuracy (teacher-forced, held out)', condition => { const d = metrics[condition].diagnostic; return d.total ? [`${percent(d.accuracy)} · ${d.correct}/${d.total}`, `95% CI ${percent(d.wilson95[0])}–${percent(d.wilson95[1])} · chance ${percent(d.chance)}`] : null; }],
      ['Recall chain lengths', condition => { const r = metrics[condition].recall; return r.episodes ? [r.chainLengths.join(' · '), `mean ${r.meanChainLength.toFixed(1)} · exact ${r.exact}/${r.episodes} · ended ${r.ended}, capped ${r.capped}`] : null; }],
      ['Recall outputs', condition => { const list = (result ? result.episodes : episodes).filter(item => item.condition === condition && item.phase === 'recall'); return list.length ? [null, list.map(item => printable(item.output) || '(nothing)').join('\n')] : null; }],
      ['Mean edit distance', condition => { const r = metrics[condition].recall; return r.episodes ? [r.meanEditDistance.toFixed(1), `from “${reference}”`] : null; }],
      ['Activity at recall snapshots', condition => { const a = metrics[condition].recall.activity; return a.decisions ? [`${a.meanCueSpikes.toFixed(0)} spikes per cue`, `${a.meanNonrestVoltageNeurons.toFixed(0)} neurons off rest · ${a.silentSnapshots}/${a.decisions} silent`] : null; }]
    ];
    const fragment = document.createDocumentFragment();
    for (const [name, cellFor] of rows) {
      const row = document.createElement('tr'), label = document.createElement('th'); label.scope = 'row'; label.textContent = name; row.append(label);
      for (const condition of CONDITIONS) {
        const cell = document.createElement('td'), value = cellFor(condition);
        if (!value) cell.textContent = '—';
        else {
          if (value[0] !== null) { const b = document.createElement('b'); b.textContent = value[0]; cell.append(b); }
          const small = document.createElement('small'); small.textContent = value[1]; if (name === 'Recall outputs') small.className = 'chain-outputs'; cell.append(small);
        }
        row.append(cell);
      }
      fragment.append(row);
    }
    $('delay-results').replaceChildren(fragment); drawChart($('chart')); syncControls();
  }
  function updateProgress(next) {
    progress = next; $('progress').max = next.totalTrials; $('progress').value = next.completed;
    text('progress-text', `${next.completed} / ${next.totalTrials} decisions`); text('time', `${next.simTimeS.toFixed(2)} s simulated`);
    text('cue', symbol(next.cue)); text('drive', next.inputActive ? 'ON' : 'OFF');
    text('stage', next.stage === 'cue' ? 'Presenting cue' : next.stage === 'gap' ? 'Gap · reading state' : next.stage === 'complete' ? 'Experiment complete' : 'Awaiting episode');
    text('trial-time', `${next.cueTimeMs.toFixed(0)} / ${decisionMs} ms`); $('clock').style.left = `${Math.min(100, next.cueTimeMs / decisionMs * 100)}%`;
    text('episode', next.phase === 'complete' ? 'All conditions complete' : `${labels[next.condition]} · ${next.phase} episode ${next.episode}/${next.episodeTotal} · step ${next.step}`);
    const conditionIndex = next.phase === 'complete' ? 3 : CONDITIONS.indexOf(next.condition);
    document.querySelectorAll('.phase-track li').forEach((element, index) => {
      element.classList.toggle('is-current', index === conditionIndex); element.classList.toggle('is-complete', index < conditionIndex);
    });
    text('lock', next.phase === 'train' ? 'Collecting training snapshots' : next.phase === 'complete' ? 'All conditions complete' : 'Current condition: fitted readout frozen');
    if (running && next.condition) status(`${labels[next.condition]} · ${next.phase} ${next.episode}/${next.episodeTotal}`, 'playing');
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
    const link = document.createElement('a'); link.href = url; link.download = `flyhamlet-chain-${data.seed}${data.complete ? '' : '-partial'}.json`;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function verifyFrozen(data) {
    if (!data.complete) return false;
    return CONDITIONS.every(condition => {
      const model = data.models[condition];
      return model?.fitCount === 1 && model.fingerprint === model.finalFingerprint && model.scaler?.fingerprint === model.scaler?.finalFingerprint &&
        JSON.stringify(model.weights) === JSON.stringify(model.frozenWeights);
    });
  }
  function conclusion(data) {
    const describe = condition => {
      const r = data.metrics[condition].recall, d = data.metrics[condition].diagnostic;
      return `${labels[condition]}: next-letter accuracy ${percent(d.accuracy)} teacher-forced; autonomous chains ${r.chainLengths.join(', ')} (mean ${r.meanChainLength.toFixed(1)} of ${reference.length}, exact ${r.exact}/${r.episodes}).`;
    };
    return `${describe('original')} ${describe('slow')} ${describe('reset')} The no-memory comparator reaches ${comparatorLength}. Readouts stayed frozen. The slower model is an imposed hypothesis, and the readout is external; a long chain shows position carried in this simulation’s state, not learning by the connectome or memory for a text.`;
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
        if (record.trial !== records.length + 1) { fail('The decision log arrived out of order. Start a new experiment.'); return; }
        records.push(record); onRecord(record);
        if (record.phase === 'recall' && (record.prediction === END || record.step >= DEFAULTS.maxDecisions)) {
          const rows = records.filter(row => row.condition === record.condition && row.phase === 'recall' && row.episode === record.episode);
          const output = rows.map(row => row.prediction).filter(code => code !== END).join('');
          let prefix = 0; while (prefix < output.length && output[prefix] === reference[prefix]) prefix++;
          const stoppedBy = record.prediction === END ? END : 'cap';
          episodes.push({condition: record.condition, phase: 'recall', episode: record.episode, output, stoppedBy, exact: output === reference && stoppedBy === END, chainLength: prefix, editDistance: 0});
        }
      }
      updateProgress(message.progress); text('rate', `${message.realTimeRatio.toFixed(2)}× actual speed`);
      if (message.records.length) updateResults();
    } else if (message.type === 'complete') {
      if (!verifyFrozen(message.result)) { fail('A frozen-readout check failed. This run is invalid.'); return; }
      result = message.result; running = false; status('Experiment complete', 'complete'); text('drive', 'OFF');
      comparatorLength = result.comparator.episode.chainLength; text('comparator', comparatorLength);
      text('lock', 'Verified: all evaluation scalers and weights unchanged'); text('conclusion', conclusion(result)); updateResults();
    } else if (message.type === 'export') download(message.result);
  }
  function clearDisplay() {
    progress = null; selectedLetter = null; currentOutput = ''; currentEnded = false; renderTyped(); placeFly(); $('output').replaceChildren();
    text('output-phase', 'Awaiting recall'); text('observations', '0 recall episodes'); text('cue', '—'); text('choice', '—'); text('drive', 'OFF');
    text('action', 'Waiting for the first decision'); text('episode', 'Awaiting trial');
    text('stage', 'Awaiting episode'); text('trial-time', `0 / ${decisionMs} ms`); $('clock').style.left = '0%';
    $('progress').value = 0; text('progress-text', `0 / ${defaultBudget} decisions`); text('time', '0.00 s simulated');
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
    $('seed').setCustomValidity(''); seed = value; active = true; loading = true; records = []; episodes = []; result = null; clearDisplay();
    $('load').hidden = false; $('load').classList.remove('is-error'); $('retry').hidden = true;
    text('load-message', 'Loading the full connectome. First use downloads about 49 MB, then prepares the comparison network.');
    status('Preparing experiment', 'loading'); syncControls();
    try {
      if (!worker) {
        worker = new Worker(new URL('chain-worker.js?v=1', scriptURL)); generation = 0;
        worker.addEventListener('message', onMessage);
        worker.addEventListener('error', () => fail('The experiment worker could not continue. Check your connection and try again.'));
      }
      generation++;
      worker.postMessage({type: 'init', seed, speed: Number($('speed').value), autoplay: !document.hidden, manifestURL: new URL('model/manifest.json', scriptURL).href});
    } catch (error) { fail(error.message || 'Could not start this browser’s experiment worker.'); }
  }
  function reset() {
    active = false; worker?.postMessage({type: 'pause'});
    started = false; running = false; loading = false; result = null; records = []; episodes = [];
    $('seed').value = freshSeed(); $('seed').setCustomValidity(''); $('load').hidden = true;
    clearDisplay(); status('Ready for a new experiment'); syncControls();
  }
  $('run').addEventListener('click', () => { if (!started) start(); else worker?.postMessage({type: running ? 'pause' : 'resume'}); });
  $('new').addEventListener('click', reset); $('retry').addEventListener('click', start);
  $('seed').addEventListener('input', () => $('seed').setCustomValidity(''));
  $('speed').addEventListener('change', () => { worker?.postMessage({type: 'speed', value: Number($('speed').value)}); if (!started) text('rate', `Target speed ${$('speed').value}×`); });
  $('export').addEventListener('click', () => result ? download(result) : worker?.postMessage({type: 'export'}));
  document.addEventListener('visibilitychange', () => { if (document.hidden && active && (running || loading)) worker?.postMessage({type: 'pause'}); });
  const resize = () => { placeFly(); drawChart($('chart')); };
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(resize); observer.observe($('chart')); observer.observe($('keyboard'));
  } else window.addEventListener('resize', resize);
  $('speed').value = '1'; buildKeyboard(Array.from('rezndomstyawukihbpgqf xcvjl')); reset();
})();
