/* Observer UI for the in-brain mushroom-body assay. Cue and prediction labels are for
 * inspection only. No UI choice or display history reaches the readout. */
(() => {
  'use strict';
  const $ = id => document.getElementById(`lab-${id}`);
  const {summarizeMushroomBodyRun, CONDITIONS, DEFAULTS, END, START} = window.FlyHamletMushroomBody;
  const OUTPUT_CODES = window.FlyHamletRecall.OUTPUT_CODES;
  const labels = {plastic: 'Plastic', plasticReset: 'Plastic, reset', frozen: 'Frozen'};
  const colors = {plastic: '#78e5ed', plasticReset: '#ffba79', frozen: '#d0aff6'};
  const scriptURL = document.currentScript.src;
  const reference = DEFAULTS.phrase, decisionMs = DEFAULTS.cueMs + DEFAULTS.gapMs;
  const defaultBudget = 3 * ((DEFAULTS.trainEpisodes + DEFAULTS.diagnosticEpisodes) * (reference.length + 1) + DEFAULTS.recallEpisodes * DEFAULTS.maxDecisions);
  const defaultConclusion = 'No result yet. This experiment tests whether existing synapses of the simulated mushroom body can store one phrase under an imposed slower model and an external teacher. It does not show how flies learn, and it is not memory for the full play.';
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = value; };
  const symbol = value => value === ' ' ? '␣' : value === START ? 'START' : value === END ? 'END' : (value ?? '—');
  const printable = output => output.replace(/ /g, '␣');
  const percent = value => `${(100 * value).toFixed(1)}%`;
  const freshSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
  let worker = null, generation = 0, seed = null, active = false, started = false, comparatorLength = 6;
  let running = false, loading = false, result = null, progress = null, records = [], episodes = [];
  let metrics = summarizeMushroomBodyRun([], [], reference), keys = [], layout = [], selectedLetter = null, currentOutput = '', currentEnded = false;

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
  function renderVote(counts, target, winner) {
    const cells = $('vote').children;
    const max = Math.max(1, ...counts);
    OUTPUT_CODES.forEach((code, g) => {
      const cell = cells[g]; cell.querySelector('i').style.height = `${100 * counts[g] / max}%`;
      cell.querySelector('b').textContent = counts[g] ? counts[g] : '';
      cell.classList.toggle('is-target', target === code); cell.classList.toggle('is-winner', winner === code);
    });
  }
  function buildVote() {
    const fragment = document.createDocumentFragment();
    for (const code of OUTPUT_CODES) {
      const cell = document.createElement('div'), bar = document.createElement('i'), label = document.createElement('span'), count = document.createElement('b');
      label.textContent = symbol(code); cell.append(count, bar, label); fragment.append(cell);
    }
    $('vote').replaceChildren(fragment);
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
      selectedLetter = record.prediction !== END ? record.prediction : null;
      text('choice', symbol(record.prediction));
      text('action', record.phase === 'train' ? `${labels[record.condition]} · training ${record.episode}: cue ${symbol(record.cue)} → ${symbol(record.prediction)} (target ${symbol(record.target)}${record.learningApplied ? `, ${record.potentiated + record.depressed} synapses moved` : ', no update'})`
        : `${labels[record.condition]} · diagnostic ${record.episode}: cue ${symbol(record.cue)} → ${symbol(record.prediction)} (${record.correct ? 'correct' : 'wrong'})`);
    }
    renderVote(record.groupCounts, record.target, record.prediction);
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
  function drawCurve(canvas) {
    const ctx = canvas.getContext('2d'), width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
    const left = 40, right = 14, top = 12, bottom = 30, episodes = DEFAULTS.trainEpisodes;
    const X = episode => left + (episode - 1) / Math.max(1, episodes - 1) * (width - left - right);
    const Y = value => top + (1 - value) * (height - top - bottom);
    ctx.font = '10px monospace'; ctx.textAlign = 'right'; ctx.lineWidth = 1;
    for (const value of [0, .5, 1]) {
      ctx.strokeStyle = '#345566'; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(X(1), Y(value)); ctx.lineTo(X(episodes), Y(value)); ctx.stroke();
      ctx.fillStyle = '#b8cbd7'; ctx.fillText(`${Math.round(value * 100)}%`, left - 6, Y(value) + 3);
    }
    ctx.strokeStyle = '#91a6b3'; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(X(1), Y(1 / 8)); ctx.lineTo(X(episodes), Y(1 / 8)); ctx.stroke();
    ctx.setLineDash([]); ctx.textAlign = 'center';
    for (let episode = 1; episode <= episodes; ++episode) ctx.fillText(episode, X(episode), height - 16);
    ctx.fillStyle = '#91aebb'; ctx.font = '9px monospace'; ctx.fillText('Training episode (chance 12.5%)', (left + width - right) / 2, height - 4);
    for (const condition of CONDITIONS) {
      const rows = metrics[condition].training.curve.filter(row => row.total);
      ctx.strokeStyle = colors[condition]; ctx.fillStyle = colors[condition]; ctx.lineWidth = 1.7;
      ctx.setLineDash(condition === 'plasticReset' ? [5, 3] : condition === 'frozen' ? [2, 3] : []); ctx.beginPath();
      rows.forEach((row, i) => { const x = X(row.episode), y = Y(row.accuracy); if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke(); ctx.setLineDash([]);
      for (const row of rows) { ctx.beginPath(); ctx.arc(X(row.episode), Y(row.accuracy), 2.5, 0, Math.PI * 2); ctx.fill(); }
    }
  }
  function updateResults() {
    metrics = result ? result.metrics : summarizeMushroomBodyRun(episodes, records, reference);
    for (const condition of CONDITIONS) {
      const recall = metrics[condition].recall;
      text(`${condition}-score`, recall.episodes ? recall.meanChainLength.toFixed(1) : '—');
      text(`${condition}-note`, recall.episodes ? `of ${reference.length} · longest ${recall.maxChainLength} · exact ${recall.exact}/${recall.episodes}`
        : condition === 'plastic' ? 'Of 18 characters, 6 recall episodes' : condition === 'plasticReset' ? 'No state carried between cues' : 'Same readout, no learning');
    }
    const rows = [
      ['Training accuracy by episode (learning on)', condition => { const t = metrics[condition].training; return t.curve.length ? [t.curve.map(row => percent(row.accuracy)).join(' · '), `${t.updates} synaptic updates`] : null; }],
      ['Next-letter accuracy (teacher-forced, learning off)', condition => { const d = metrics[condition].diagnostic; return d.total ? [`${percent(d.accuracy)} · ${d.correct}/${d.total}`, `95% CI ${percent(d.wilson95[0])}–${percent(d.wilson95[1])} · chance ${percent(d.chance)}`] : null; }],
      ['Recall chain lengths', condition => { const r = metrics[condition].recall; return r.episodes ? [r.chainLengths.join(' · '), `mean ${r.meanChainLength.toFixed(1)} · exact ${r.exact}/${r.episodes} · ended ${r.ended}, capped ${r.capped}`] : null; }],
      ['Recall outputs', condition => { const list = (result ? result.episodes : episodes).filter(item => item.condition === condition && item.phase === 'recall'); return list.length ? [null, list.map(item => printable(item.output) || '(nothing)').join('\n')] : null; }],
      ['Learned synapses', condition => { const m = result?.models?.[condition]; return m?.weightStats ? [`${m.weightStats.changed.toLocaleString()} changed`, `mean ${m.weightStats.mean.toFixed(2)} mV · max ${m.weightStats.max.toFixed(2)} mV · ${m.weightStats.atZero.toLocaleString()} at zero · ${m.weightStats.atMax.toLocaleString()} at the cap`] : null; }],
      ['Mushroom-body activity at recall', condition => { const a = metrics[condition].recall.activity; return a.decisions ? [`${(100 * a.meanKcFraction).toFixed(1)}% Kenyon cells per cue`, `${a.meanMbonSpikes.toFixed(1)} MBON spikes per cue · ${a.silentDecisions}/${a.decisions} silent · ${a.tiedDecisions} tied`] : null; }]
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
    $('delay-results').replaceChildren(fragment); drawChart($('chart')); drawCurve($('curve')); syncControls();
  }
  function updateProgress(next) {
    progress = next; $('progress').max = next.totalTrials; $('progress').value = next.completed;
    text('progress-text', `${next.completed} / ${next.totalTrials} decisions`); text('time', `${next.simTimeS.toFixed(2)} s simulated`);
    text('cue', symbol(next.cue)); text('drive', next.learning ? 'ON' : 'OFF');
    if (next.groupCounts && next.stage !== 'complete') renderVote(next.groupCounts, null, null);
    text('stage', next.stage === 'cue' ? 'Presenting cue' : next.stage === 'gap' ? 'Gap · reading state' : next.stage === 'complete' ? 'Experiment complete' : 'Awaiting episode');
    text('trial-time', `${next.cueTimeMs.toFixed(0)} / ${decisionMs} ms`); $('clock').style.left = `${Math.min(100, next.cueTimeMs / decisionMs * 100)}%`;
    text('episode', next.phase === 'complete' ? 'All conditions complete' : `${labels[next.condition]} · ${next.phase} episode ${next.episode}/${next.episodeTotal} · step ${next.step}`);
    const conditionIndex = next.phase === 'complete' ? 3 : CONDITIONS.indexOf(next.condition);
    document.querySelectorAll('.phase-track li').forEach((element, index) => {
      element.classList.toggle('is-current', index === conditionIndex); element.classList.toggle('is-complete', index < conditionIndex);
    });
    text('lock', next.phase === 'train' ? (next.learning ? 'Synapses learning from the teacher' : 'Frozen synapses, teacher shown') : next.phase === 'complete' ? 'All conditions complete' : 'Current condition: synapses frozen');
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
    const link = document.createElement('a'); link.href = url; link.download = `flyhamlet-synapse-${data.seed}${data.complete ? '' : '-partial'}.json`;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function verifyFrozen(data) {
    if (!data.complete) return false;
    return CONDITIONS.every(condition => {
      const model = data.models[condition];
      return model?.frozenAfterTraining && model.fingerprint === model.finalFingerprint && model.fingerprint === model.frozenFingerprint &&
        (condition !== 'frozen' || JSON.stringify(model.learnedPlasticWeights) === JSON.stringify(data.initialPlasticWeights));
    });
  }
  function conclusion(data) {
    const describe = condition => {
      const r = data.metrics[condition].recall, d = data.metrics[condition].diagnostic;
      const t = data.metrics[condition].training.curve;
      return `${labels[condition]}: training accuracy ${percent(t[0].accuracy)} → ${percent(t[t.length - 1].accuracy)}; next-letter accuracy ${percent(d.accuracy)} with learning off; autonomous chains ${r.chainLengths.join(', ')} (mean ${r.meanChainLength.toFixed(1)} of ${reference.length}, exact ${r.exact}/${r.episodes}).`;
    };
    return `${describe('plastic')} ${describe('plasticReset')} ${describe('frozen')} The no-memory comparator reaches ${comparatorLength}. Only Kenyon-cell-to-MBON synapses changed, and only during training. The model, codes and teacher are imposed; a chain beyond the comparator shows position written into existing synapses of this simulation, not how flies learn or memory for a text.`;
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
      text('lock', 'Verified: synapses unchanged after training; frozen control never learned'); text('conclusion', conclusion(result)); updateResults();
    } else if (message.type === 'export') download(message.result);
  }
  function clearDisplay() {
    progress = null; selectedLetter = null; currentOutput = ''; currentEnded = false; renderTyped(); placeFly(); $('output').replaceChildren();
    text('output-phase', 'Awaiting recall'); text('observations', '0 recall episodes'); text('cue', '—'); text('choice', '—'); text('drive', 'OFF');
    text('action', 'Waiting for the first decision'); text('episode', 'Awaiting trial');
    text('stage', 'Awaiting episode'); text('trial-time', `0 / ${decisionMs} ms`); $('clock').style.left = '0%'; renderVote(Array(8).fill(0), null, null);
    $('progress').value = 0; text('progress-text', `0 / ${defaultBudget} decisions`); text('time', '0.00 s simulated');
    text('lock', 'Synapses have not been trained'); text('rate', `Target speed ${$('speed').value}×`);
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
    text('load-message', 'Loading the full connectome and mushroom-body annotations. First use downloads about 49 MB, then prepares the rescaled network.');
    status('Preparing experiment', 'loading'); syncControls();
    try {
      if (!worker) {
        worker = new Worker(new URL('mb-worker.js?v=1', scriptURL)); generation = 0;
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
  const resize = () => { placeFly(); drawChart($('chart')); drawCurve($('curve')); };
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(resize); observer.observe($('chart')); observer.observe($('curve')); observer.observe($('keyboard'));
  } else window.addEventListener('resize', resize);
  $('speed').value = '1'; buildVote(); buildKeyboard(Array.from('rezndomstyawukihbpgqf xcvjl')); reset();
})();
