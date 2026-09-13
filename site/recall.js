/* The worker owns every decision. This page displays the audit trail and guides
 * the fly to the chosen key; reference text is used only for display/scoring. */
(() => {
  'use strict';
  const $ = id => document.getElementById(`lab-${id}`);
  const {ExplicitTypewriter} = window.FlyHamletLearning;
  const scriptURL = document.currentScript.src;
  const media = matchMedia('(prefers-reduced-motion: reduce)');
  const labels = {train: 'Teaching', recall: 'Neural + memory', ablated: 'History removed', symbol: 'Symbol memory'};
  const phaseOrder = ['train', 'recall', 'ablated'];
  const reference = 'to be or not to be'; // observer only; never sent to the worker
  const symbol = value => value === ' ' ? 'SPACE' : (value ?? '—');
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = value; };
  let worker = null, generation = 0, seed = null, active = false, started = false;
  let running = false, loading = false, result = null, progress = null, records = [];
  let queue = [], motion = null, raf = null, lastFrame = null, memory = [], memoryEpisode = '';
  let typewriter, keys = [], position = {x: .5, y: .5};
  const outputNodes = new Map();
  const freshSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];

  function status(message, state = 'paused') {
    text('status', message); $('status').dataset.state = state;
  }
  function syncControls() {
    $('run').disabled = loading || Boolean(result);
    text('run', result ? 'Experiment complete' : (running ? 'Pause' : started ? 'Resume' : 'Run experiment'));
    $('run').setAttribute('aria-pressed', String(running));
    $('seed').disabled = loading || started;
    $('export').disabled = !records.length || (!worker && !result);
  }
  function placeFly() {
    const grid = $('keyboard');
    $('fly').style.left = `${grid.offsetLeft + position.x * grid.clientWidth}px`;
    $('fly').style.top = `${grid.offsetTop + position.y * grid.clientHeight}px`;
  }
  function buildKeyboard(layout) {
    typewriter = new ExplicitTypewriter(layout); keys = Array(layout.length);
    const fragment = document.createDocumentFragment();
    for (let row = 2; row >= 0; --row) for (let col = 0; col < 9; ++col) {
      const index = row * 9 + col, letter = layout[index], key = document.createElement('div');
      key.className = `keycap${reference.includes(letter) ? ' task-key' : ''}`;
      const face = document.createElement('span'); face.className = 'key-face';
      const label = document.createElement('span'); label.className = 'key-label';
      label.textContent = letter === ' ' ? '␣' : letter.toUpperCase();
      face.append(label); key.append(face); fragment.append(key); keys[index] = key;
    }
    $('keyboard').replaceChildren(fragment); placeFly();
  }
  function showMemory(record) {
    const episode = `${record.phase}-${record.episode}`;
    if (episode !== memoryEpisode) { memory = []; memoryEpisode = episode; }
    memory.push(record.cue); memory = memory.slice(-6);
    const shown = [...Array(6 - memory.length).fill(null), ...memory];
    $('memory').querySelectorAll('li').forEach((element, index) => {
      const value = record.phase === 'ablated' && index < 5 ? null : shown[index];
      element.textContent = value === ' ' ? '␣' : symbol(value);
      element.classList.toggle('empty', value === null);
    });
    text('memory-note', record.phase === 'ablated' ? 'Older neural responses are masked; only the current response reaches the decoder.' : 'Each slot stores neural activity. Cue labels are shown here for inspection.');
    text('cue', symbol(record.cue)); text('choice', symbol(record.prediction));
    if (record.phase === 'train') text('action', `Teaching ${record.episode}/6 · collect response to ${symbol(record.cue)} → target ${symbol(record.target)}`);
  }
  function outputNode(record) {
    const key = `${record.phase}-${record.episode}`;
    if (!outputNodes.has(key)) {
      const group = document.createElement('span'); group.className = 'lab-transcript-group';
      const label = document.createElement('span'); label.className = 'lab-transcript-label';
      label.textContent = `${labels[record.phase]} / run ${record.episode}`;
      const content = document.createElement('span'); content.className = 'lab-transcript-text';
      group.append(label, content); $('output').append(group); outputNodes.set(key, content);
    }
    return outputNodes.get(key);
  }
  function commitDecision(record) {
    const output = outputNode(record);
    if (record.prediction === 'END') {
      const end = document.createElement('span'); end.className = 'recall-end'; end.textContent = 'END'; output.append(end);
      text('action', `${labels[record.phase]} ${record.episode} · predicted END`);
    } else {
      typewriter.moveTo(record.prediction);
      const press = typewriter.press();
      output.append(document.createTextNode(press.letter));
      text('presses', `${typewriter.events.length} presses`);
      text('action', `Pressed ${symbol(record.prediction)} · this choice becomes the next cue`);
      keys.forEach(key => key.classList.remove('is-pressed')); keys[press.keyIndex].classList.add('is-pressed');
      if (record.step === 48) {
        const cap = document.createElement('span'); cap.className = 'recall-end'; cap.textContent = 'CAP'; output.append(cap);
      }
    }
    text('output-phase', `${labels[record.phase]} / ${record.episode}`);
    $('output').scrollTop = $('output').scrollHeight;
  }
  function beginMotion(record) {
    showMemory(record);
    if (record.prediction === 'END') { commitDecision(record); return; }
    const index = typewriter.layout.indexOf(record.prediction);
    if (index < 0) throw new Error('Unknown output key.');
    motion = {record, from: {...position}, to: {x: (index % 9 + .5) / 9, y: (2 - Math.floor(index / 9) + .5) / 3}, elapsed: 0};
  }
  function frame(time) {
    raf = null;
    if ((!running && !result) || document.hidden) { lastFrame = null; return; }
    const elapsed = lastFrame === null ? 0 : Math.min(50, time - lastFrame); lastFrame = time;
    if (media.matches) {
      if (motion) { position = motion.to; commitDecision(motion.record); motion = null; }
      while (queue.length) {
        beginMotion(queue.shift());
        if (motion) { position = motion.to; commitDecision(motion.record); motion = null; }
      }
      placeFly();
    } else {
      while (!motion && queue.length) beginMotion(queue.shift());
      if (motion) {
        motion.elapsed += elapsed;
        const fraction = Math.min(1, motion.elapsed / (queue.length > 5 ? 35 : 130));
        const smooth = fraction * fraction * (3 - 2 * fraction);
        position.x = motion.from.x + (motion.to.x - motion.from.x) * smooth;
        position.y = motion.from.y + (motion.to.y - motion.from.y) * smooth;
        const angle = Math.atan2(motion.to.y - motion.from.y, motion.to.x - motion.from.x) * 180 / Math.PI;
        $('fly').style.transform = `translate(-50%,-50%) rotate(${angle + 90}deg)`; placeFly();
        if (fraction === 1) { commitDecision(motion.record); motion = null; }
      }
    }
    if (motion || queue.length) raf = requestAnimationFrame(frame);
    else { lastFrame = null; if (result) text('action', 'All autonomous decisions retained'); }
  }
  function animate() {
    if (raf === null && (motion || queue.length) && (running || result) && !document.hidden) raf = requestAnimationFrame(frame);
  }
  function editDistance(a, b) {
    let previous = Array.from({length: b.length + 1}, (_, i) => i);
    for (let i = 1; i <= a.length; ++i) {
      const next = [i];
      for (let j = 1; j <= b.length; ++j) next[j] = Math.min(next[j - 1] + 1, previous[j] + 1, previous[j - 1] + Number(a[i - 1] !== b[j - 1]));
      previous = next;
    }
    return previous[b.length];
  }
  function finishedEpisodes() {
    if (result) return [...result.episodes, ...result.comparator.episodes.map(episode => ({...episode, phase: 'symbol'}))];
    const groups = new Map();
    for (const record of records) if (record.phase !== 'train') {
      const key = `${record.phase}-${record.episode}`;
      if (!groups.has(key)) groups.set(key, []); groups.get(key).push(record);
    }
    const episodes = [];
    for (const group of groups.values()) {
      const last = group[group.length - 1];
      if (last.prediction !== 'END' && last.step !== 48) continue;
      const output = group.filter(record => record.prediction !== 'END').map(record => record.prediction).join('');
      const stoppedBy = last.prediction === 'END' ? 'END' : 'cap';
      episodes.push({phase: last.phase, episode: last.episode, output, stoppedBy, editDistance: editDistance(output, reference), exact: stoppedBy === 'END' && output === reference});
    }
    return episodes;
  }
  function updateResults() {
    const episodes = finishedEpisodes();
    for (const phase of ['recall', 'ablated', 'symbol']) {
      const items = episodes.filter(episode => episode.phase === phase);
      text(`${phase}-score`, phase === 'symbol' ? (items.length ? (items[0].exact ? 'Exact' : 'Not exact') : '—') : (items.length ? `${items.filter(episode => episode.exact).length} / ${items.length}` : '— / 3'));
    }
    const fragment = document.createDocumentFragment();
    for (const episode of episodes) {
      const row = document.createElement('tr');
      const name = document.createElement('th'); name.scope = 'row'; name.textContent = `${labels[episode.phase]} / ${episode.episode}`; row.append(name);
      for (const value of [episode.output || '(empty)', episode.stoppedBy === 'END' ? 'END' : '48-step cap', episode.editDistance, episode.exact ? 'Yes' : 'No']) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      row.lastChild.className = episode.exact ? 'exact-pass' : 'exact-fail'; fragment.append(row);
    }
    if (!episodes.length) {
      const row = document.createElement('tr'), cell = document.createElement('td'); cell.colSpan = 5;
      cell.textContent = 'Results appear here as each rollout finishes.'; row.append(cell); fragment.append(row);
    }
    $('rollouts').replaceChildren(fragment); syncControls();
  }
  function updateProgress(next) {
    progress = next;
    $('progress').max = Math.max(1, next.totalTrials); $('progress').value = next.completed;
    text('progress-text', next.phase === 'complete' ? `${next.completed} neural responses · complete` : `${labels[next.phase]} ${next.episode} · step ${next.step}`);
    text('time', `${next.simTimeS.toFixed(2)} s simulated`);
    text('teacher', next.phase === 'train' ? 'ON' : 'OFF');
    const phaseIndex = next.phase === 'complete' ? 3 : phaseOrder.indexOf(next.phase);
    document.querySelectorAll('.phase-track li').forEach((element, index) => {
      element.classList.toggle('is-current', index === phaseIndex); element.classList.toggle('is-complete', index < phaseIndex);
    });
    text('lock', phaseIndex > 0 ? 'Decoder fitted · weights frozen' : 'Collecting neural responses for fitting');
    if (running) status(next.phase === 'train' ? 'Teaching · collecting neural responses' : next.phase === 'ablated' ? 'Testing with history removed' : 'Recalling · teaching cues off', 'playing');
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
    const link = document.createElement('a'); link.href = url; link.download = `flyhamlet-recall-${data.seed}${data.complete ? '' : '-partial'}.json`;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function onMessage(event) {
    const message = event.data;
    if (!active || message.seed !== seed || message.generation !== generation) return;
    if (message.type === 'progress') {
      text('load-message', message.message.startsWith('Loading') ? 'Loading the full connectome · 49 MB on first visit' : 'Preparing and verifying the brain wiring…'); return;
    }
    if (message.type === 'error') { fail(message.message); return; }
    if (message.type === 'ready') {
      loading = false; started = true; $('load').hidden = true;
      buildKeyboard(message.metadata.layout); updateProgress(message.progress); syncControls();
    } else if (message.type === 'state') {
      running = message.running; syncControls();
      if (!running && !result && started) status('Paused · experiment state retained');
      if (running && progress) { updateProgress(progress); animate(); }
    } else if (message.type === 'tick') {
      for (const record of message.records) {
        if (record.trial !== records.length + 1) { fail('The decision log arrived out of order. Start a new experiment.'); return; }
        records.push(record);
        if (record.phase === 'train') showMemory(record); else queue.push(record);
      }
      updateProgress(message.progress); text('rate', `${message.realTimeRatio.toFixed(2)}× actual speed`);
      if (message.records.length) updateResults(); animate();
    } else if (message.type === 'complete') {
      result = message.result; running = false;
      if (!result.complete || !result.frozenWeights || JSON.stringify(result.finalWeights) !== JSON.stringify(result.frozenWeights)) {
        result = null; fail('The frozen decoder check failed. This run is invalid.'); return;
      }
      status('Experiment complete', 'complete'); text('teacher', 'OFF'); text('lock', 'Verified: fitted weights unchanged');
      const good = result.episodes.filter(episode => episode.phase === 'recall' && episode.exact).length;
      const conventional = result.comparator.episodes.filter(episode => episode.exact).length;
      text('conclusion', `Neural encoder + external memory recalled the exact phrase and END in ${good}/3 runs. The deterministic symbol-memory comparison ${conventional ? 'recalled it exactly' : 'did not recall it exactly'}. All outputs are shown above; weights stayed frozen. This measures memorization of one taught phrase, not language understanding.`);
      updateResults(); animate();
    } else if (message.type === 'export') download(message.result);
  }
  function clearDisplay() {
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null; queue = []; motion = null; lastFrame = null; progress = null; memory = []; memoryEpisode = '';
    typewriter?.reset(); position = {x: .5, y: .5}; placeFly(); keys.forEach(key => key.classList.remove('is-pressed'));
    outputNodes.clear(); $('output').replaceChildren();
    $('memory').querySelectorAll('li').forEach(element => { element.textContent = '—'; element.classList.add('empty'); });
    text('memory-note', 'Each slot stores neural activity. Cue labels are shown here for inspection.');
    for (const id of ['cue', 'choice']) text(id, '—');
    text('teacher', 'OFF'); text('presses', '0 presses'); text('output-phase', 'Awaiting recall'); text('action', 'Waiting to run');
    $('progress').value = 0; text('progress-text', 'Waiting to run'); text('time', '0.00 s simulated');
    text('lock', 'Decoder has not been fitted'); text('rate', `Target speed ${$('speed').value}×`);
    text('conclusion', 'This is a memorization benchmark using seven symbols. Success would show recall by the combined system; the conventional comparison uses direct symbol memory.');
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
    text('load-message', 'Loading the full connectome. The first visit downloads 49 MB of wiring.');
    status('Loading brain wiring', 'loading'); syncControls();
    try {
      if (!worker) {
        worker = new Worker(new URL('recall-worker.js?v=1', scriptURL)); generation = 0;
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
  $('speed').addEventListener('change', () => { worker?.postMessage({type: 'speed', value: Number($('speed').value)}); if (!started) text('rate', `Target speed ${$('speed').value}×`); });
  $('export').addEventListener('click', () => result ? download(result) : worker?.postMessage({type: 'export'}));
  document.addEventListener('visibilitychange', () => { if (document.hidden && active && (running || loading)) worker?.postMessage({type: 'pause'}); else animate(); });
  media.addEventListener('change', animate);
  if ('ResizeObserver' in window) new ResizeObserver(placeFly).observe($('keyboard'));
  else window.addEventListener('resize', placeFly);
  $('speed').value = '1'; buildKeyboard(Array.from('rezndomstyawukihbpgqf xcvjl')); reset();
})();
