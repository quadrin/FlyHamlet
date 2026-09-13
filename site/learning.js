/* Live cue-decoding bench. The worker computes decisions; this controller only
 * presents them and assists movement to the chosen key. It never picks a target.
 */
(() => {
  'use strict';
  const $ = id => document.getElementById(`lab-${id}`);
  const {ExplicitTypewriter, summarizeTrials} = window.FlyHamletLearning;
  const phaseNames = {baseline: 'Baseline', train: 'Training', evaluation: 'Evaluation'};
  const phaseOrder = ['baseline', 'train', 'evaluation'];
  const phaseTotals = {baseline: 20, train: 60, evaluation: 40};
  const scriptURL = document.currentScript.src;
  const media = matchMedia('(prefers-reduced-motion: reduce)');
  const chart = $('chart'), ctx = chart.getContext('2d');
  let worker = null, generation = 0, seed = null, active = false;
  let started = false, running = false, loading = false, result = null;
  let records = [], progress = null, queue = [], motion = null, raf = null, lastFrame = null;
  let typewriter, keyElements = [], position = {x: .5, y: .5};
  let reduced = media.matches;
  const outputNodes = {};
  const freshSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
  const text = (id, value) => { if ($(id).textContent !== String(value)) $(id).textContent = value; };
  const pct = value => `${(value * 100).toFixed(0)}%`;

  function status(message, state = 'paused') {
    text('status', message);
    $('status').dataset.state = state;
  }
  function syncControls() {
    $('run').disabled = loading || Boolean(result);
    text('run', result ? 'Experiment complete' : (running ? 'Pause' : (started ? 'Resume' : 'Run experiment')));
    $('run').setAttribute('aria-pressed', String(running));
    $('seed').disabled = started || loading;
    $('export').disabled = !records.length || (!worker && !result);
  }
  function buildKeyboard(layout) {
    typewriter = new ExplicitTypewriter(layout);
    keyElements = Array(layout.length);
    const fragment = document.createDocumentFragment();
    for (let row = 2; row >= 0; --row) for (let col = 0; col < 9; ++col) {
      const index = row * 9 + col, letter = layout[index];
      const key = document.createElement('div');
      key.className = `keycap${['t', 'o'].includes(letter) ? ' task-key' : ''}`;
      key.dataset.letter = letter;
      const face = document.createElement('span'); face.className = 'key-face';
      const label = document.createElement('span'); label.className = 'key-label';
      label.textContent = letter === ' ' ? '␣' : letter.toUpperCase();
      face.append(label); key.append(face); fragment.append(key); keyElements[index] = key;
    }
    $('keyboard').replaceChildren(fragment);
    position = {x: .5, y: .5}; placeFly();
  }
  function placeFly() {
    const grid = $('keyboard');
    $('fly').style.left = `${grid.offsetLeft + position.x * grid.clientWidth}px`;
    $('fly').style.top = `${grid.offsetTop + position.y * grid.clientHeight}px`;
  }
  function clearPresentation() {
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null; queue = []; motion = null; lastFrame = null;
    typewriter?.reset();
    position = {x: .5, y: .5}; placeFly();
    keyElements.forEach(key => key.classList.remove('is-pressed'));
    $('output').replaceChildren();
    for (const phase of phaseOrder) {
      const group = document.createElement('span'); group.className = 'lab-transcript-group';
      const label = document.createElement('span'); label.className = 'lab-transcript-label'; label.textContent = phaseNames[phase];
      const content = document.createElement('span'); content.className = 'lab-transcript-text';
      group.append(label, content); $('output').append(group); outputNodes[phase] = content;
    }
    text('presses', '0 presses'); text('output-phase', 'Awaiting experiment');
    text('cue', '—'); text('choice', '—'); text('updates', 'OFF'); text('action', 'Waiting to run');
  }
  function commitPress(record) {
    // Travel and pressing are separate operations. Only the latter appends text,
    // including repeated choices and mistakes; the target is never substituted.
    typewriter.moveTo(record.prediction);
    const press = typewriter.press();
    const span = document.createElement('span');
    span.textContent = press.letter.toUpperCase();
    span.title = `Trial ${record.trial}: cue ${record.target}, chose ${record.prediction}; ${record.correct ? 'correct' : 'incorrect'}`;
    outputNodes[record.phase].append(span);
    text('presses', `${typewriter.events.length} presses`);
    text('output-phase', phaseNames[record.phase]);
    text('action', `Pressed ${record.prediction} · ${record.correct ? 'correct' : `cue was ${record.target}`}`);
    keyElements.forEach(key => key.classList.remove('is-pressed'));
    keyElements[press.keyIndex].classList.add('is-pressed');
    $('output').scrollTop = $('output').scrollHeight;
  }
  function beginMotion(record) {
    const index = typewriter.layout.indexOf(record.prediction.toLowerCase());
    if (index < 0) throw new Error('A decoder choice is missing from the keyboard.');
    motion = {record, from: {...position}, to: {x: (index % 9 + .5) / 9, y: (2 - Math.floor(index / 9) + .5) / 3}, elapsed: 0};
    text('cue', record.target); text('choice', record.prediction);
    text('action', `Moving to ${record.prediction} · no key pressed in transit`);
    keyElements.forEach(key => key.classList.remove('is-pressed'));
  }
  function presentationFrame(time) {
    raf = null;
    if ((!running && !result) || document.hidden) { lastFrame = null; return; }
    const elapsed = lastFrame === null ? 0 : Math.min(50, time - lastFrame);
    lastFrame = time;
    if (reduced) {
      if (motion) { position = motion.to; commitPress(motion.record); motion = null; }
      while (queue.length) { beginMotion(queue.shift()); position = motion.to; commitPress(motion.record); motion = null; }
      placeFly();
    } else {
      if (!motion && queue.length) beginMotion(queue.shift());
      if (motion) {
        motion.elapsed += elapsed;
        // The animation only depicts computed decisions. Speed never changes
        // their order, neural simulation time, or the raw experimental record.
        const duration = queue.length > 5 ? 35 : 130;
        const fraction = Math.min(1, motion.elapsed / duration);
        const smooth = fraction * fraction * (3 - 2 * fraction);
        position.x = motion.from.x + (motion.to.x - motion.from.x) * smooth;
        position.y = motion.from.y + (motion.to.y - motion.from.y) * smooth;
        const angle = Math.atan2(motion.to.y - motion.from.y, motion.to.x - motion.from.x) * 180 / Math.PI;
        $('fly').style.transform = `translate(-50%,-50%) rotate(${angle + 90}deg)`;
        placeFly();
        if (fraction === 1) { commitPress(motion.record); motion = null; }
      }
    }
    if (motion || queue.length) raf = requestAnimationFrame(presentationFrame);
    else { lastFrame = null; if (result) text('action', 'All decisions pressed · full output retained'); }
  }
  function animate() {
    if (raf === null && (queue.length || motion) && (running || result) && !document.hidden)
      raf = requestAnimationFrame(presentationFrame);
  }
  function updateProgress(next) {
    progress = next;
    $('progress').max = next.totalTrials;
    $('progress').value = next.completed;
    text('progress-text', `${next.completed} / ${next.totalTrials} trials`);
    text('time', `${next.simTimeS.toFixed(2)} s simulated`);
    text('updates', next.phase === 'train' ? 'ON' : 'OFF');
    text('update-count', next.updateCount);
    const phaseIndex = next.phase === 'complete' ? 3 : phaseOrder.indexOf(next.phase);
    document.querySelectorAll('.phase-track li').forEach((element, index) => {
      element.classList.toggle('is-current', index === phaseIndex);
      element.classList.toggle('is-complete', index < phaseIndex);
    });
    text('lock', phaseIndex >= 2 ? 'Evaluation: decoder weights locked' : (next.phase === 'train' ? 'Supervised decoder training' : 'Decoder has not been trained'));
    if (running) status(next.phase === 'train' ? 'Training decoder' : (next.phase === 'evaluation' ? 'Evaluating · weights frozen' : 'Measuring baseline'), 'playing');
  }
  function updateMetrics() {
    for (const phase of phaseOrder) {
      const trials = records.filter(record => record.phase === phase);
      const trained = summarizeTrials(trials), control = summarizeTrials(trials, 'controlPrediction');
      const describe = metric => metric.total ? `${pct(metric.accuracy)} · ${metric.correct}/${metric.total}` : '—';
      text(`${phase}-accuracy`, describe(trained)); text(`${phase}-control`, describe(control));
      if (phase === 'evaluation' && trained.total) {
        text('eval-score', describe(trained)); text('control-score', describe(control));
        text('eval-detail', trained.total === phaseTotals.evaluation ? `95% interval ${pct(trained.wilson95[0])}–${pct(trained.wilson95[1])}` : 'Evaluation in progress');
        text('control-detail', control.total === phaseTotals.evaluation ? `95% interval ${pct(control.wilson95[0])}–${pct(control.wilson95[1])}` : 'Same cues and neural activity');
      }
    }
    drawChart(); syncControls();
  }
  function drawChart() {
    const ratio = devicePixelRatio || 1, width = chart.clientWidth, height = chart.clientHeight;
    if (!width || !height) return;
    if (chart.width !== Math.round(width * ratio) || chart.height !== Math.round(height * ratio)) {
      chart.width = Math.round(width * ratio); chart.height = Math.round(height * ratio);
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
    const left = 39, right = 12, top = 26, bottom = 29;
    const X = trial => left + trial / 120 * (width - left - right);
    const Y = score => top + (1 - score) * (height - top - bottom);
    [[0,20,'#a1b9ca0e','Base'],[20,80,'#ffba790e','Train'],[80,120,'#c1ec8c0e','Evaluate']].forEach(([from,to,color,label]) => {
      ctx.fillStyle = color; ctx.fillRect(X(from),top,X(to)-X(from),height-top-bottom);
      ctx.fillStyle = '#b8d8e4'; ctx.textAlign = 'center'; ctx.font = '10px monospace'; ctx.fillText(label,(X(from)+X(to))/2,14);
    });
    for (const value of [0,.5,1]) {
      ctx.strokeStyle = value === .5 ? '#a1b9ca' : '#345566'; ctx.setLineDash(value === .5 ? [4,4] : []);
      ctx.beginPath(); ctx.moveTo(X(0),Y(value)); ctx.lineTo(X(120),Y(value)); ctx.stroke();
      ctx.fillStyle = '#b8cbd7'; ctx.textAlign = 'right'; ctx.fillText(pct(value),left-7,Y(value)+3);
    }
    ctx.setLineDash([]); ctx.textAlign = 'center';
    for (const trial of [0,20,40,60,80,100,120]) ctx.fillText(trial,X(trial),height-12);
    for (const [field,color] of [['correct','#78e5ed'],['controlCorrect','#ffba79']]) {
      const points = [];
      for (let start=0; start+10<=records.length; start+=10) points.push({x:start+10,y:records.slice(start,start+10).filter(record=>record[field]).length/10});
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.7; ctx.beginPath();
      points.forEach((point,index) => index ? ctx.lineTo(X(point.x),Y(point.y)) : ctx.moveTo(X(point.x),Y(point.y))); ctx.stroke();
      for (const point of points) { ctx.beginPath(); ctx.arc(X(point.x),Y(point.y),2.6,0,Math.PI*2); ctx.fill(); }
    }
    ctx.lineWidth = 1;
  }
  function fail(message) {
    running = false; loading = false; started = false; active = false;
    worker?.terminate(); worker = null; generation = 0;
    status('Experiment stopped', 'error');
    $('load').hidden = false; $('load').classList.add('is-error'); $('retry').hidden = false;
    text('load-message', message); syncControls();
  }
  function download(data) {
    if (!data) return;
    const file = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(file), link = document.createElement('a');
    link.href = url; link.download = `flyhamlet-learning-${data.seed}${data.complete ? '' : '-partial'}.json`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function onMessage(event) {
    const message = event.data;
    if (!active || message.seed !== seed || message.generation !== generation) return;
    if (message.type === 'progress') {
      text('load-message', message.message.startsWith('Loading') ? 'Loading the full connectome · 49 MB on first visit' : 'Preparing and verifying the brain wiring…');
      return;
    }
    if (message.type === 'error') { fail(message.message); return; }
    if (message.type === 'ready') {
      loading = false; started = true;
      $('load').hidden = true; buildKeyboard(message.metadata.layout); updateProgress(message.progress); syncControls();
    } else if (message.type === 'state') {
      running = message.running; syncControls();
      if (!running && !result && started) status(progress?.phase === 'complete' ? 'Finishing experiment' : 'Paused · experiment state retained');
      if (running && progress) { updateProgress(progress); animate(); }
    } else if (message.type === 'tick') {
      for (const record of message.records) {
        // Worker records arrive once and in decision order, including wrong choices.
        if (record.trial !== records.length + 1) { fail('The trial log arrived out of order. Please start a new experiment.'); return; }
        records.push(record); queue.push(record);
      }
      updateProgress(message.progress); text('rate', `${message.realTimeRatio.toFixed(2)}× actual speed`);
      if (message.records.length) updateMetrics();
      animate();
    } else if (message.type === 'complete') {
      result = message.result; running = false;
      const frozen = result.frozenWeights;
      const evaluation = result.trials.filter(record => record.phase === 'evaluation');
      if (!frozen || evaluation.some(record => record.feedbackApplied || record.updateCountBefore !== record.updateCountAfter ||
          record.weightsBefore.some((weight,index) => weight !== frozen[index]) || record.weightsAfter.some((weight,index) => weight !== frozen[index]))) {
        result = null; fail('Evaluation weights changed unexpectedly. This run is invalid.'); return;
      }
      status('Experiment complete', 'complete'); text('updates','OFF'); text('lock','Evaluation verified: weights unchanged');
      text('update-note', `${result.updateCount} training updates · 0 evaluation updates`);
      text('conclusion', `Evaluation complete: ${result.metrics.evaluation.correct}/${result.metrics.evaluation.total} correct with the trained decoder, ${result.metrics.evaluation.control.correct}/${result.metrics.evaluation.control.total} with the untrained control. Decoder weights stayed frozen. Export the full run to inspect every decision; this is cue decoding, not phrase recall.`);
      updateMetrics(); animate();
    } else if (message.type === 'export') download(message.result);
  }
  function start() {
    const value = Number($('seed').value);
    if (!$('seed').value.trim() || !Number.isInteger(value) || value < 0 || value > 4294967295) {
      $('seed').setCustomValidity('Enter an integer from 0 to 4294967295.'); $('seed').reportValidity(); return;
    }
    $('seed').setCustomValidity(''); seed = value; active = true; loading = true;
    records = []; result = null; clearRunDisplay();
    $('load').hidden = false; $('load').classList.remove('is-error'); $('retry').hidden = true;
    text('load-message','Loading the full connectome. The first visit downloads 49 MB of wiring.');
    status('Loading brain wiring','loading'); syncControls();
    try {
      if (!worker) {
        worker = new Worker(new URL('learning-worker.js?v=1',scriptURL)); generation = 0;
        worker.addEventListener('message',onMessage);
        worker.addEventListener('error',() => fail('The experiment worker could not continue. Check your connection and try again.'));
      }
      generation++;
      worker.postMessage({type:'init',seed,speed:Number($('speed').value),autoplay:!document.hidden,
        manifestURL:new URL('model/manifest.json',scriptURL).href});
    } catch (error) { fail(error.message || 'Could not start this browser’s experiment worker.'); }
  }
  function clearRunDisplay() {
    progress = null; clearPresentation(); updateMetrics();
    $('progress').value = 0; text('progress-text','0 / 120 trials'); text('time','0.00 s simulated');
    text('eval-score','—'); text('control-score','—'); text('eval-detail','40 trials after training'); text('control-detail','Same cues and neural activity');
    text('update-count','0'); text('update-note','Only the training phase changes weights'); text('lock','Decoder has not been trained');
    text('conclusion','Evaluation keeps the sensory cues available and freezes the decoder. This tests cue decoding, not phrase recall or language understanding.');
    text('rate',`Target speed ${$('speed').value}×`);
    document.querySelectorAll('.phase-track li').forEach(element=>element.classList.remove('is-current','is-complete'));
  }
  function reset() {
    active = false; worker?.postMessage({type:'pause'});
    started = false; running = false; loading = false; result = null; records = [];
    $('seed').value = freshSeed(); $('seed').setCustomValidity('');
    $('load').hidden = true; clearRunDisplay();
    status('Ready for a new experiment'); syncControls();
  }
  $('run').addEventListener('click',() => {
    if (!started) start();
    else { worker?.postMessage({type:running ? 'pause' : 'resume'}); }
  });
  $('new').addEventListener('click',reset);
  $('retry').addEventListener('click',start);
  $('seed').addEventListener('input',() => $('seed').setCustomValidity(''));
  $('speed').addEventListener('change',() => {
    worker?.postMessage({type:'speed',value:Number($('speed').value)});
    if (!started) text('rate',`Target speed ${$('speed').value}×`);
  });
  $('export').addEventListener('click',() => result ? download(result) : worker?.postMessage({type:'export'}));
  document.addEventListener('visibilitychange',() => {
    if (document.hidden && active && (running || loading)) worker?.postMessage({type:'pause'});
    else animate();
  });
  media.addEventListener('change',event => { reduced=event.matches; animate(); });
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(()=>{drawChart();placeFly();}); observer.observe(chart); observer.observe($('keyboard'));
  } else window.addEventListener('resize',()=>{drawChart();placeFly();});
  $('speed').value = '1';
  buildKeyboard(Array.from('rezndomstyawukihbpgqf xcvjl'));
  reset();
})();
