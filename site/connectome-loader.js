/* Load the complete, immutable CSR graph used by the browser experiments.
 * Sequential decompression bounds peak temporary memory. The model manifest's
 * uncompressed sizes and SHA-256 digests are checked before integration starts.
 */
(function (root) {
  'use strict';

  const clock = () => typeof performance === 'undefined' ? Date.now() : performance.now();

  async function fetchArray(url, name, integrity, progress) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${name}: HTTP ${response.status}.`);
    const total = Number(response.headers.get('content-length')) || integrity?.bytes || 0;
    let loaded = 0;
    let bytes;
    if (response.body) {
      const chunks = [];
      const reader = response.body.getReader();
      let lastProgress = -Infinity;
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        if (clock() - lastProgress >= 150) {
          progress(`Loading ${name}`, loaded, total);
          lastProgress = clock();
        }
      }
      bytes = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
      }
    } else {
      bytes = new Uint8Array(await response.arrayBuffer());
      loaded = bytes.byteLength;
    }
    progress(`Expanding ${name}`, loaded, total);
    let buffer = bytes.buffer;
    // Some HTTP hosts already decompress .gz assets; inspect the actual bytes.
    if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
      if (typeof DecompressionStream === 'undefined')
        throw new Error('This browser cannot expand the connectome. Please use a current Chrome, Edge, Firefox, or Safari.');
      buffer = await new Response(new Blob([bytes]).stream()
        .pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    }
    if (integrity?.rawBytes !== undefined && buffer.byteLength !== integrity.rawBytes)
      throw new Error(`The downloaded ${name} has the wrong size. Please reload.`);
    if (buffer.byteLength % 4 !== 0)
      throw new Error(`The downloaded ${name} has an invalid array size.`);
    if (integrity?.rawSha256) {
      if (!root.crypto?.subtle)
        throw new Error('Connectome verification needs a secure HTTPS connection or localhost.');
      const digest = await root.crypto.subtle.digest('SHA-256', buffer);
      const hex = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
      if (hex !== integrity.rawSha256)
        throw new Error(`The downloaded ${name} failed its integrity check. Please reload.`);
    }
    return buffer;
  }

  async function load(manifestURL, options = {}) {
    const progress = options.progress || (() => {});
    progress('Loading the full connectome manifest', 0, 0);
    const url = new URL(manifestURL || 'model/manifest.json', options.baseURL || root.location?.href);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load the model: HTTP ${response.status}.`);
    const manifest = await response.json();
    if (manifest.format !== 'flyhamlet-csr-v1') throw new Error('Unsupported connectome format.');
    if (!Number.isSafeInteger(manifest.n) || manifest.n <= 0 ||
        !Number.isSafeInteger(manifest.edgeCount) || manifest.edgeCount < 0)
      throw new Error('Invalid connectome dimensions.');
    const arrays = {};
    for (const name of ['indptr', 'indices', 'weights']) {
      const file = manifest.files?.[name];
      const path = typeof file === 'string' ? file : file?.url;
      if (!path) throw new Error(`The connectome manifest is missing ${name}.`);
      const buffer = await fetchArray(new URL(path, url), name, manifest.integrity?.[name], progress);
      arrays[name] = name === 'weights' ? new Float32Array(buffer) : new Uint32Array(buffer);
    }
    if (arrays.indptr.length !== manifest.n + 1 || arrays.indices.length !== manifest.edgeCount ||
        arrays.weights.length !== manifest.edgeCount || arrays.indptr[0] !== 0 ||
        arrays.indptr[manifest.n] !== manifest.edgeCount)
      throw new Error('The full connectome data is incomplete.');
    for (let i = 0; i < manifest.n; ++i)
      if (arrays.indptr[i] > arrays.indptr[i + 1]) throw new Error('Invalid connectome row offsets.');
    for (const index of arrays.indices)
      if (index >= manifest.n) throw new Error('Invalid connectome target index.');
    for (const weight of arrays.weights)
      if (!Number.isFinite(weight)) throw new Error('Invalid connectome synaptic weight.');
    progress('Full connectome ready', manifest.edgeCount, manifest.edgeCount);
    return {manifest, connectome: Object.freeze({n: manifest.n, ...arrays})};
  }

  const api = {load};
  root.FlyHamletConnectome = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : globalThis);
