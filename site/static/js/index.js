/** Generic behavior for generated project pages. */

function honorReducedMotion() {
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  document.querySelectorAll('video[autoplay]').forEach((video) => {
    video.removeAttribute('autoplay');
    video.pause();
    video.controls = true;
  });
}

async function mountInteractiveApp(container) {
  const entry = container.dataset.entry;
  if (!entry) return;

  const mountPoint = container.querySelector('.interactive-mount');
  const errorElement = container.querySelector('.interactive-error');

  try {
    // Resolve against the document, not this file under static/js/. This keeps
    // ./static/wasm/demo.js portable on both root and project GitHub Pages URLs.
    const moduleUrl = new URL(entry, document.baseURI).href;
    const module = await import(moduleUrl);
    const exportName = container.dataset.export || 'mount';
    const mount = exportName === 'default' ? module.default : module[exportName];
    if (typeof mount !== 'function') {
      throw new TypeError(`Module does not export a ${exportName}() function`);
    }
    const options = JSON.parse(container.dataset.options || '{}');
    mountPoint.replaceChildren();
    await mount(mountPoint, options);
  } catch (error) {
    console.error('Could not start interactive application:', error);
    errorElement.textContent = 'The interactive demo could not be loaded.';
    errorElement.hidden = false;
  }
}

function startSyncedGifs(container) {
  const images = [...container.querySelectorAll('img')];
  if (images.length < 2) return;
  const sources = images.map((image) => image.getAttribute('src'));
  // Keep the authored relative paths. An absolute or root path would miss
  // the project prefix on GitHub Pages.
  if (sources.some((src) => !src || /^(?:[a-z][a-z0-9+.-]*:|\/)/i.test(src))) return;

  let pending = images.length;
  let failed = false;
  const finished = (ok) => {
    if (!ok) failed = true;
    pending -= 1;
    if (pending > 0 || failed) return;
    images.forEach((image) => image.setAttribute('src', ''));
    images.forEach((image, index) => image.setAttribute('src', sources[index]));
  };
  images.forEach((image) => {
    if (image.complete && image.naturalWidth > 0) finished(true);
    else {
      image.addEventListener('load', () => finished(true), { once: true });
      image.addEventListener('error', () => finished(false), { once: true });
    }
  });
}

honorReducedMotion();
document.querySelectorAll('[data-sync-gifs]').forEach(startSyncedGifs);
document.querySelectorAll('.interactive-app').forEach(mountInteractiveApp);
