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

honorReducedMotion();
document.querySelectorAll('.interactive-app').forEach(mountInteractiveApp);
