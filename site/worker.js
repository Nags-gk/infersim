// Runs the InferSim Python package inside Pyodide, off the main thread.
let pyodide = null;
let call = null;

async function boot(indexURL) {
  const { loadPyodide } = await import(indexURL + "pyodide.mjs");
  pyodide = await loadPyodide({ indexURL });
  const files = await (await fetch("py/manifest.json")).json();
  for (const f of files) {
    const text = await (await fetch("py/" + f)).text();
    const path = "/home/pyodide/" + f;
    pyodide.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    pyodide.FS.writeFile(path, text);
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  call = pyodide.runPython("from infersim.web import call; call");
  const version = pyodide.runPython("import sys; sys.version.split()[0]");
  return { python: version };
}

self.onmessage = async (e) => {
  const { id, type, fn, kwargs, indexURL } = e.data;
  try {
    if (type === "boot") {
      const info = await boot(indexURL);
      self.postMessage({ id, ok: true, result: info });
      return;
    }
    const t0 = performance.now();
    const out = JSON.parse(call(fn, JSON.stringify(kwargs)));
    self.postMessage({ id, ok: true, result: out, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};
