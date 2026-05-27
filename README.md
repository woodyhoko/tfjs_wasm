# TensorFlow.js WebAssembly Benchmarking

A benchmarking suite that **compares TensorFlow.js inference performance** across WebAssembly (WASM) and CPU backends, tested across Chrome, Edge, and Firefox.

---

## What It Measures

TensorFlow.js can run inference via three backends:
- **WebGL** (GPU-accelerated)
- **WASM** (WebAssembly, CPU-optimized)
- **Plain JS** (CPU fallback)

This project measures and compares WASM vs. CPU for various operation sizes and reports timing results across browsers.

---

## Results Included

Pre-collected benchmark logs from three browsers:
- `15g_chrome.pickle` — Chrome results
- `15g_edge.pickle` — Edge results
- `15g_firefox.pickle` — Firefox results

---

## Run Your Own Benchmark

### Step 1 — Host the test server

```bash
python3 -m http.server 8000
```

### Step 2 — Install automation dependencies

```bash
pip install -U selenium numpy tqdm matplotlib
```

### Step 3 — Install Chrome, Edge, and Firefox

Make sure WebDriver binaries (chromedriver, geckodriver, etc.) are in your PATH.

### Step 4 — Run the notebook

```bash
jupyter notebook test.ipynb
```

The notebook automates browser launches via Selenium, runs inference benchmarks, collects timing data, and plots comparison charts.

