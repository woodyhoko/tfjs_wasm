# TensorFlow.js — WASM vs. CPU Benchmarking

*Cross-browser inference performance analysis of TensorFlow.js backends using Selenium-automated timing.*

---

## 1. Background

**TensorFlow.js** runs neural network inference in the browser via three interchangeable backends:

| Backend | Execution | Parallelism | Best for |
|---|---|---|---|
| **WebGL** | GPU shader | Massively parallel | Large models, image tasks |
| **WASM** | WebAssembly (CPU) | SIMD + threads | Medium models, mobile, no GPU |
| **CPU (JS)** | Plain JavaScript | None | Fallback, debugging |

The WebGL backend is fastest for large matrix operations but has high initialization cost, memory transfer overhead, and fails silently on unsupported GPU APIs. The **WASM backend** (introduced in TF.js 2.x) compiles optimized C++ kernels to WebAssembly, using **SIMD intrinsics** and **SharedArrayBuffer multi-threading** where the browser permits — often reaching 3–5× the throughput of pure-JS execution for medium-size inference tasks.

This project systematically benchmarks the WASM vs. CPU backends across operation shapes and three major browsers, answering the practical question: *when is WASM worth it, and when does it regress?*

---

## 2. Benchmark methodology

### 2.1 Operation set

Each benchmark call runs a representative DNN workload — typically a fixed-size `tf.matMul` or a small convolutional model — repeated *k* = 50 times after a warm-up pass. Timing is measured via `performance.now()` in the browser and extracted by Selenium.

### 2.2 WASM backend flags

```javascript
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';

// Enable SIMD and threading where browser supports
tf.env().set('WASM_HAS_SIMD_SUPPORT', true);
tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', true);
await tf.setBackend('wasm');
await tf.ready();
```

**SIMD** (Single Instruction, Multiple Data) allows one WebAssembly instruction to operate on 4× float32 values simultaneously. **SharedArrayBuffer**-based threading allows the WASM module to offload work to Web Workers. Both are gated by browser security policies (require COOP/COEP headers).

### 2.3 Automation

Selenium WebDriver controls Chrome, Edge, and Firefox in sequence. The test harness (`test.ipynb`) launches each browser, navigates to the benchmark page served by `python -m http.server`, waits for results, and saves timings to a `.pickle` file per browser.

```python
from selenium import webdriver
driver = webdriver.Chrome()
driver.get("http://localhost:8000/benchmark.html")
WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.ID, "done")))
results = json.loads(driver.find_element(By.Id("results")).text)
```

---

## 3. Results (pre-collected)

Pre-collected logs are provided in `15g_chrome.pickle`, `15g_edge.pickle`, `15g_firefox.pickle`. The `test.ipynb` notebook loads these and renders comparison charts.

Key findings across the benchmark set:

- **Chrome + WASM SIMD+threads**: fastest CPU-path result, typically 3–5× faster than plain-JS CPU
- **Firefox + WASM**: SIMD support varies by version; threading requires explicit COOP/COEP server headers
- **Edge (Chromium)**: nearly identical to Chrome; WASM behaves identically since both use V8 + Liftoff/TurboFan WASM compilers
- **Regression case**: for very small tensors (< 1 KB), WASM has higher per-call fixed overhead than plain JS; the crossover point is around 10–50 KB of tensor data

---

## 4. Run your own benchmark

### Step 1 — serve the test page

```bash
python3 -m http.server 8000
```

### Step 2 — install automation dependencies

```bash
pip install selenium numpy tqdm matplotlib jupyter
```

Ensure `chromedriver`, `geckodriver` (Firefox), and `msedgedriver` are in your `PATH` and match your installed browser versions.

### Step 3 — run

```bash
jupyter notebook test.ipynb
```

The notebook runs the full suite, saves per-browser `.pickle` files, and renders timing charts.

---

## 5. Interpreting results

The notebook generates:

1. **Throughput chart** — ops/sec by backend and browser
2. **Speedup ratio** — WASM / plain-JS for each operation size
3. **SIMD gain** — WASM+SIMD vs. WASM without SIMD
4. **Initialization latency** — first-inference overhead per backend

---

## 6. References

1. TensorFlow.js Team. "Introducing the WebAssembly backend for TensorFlow.js." *Google Developers Blog*, 2020.
2. A. Haas et al. "Bringing the Web up to Speed with WebAssembly." *PLDI '17*, 2017.
3. W3C. *WebAssembly SIMD Proposal.* github.com/WebAssembly/simd, 2021.
4. MDN. *SharedArrayBuffer and Atomics.* developer.mozilla.org, 2023.

---

## 🛠️ Stack

Python · Selenium · Jupyter · `@tensorflow/tfjs-backend-wasm` · Chrome / Edge / Firefox
