# **Benchmarking Llama 3.1 8B on CPU: Transformers vs. vLLM vs. llama.cpp**

## **Objective**

Compare the **peak memory usage (Max RSS)** and **latency** of running the Meta **Llama 3.1 8B** model on **CPU-only** across three implementation stacks:

* **A) Hugging Face Transformers (PyTorch, CPU-only)**

* **B) vLLM (CPU backend)**

* **C) llama.cpp (GGUF weights, CPU)**

We will measure memory footprint and speed for each stack under consistent conditions, and provide a clear recommendation based on the results.

## **Environment Setup**

* **Platform:** Ubuntu 22.04 LTS (AMD64 devcontainer). The environment is fixed to Ubuntu on x86\_64, so no special OS detection is needed.

* **CPU Hardware:** Record the CPU model, number of sockets, physical cores, and threads. Disable frequency scaling (set CPU governor to **performance** if possible) to ensure stable results. If the machine has multiple sockets, use `numactl` to bind the process and memory to one NUMA node (document the binding).

* **Threading Policy:** Use a single-process for each test. For fair comparison, set all relevant thread controls to the same value:

  1. Environment variables: `OMP_NUM_THREADS`, `MKL_NUM_THREADS` (if applicable).

  2. In PyTorch, call `torch.set_num_threads(n)`.

  3. We will test with **three thread counts**: `1`, `n_physical_cores`, and `n_physical_cores/2` (half the cores). This helps gauge scaling efficiency.

* **Virtual Environments:** Use **separate virtual environments** for each stack to avoid dependency conflicts (especially since vLLM’s CPU backend may require specific builds). We will create:

  1. `venv-hf` – for Transformers/PyTorch stack

  2. `venv-vllm` – for vLLM stack

  3. `venv-llamacpp` – for llama.cpp (may use for any Python helper scripts; llama.cpp itself is a compiled binary)

* **Package Installation:** Use the **UV package manager** (faster alternative to pip) in the devcontainer. For example, use `uv pip install ...` instead of `pip install ...` when installing dependencies in each venv. This ensures quicker, deterministic installs. If `uv` is not available for some reason, fall back to regular pip, but in our devcontainer `uv` is set up by default.

## **Model and Versions**

* **Model:** `meta-llama/Llama-3.1-8B` (the base model, or use the `...-8B-Instruct` variant if a chat-style prompt is needed). Make sure you have access to this gated model (accept the license on Hugging Face and set up credentials if required).

* **Weights & Precision Tracks:** We will evaluate two scenarios for model precision:

   **Track 1 – Full-Precision Parity:** All stacks use equivalent high precision.

  * *Transformers*: Load in BF16 if CPU supports it (bfloat16) for efficiency; otherwise use FP32.

  * *vLLM*: Uses BF16 by default on CPU (requires AVX512\_BF16 support or it will fall back appropriately).

  * *llama.cpp*: Use **F16 GGUF** weights (16-bit float) for the 8B model. This file is \~16 GB. (GGUF is the latest weight format for llama.cpp, successor to GGML.)

* **Track 2 – Low-Memory Quantization:** Highlight practical memory savings using quantized weights (llama.cpp only, since Transformers/vLLM do not natively support these exact quantizations).

  * *llama.cpp:* Use 4-bit and 5-bit weights. Specifically, test **Q4\_K\_M** and **Q5\_K\_M** quantized GGUF variants (approximately 4.9 GB and 5.7 GB files for the 8B model, respectively). These represent balanced quantization with minimal accuracy loss.

  * *(Transformers and vLLM remain in full precision for fairness; their memory use would be much higher and not directly comparable to quantized llama.cpp, so Track 2 is mainly to show llama.cpp’s advantage in memory usage.)*

* *Rationale:* Track 1 ensures all stacks are using similar precision (so accuracy and memory usage are comparable at \~16 GB). Track 2 demonstrates how llama.cpp can trade off some precision for drastically lower memory, which is useful in practice. We will explain this distinction in the report.

* **Version Pins:** Use the specified versions for reproducibility:

  * **Transformers:** 4.57.1 (latest stable, which supports Llama 3.1)

  * **PyTorch:** 2.9.x (latest stable 2.9 CPU build). If a 2.9 wheel isn’t available for our environment, use PyTorch ≥2.5 at minimum.

  * **vLLM:** 0.11.0 (latest stable release for vLLM). This version introduces the CPU backend. We will build or install it with CPU support (requires AVX-512 instructions).

  * **llama.cpp:** latest commit of ggml-org/llama.cpp (build from source). Use the latest release or HEAD to get all GGUF improvements.

## **Test Data and Prompt Design**

We want realistic usage scenarios with controlled conditions:

* **Deterministic generation:** Set *temperature \= 0.0* and *top\_p \= 1.0* for all tests. This forces deterministic, greedy decoding so that each run produces the same output given the same prompt (useful for verifying consistency and timing). No sampling or randomization.

* **Prompt lengths:** Test with three prompt sizes to see how initial prompt length affects memory and latency:

  * \~256 tokens (short prompt)

  * \~1024 tokens (medium prompt)

  * \~2048 tokens (long prompt close to model context limit for 8B)  
     These can be a mix of use cases (e.g., a short Q\&A prompt vs. a long document or code snippet). Use neutral, consistent content across tests. For example, a short prompt could be a question, and a long prompt could be a chunk of Wikipedia text or code. The exact content isn’t critical as long as token lengths are in these ranges and the same for each stack.

* **Generation length:** In all cases, generate **128 tokens** of output from the model. (We will measure time to first token and throughput over these 128 tokens.)

* **Tokenizer consistency:** Use the same tokenizer across stacks for counting tokens in prompts if possible. The Hugging Face tokenizer for Llama 3.1 can be used to pre-count prompt tokens (prompt length in tokens) so we accurately compare “256 tokens” etc. vLLM and HF will use that tokenizer by default when given the model, and llama.cpp’s GGUF will have the same tokenizer baked in, so the tokenization should align.

## **Metrics to Measure**

For each test run, collect the following metrics (with precise definitions and how to measure them):

* **Peak Memory Usage (Max RSS):** The maximum resident set size of the process, measured in MiB. We will use the system utility `/usr/bin/time -v` to run each benchmark process; it reports “Maximum resident set size (kbytes)”. Convert that to MiB (divide by 1024). This is our ground-truth memory footprint measurement for the model and framework. (*Note:* This captures the full process memory; for Python-based stacks this includes Python interpreter overhead, etc., but it’s the fairest external measure.)

* **Time to First Token (TTFT):** The wall-clock time from when a request is made (prompt submitted for generation) to when the **first output token** is produced. We will measure this in milliseconds. Implementation: record a timestamp immediately before generation starts, then another when the first token is emitted by the model. In each stack:

  * **Transformers:** We can hook into the generate loop or simply measure the elapsed time until `generate()` returns the first token in its output (in a greedy deterministic setting, `generate()` might produce the whole sequence at once; to get first-token latency, we may need to generate step by step or modify the generation to yield tokens). If using plain `model.generate`, TTFT will roughly equal total generation latency, so instead we might simulate by generating token-by-token in a loop.

  * **vLLM:** vLLM is optimized for streaming generation. If using the vLLM API, we should retrieve the first token from the `LLM.generate` output (which might stream tokens) and timestamp that. If using a server mode, measure from request to first chunk in response. We prefer the Python API for a direct measurement in-process.

  * **llama.cpp:** The CLI can stream tokens to stdout. We will instrument this by launching the process and noting when the first token appears in the output stream (or adding a timer in code). The `--nolog` flag ensures minimal overhead. We might need to parse the llama.cpp output: it typically prints each generated token (unless `--silent` or similar is used). We will capture a timestamp when the first token text is output. Alternatively, run llama.cpp in a wrapper that prints timestamps.

* **Decode Throughput (tokens/s after first token):** The generation rate after the first token. We define this as `generated_tokens / (total_time - TTFT)`. In practice, since we generate 128 tokens, this would be approximately 128 tokens divided by the time from first token to the last token. If the stack provides a tokens/sec metric (e.g., llama.cpp prints an overall token/s at end), we will still calculate it ourselves to keep methodology consistent across stacks. This measures how fast the model produces the remaining tokens once it gets going (it excludes the one-time overhead before the first token, which often includes caching the prompt or other initialization).

* **Latency distribution:** Each test scenario (stack \+ thread count \+ prompt length \+ track) will be run **3 times**. From those runs, we will report the **median (p50)** and **95th percentile (p95)** for the two latency metrics (TTFT and throughput). The median gives typical performance, while p95 (worst of three in this case, since N=3) indicates run-to-run variability or any occasional stalls. For memory (Max RSS), which is usually stable run-to-run, we will report the median value across the 3 runs for each scenario.

## **Benchmark Procedure by Stack**

We will run the tests for each stack in isolation, then aggregate results. Ensure that for each stack the only active threads/processes are the ones under test (to avoid CPU contention). All runs are CPU-only (no GPU usage).

### **A. Hugging Face Transformers (PyTorch) – CPU Inference**

**Environment & Install:** Activate the `venv-hf` virtual environment. Install the required packages using UV:

 `uv pip install torch==2.9.0 torchvision==2.9.0 torchaudio==2.9.0   # PyTorch CPU latest`    
`uv pip install transformers==4.57.1 accelerate==0.23.0             # Transformers and optional accelerate`

1.  *(Note: If PyTorch 2.9.0 is not available for your CPU/OS, use the latest 2.5+ CPU version. `accelerate` is not strictly required but can help if we needed to dispatch model to CPU. We will use pure transformers for simplicity.)*

**Model Loading:** Use the Transformers API to load the Llama 3.1 8B model and tokenizer. Example in code:

 `from transformers import AutoModelForCausalLM, AutoTokenizer`  
`import torch`  
`model_id = "meta-llama/Llama-3.1-8B"  # or 8B-Instruct if needed`  
`tokenizer = AutoTokenizer.from_pretrained(model_id)`  
`model = AutoModelForCausalLM.from_pretrained(`  
    `model_id,`  
    `torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() or torch.cpu.is_bf16_supported() else torch.float32,`  
    `device_map={"": "cpu"}   # load on CPU`  
`)`  
`model.eval()`  
`torch.set_num_threads(<THREAD_COUNT>)`

2.  This will load the model weights (\~16 GB in BF16) into CPU memory. If BF16 is not supported by the CPU, it will use FP32 (\~32 GB, but on most modern CPUs bfloat16 is supported).

3. **Inference Settings:** We use greedy decoding due to `temperature=0`. Prepare the prompt text and tokenize it. We will not use `model.generate` directly in one go because we want to measure time to first token. Instead, consider generating step-by-step: e.g., use `model(**inputs)` to get logits for next token repeatedly. However, for simplicity, you can measure end-to-end generate time as TTFT (since in non-streaming API it's effectively the first and only output time). In this context, we'll approximate TTFT \= total generation time for HF (we will note this assumption in the report).

Ensure to wrap generation in a timing context. For example, in Python:

 `import time`  
`input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids`  
`start_time = time.time()`  
`output_ids = model.generate(input_ids, max_new_tokens=128, do_sample=False, temperature=0.0)`  
`first_token_time = start_time  # (In Transformers generate, first token is at completion)`  
`end_time = time.time()`  
`total_time = end_time - start_time`

*  Since Transformers doesn’t easily give first token timing without deeper hooks, we will treat `total_time` as the single-shot latency and use that as TTFT (worst-case). We’ll note that HF isn’t streaming in our test.

  * After generation, gather the output to ensure the model actually generated 128 tokens (to be consistent across stacks).

**Memory Measurement:** Run the entire inference process (loading model \+ generation) under `/usr/bin/time -v`. For example:

 `/usr/bin/time -v python run_hf_inference.py --prompt prompt.txt --threads <N>`

4.  The Python script `run_hf_inference.py` will handle steps 2-3 for a given prompt and thread count. The time command will report Max RSS. We capture that from stderr.

5. **Repeat Runs:** Do the above for each prompt length (256, 1024, 2048 tokens) and each thread count (1, half cores, full cores). Each combination run 3 times. Log the timings (start, first token, end) for each run in a log file (e.g., `results/logs/hf_<threads>_<promptlen>.log`).

### **B. vLLM (CPU Backend) – CPU Inference**

**Environment & Install:** Activate `venv-vllm`. Install vLLM and its dependencies. Using UV:

 `uv pip install vllm==0.11.0`

1.  *Note:* vLLM 0.11.0’s PyPI might ship with a CPU backend, but it may also try to install GPU deps by default. If needed, build from source enabling CPU support. Ensure the system has **AVX-512** instructions (vLLM CPU backend requires AVX512 for optimized kernels; if not present, this stack may not run). We will document any CPU capability checks (e.g., using `lscpu` to see if avx512 is in flags).

**Setup vLLM for CPU:** Set environment variable for the process:

 `export VLLM_TARGET_DEVICE=cpu`

2.  This tells vLLM to use its CPU KV-cache backend instead of trying GPU. Optionally, set `VLLM_CPU_KVCACHE_SPACE=<MB>` if we need to increase default key/value cache memory (for long prompts). By default it might allocate enough based on model size; we will monitor and adjust if needed.

**Using the vLLM API:** We will use vLLM’s Python interface for a single request (to avoid overhead of running a server). For example, a `run_vllm_inference.py` script:

 `from vllm import LLM, SamplingParams`  
`llm = LLM(model="meta-llama/Llama-3.1-8B", tokenizer="meta-llama/Llama-3.1-8B", dtype="bfloat16")`  
`prompt = open("prompt.txt").read()`  
`sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)`  
`import time`  
`start_time = time.time()`  
`outputs = llm.generate([prompt], sampling_params)`  
`first_token_time = outputs[0].timestamps[0]  # hypothetical: if vLLM provides timestamps per token`  
`end_time = time.time()`

3.  In practice, `LLM.generate` might stream tokens under the hood but returns the full output at the end. We might not have an easy hook for first token time; if needed, we could modify vLLM to print timestamps or measure difference between start and when `outputs` becomes available (which again is full completion). If vLLM prints its own throughput (it might log something), we’ll ignore those and use our measurements.

   * Pin threads for vLLM as well. vLLM likely uses multiple threads internally (possibly via oneDNN or PyTorch). We should also set `OMP_NUM_THREADS` before running to `<THREAD_COUNT>` to ensure vLLM doesn’t spawn more threads than intended.

**Memory & Runs:** Similar to HF, run the vLLM inference script under `/usr/bin/time -v` for each scenario. For example:

 `/usr/bin/time -v python run_vllm_inference.py --prompt prompt.txt --threads <N>`

4.  Ensure that vLLM loads the model fresh in each run (to measure memory properly per process). Each run will load 16GB of model in BF16.

5. **Repeat for all prompt lengths and thread counts**, 3 runs each. Log timings and results in `results/logs/vllm_<threads>_<promptlen>.log`. Watch for any issues (vLLM is newer on CPU; ensure it does not crash on long prompts or consume excessive memory beyond model size). Document any such anomalies.

### **C. llama.cpp (GGUF) – CPU Inference**

**Build llama.cpp:** Ensure the system has the necessary build tools (CMake, Make, a C++ compiler). In `venv-llamacpp` (or outside venv, as it’s a system build):

 `git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp`  
`make -j$(nproc)`

1.  This builds the `llama` binary (or `main` binary) with default flags (which usually auto-detect AVX2/AVX512, etc., for optimal performance). The devcontainer being Ubuntu 22.04 on AMD64 likely supports at least AVX2; if AVX-512 is available, llama.cpp will use it too.

2. **Prepare GGUF model files:** Convert the Llama 3.1 8B model to GGUF format. You will need the original HF model files (from `meta-llama/Llama-3.1-8B`). Use the conversion script provided by llama.cpp or the `transformers` conversion utility. For example, llama.cpp has `convert.py` that can take a HuggingFace model and output GGUF. Generate:

   * **F16 GGUF** for Track 1: (16GB).

   * **Q4\_K\_M GGUF** and **Q5\_K\_M GGUF** for Track 2: (approx 5GB each).  
      Ensure these `.gguf` files are in a known directory (e.g., `./models/llama-3.1-8b/` with distinct filenames for f16, q4\_k\_m, q5\_k\_m).

**Run Inference with llama.cpp CLI:** We will use the command-line program to load the model and generate text. The basic usage:

 `./llama.cpp/build/bin/main -m /path/to/model.gguf -t <THREADS> -p "<prompt_text>" -n 128 --temp 0.0 --nolog`  
 We might create a small wrapper script `run_llama_cpp.sh` to simplify passing prompts from a file and capturing output. For instance:

 `#!/usr/bin/env bash`  
`PROMPT_FILE=$1`  
`THREADS=$2`  
`MODEL_FILE=$3`  
`/usr/bin/time -v ./llama.cpp/build/bin/main -m $MODEL_FILE -t $THREADS -n 128 --temp 0.0 --nolog -p "$(cat $PROMPT_FILE)"`

3.  This will print tokens to stdout; `/usr/bin/time -v` will print memory usage after the process exits. We need to capture timestamps for first token and last token:

   * If llama.cpp doesn’t output a timestamp, we can estimate TTFT by inserting our own timing. One approach: run the process and monitor its output in a parent script, recording the time when the first byte of output appears. Alternatively, modify llama.cpp to print a line when the first token is generated along with a timestamp. For simplicity, we might run the CLI without `--nolog` (so it logs each token with a timestamp by default). Actually, llama.cpp by default prints each token; it might not include timestamps. We can use `stdbuf -oL` to ensure we get output line by line and use a Python or expect script to timestamp the first line of output.

   * Given complexity, we might approximate TTFT for llama.cpp as the time from start to when the first character of generated text is output. This likely is very short after prompt processing, but we will attempt to measure it accurately.

**Memory & Runs:** Use the above script for each model variant and thread count. For example:

 `./run_llama_cpp.sh prompt.txt <N_THREADS> models/llama-3.1-8b-f16.gguf`

4.  and similarly for q4\_K\_M and q5\_K\_M models (Track 2). Each run yields Max RSS from `/usr/bin/time`. Collect TTFT and throughput from the timestamps or by post-processing the output logs. llama.cpp often prints an overall performance like “X tokens/s” at the end; you can use that as a sanity check for throughput but compute our own as described.

5. **Repeat for each prompt length** (256, 1024, 2048\) and each thread count, 3 runs each. Log outputs in `results/logs/llama_<threads>_<promptlen>_<model>.log`.

## **Orchestration and Automation**

To ensure reproducibility and avoid manual error, we will automate the entire benchmark with a master script. **Option 2 (isolated envs \+ bash orchestrator)** will be used as recommended:

* Create three separate virtual environments as described (ensuring UV is available in each).

* Write stack-specific scripts (`run_hf_inference.py`, `run_vllm_inference.py`, `run_llama_cpp.sh`, etc.) to execute one round of inference given a prompt and thread count. Each script focuses on its stack’s logic.

* Write an orchestration bash script **`orchestrate.sh`** that:

  * Activates `venv-hf`, then loops over prompt files and thread counts, calling the HF script with `/usr/bin/time -v` and capturing metrics (perhaps the script itself prints a JSON or CSV line of metrics for each run). Deactivate venv.

  * Activates `venv-vllm` and does the same with vLLM script.

  * Activates `venv-llamacpp` and runs llama.cpp tests for F16, Q4\_K\_M, Q5\_K\_M models.

  * Ensures sequential execution (to not overload CPU or memory by parallel runs).

  * Optionally takes an argument to run only a specific stack’s tests (for debugging or if we want to rerun just one). For example, `./orchestrate.sh llama.cpp` could just run the llama.cpp part. By default, with no args, it runs all stacks sequentially.

* Make sure the orchestrator captures the outputs. One approach: have each stack’s script output a line in a consistent JSON or CSV format containing all relevant fields (stack name, track, precision/quant, threads, prompt\_length, TTFT, throughput, memory, etc.). The orchestrator can redirect these to a combined log file, which we later convert to the final CSV.

* **Thread binding:** In the orchestrator, for each run, export `OMP_NUM_THREADS=<N>` (and same for MKL) and perhaps use `taskset` or `numactl` to pin the process to a single CPU socket if applicable. This ensures the process doesn’t migrate or use other sockets. Document the CPU binding in the system info output.

* **Repeatability:** The script should allow re-running the whole suite. Clean up any loaded models from memory between runs (since each run is a separate process, memory should be freed on exit).

## **Data Recording and Outputs**

After running all scenarios, gather the results:

* **Benchmark Results CSV:** Save a CSV file at `results/bench.csv` with columns:  
   `stack, track, quant_or_dtype, threads, prompt_tokens, gen_tokens, ttft_ms_p50, ttft_ms_p95, decode_tps_p50, decode_tps_p95, max_rss_mib_p50, cpu_model, sockets, cores, os, torch_ver, transformers_ver, vllm_ver, llamacpp_git`

   Each row is a scenario (e.g., “Transformers, Track1, BF16, threads=16, prompt=1024 tokens, gen=128 tokens, TTFT median=XYZ, TTFT p95=..., throughput median=..., throughput p95=..., memory median=..., plus system and version info repeated or in separate columns”). The hardware and software version columns can be constant across all rows since they’re the same environment (or just filled in for reference on each).

* **System Info:** Save `results/system_info.json` containing details about the environment: output of `lscpu` (for CPU model, cores, flags), `uname -a` (OS kernel), `/proc/meminfo` (total RAM), and versions of each relevant package (torch, transformers, vLLM, commit hash of llama.cpp). This helps others reproduce the setup exactly.

* **Logs:** Keep raw logs in `results/logs/` for transparency. For example:

  * `hf_8b_prompt256_thr16_run1.log` (raw stdout/stderr of that run including time outputs)

  * Similarly for vLLM and llama.cpp runs.  
     These logs are useful to verify that each run completed successfully and for any debugging.

* **Summary Report:** Write a one-page summary of conclusions (this can be a Markdown or PDF in `results/summary.md`). In this report, include a **ranked summary for each track**: e.g., which stack had the lowest latency in Track 1, which had the lowest memory, etc. We will provide a recommendation based on the results. (This summary can be prepared after running, using the data to back the statements.)

## **Conclusion and Recommendation (Draft)**

*After conducting the above benchmarks, we will form conclusions.* For example, we expect:

* **Latency (Track 1, full precision):** On CPU, **vLLM** might achieve the fastest generation due to its optimized streaming architecture and AVX-512 fused ops, **if** running on a CPU that supports AVX-512 and BF16. Transformers (PyTorch) will serve as a baseline – straightforward but possibly slower due to less optimization for long context. **llama.cpp** in F16 might be slightly slower per token because it’s highly optimized in C++ but doesn’t use some of the fused kernels that vLLM or oneDNN might; however, llama.cpp can scale well with threads. We’ll confirm with data. TTFT for Transformers and llama.cpp includes full prompt processing, whereas vLLM’s design might amortize that better – expect vLLM to have a lower TTFT relative to total time.

* **Memory:** In Track 1, all approaches load the full model in \~16 GB. However, overhead can differ: Transformers (Python) might use extra space for Python objects and overhead (possibly a few hundred MB more). vLLM might use additional memory for its KV cache (especially with long prompts, since it keeps a cache for fast token generation). llama.cpp is quite memory efficient in pure C++, with memory mostly for the model and context. We anticipate vLLM’s Max RSS could be slightly higher if it pre-allocates a large KV cache in memory for 2048-token context (which might be a few GB). We will report if any stack exceeded the base model size significantly.

* **Quantized Performance (Track 2):** llama.cpp’s 4-bit and 5-bit models will **drastically reduce memory** – down to \~5 GB – which is a clear advantage if memory is limited. The speed of quantized models can be a trade-off: 4-bit might be faster due to less data movement, or slightly slower if it requires extra bit unpacking computation; 5-bit (Q5\_K\_M) often has a good balance of speed and accuracy. We will see in results if Q4 vs Q5 has any latency difference. Regardless, neither HF Transformers nor vLLM natively support such low-bit quantization out-of-the-box at the moment, making llama.cpp the go-to solution for low-memory CPU inference of Llama 3.1.

* **Recommendation:** If **memory footprint** is the primary concern, **llama.cpp with Q4\_K\_M quantization** is highly recommended (≈5 GB RAM vs 16 GB for full precision) with only a small impact on model quality. If **latency** is the priority and the CPU supports the required instructions, **vLLM’s CPU backend** might offer the best throughput and lower latency to first token due to its optimized architecture (it’s designed for high-throughput serving). However, vLLM CPU requires an AVX-512 capable processor; on older CPUs without AVX-512, it may not work or will be significantly slower, in which case vanilla Transformers or llama.cpp are the alternatives. **Transformers (PyTorch)** provides a reliable baseline and easy implementation, but is likely the slowest in this comparison for CPU inference. In summary: for cutting-edge CPU servers with AVX-512, try vLLM; for low-memory or edge devices, use llama.cpp quantized; for simplicity and general compatibility, Transformers is acceptable but not optimal for performance.

We will validate these points with the actual data collected. All version pins and hardware details are recorded to ensure reproducibility. The final report will include the data and a ranked summary of which stack performed best under each track scenario, along with the above recommendations and notes on requirements (like AVX-512 for vLLM).

