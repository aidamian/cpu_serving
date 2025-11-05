=== Running backend: vllm ===
Executing: /opt/venvs/venv-vllm/bin/python /workspaces/cpu_serving/scripts/benchmark_vllm.py --output /workspaces/cpu_serving/artifacts/simple_test/vllm/20251105T073258Z_simple-test.json --prompt Quick CPU smoke-test prompt for the Hugging Face backend. --max-new-tokens 8 --temperature 0.0 --top-p 1.0 --repetition-penalty 1.0 --model-id /workspaces/cpu_serving/models/meta-llama--Llama-3.2-1B --dtype float32 --tensor-parallel-size 1 --enforce-eager --num-threads 2 --warmup-tokens 32
/opt/venvs/venv-vllm/lib/python3.12/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 11-05 07:33:13 [__init__.py:220] No platform detected, vLLM is running on UnspecifiedPlatform
WARNING 11-05 07:33:16 [_custom_ops.py:20] Failed to import from vllm._C with ImportError('libcudart.so.12: cannot open shared object file: No such file or directory')
INFO 11-05 07:33:16 [arg_utils.py:504] HF_HUB_OFFLINE is True, replace model_id [/workspaces/cpu_serving/models/meta-llama--Llama-3.2-1B] to model_path [/workspaces/cpu_serving/models/meta-llama--Llama-3.2-1B]
INFO 11-05 07:33:16 [arg_utils.py:504] HF_HUB_OFFLINE is True, replace model_id [/workspaces/cpu_serving/models/meta-llama--Llama-3.2-1B] to model_path [/workspaces/cpu_serving/models/meta-llama--Llama-3.2-1B]
INFO 11-05 07:33:16 [utils.py:233] non-default args: {'trust_remote_code': True, 'dtype': 'float32', 'disable_log_stats': True, 'enforce_eager': True, 'model': '/workspaces/cpu_serving/models/meta-llama--Llama-3.2-1B'}
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
INFO 11-05 07:33:16 [model.py:547] Resolved architecture: LlamaForCausalLM
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 11-05 07:33:16 [model.py:1727] Upcasting torch.bfloat16 to torch.float32.
INFO 11-05 07:33:16 [model.py:1510] Using max model len 131072
WARNING 11-05 07:33:16 [cpu.py:117] Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) for CPU backend is not set, using 4 by default.
INFO 11-05 07:33:16 [scheduler.py:205] Chunked prefill is enabled with max_num_batched_tokens=4096.
/opt/venvs/venv-vllm/lib/python3.12/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 11-05 07:33:22 [__init__.py:220] No platform detected, vLLM is running on UnspecifiedPlatform
WARNING 11-05 07:33:24 [_custom_ops.py:20] Failed to import from vllm._C with ImportError('libcudart.so.12: cannot open shared object file: No such file or directory')
(EngineCore_DP0 pid=12692) INFO 11-05 07:33:24 [core.py:644] Waiting for init message from front-end.
(EngineCore_DP0 pid=12692) INFO 11-05 07:33:24 [core.py:77] Initializing a V1 LLM engine (v0.11.0) with config: model='/workspaces/cpu_serving/models/meta-llama--Llama-3.2-1B', speculative_config=None, tokenizer='/workspaces/cpu_serving/models/meta-llama--Llama-3.2-1B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float32, max_seq_len=131072, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=True, kv_cache_dtype=auto, device_config=cpu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/workspaces/cpu_serving/models/meta-llama--Llama-3.2-1B, enable_prefix_caching=True, chunked_prefill_enabled=True, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":null,"use_inductor":true,"compile_sizes":null,"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":0,"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":[],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"use_inductor_graph_partition":false,"pass_config":{},"max_capture_size":null,"local_cache_dir":null}
(EngineCore_DP0 pid=12692) INFO 11-05 07:33:24 [importing.py:43] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
(EngineCore_DP0 pid=12692) INFO 11-05 07:33:24 [importing.py:63] Triton not installed or not compatible; certain GPU-related functions will not be available.
(EngineCore_DP0 pid=12692) WARNING 11-05 07:33:25 [interface.py:381] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708] EngineCore failed to start.
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708] Traceback (most recent call last):
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 699, in run_engine_core
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     engine_core = EngineCoreProc(*args, **kwargs)
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 498, in __init__
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     super().__init__(vllm_config, executor_class, log_stats,
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 83, in __init__
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     self.model_executor = executor_class(vllm_config)
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/executor/executor_base.py", line 54, in __init__
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     self._init_executor()
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/executor/uniproc_executor.py", line 54, in _init_executor
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     self.collective_rpc("init_device")
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/executor/uniproc_executor.py", line 83, in collective_rpc
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     return [run_method(self.driver_worker, method, args, kwargs)]
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/utils/__init__.py", line 3122, in run_method
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     return func(*args, **kwargs)
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/worker/worker_base.py", line 259, in init_device
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     self.worker.init_device()  # type: ignore
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/worker/cpu_worker.py", line 49, in init_device
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     self.local_omp_cpuid = self._get_autobind_cpu_ids(
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/worker/cpu_worker.py", line 116, in _get_autobind_cpu_ids
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     CpuPlatform.get_allowed_cpu_core_node_list()
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/platforms/cpu.py", line 284, in get_allowed_cpu_core_node_list
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     logical_cpu_list: list[LogicalCPUInfo] = json.loads(
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]                                              ^^^^^^^^^^^
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/usr/lib/python3.12/json/__init__.py", line 359, in loads
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     return cls(**kw).decode(s)
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]            ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/usr/lib/python3.12/json/decoder.py", line 337, in decode
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     obj, end = self.raw_decode(s, idx=_w(s, 0).end())
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]   File "/usr/lib/python3.12/json/decoder.py", line 355, in raw_decode
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708]     raise JSONDecodeError("Expecting value", s, err.value) from None
(EngineCore_DP0 pid=12692) ERROR 11-05 07:33:26 [core.py:708] json.decoder.JSONDecodeError: Expecting value: line 6 column 18 (char 79)
(EngineCore_DP0 pid=12692) Process EngineCore_DP0:
(EngineCore_DP0 pid=12692) Traceback (most recent call last):
(EngineCore_DP0 pid=12692)   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore_DP0 pid=12692)     self.run()
(EngineCore_DP0 pid=12692)   File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
(EngineCore_DP0 pid=12692)     self._target(*self._args, **self._kwargs)
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 712, in run_engine_core
(EngineCore_DP0 pid=12692)     raise e
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 699, in run_engine_core
(EngineCore_DP0 pid=12692)     engine_core = EngineCoreProc(*args, **kwargs)
(EngineCore_DP0 pid=12692)                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 498, in __init__
(EngineCore_DP0 pid=12692)     super().__init__(vllm_config, executor_class, log_stats,
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 83, in __init__
(EngineCore_DP0 pid=12692)     self.model_executor = executor_class(vllm_config)
(EngineCore_DP0 pid=12692)                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/executor/executor_base.py", line 54, in __init__
(EngineCore_DP0 pid=12692)     self._init_executor()
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/executor/uniproc_executor.py", line 54, in _init_executor
(EngineCore_DP0 pid=12692)     self.collective_rpc("init_device")
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/executor/uniproc_executor.py", line 83, in collective_rpc
(EngineCore_DP0 pid=12692)     return [run_method(self.driver_worker, method, args, kwargs)]
(EngineCore_DP0 pid=12692)             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/utils/__init__.py", line 3122, in run_method
(EngineCore_DP0 pid=12692)     return func(*args, **kwargs)
(EngineCore_DP0 pid=12692)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/worker/worker_base.py", line 259, in init_device
(EngineCore_DP0 pid=12692)     self.worker.init_device()  # type: ignore
(EngineCore_DP0 pid=12692)     ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/worker/cpu_worker.py", line 49, in init_device
(EngineCore_DP0 pid=12692)     self.local_omp_cpuid = self._get_autobind_cpu_ids(
(EngineCore_DP0 pid=12692)                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/worker/cpu_worker.py", line 116, in _get_autobind_cpu_ids
(EngineCore_DP0 pid=12692)     CpuPlatform.get_allowed_cpu_core_node_list()
(EngineCore_DP0 pid=12692)   File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/platforms/cpu.py", line 284, in get_allowed_cpu_core_node_list
(EngineCore_DP0 pid=12692)     logical_cpu_list: list[LogicalCPUInfo] = json.loads(
(EngineCore_DP0 pid=12692)                                              ^^^^^^^^^^^
(EngineCore_DP0 pid=12692)   File "/usr/lib/python3.12/json/__init__.py", line 359, in loads
(EngineCore_DP0 pid=12692)     return cls(**kw).decode(s)
(EngineCore_DP0 pid=12692)            ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692)   File "/usr/lib/python3.12/json/decoder.py", line 337, in decode
(EngineCore_DP0 pid=12692)     obj, end = self.raw_decode(s, idx=_w(s, 0).end())
(EngineCore_DP0 pid=12692)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=12692)   File "/usr/lib/python3.12/json/decoder.py", line 355, in raw_decode
(EngineCore_DP0 pid=12692)     raise JSONDecodeError("Expecting value", s, err.value) from None
(EngineCore_DP0 pid=12692) json.decoder.JSONDecodeError: Expecting value: line 6 column 18 (char 79)
Traceback (most recent call last):
  File "/workspaces/cpu_serving/scripts/benchmark_vllm.py", line 168, in <module>
    main()
  File "/workspaces/cpu_serving/scripts/benchmark_vllm.py", line 152, in main
    result = run_vllm_benchmark(config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/cpu_serving/src/cpu_serving/benchmarks.py", line 396, in run_vllm_benchmark
    llm = LLM(
          ^^^^
  File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 297, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 177, in from_engine_args
    return cls(vllm_config=vllm_config,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 114, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 80, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 602, in __init__
    super().__init__(
  File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 448, in __init__
    with launch_core_engines(vllm_config, executor_class,
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 732, in launch_core_engines
    wait_for_engine_startup(
  File "/opt/venvs/venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 785, in wait_for_engine_startup
    raise RuntimeError("Engine core initialization failed. "
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore_DP0': 1}
Backend 'vllm' failed with exit code 1.






introduce log_color(s, color) with colors such as "g" (green), "b" (blue), "y" (yellow), "r" (red), "d" (dark grey) that would also use print(output, flush=True) so that the output warning/lib info gets separated from script/process/benchmark actual information. Chose color accordingly. Shorten the "model" column in the outputs to just the final part of the file/path (out of "workspaces/cpu_serving/models/hugging-quants--Llama-3.2-1B-Instruct-Q4_K_M-GGUF/llama-3.2-1b-instruct-q4_k_m.gguf" use just "llama-3.2-1b-instruct-q4_k_m.gguf")






Change the prompt for a more suitable task for on-edge devices such as SQL generation that would better simulate live and real use cases. 
Use a bigger prompt such as "Write the DDL SQL for the definition of user accounts table. Output only the viable SQL". 
Allow generation of up to 250 tokens.






Modify the llama.cpp benchmark so that it also tests speed and memory usage using 






Add a new test (including the venv if required) that would use hf and bitsandbites with int4 as well as int8 quantization to benchmark the given model (two runs with two results)

