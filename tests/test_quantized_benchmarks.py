from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import List
from unittest import mock


from cpu_serving.benchmarks import (
    BenchmarkResult,
    HFBenchmarkConfig,
    LlamaCppBenchmarkConfig,
    format_results_table,
    run_hf_quantized_benchmarks,
    run_llamacpp_quantized_benchmarks,
)


class FakeTensor:
    def __init__(self, data, dtype: str = "float32") -> None:
        self.data = [list(row) for row in data]
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, int]:
        if not self.data:
            return (0, 0)
        return (len(self.data), len(self.data[0]))

    def to(self, device: str) -> "FakeTensor":  # pragma: no cover - parity shim
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key):  # pragma: no cover - minimal indexing
        if isinstance(key, int):
            return FakeTensor([self.data[key]], dtype=self.dtype)
        if isinstance(key, slice):
            return FakeTensor(self.data[key], dtype=self.dtype)
        if isinstance(key, tuple) and len(key) == 2:
            row, column = key
            return self[row][column]
        raise TypeError(f"Unsupported index type: {type(key)!r}")

    def tolist(self) -> List[List[int]]:
        return [list(row) for row in self.data]


class _InferenceMode:
    def __enter__(self) -> None:  # pragma: no cover - control flow shim
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _tensor(data, dtype=None) -> FakeTensor:
    dtype_name = dtype if isinstance(dtype, str) else "float32"
    return FakeTensor(data, dtype=dtype_name)


def _ones_like(tensor: FakeTensor) -> FakeTensor:
    return FakeTensor([[1 for _ in row] for row in tensor.data], dtype=tensor.dtype)


def _full(shape: tuple[int, int], value: int, dtype=None) -> FakeTensor:
    rows, cols = shape
    dtype_name = dtype if isinstance(dtype, str) else "float32"
    return FakeTensor([[value] * cols for _ in range(rows)], dtype=dtype_name)


def _cat(tensors: List[FakeTensor], dim: int = 1) -> FakeTensor:
    if dim != 1:  # pragma: no cover - defensive
        raise NotImplementedError("Only dim=1 concatenation is supported in tests.")
    rows = len(tensors[0].data)
    merged = []
    for row_idx in range(rows):
        row: List[int] = []
        for tensor in tensors:
            row.extend(tensor.data[row_idx])
        merged.append(row)
    return FakeTensor(merged, dtype=tensors[0].dtype)


_fake_torch = types.ModuleType("torch")
_fake_torch.float32 = "float32"
_fake_torch.float16 = "float16"
_fake_torch.bfloat16 = "bfloat16"
_fake_torch.tensor = _tensor
_fake_torch.ones_like = _ones_like
_fake_torch.full = _full
_fake_torch.cat = _cat
_fake_torch.inference_mode = lambda: _InferenceMode()


class DummyMonitor:
    def __init__(self, *args, **kwargs) -> None:
        self.max_rss_bytes = 1_048_576

    def __enter__(self) -> "DummyMonitor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeLlama:
    instances: List["FakeLlama"] = []
    model_paths: List[str] = []

    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs
        FakeLlama.instances.append(self)
        FakeLlama.model_paths.append(kwargs.get("model_path"))

    def __call__(self, *args, **kwargs):
        return {
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            "choices": [{"text": "quantized output"}],
        }


class DummyTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt"):
        input_ids = _fake_torch.tensor([[1, 2, 3]], dtype="float32")
        attention_mask = _fake_torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:  # pragma: no cover - interface
        if hasattr(tokens, "tolist"):
            raw = tokens.tolist()[0] if tokens.tolist() else []
            return " ".join(str(value) for value in raw) or "dummy completion"
        return "dummy completion"

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "DummyTokenizer":
        return cls()


class DummyModel:
    calls: List[dict] = []

    def __init__(self, kwargs: dict) -> None:
        self.kwargs = kwargs

    def generate(self, input_ids, attention_mask=None, max_new_tokens=None, **kwargs):
        prompt_len = input_ids.shape[-1]
        extra_tokens = max_new_tokens or 2
        if extra_tokens <= 0:
            extra_tokens = 2
        extra = _fake_torch.full((input_ids.shape[0], extra_tokens), 4, dtype=input_ids.dtype)
        return _fake_torch.cat([input_ids, extra], dim=1)


class FakeAutoModel:
    calls: List[dict] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> DummyModel:
        cls.calls.append(kwargs)
        return DummyModel(kwargs)


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class QuantizedBenchmarksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor_patch = mock.patch("cpu_serving.benchmarks.MemoryMonitor", DummyMonitor)
        self.threads_patch = mock.patch("cpu_serving.benchmarks._ensure_threads_config", lambda *_: None)
        self.torch_patch = mock.patch.dict(sys.modules, {"torch": _fake_torch})
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = DummyTokenizer
        fake_transformers.AutoModelForCausalLM = FakeAutoModel
        fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
        self.transformers_patch = mock.patch.dict(sys.modules, {"transformers": fake_transformers})
        fake_tabulate_module = types.ModuleType("tabulate")

        def _fake_tabulate(rows, headers, tablefmt=None):
            header_line = " | ".join(headers)
            body_lines = [" | ".join(str(value) for value in row) for row in rows]
            return "\n".join([header_line, *body_lines])

        fake_tabulate_module.tabulate = _fake_tabulate
        self.tabulate_patch = mock.patch.dict(sys.modules, {"tabulate": fake_tabulate_module})
        self.monitor_patch.start()
        self.threads_patch.start()
        self.torch_patch.start()
        self.transformers_patch.start()
        self.tabulate_patch.start()
        self.addCleanup(self.monitor_patch.stop)
        self.addCleanup(self.threads_patch.stop)
        self.addCleanup(self.torch_patch.stop)
        self.addCleanup(self.transformers_patch.stop)
        self.addCleanup(self.tabulate_patch.stop)

    def test_llamacpp_quantized_benchmarks(self) -> None:
        FakeLlama.instances = []
        FakeLlama.model_paths = []
        fake_module = types.SimpleNamespace(Llama=FakeLlama)

        with tempfile.TemporaryDirectory() as tmp_dir:
            q4_path = Path(tmp_dir) / "llama-q4.gguf"
            q8_path = Path(tmp_dir) / "llama-q8.gguf"
            q4_path.write_bytes(b"q4")
            q8_path.write_bytes(b"q8")

            config = LlamaCppBenchmarkConfig(
                model_path=str(q4_path),
                quantizations={"int8": str(q8_path)},
                auto_discover_quantizations=False,
            )

            with mock.patch.dict(sys.modules, {"llama_cpp": fake_module}):
                results = run_llamacpp_quantized_benchmarks(
                    config,
                    quantization_order=["int4", "int8"],
                )

        self.assertEqual(len(results), 2)
        quantizations = [result.parameters.get("quantization") for result in results]
        self.assertEqual(quantizations, ["int4", "int8"])
        self.assertEqual(FakeLlama.model_paths, [str(q4_path), str(q8_path)])

    def test_hf_bitsandbytes_quantized_benchmarks(self) -> None:
        FakeAutoModel.calls = []

        config = HFBenchmarkConfig(
            model_id="dummy/model",
            tokenizer_id="dummy/model",
            prompt="Hello",
            bitsandbytes_compute_dtype="float16",
            bitsandbytes_quant_type="nf4",
        )

        results = run_hf_quantized_benchmarks(
            config,
            quantizations=["int4", "int8"],
        )

        self.assertEqual(len(results), 2)
        quantizations = [result.parameters.get("quantization") for result in results]
        self.assertEqual(quantizations, ["int4", "int8"])
        self.assertEqual(results[0].parameters.get("quantization_detail"), "nf4")
        self.assertEqual(results[1].parameters.get("quantization_detail"), "standard")
        self.assertEqual(len(FakeAutoModel.calls), 2)
        four_bit_config = FakeAutoModel.calls[0]["quantization_config"].kwargs
        eight_bit_config = FakeAutoModel.calls[1]["quantization_config"].kwargs
        self.assertTrue(four_bit_config.get("load_in_4bit"))
        self.assertTrue(eight_bit_config.get("load_in_8bit"))

    def test_hf_bitsandbytes_int8_offload_detail(self) -> None:
        FakeAutoModel.calls = []

        config = HFBenchmarkConfig(
            model_id="dummy/model",
            tokenizer_id="dummy/model",
            prompt="Hi",
            bitsandbytes_int8_cpu_offload=True,
        )

        results = run_hf_quantized_benchmarks(
            config,
            quantizations=["int8"],
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.parameters.get("quantization"), "int8")
        self.assertEqual(result.parameters.get("quantization_detail"), "fp32-offload")

    def test_format_results_table_includes_quantization_detail(self) -> None:
        result = BenchmarkResult(
            backend="huggingface",
            model="dummy/model",
            prompt="Hi",
            prompt_tokens=5,
            completion="completion",
            completion_tokens=7,
            max_new_tokens=10,
            load_time_s=1.23,
            generate_time_s=0.5,
            peak_memory_bytes=2_000_000,
            num_threads=2,
            parameters={"quantization": "int4", "quantization_detail": "nf4"},
        )
        table = format_results_table([result])
        self.assertIn("int4 (nf4)", table)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
