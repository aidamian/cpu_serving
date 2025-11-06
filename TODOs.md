Perform the following tasks:

1. Modify the llama.cpp benchmark so that it also tests speed and memory usage using at least two different quantization methods (for example, int4 and int8) if supported by the model.

2. Add a new test (including the venv if required) that would use hf and bitsandbites with int4 as well as int8 quantization to benchmark the given model (two runs with two results for int4 and int8).

Make sure that all the test are properly integrated into the existing benchmarking framework and tested and that the results are clearly reported for each quantization method used.





rename simple_test.sh to test.sh and make sure it runs all tests including the new quantization tests. Review all functions, methods and files and complete them with detailed numpy-style docstrings.
