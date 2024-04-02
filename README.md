# spInfer
Speculative Inference for Gemma (and more to come).

## Installation
```console
pip install .
```

## Run example for Gemma
```console
run_gemma_spinfer
```
1. It will initiate 2 models -- HuggingFace token required:\
    a. Pre-trained `google/gemma-2b` as guessing model (`GemmaForCausalLM`)\
    b. Pre-trained `google/gemma-7b` as main model (`spGemmaForCausalLM`) which includes speculative functions.
2. Input sentences are tokenized as a batch.
3. Each sentence is processed individually using speculative inference technique. Specifically, the guessing model will predict for next `k` tokens (default to `5`), and the main model will do parallel prediction over the `k` new tokens. 
4. If the prediction of guessing model and main model are the same, multiple tokens can be accepted in single forward pass.

Example generated sequence:\
<span style="color:gray">[original]</span> [...][...]  -- [...]

<span style="color:gray">[2, 1841, 603, 573, 6996, 576, 1913, 235336]</span> [109, 1841] [603, 573, 6996, 576, 1913] [235336, 109, 1841, 603, 573, 6996] [576, 1913]

<span style="color:gray">[0, 2, 5559, 476, 3904, 1105, 8778, 235292]</span> [109] [22066, 603, 573, 1370, 1744, 576] [573, 2788, 235265, 1165, 603, 573] [1744, 1185]
