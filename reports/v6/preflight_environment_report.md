# CEN-TaTS v6 P0 预阶段环境验收报告

## 1. 执行摘要
P0_STATUS: PASS

本阶段只执行环境审计、GPT-2 本地权重验证、CUDA 验证、TaTS+iTransformer 单批次 smoke 与自动测试。未运行 M0-M5、完整数据集训练、事件抽取、因果图、verifier、APO 或任何付费 LLM API。

## 2. Git 分支和 commit
- branch: experiment/cen-tats-formal-v6
- base_commit: 7f4bac0721989d46a0d53c45aa1eb2d3e64683ac

## 3. Python/conda 环境
- conda env: tats
- expected python: D:/Miniconda/envs/tats/python.exe
- actual python: D:\Miniconda\envs\tats\python.exe
- note: 当前 shell 的 conda 不在 PATH，正式命令统一使用显式解释器路径。

## 4. 实际 python、pip、hf 路径
详见 $resultDir/environment_paths.txt。硬性解释器检查通过，sys.executable 指向 D:\Miniconda\envs\tats\python.exe。

## 5. GPT-2 下载状态
D:/models/gpt2 已包含全部 7 个必需文件，本轮未启动新的 hf download。

## 6. GPT-2 文件清单
| file | exists | size_bytes | sha256 |
|---|---:|---:|---|
| config.json | True | 665 | 0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4 |
| generation_config.json | True | 124 | ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b |
| merges.txt | True | 456318 | 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5 |
| model.safetensors | True | 548105171 | 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707 |
| tokenizer.json | True | 1355256 | 8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6 |
| tokenizer_config.json | True | 26 | 5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f |
| vocab.json | True | 1042301 | 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783 |

## 7. GPT-2 本地加载结果
- success: True
- model_class: GPT2Model
- hidden_size: 768
- parameter_count: 124439808
- local_files_only: True
- random_init: False
- embedding_variance: 57.44138717651367
- embeddings_not_identical: True

## 8. Transformers 和 Hugging Face 版本
`	ext
transformers= 5.13.1
huggingface_hub= 1.23.0
safetensors= 0.8.0

`

## 9. NVIDIA 驱动和 GPU
`	ext
Mon Jul 13 21:07:36 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.29                 Driver Version: 581.29         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti   WDDM  |   00000000:01:00.0  On |                  N/A |
| 31%   44C    P0             23W /  180W |     580MiB /  16311MiB |      9%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

`

## 10. PyTorch 与 CUDA 版本
修复前：
`	ext
python= 3.11.15 | packaged by Anaconda, Inc. | (main, Jun 11 2026, 15:12:53) [MSC v.1942 64 bit (AMD64)]
executable= D:\Miniconda\envs\tats\python.exe
torch= 2.13.0+cpu
torch_cuda= None
cuda_available= False
device_count= 0

`

修复后：
`	ext
python= 3.11.15 | packaged by Anaconda, Inc. | (main, Jun 11 2026, 15:12:53) [MSC v.1942 64 bit (AMD64)]
executable= D:\Miniconda\envs\tats\python.exe
torch= 2.7.0+cu128
torch_cuda= 12.8
cuda_available= True
device_count= 1
device_name= NVIDIA GeForce RTX 5060 Ti

`

## 11. CUDA 矩阵测试
- cuda_available: True
- device_count: 1
- gpu_matmul_ok: True
- allocated_bytes: 510916608
- reserved_bytes: 836763648

## 12. GPT-2 GPU forward
- gpu_forward_ok: True
- gpt2_model_class: GPT2Model
- device: cuda:0

## 13. TaTS iTransformer 最小 forward
- success: True
- backbone: iTransformer
- output_shape: 2,2,1
- device: cuda:0
- all_tensors_on_expected_gpu: True

## 14. 文本 projection 梯度
- raw_text_embedding_variance: 66.03424835205078
- projected_text_variance: 3.6237175464630127
- projection_gradient_norm: 58.87201605737209

## 15. real/zero/shuffle 预测差异
- prediction_real_vs_zero_max_diff: 0.015337973833084106
- prediction_real_vs_shuffle_max_diff: 0.007808566093444824

## 16. 自动测试结果
`	ext
.........                                                                [100%]
9 passed in 8.54s

`

## 17. 已修改文件
`	ext
 M .gitignore
?? configs/v6/
?? prompt/
?? reports/v6/
?? results/v6/
?? scripts/v6/
?? src/cen_tats/runtime/
?? tests/v6/

`

`	ext
 .gitignore | 7 +++++++
 1 file changed, 7 insertions(+)

`

## 18. 环境风险
- shell PATH 里的 python、pip、hf 指向系统 Python 或不可由 where.exe 发现，不能依赖 PowerShell 提示符或 PATH。
- 已按 P0 规则将 CPU-only torch 修复为官方 CUDA wheel：torch 2.7.0+cu128、torchvision 0.22.0+cu128、torchaudio 2.7.0+cu128。
- configs/v6/preflight.local.yaml 已加入 .gitignore，不会提交本地绝对路径配置。

## 19. P0 验收结论
P0_STATUS: PASS

## 20. 是否允许进入 P1
允许进入 P1，但本阶段按要求在 P0 报告和提交后停止。
