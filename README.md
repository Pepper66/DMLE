# DMLE: Distributed Multi-Layer Editing for Rule-Level Knowledge in Large Language Models

This repository contains the code and data for **Distributed Multi-Layer Editing (DMLE)**, a rule-level knowledge editing method for large language models.

DMLE is designed for editing structured rules that must remain consistent across multiple forms: symbolic formulas, natural language descriptions, and numerical instances. Instead of applying one uniform edit to a single layer or a contiguous block of layers, DMLE applies form-aware updates to different layer groups: formulas and descriptions share an early-layer update, while instances use a separate middle-layer update.

Paper: [Distributed Multi-Layer Editing for Rule-Level Knowledge in Large Language Models](https://arxiv.org/pdf/2604.08284)  
Code: [https://github.com/Pepper66/DMLE](https://github.com/Pepper66/DMLE)

## Overview

Most model editing methods are developed for fact-level knowledge, where a target edit is usually represented as a single association. Rule-level knowledge is more complex: the same rule may appear as a formula, a textual explanation, or a concrete numerical application. A successful edit should update all of these forms coherently while preserving unrelated model behavior.

The paper studies where different forms of rule knowledge are represented inside transformer models. Causal tracing shows a form-specific layer-wise pattern:

- Formula and description knowledge tend to concentrate in earlier layers.
- Instance-level applications tend to rely more on middle layers.
- A single-layer or uniform contiguous-layer edit is therefore often insufficient for coherent rule-level editing.

Based on this observation, DMLE performs:

- a shared early-layer update for formula and description forms;
- a separate middle-layer update for instance forms;
- MEMIT-style closed-form parameter updates adapted to rule-level editing.

## Repository Structure

```text
.
+-- run_new.py          # Main script for running multi-stage rule editing and evaluation
+-- cal_avg.py          # Utility script for aggregating saved metric files
+-- RuleEdit-200.zip    # RuleEdit-200 dataset archive
```

After extracting `RuleEdit-200.zip`, the dataset contains 200 rule folders. Each rule folder includes:

```text
<rule_id>/
+-- f_d.json    # Formula and description editing samples
+-- ins.json    # Instance editing samples
```

Each editing case contains the edited prompt and target, rephrase prompts, locality prompts, rule-understanding prompts, and instance-portability prompts.

## Dataset

This repository includes **RuleEdit-200**, an extension of the RuleEdit benchmark from 80 to 200 manually verified mathematics and physics rules. Each rule is represented in three aligned forms:

- **Formula**: the symbolic expression of the rule.
- **Description**: a natural language statement of the rule.
- **Instance**: a numerical application of the rule.

RuleEdit-200 is intended to evaluate whether an edited model can update a rule consistently across these forms, rather than only memorizing a single rewritten prompt.

## Installation

Create a Python environment and install the core dependencies:

```bash
conda create -n dmle python=3.10
conda activate dmle
pip install numpy
```

This code is built on top of [EasyEdit](https://github.com/zjunlp/EasyEdit). Install EasyEdit and its model dependencies following the EasyEdit documentation:

```bash
pip install git+https://github.com/zjunlp/EasyEdit.git
```

Depending on the backbone model and editor configuration, you may also need packages such as `torch`, `transformers`, `accelerate`, and CUDA-compatible dependencies.

## Data Preparation

Extract the dataset archive:

```bash
unzip RuleEdit-200.zip
```

The main script expects `--base_data_dir` to point to the extracted dataset directory. It also expects a global `config.json` file inside that directory. The current repository does not include this file, so you should create one according to the layer groups you want to edit.

Example:

```json
[
  {
    "name": "f_d",
    "layers": [3, 4, 5]
  },
  {
    "name": "ins",
    "layers": [12, 13, 14]
  }
]
```

The `f_d` stage edits formula and description samples, while the `ins` stage edits instance samples. The exact layer choices should match the target backbone and the causal-tracing configuration used in your experiment.

## Usage

Run DMLE-style multi-stage editing with:

```bash
python run_new.py \
  --editing_method MEMIT \
  --hparams_dir /path/to/easyedit/hparams \
  --metrics_save_dir ./results \
  --base_data_dir ./RuleEdit-200 \
  --rule_edit True \
  --evaluation_type exact_match \
  --api_key YOUR_API_KEY
```

Arguments:

- `--editing_method`: editing algorithm name. Supported values in the script include `IKE`, `ICE`, `MEMIT`, `ROME`, `LoRA`, and `GRACE`.
- `--hparams_dir`: path to the EasyEdit hyperparameter configuration.
- `--metrics_save_dir`: directory where metric JSON files will be saved.
- `--base_data_dir`: path to the extracted RuleEdit-200 dataset directory.
- `--rule_edit`: whether to enable rule-editing behavior in the hyperparameters.
- `--evaluation_type`: evaluation mode passed into the hyperparameters.
- `--api_key`: API key used by the LLM judge for locality evaluation.

The script processes each numbered rule folder, applies staged edits, evaluates the final edited model on the union of formula/description and instance samples, and saves per-rule metrics as:

```text
metrics_rule_<rule_id>.json
```

## Metric Aggregation

To aggregate saved metric files, use `cal_avg.py` by setting the result directory in the script:

```python
calculate_metrics("./results")
```

Then run:

```bash
python cal_avg.py
```

The aggregation reports rewrite accuracy, rephrase accuracy, locality, portability, and rule-understanding scores.

## Evaluation Metrics

Following the paper, DMLE is evaluated with both standard model-editing metrics and rule-specific metrics:

- **Reliability / Rewrite Accuracy**: whether the model produces the edited target for the edited prompt.
- **Generalization / Rephrase Accuracy**: whether the edit generalizes to rephrased prompts.
- **Locality**: whether unrelated knowledge remains unchanged.
- **Instance Portability (IP)**: whether the edited rule can be applied to numerical instances.
- **Rule Understanding (RU)**: whether the edited rule is reflected consistently across formula, description, and instance forms.

## Main Results

In the paper, DMLE is evaluated on GPT-J-6B, Qwen2.5-7B, Qwen2-7B, and LLaMA-3-8B. Compared with representative baselines including ROME, MEMIT, GRACE, and prompting-based editing, DMLE achieves stronger rule-level performance, especially on instance portability and rule understanding.

On average, the paper reports that DMLE improves instance portability and rule understanding by **13.91** and **50.19** percentage points, respectively, over the strongest baseline across the evaluated backbones.

## Notes

- This repository contains the core experiment script and RuleEdit-200 data, but does not include full EasyEdit hyperparameter files.
- The included script expects a `config.json` stage file in the extracted dataset directory.
- `cal_avg.py` contains some garbled comments due to text encoding, but the metric computation logic is still readable.
- Large model editing experiments require substantial GPU memory. Please adjust model, editor, batch size, and layer settings according to your hardware.

## Citation

If you use this repository or RuleEdit-200 in your work, please cite:

```bibtex
@article{wang2026dmle,
  title={Distributed Multi-Layer Editing for Rule-Level Knowledge in Large Language Models},
  author={Wang, Yating and Zhao, Wenting and Zhao, Yaqi and Gong, Yongshun and Yin, Yilong and Sun, Haoliang},
  journal={arXiv preprint arXiv:2604.08284},
  year={2026}
}
```

## License

No license file is currently included in this repository. Please contact the authors before using the code or dataset for purposes beyond academic research and reproduction.
