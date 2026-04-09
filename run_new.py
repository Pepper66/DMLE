#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from easyeditor import (
    IKEHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    LoRAHyperParams,
    GraceHyperParams,
)
from easyeditor import BaseEditor

from easyeditor.editors.utils import _prepare_requests
from easyeditor.evaluate.evaluate_utils import llm_judge

# More robust import for compute_edit_quality across different EasyEdit forks
try:
    from easyeditor.evaluate.evaluate import compute_edit_quality
except Exception:
    from easyeditor.evaluate import compute_edit_quality


# -------------------------
# 1) Summary printing
# -------------------------
def _safe_mean(x: Any) -> float:
    try:
        arr = np.array(x, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.mean(arr))
    except Exception:
        return float("nan")


def eval_summary(metrics_path: str) -> None:
    if not os.path.exists(metrics_path):
        print("[eval] file not found:", metrics_path)
        return
    with open(metrics_path, "r") as f:
        datas = json.load(f)

    rewrite_list = []
    rephrase_list = []

    for d in datas:
        ra = d.get("post", {}).get("rewrite_acc", None)
        if isinstance(ra, list) and len(ra) > 0:
            rewrite_list.append(float(ra[0]))
        elif ra is not None:
            rewrite_list.append(float(ra))

        rpa = d.get("post", {}).get("rephrase_acc", None)
        if isinstance(rpa, list) and len(rpa) > 0:
            rephrase_list.append(float(rpa[0]))
        elif rpa is not None:
            rephrase_list.append(float(rpa))

    if len(rewrite_list) > 0:
        print("Edit_Succ:", np.mean(rewrite_list) * 100.0)
    if len(rephrase_list) > 0:
        print("Overall_generality:", np.mean(rephrase_list) * 100.0)

    locality_case_scores = []
    for d in datas:
        loc = d.get("post", {}).get("locality", {})
        per_case = []
        for k, v in loc.items():
            if k.endswith("_acc"):
                per_case.append(_safe_mean(v) * 100.0)
        if len(per_case) > 0:
            locality_case_scores.append(float(np.mean(per_case)))
    if len(locality_case_scores) > 0:
        print("Overall_locality:", float(np.mean(locality_case_scores)))

    inst_scores = []
    rule_scores = []
    for d in datas:
        por = d.get("post", {}).get("portability", {})
        if "Instance_acc" in por:
            inst_scores.append(_safe_mean(por["Instance_acc"]) * 100.0)
        if "Rule_acc" in por:
            rule_scores.append(_safe_mean(por["Rule_acc"]) * 100.0)

    if len(inst_scores) > 0:
        print("Instance_portability:", float(np.mean(inst_scores)))
    if len(rule_scores) > 0:
        print("Rule_understanding:", float(np.mean(rule_scores)))


# -------------------------
# 2) Build inputs
# -------------------------
def build_inputs_from_dataset(dataset: List[Dict]) -> Tuple:
    prompts, subjects, ground_truths, target_news, rephrase_prompts = [], [], [], [], []
    neighborhood_prompts, neighborhood_truths = [], []
    distracting_prompts, distracting_truths = [], []
    instance_prompts, instance_truths = [], []
    rule_prompts, rule_truths = [], []

    for data in dataset:
        prompts.append(data["prompt"])
        subjects.append(data["subject"])
        ground_truths.append(data["ground_truth"] if "ground_truth" in data else "")
        target_news.append(data["target_new"])
        rephrase_prompts.append(data.get("rephrase_prompt", ""))

        neighborhood_prompts.append(data["locality"]["neighborhood"]["prompt"])
        neighborhood_truths.append(data["locality"]["neighborhood"]["ground_truth"])
        distracting_prompts.append(data["locality"]["distracting"]["prompt"])
        distracting_truths.append(data["locality"]["distracting"]["ground_truth"])

        instance_prompts.append(data["Instance"]["prompt"])
        instance_truths.append(data["Instance"]["target_new"])
        rule_prompts.append(data["Rule_Understanding"]["prompt"])
        rule_truths.append(data["Rule_Understanding"]["target_new"])

    locality_inputs = {
        "neighborhood": {"prompt": neighborhood_prompts, "ground_truth": neighborhood_truths},
        "distracting": {"prompt": distracting_prompts, "ground_truth": distracting_truths},
    }
    portability_inputs = {
        "Instance": {"prompt": instance_prompts, "ground_truth": instance_truths},
        "Rule": {"prompt": rule_prompts, "ground_truth": rule_truths},
    }

    return (
        prompts,
        subjects,
        ground_truths,
        target_news,
        rephrase_prompts,
        locality_inputs,
        portability_inputs,
    )


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# -------------------------
# 3) True batch edit via apply_algo
# -------------------------
def batch_apply_one_stage(editor: BaseEditor, hparams, stage_requests: List[Dict]) -> None:
    edited_model, _weights_copy = editor.apply_algo(
        editor.model,
        editor.tok,
        stage_requests,
        hparams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        train_ds=None,
    )
    editor.model = edited_model


# -------------------------
# 4) Compatibility helpers
# -------------------------
def prepare_requests_compat(
    prompts,
    target_new,
    ground_truth,
    rephrase_prompts,
    locality_inputs,
    portability_inputs,
    subjects,
):
    try:
        return _prepare_requests(
            prompts=prompts,
            target_new=target_new,
            ground_truth=ground_truth,
            target_neg=None,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            subject=subjects,
        )
    except TypeError:
        return _prepare_requests(
            prompts=prompts,
            target_new=target_new,
            ground_truth=ground_truth,
            target_neg=None,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
        )


def calculate_final_grand_average(metrics_files: List[str]):
    all_summaries = []

    for path in metrics_files:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            datas = json.load(f)

        rewrite_list = [float(
            d["post"].get("rewrite_acc", [0])[0]
            if isinstance(d["post"].get("rewrite_acc"), list)
            else d["post"].get("rewrite_acc", 0)
        ) for d in datas]

        rephrase_list = [float(
            d["post"].get("rephrase_acc", [0])[0]
            if isinstance(d["post"].get("rephrase_acc"), list)
            else d["post"].get("rephrase_acc", 0)
        ) for d in datas]

        loc_scores = []
        for d in datas:
            loc = d["post"].get("locality", {})
            per_case = [_safe_mean(v) * 100.0 for k, v in loc.items() if k.endswith("_acc")]
            if per_case:
                loc_scores.append(np.mean(per_case))

        inst_scores = [_safe_mean(d["post"].get("portability", {}).get("Instance_acc", 0)) * 100.0 for d in datas]
        rule_scores = [_safe_mean(d["post"].get("portability", {}).get("Rule_acc", 0)) * 100.0 for d in datas]

        all_summaries.append({
            "Edit_Succ": np.mean(rewrite_list) * 100.0,
            "Generality": np.mean(rephrase_list) * 100.0,
            "Locality": np.mean(loc_scores) if loc_scores else 0,
            "Portability": np.mean(inst_scores),
            "Rule_Logic": np.mean(rule_scores)
        })

    print("\n" + "=" * 85)
    print(f"{'Rule_Folder':<15} | {'Succ':<8} | {'Gen':<8} | {'Loc':<8} | {'Port':<8} | {'Logic':<8}")
    print("-" * 85)

    averages = {k: [] for k in all_summaries[0].keys()}
    for i, s in enumerate(all_summaries):
        print(
            f"Rule_{i:<10} | {s['Edit_Succ']:<8.2f} | {s['Generality']:<8.2f} | "
            f"{s['Locality']:<8.2f} | {s['Portability']:<8.2f} | {s['Rule_Logic']:<8.2f}"
        )
        for k, v in s.items():
            averages[k].append(v)

    print("-" * 85)
    print(
        f"{'GRAND AVERAGE':<15} | {np.mean(averages['Edit_Succ']):<8.2f} | "
        f"{np.mean(averages['Generality']):<8.2f} | {np.mean(averages['Locality']):<8.2f} | "
        f"{np.mean(averages['Portability']):<8.2f} | {np.mean(averages['Rule_Logic']):<8.2f}"
    )
    print("=" * 85 + "\n")


# -------------------------
# 5) Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", type=str, default="")
    parser.add_argument("--hparams_dir", type=str, default="")
    parser.add_argument("--metrics_save_dir", type=str, default="")
    parser.add_argument("--rule_edit", default=True, type=str)
    parser.add_argument("--evaluation_type", default="", type=str)
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--base_data_dir", type=str, default="")
    parser.add_argument("--union_save_name", type=str, default="")
    args = parser.parse_args()

    # select hparams
    if args.editing_method == "IKE":
        editing_hparams = IKEHyperParams
    elif args.editing_method == "ICE":
        editing_hparams = IKEHyperParams
    elif args.editing_method == "MEMIT":
        editing_hparams = MEMITHyperParams
    elif args.editing_method == "ROME":
        editing_hparams = ROMEHyperParams
    elif args.editing_method == "LoRA":
        editing_hparams = LoRAHyperParams
    elif args.editing_method == "GRACE":
        editing_hparams = GraceHyperParams
    else:
        raise NotImplementedError(args.editing_method)

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    hparams.fp16 = True
    hparams.rule_edit = args.rule_edit
    hparams.evaluation_type = args.evaluation_type
    if args.api_key is not None:
        hparams.api_key = args.api_key

    base_data_dir = args.base_data_dir

    # 只取数字文件夹
    rule_folders = [
        f for f in os.listdir(base_data_dir)
        if os.path.isdir(os.path.join(base_data_dir, f)) and f.isdigit()
    ]
    rule_folders.sort(key=lambda x: int(x))

    # 统一读取全局 config.json
    global_config_path = os.path.join(base_data_dir, "config.json")
    global_stages = load_json(global_config_path)

    assert isinstance(global_stages, list) and len(global_stages) > 0
    for s in global_stages:
        assert "name" in s and "layers" in s, "Each stage must have keys: name, layers"

    all_rule_summaries = []

    for folder in rule_folders:
        print(f"\n{'=' * 20} Processing Rule: {folder} {'=' * 20}")
        current_dir = os.path.join(base_data_dir, folder)

        rule_output_path = os.path.join(args.metrics_save_dir, f"metrics_rule_{folder}.json")
        if os.path.exists(rule_output_path):
            print(f"[SKIP] Found existing metrics: {rule_output_path} -> skip rule {folder}")
            all_rule_summaries.append(rule_output_path)
            continue

        editor = BaseEditor.from_hparams(hparams)

        # 根据统一 config.json 自动构造当前规则的 stage
        stages = []
        for s in global_stages:
            stage_name = s["name"]

            if stage_name == "f_d":
                data_path = os.path.join(current_dir, "f_d.json")
            elif stage_name == "ins":
                data_path = os.path.join(current_dir, "ins.json")
            else:
                raise ValueError(f"Unknown stage name in global config: {stage_name}")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Missing stage data: {data_path}")

            stages.append({
                "name": stage_name,
                "data": data_path,
                "layers": s["layers"]
            })

        # union dataset = concat all stage datasets in given order
        union_dataset: List[Dict] = []
        for s in stages:
            ds = load_json(s["data"])
            if not isinstance(ds, list):
                raise ValueError(f"dataset must be a list, got {type(ds)} from {s['data']}")
            union_dataset.extend(ds)

        (
            u_prompts,
            u_subjects,
            u_ground_truths,
            u_target_news,
            u_rephrase_prompts,
            u_locality_inputs,
            u_portability_inputs,
        ) = build_inputs_from_dataset(union_dataset)

        union_requests = prepare_requests_compat(
            prompts=u_prompts,
            target_new=u_target_news,
            ground_truth=u_ground_truths,
            rephrase_prompts=u_rephrase_prompts,
            locality_inputs=u_locality_inputs,
            portability_inputs=u_portability_inputs,
            subjects=u_subjects,
        )

        print("[INFO] stages =", len(stages))
        for i, s in enumerate(stages):
            print("[INFO] stage", i, "name=", s["name"], "data=", s["data"], "layers=", s["layers"])

        # PRE cache on union
        print("[INFO] PRE evaluate on UNION (original model) ...")
        pre_cache = []
        for req in union_requests:
            pre = compute_edit_quality(
                editor.model,
                hparams.model_name,
                hparams,
                editor.tok,
                req,
                hparams.device,
                eval_metric="exact match",
                test_generation=False,
            )
            pre_cache.append(pre)

        # multi-stage cumulative batch edit
        print("[INFO] Start multi-stage cumulative BATCH editing ...")
        for stage_id, st in enumerate(stages):
            data_path = st["data"]
            layers = st["layers"]

            hparams.layers = layers
            if hasattr(hparams, "layer_selection"):
                hparams.layer_selection = "manual"

            stage_dataset = load_json(data_path)

            (
                prompts,
                subjects,
                ground_truths,
                target_news,
                rephrase_prompts,
                locality_inputs,
                portability_inputs,
            ) = build_inputs_from_dataset(stage_dataset)

            stage_requests = prepare_requests_compat(
                prompts=prompts,
                target_new=target_news,
                ground_truth=ground_truths,
                rephrase_prompts=rephrase_prompts,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                subjects=subjects,
            )

            print("--- Stage", stage_id, "---")
            print("name:", st["name"])
            print("data:", data_path)
            print("layers:", layers)
            print("requests:", len(stage_requests))

            batch_apply_one_stage(editor, hparams, stage_requests)

        # POST on union
        print("[INFO] POST evaluate on UNION (final edited model) ...")
        post_list = []
        for req in union_requests:
            post = compute_edit_quality(
                editor.model,
                hparams.model_name,
                hparams,
                editor.tok,
                req,
                hparams.device,
                eval_metric="exact match",
                test_generation=False,
            )
            post_list.append(post)

        # merge metrics + locality output->acc
        rule_metrics: List[Dict] = []
        for i, req in enumerate(union_requests):
            pre = pre_cache[i]
            post = post_list[i]

            if isinstance(post, dict) and "locality" in post and isinstance(post["locality"], dict):
                post_loc = post["locality"]
                new_loc = {}

                for loc_name in ["neighborhood", "distracting"]:
                    out_key = f"{loc_name}_output"
                    if out_key not in post_loc:
                        continue

                    preds = post_loc[out_key]
                    qs = req["locality"][loc_name]["prompt"]
                    gts = req["locality"][loc_name]["ground_truth"]

                    accs = []
                    for q, gt, pred in zip(qs, gts, preds):
                        accs.append(llm_judge(q, gt, pred, hparams.api_key))

                    new_loc[out_key] = preds
                    new_loc[f"{loc_name}_acc"] = accs

                post["locality"] = new_loc

                if isinstance(pre, dict):
                    pre.pop("locality", None)

            rule_metrics.append(
                {
                    "case_id": i,
                    "requested_rewrite": req,
                    "pre": pre,
                    "post": post,
                }
            )

        os.makedirs(args.metrics_save_dir, exist_ok=True)
        with open(rule_output_path, "w") as f:
            json.dump(rule_metrics, f, indent=2, ensure_ascii=False)

        all_rule_summaries.append(rule_output_path)
        print(f">>> 规则 {folder} 完成。指标摘要:")
        eval_summary(rule_output_path)

    calculate_final_grand_average(all_rule_summaries)


if __name__ == "__main__":
    main()