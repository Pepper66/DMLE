import json
import os
import glob


def calculate_metrics(folder_path):
    all_rewrite_acc = []
    all_rephrase_acc = []
    all_locality_acc = []
    all_portability_acc = []
    all_rule_acc = []

    # 1. 获取并排序文件列表 (按数字顺序排序)
    file_pattern = os.path.join(folder_path, "metrics_rule_*.json")
    files = glob.glob(file_pattern)

    # 根据文件名中的数字进行排序，避免 1, 10, 2 这种乱序
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

    if not files:
        print(f"错误：在路径 '{folder_path}' 下未发现文件。")
        return

    print(
        f"{'文件名':<25} | {'Rewrite':<10} | {'Rephrase':<10} | {'Locality':<10} | {'Portability':<12} | {'RuleUnd'}")
    print("-" * 95)

    for file_path in files:
        file_name = os.path.basename(file_path)

        # 每个文件内部的临时统计
        file_rw, file_rp, file_loc, file_port, file_rule = [], [], [], [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                continue

            for case in data:
                post = case.get("post", {})

                # 计算 Rewrite
                rw = post.get("rewrite_acc", [])
                if rw:
                    val = sum(rw) / len(rw)
                    file_rw.append(val)
                    all_rewrite_acc.append(val)

                # 计算 Rephrase
                rp = post.get("rephrase_acc", [])
                if rp:
                    val = sum(rp) / len(rp)
                    file_rp.append(val)
                    all_rephrase_acc.append(val)

                # 计算 Locality (neighborhood + distracting)
                loc = post.get("locality", {})
                loc_list = loc.get("neighborhood_acc", []) + loc.get("distracting_acc", [])
                if loc_list:
                    val = sum(loc_list) / len(loc_list)
                    file_loc.append(val)
                    all_locality_acc.append(val)

                # 计算 Portability (Instance + Rule)
                port = post.get("portability", {})
                port_list = port.get("Instance_acc", []) + port.get("Rule_acc", [])
                if port_list:
                    val = sum(port_list) / len(port_list)
                    file_port.append(val)
                    all_portability_acc.append(val)

                rule_acc = port.get("Rule_acc", [])
                if rule_acc:
                    val = sum(rule_acc) / len(rule_acc)
                    file_rule.append(val)
                    all_rule_acc.append(val)

        # 计算该文件(3个case)的平均指标
        avg_f = lambda x: sum(x) / len(x) if x else 0.0
        f_rw = avg_f(file_rw)
        f_rp = avg_f(file_rp)
        f_loc = avg_f(file_loc)
        f_port = avg_f(file_port)
        f_rule = avg_f(file_rule)
        f_total = (f_rw + f_rp + f_loc + f_port) / 4

        # 打印当前文件的行
        print(f"{file_name:<25} | {f_rw:<10.2f} | {f_rp:<10.2f} | {f_loc:<10.2f} | {f_port:<12.2f} | {f_rule:<12.2f}")

    # 3. 输出全局汇总
    def get_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    avg_rw = get_avg(all_rewrite_acc)
    avg_rp = get_avg(all_rephrase_acc)
    avg_loc = get_avg(all_locality_acc)
    avg_port = get_avg(all_portability_acc)
    avg_rule = get_avg(all_rule_acc)
    #total_score = (avg_rw + avg_rp + avg_loc + avg_port) / 4

    print("-" * 95)
    print(
        f"{'所有文件汇总平均值':<25} | {avg_rw:<10.4f} | {avg_rp:<10.4f} | {avg_loc:<10.4f} | {avg_port:<12.4f} | {avg_rule:<12.4f}")
    print("=" * 95)


# 使用时替换为你的文件夹路径
calculate_metrics('')