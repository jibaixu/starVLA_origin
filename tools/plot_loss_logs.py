import argparse
import ast
import re
import matplotlib.pyplot as plt


def plot_loss_logs(log_path, keys_to_plot, output_image_path):
    steps = []
    data = {key: [] for key in keys_to_plot}

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配类似 ">> Step 2700, Loss: { ... }" 的模式（支持跨行）
    pattern = re.compile(r">> Step\s+(\d+),\s+Loss:\s*(\{.*?\})", re.DOTALL)
    matches = pattern.findall(content)

    print(f"[-] 正则匹配到的记录数: {len(matches)}")

    found_keys_sample = set()
    valid_record_count = 0

    for step_str, dict_str in matches:
        step = int(step_str)
        try:
            # 1. 核心修复：移除日志库注入的源码位置标记（例如 "train_starvla.py:477"）
            cleaned_str = re.sub(r"[a-zA-Z0-9_]+\.py:\d+\s*", "", dict_str)
            # 2. 将换行符和多余空白压缩为单空格
            cleaned_str = re.sub(r"\s+", " ", cleaned_str)

            # 安全解析字典
            loss_dict = ast.literal_eval(cleaned_str)
            found_keys_sample.update(loss_dict.keys())
        except Exception as e:
            continue

        # 检查所选的 key 是否全部存在于当前记录中
        if all(k in loss_dict for k in keys_to_plot):
            valid_record_count += 1
            steps.append(step)
            for k in keys_to_plot:
                data[k].append(float(loss_dict[k]))

    print(f"[-] 成功匹配到指定 keys 的记录数: {valid_record_count}")

    if not steps:
        print("\n[警告] 未找到包含指定 keys 的有效记录！")
        print(
            "当前日志中实际包含的所有 Keys 示例（供参考）:",
            list(found_keys_sample)[:30],
        )
        print("请检查你输入的 --keys 参数拼写是否正确。")
        return

    # 开始绘图
    plt.figure(figsize=(10, 6), dpi=300)

    for k in keys_to_plot:
        plt.plot(steps, data[k], label=k, linewidth=1.5)

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Training Metrics Visualization", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_image_path)
    print(f"[成功] 可视化图表已成功保存至: {output_image_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot specific metrics from filtered training logs."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="loss_filtered.log",
        help="Path to the filtered loss log file.",
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="+",
        default=["gradient_cosine/spatial_vs_action", "gradient_cosine/future_vs_action", "gradient_cosine/plan_vs_action"],
        help="List of keys to plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="loss_curve.png",
        help="Path to save the output visualization image.",
    )
    args = parser.parse_args()

    plot_loss_logs(args.input, args.keys, args.output)
