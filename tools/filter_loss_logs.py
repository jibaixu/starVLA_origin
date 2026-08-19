import argparse
import re


def filter_loss_logs(input_path, output_path):
    # 用于匹配日志起始行的正则：例如 "08/16 [00:20:16] INFO     | >> Step 2700, Loss: {"
    start_pattern = re.compile(r".*INFO\s+\|\s+>>\s+Step\s+\d+.*Loss:\s*\{")

    saved_lines = []
    recording = False
    brace_count = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            # 如果当前不在记录状态，检查是否遇到了 Loss 日志的开头
            if not recording:
                if start_pattern.search(line):
                    recording = True
                    saved_lines = [line.strip()]
                    # 计算当前行中大括号的配对情况（防止单行闭合的情况）
                    brace_count += line.count("{") - line.count("}")
                    if brace_count == 0:
                        f_out.write(" ".join(saved_lines) + "\n")
                        recording = False
            else:
                # 处于记录状态中，持续收集后续行直到大括号闭合
                saved_lines.append(line.strip())
                brace_count += line.count("{") - line.count("}")

                # 当大括号完全闭合时，说明一条完整的 loss 记录收集完毕
                if brace_count <= 0:
                    # 将多行合并为紧凑的一行写入新文件（也可以选择保持原样写入）
                    f_out.write(" ".join(saved_lines) + "\n")
                    recording = False
                    brace_count = 0

    print(f"过滤完成！Loss 信息已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter loss logs from training log files."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="train.log",
        help="Path to the original training log file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="loss_filtered.log",
        help="Path to save the filtered loss log file.",
    )
    args = parser.parse_args()

    filter_loss_logs(args.input, args.output)
