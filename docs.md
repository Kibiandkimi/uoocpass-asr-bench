# 怎么在 Codespace 里跑

本机 `/home` 常年只剩几百 MB，medium / float32 / big-cn 这些配置根本放不下，
所以基准主要跑在 GitHub 上。两种方式：

## A. Codespaces（交互式，推荐调参用）

```bash
gh codespace create -R Kibiandkimi/uoocpass-asr-bench -m standardLinux32gb
gh codespace ssh -R Kibiandkimi/uoocpass-asr-bench
# 进去之后：
bash scripts/fetch_models.sh
python3 scripts/bench.py --list
python3 scripts/bench.py --engines fw-small-int8,vosk-big --repeats 3
```

同一仓库可以开多个 Codespace **并排跑不同候选**（每个独立机器，天然隔离）：

```bash
for e in fw-medium-int8 vosk-big wcpp-small; do
  gh codespace create -R Kibiandkimi/uoocpass-asr-bench -m standardLinux32gb -d
done
gh codespace list
```

用完记得删，否则配额会一直烧：

```bash
gh codespace delete -c <name> --force
```

## B. Actions（一键全跑，最省事）

```bash
gh workflow run bench.yml -f repeats=3
gh run watch
gh run download <run-id>          # 拿到所有 job 的 results.json
```

10 个候选是 10 个独立 job **并排**跑的，最后 `merge` job 合并成全局帕累托前沿。

## 指标口径

| 指标 | 定义 | 取向 |
| --- | --- | --- |
| **CER** | 字符错误率 = 编辑距离 / 参考长度。先剥标点和空格，再繁转简 | 越低越准 |
| **体积** | 包本体 + 模型权重的实际占用（不含系统共享库） | 越小越好 |
| **RTF** | 实时倍率 = 音频秒数 / 转写秒数 | 越大越快 |

优先级 **准 > 小 > 快**，但要的是**帕累托前沿**：列出那些「没有别的方案能在三项上同时不差」的点，再在它们之间按偏好挑。
