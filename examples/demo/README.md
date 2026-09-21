# ⚠️ 这是合成数据，不是真实测量结果

本目录中的文件由 **`latency_probe.py` 的 `dummy` 模式**生成，用于演示工具的输出格式，
**不代表任何真实硬件或模型的性能**。

生成方式：

```bash
python latency_probe.py --runs 50 --warmup 5 --dummy-ms 8 --jitter 0.45 \
       --out examples/demo --label demo-synthetic
python plot_hist.py examples/demo/latency.csv --bins 16
```

`dummy` 模式让每次调用的目标耗时在 `8 ms × (1 ± 0.45)` 之间均匀随机抖动，
纯粹是为了让测量链路和绘图链路都有真实形态的数据可处理。

---

## 为什么放这个目录

因为这个仓库目前**只有工具、没有成果**。我不想用一张来路不明的图冒充实验数据，
但也需要让人一眼看出工具确实能跑通、能出图。

所以这里放的是**明确标注为合成**的示例——它的作用是证明管线通，不是证明性能。

---

## 真实测量应该怎么产生

把 `--model dummy` 换成你自己的模型插件：

```python
# my_model.py
def step():
    return model.infer(observation)
```

```bash
python latency_probe.py --runs 50 --model plugin:my_model:step --out results --label pi0.5-jetson
python plot_hist.py results/latency.csv
```

**`results/` 目录已在 `.gitignore` 中，真实数据默认不入库。**
等我拿到真实数据后，会先确认它可复现、硬件与配置写清楚，再决定是否放进来。

---

## 本目录文件

| 文件 | 说明 |
|---|---|
| `latency.csv` | 50 次合成调用的原始延迟数据 |
| `latency_hist.svg` | 由上面的 CSV 生成的延迟分布直方图 |
