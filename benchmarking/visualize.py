import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

with open(Path(__file__).parent / "results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    data = list(reader)

hypotheses = [int(row["n"]) for row in data]
sacreBLEU = 100 * np.array([float(row["sacrebleu"]) for row in data])
fastchrf_pairwise = 100 * np.array([float(row["pairwise"]) for row in data])
fastchrf_aggregate = 100 * np.array([float(row["aggregate"]) for row in data])

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 6))

plt.plot(hypotheses, sacreBLEU, label='SacreBLEU', marker='o')
plt.plot(hypotheses, fastchrf_pairwise, label='fastchrf.pairwise_chrf', marker='s')
plt.plot(hypotheses, fastchrf_aggregate, label='fastchrf.aggregate_chrf', marker='^')

plt.xlabel('Number of Hypotheses (log scale)')
plt.ylabel('Time in milliseconds (log scale)')
plt.xscale('log')
plt.yscale('log')
plt.xticks(hypotheses, labels=[f"{h}" for h in hypotheses])
plt.yticks([0.1, 1, 10, 100, 1000, 10000], labels=[f"{t}" for t in [0.1, 1, 10, 100, 1000, 10000]])

plt.legend()

plt.savefig(Path(__file__).parent / "results.png", dpi=60)
