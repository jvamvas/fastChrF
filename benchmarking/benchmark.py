import timeit
from pathlib import Path
from typing import List

import numpy as np
from sacrebleu import CHRF as sacrebleu_Chrf

from fastchrf import aggregate_chrf, pairwise_chrf


samples_path = Path(__file__).parent / "samples.txt"
samples = samples_path.read_text().splitlines()

BATCH_SIZE = 1
NUM_REPEATS = 5
CHAR_ORDER = 6
BETA = 2
REMOVE_WHITESPACE = True
EPS_SMOOTHING = False

log_lines = []

for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    print(f"Number of hypotheses: {n}.")
    assert len(samples) >= n
    hypotheses = samples[:n]
    hypotheses = BATCH_SIZE * [hypotheses]
    references = hypotheses

    print("SacreBLEU:")

    sacrebleu_chrf = sacrebleu_Chrf(
        char_order=CHAR_ORDER,
        word_order=0,
        beta=BETA,
        whitespace=not REMOVE_WHITESPACE,
        eps_smoothing=EPS_SMOOTHING,
    )

    def sacrebleu_average_chrf(hypotheses_: List[List[str]], references_: List[List[str]]) -> List[List[float]]:
        batch_size = len(hypotheses_)
        num_hypotheses = len(hypotheses_[0])
        num_references = len(references_[0])
        pairwise_scores = np.zeros((batch_size, num_hypotheses, num_references))
        for i in range(batch_size):
            for j in range(num_hypotheses):
                for k in range(num_references):
                    pairwise_scores[i, j, k] = sacrebleu_chrf.sentence_score(
                        hypotheses_[i][j],
                        [references_[i][k]],
                    ).score
        average_scores = np.average(pairwise_scores, axis=2)
        return average_scores.tolist()

    sacrebleu_time = min(timeit.repeat(lambda: sacrebleu_average_chrf(hypotheses, references), repeat=NUM_REPEATS, number=1))
    print(sacrebleu_time)

    print("fastchrf.pairwise_chrf:")
    
    def fastchrf_average_chrf(hypotheses_: List[List[str]], references_: List[List[str]]) -> List[List[float]]:
        pairwise_scores = pairwise_chrf(hypotheses_, references_, char_order=CHAR_ORDER, beta=BETA,
                                        remove_whitespace=REMOVE_WHITESPACE, eps_smoothing=EPS_SMOOTHING)
        average_scores = np.average(pairwise_scores, axis=2)
        return average_scores.tolist()

    fastchrf_pairwise_time = min(timeit.repeat(lambda: fastchrf_average_chrf(hypotheses, references), repeat=NUM_REPEATS, number=1))
    print(fastchrf_pairwise_time)

    print("fastchrf.aggregate_chrf:")

    def fastchrf_aggregate_chrf(hypotheses_: List[List[str]], references_: List[List[str]]) -> List[List[float]]:
        return aggregate_chrf(hypotheses_, references_, char_order=CHAR_ORDER, beta=BETA,
                              remove_whitespace=REMOVE_WHITESPACE, eps_smoothing=EPS_SMOOTHING)

    fastchrf_aggregate_time = min(timeit.repeat(lambda: fastchrf_aggregate_chrf(hypotheses, references), repeat=NUM_REPEATS, number=1))
    print(fastchrf_aggregate_time)

    log_lines.append(f"{n}\t{sacrebleu_time}\t{fastchrf_pairwise_time}\t{fastchrf_aggregate_time}\n")

with open("results.tsv", "w") as f:
    f.write("n\tsacrebleu\tpairwise\taggregate\n")
    f.writelines(log_lines)
