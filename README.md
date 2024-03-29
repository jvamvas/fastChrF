[![Main](https://github.com/jvamvas/fastchrf/workflows/unittest/badge.svg)](https://github.com/jvamvas/fastchrf/actions/workflows/unittest.yml)
[![PyPI](https://img.shields.io/pypi/v/fastchrf)](https://pypi.python.org/pypi/fastchrf/)

# fastChrF

Fast computation of sentence-level ChrF, motivated by Minimum Bayes Risk decoding.

* **ChrF** ([Popović, 2015](https://aclanthology.org/W15-3049/)) is a string similarity metric based on character overlap.
* **Minimum Bayes Risk (MBR) decoding** is a strategy for generating text from a language model that requires many pairwise comparisons of strings.

In addition to the standard ChrF metric, we provide a streamlined variant that is faster to compute if there are many hypotheses and references, which is especially useful for MBR decoding. The streamlined variant is described in the research paper ["Linear-time Minimum Bayes Risk Decoding with Reference Aggregation"](https://arxiv.org/abs/2402.04251).

## Installation
```bash
pip install fastchrf
```

## Usage
### Parallelized computation of pairwise ChrF scores
Use the `fastchrf.pairwise_chrf` function to compute the ChrF score between each hypothesis and each reference in a set of hypotheses and references:

```python
from fastchrf import pairwise_chrf

hypotheses = ["The cat sat on the mat.", "The cat sat on the hat."]
references = ["The cat sat on the mat.", "The fat cat sat on the mat.", "A cat sat on a mat."]
pairwise_scores = pairwise_chrf([hypotheses], [references])

print(np.array(pairwise_scores))
# [[[100.          74.6319046   55.77074432]
#   [ 79.65373993  57.15287399  50.72182846]]]
```

* `pairwise_chrf` works with a **batch dimension**, so pass a list of lists of hypotheses and a list of lists of references.
* For each row in the batch, the function calculates the segment-level ChrF score between each hypothesis and each reference.
* The output has shape `(batch_size, num_hypotheses, num_references)`.

`fastchrf.pairwise_chrf` compares each hypothesis to each reference.
This is slow when the number of hypotheses and references is large, as is the case in MBR decoding.

### Faster alternative: A streamlined ChrF variant for MBR
`fastchrf.aggregate_chrf` computes a streamlined variant of ChrF that is faster to compute:

```python
from fastchrf import aggregate_chrf

hypotheses = ["The cat sat on the mat.", "The cat sat on the hat."]
references = ["The cat sat on the mat.", "The fat cat sat on the mat.", "A cat sat on a mat."]
aggregate_scores = aggregate_chrf([hypotheses], [references])

print(np.array(aggregate_scores))
# [[78.56389618 63.3719368 ]]
```

* `aggregate_chrf` does not output individual scores for each reference. Instead, it outputs an **aggregate score across references**.
* The output has shape `(batch_size, num_hypotheses)`.
* The aggregate score is **not equal** to the average of the individual scores, nor is it equal to standard multi-reference ChrF. See our paper for a formal description.

## Function Signatures

```python
def pairwise_chrf(hypotheses: List[List[str]], references: List[List[str]], char_order: int=6, beta: float=2.0, remove_whitespace: bool=True, eps_smoothing: bool=False) -> List[List[List[float]]]:
    """
    Returns a matrix of pairwise ChrF scores of shape batch_size x num_hypotheses x num_references
    
    :param hypotheses: A list of lists of hypotheses of shape batch_size x num_hypotheses
    :param references: A list of lists of references of shape batch_size x num_references
    :param char_order: An integer indicating the maximum order of the character n-grams. Defaults to 6.
    :param beta: A float indicating the beta parameter of the F-score. Defaults to 2.0.
    :param remove_whitespace: If `True`, remove whitespace when extracting character n-grams. Defaults to `True`.
    :param eps_smoothing: If `True`, add epsilon smoothing to the ChrF score. Defaults to `False`.
    :return: A list of lists of lists of floats.
    """

def aggregate_chrf(hypotheses: List[List[str]], references: List[List[str]], char_order: int=6, beta: float=2.0, remove_whitespace: bool=True, eps_smoothing: bool=False) -> List[List[float]]:
    """
    Returns a matrix of fastChrF scores of shape batch_size x num_hypotheses

    :param hypotheses: A list of lists of hypotheses of shape batch_size x num_hypotheses
    :param references: A list of lists of references of shape batch_size x num_references
    :param char_order: An integer indicating the maximum order of the character n-grams. Defaults to 6.
    :param beta: A float indicating the beta parameter of the F-score. Defaults to 2.0.
    :param remove_whitespace: If `True`, remove whitespace when extracting character n-grams. Defaults to `True`.
    :param eps_smoothing: If `True`, add epsilon smoothing to the ChrF score. Defaults to `False`.
    :return: A list of lists of lists of floats.
    """
```

## Benchmarking

* Up to 1024 medium-size hypotheses/references in German
* Batch size 1
* 64-core CPU

|    n | [SacreBLEU](https://github.com/mjpost/sacrebleu) (ms) | `fastchrf.pairwise_chrf` (ms) | `fastchrf.aggregate_chrf` (ms) |
|-----:|---------------:|----------------------------:|-----------------------------:|
|    1 |        0.49 ms |                     0.27 ms |                      0.34 ms |
|    2 |        1.77 ms |                     0.51 ms |                      0.72 ms |
|    4 |        6.56 ms |                     1.28 ms |                      1.04 ms |
|    8 |       23.28 ms |                     2.88 ms |                      2.10 ms |
|   16 |       95.18 ms |                     8.92 ms |                      3.78 ms |
|   32 |      382.58 ms |                    30.33 ms |                      6.60 ms |
|   64 |     1497.29 ms |                   106.99 ms |                     11.39 ms |
|  128 |     6062.98 ms |                   409.86 ms |                     20.44 ms |
|  256 |    24072.80 ms |                  1691.64 ms |                     40.17 ms |
|  512 |    96216.99 ms |                  7465.06 ms |                     75.94 ms |
| 1024 |   383965.22 ms |                 32262.39 ms |                    144.78 ms |

<img src='benchmarking/results.png' width=500 alt="A line graph visualizing the result in the table">

## Citation
```bibtex
@misc{vamvas-sennrich-2024-linear,
      title={Linear-time Minimum Bayes Risk Decoding with Reference Aggregation},
      author={Jannis Vamvas and Rico Sennrich},
      year={2024},
      eprint={2402.04251},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

> [!NOTE]
> The [ACL 2023 Policy on AI Writing Assistance](https://2023.aclweb.org/blog/ACL-2023-policy/) requires authors to disclose the use of AI code assistants. For this package, we used GitHub Copilot and GPT-4 to port Python code to Rust. We then used unit tests to ensure that the generated functions are equivalent to the original Python code. In addition, we adapted the [original sacreBLEU tests](https://github.com/mjpost/sacrebleu/blob/821f4b40b94e550e4cec84416dfcb584789d7af8/test/test_chrf.py) to make sure that the output of the generated functions matches the output of the sacreBLEU implementation of ChrF.

> [!CAUTION]
> fastChrF is not intended to be used as an evaluation metric. For evaluating NLG systems with the ChrF metric, use the implementation provided by [sacreBLEU](https://github.com/mjpost/sacrebleu) instead.
