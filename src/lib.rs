use std::collections::HashMap;

use pyo3::prelude::*;
use rayon::prelude::*;

/// Returns a matrix of pairwise ChrF scores of shape batch_size x num_hypotheses x num_references
///
/// :param hypotheses: A list of lists of hypotheses of shape batch_size x num_hypotheses
/// :param references: A list of lists of references of shape batch_size x num_references
/// :param char_order: An integer indicating the maximum order of the character n-grams. Defaults to 6.
/// :param beta: A float indicating the beta parameter of the F-score. Defaults to 2.0.
/// :param remove_whitespace: If `True`, remove whitespace when extracting character n-grams. Defaults to `True`.
/// :param eps_smoothing: If `True`, add epsilon smoothing to the ChrF score. Defaults to `False`.
/// :return: A list of lists of lists of floats.
#[pyfunction]
#[pyo3(name = "pairwise_chrf")]
#[pyo3(signature = (
hypotheses,
references,
char_order = 6,
beta = 2.0,
remove_whitespace = true,
eps_smoothing = false
))]
fn pairwise_chrf_py(
    py: Python<'_>,
    hypotheses: Vec<Vec<String>>,
    references: Vec<Vec<String>>,
    char_order: usize,
    beta: f64,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    validate_batch(&hypotheses, &references)?;
    // The batch is already copied into owned Rust values and the worker threads
    // never touch Python, so hold no GIL while scoring: a large call would
    // otherwise block every other Python thread for its whole duration.
    Ok(py.detach(|| chrf_pairwise_batched(hypotheses, references, char_order, beta, remove_whitespace, eps_smoothing)))
}


/// Returns a matrix of fastChrF scores of shape batch_size x num_hypotheses
///
/// :param hypotheses: A list of lists of hypotheses of shape batch_size x num_hypotheses
/// :param references: A list of lists of references of shape batch_size x num_references
/// :param char_order: An integer indicating the maximum order of the character n-grams. Defaults to 6.
/// :param beta: A float indicating the beta parameter of the F-score. Defaults to 2.0.
/// :param remove_whitespace: If `True`, remove whitespace when extracting character n-grams. Defaults to `True`.
/// :param eps_smoothing: If `True`, add epsilon smoothing to the ChrF score. Defaults to `False`.
/// :return: A list of lists of lists of floats.
#[pyfunction]
#[pyo3(name = "aggregate_chrf")]
#[pyo3(signature = (
hypotheses,
references,
char_order = 6,
beta = 2.0,
remove_whitespace = true,
eps_smoothing = false
))]
fn aggregate_chrf_py(
    py: Python<'_>,
    hypotheses: Vec<Vec<String>>,
    references: Vec<Vec<String>>,
    char_order: usize,
    beta: f64,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> PyResult<Vec<Vec<f64>>> {
    validate_batch(&hypotheses, &references)?;
    // The batch is already copied into owned Rust values and the worker threads
    // never touch Python, so hold no GIL while scoring: a large call would
    // otherwise block every other Python thread for its whole duration.
    Ok(py.detach(|| chrf_aggregate_batched(hypotheses, references, char_order, beta, remove_whitespace, eps_smoothing)))
}


/// Checks the batch dimensions that both entry points rely on.
///
/// `chrf_*_batched` iterate over the hypothesis rows and index `references` with
/// the same offset, so a shorter `references` would panic and a longer one would
/// be silently truncated. Reject both here instead.
fn validate_batch(hypotheses: &[Vec<String>], references: &[Vec<String>]) -> PyResult<()> {
    if hypotheses.is_empty() || references.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hypotheses and references must be non-empty",
        ));
    }
    if hypotheses.len() != references.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hypotheses and references must have the same batch size, but got {} and {}",
            hypotheses.len(),
            references.len(),
        )));
    }
    Ok(())
}


fn chrf_pairwise_batched(
    hypotheses: Vec<Vec<String>>,
    references: Vec<Vec<String>>,
    char_order: usize,
    beta: f64,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> Vec<Vec<Vec<f64>>> {
    hypotheses
        .par_iter()
        .zip(references.par_iter())
        .map(|(hypothesis_row, reference_row)| {
            chrf_pairwise(
                hypothesis_row,
                reference_row,
                char_order,
                beta,
                remove_whitespace,
                eps_smoothing,
            )
        })
        .collect()
}


fn chrf_pairwise(
    hypotheses: &[String],
    references: &[String],
    char_order: usize,
    beta: f64,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> Vec<Vec<f64>> {
    let num_hypotheses = hypotheses.len();
    let num_references = references.len();
    let mut metric_scores = vec![vec![0.0; num_references]; num_hypotheses];
    let ngrams_per_hypothesis: Vec<Vec<HashMap<String, u32>>> = hypotheses.par_iter().map(|hypothesis| {
        extract_all_char_ngrams(hypothesis, char_order, remove_whitespace)
    }).collect();
    let ngrams_per_reference: Vec<Vec<HashMap<String, u32>>> = references.par_iter().map(|reference| {
        extract_all_char_ngrams(reference, char_order, remove_whitespace)
    }).collect();
    let eps = 1e-16;
    let factor = beta.powi(2);
    // Each hypothesis owns one disjoint output row, so score the rows in
    // parallel. Without this the whole hypothesis x reference loop ran on a
    // single thread and `pairwise_chrf` only used more than one core when the
    // batch dimension was greater than one.
    metric_scores
        .par_iter_mut()
        .zip(ngrams_per_hypothesis.par_iter())
        .for_each(|(row_scores, hyp_ngrams)| {
            for (score_out, ref_ngrams) in row_scores.iter_mut().zip(ngrams_per_reference.iter()) {
                let mut score = 0.0;
                let mut effective_order = 0;
                let mut avg_prec = 0.0;
                let mut avg_rec = 0.0;
                for n in 0..char_order {
                    let (n_hyp, n_ref, n_match) = get_match_statistics(&hyp_ngrams[n], &ref_ngrams[n]);
                    let prec = n_match as f64 / n_hyp as f64;
                    let rec = n_match as f64 / n_ref as f64;
                    let denom = factor * prec + rec;
                    score += ((1.0 + factor) * prec * rec / denom).max(eps);
                    if n_hyp > 0 && n_ref > 0 {
                        avg_prec += prec;
                        avg_rec += rec;
                        effective_order += 1;
                    }
                }
                if eps_smoothing {
                    *score_out = 100.0 * score / char_order as f64;
                    continue;
                }
                if effective_order == 0 {
                    avg_prec = 0.0;
                    avg_rec = 0.0;
                } else {
                    avg_prec /= effective_order as f64;
                    avg_rec /= effective_order as f64;
                }
                if avg_prec + avg_rec > 0.0 {
                    score = (1.0 + factor) * avg_prec * avg_rec;
                    score /= (factor * avg_prec) + avg_rec;
                    *score_out = 100.0 * score;
                }
            }
        });
    metric_scores
}


fn chrf_aggregate_batched(
    hypotheses: Vec<Vec<String>>,
    references: Vec<Vec<String>>,
    char_order: usize,
    beta: f64,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> Vec<Vec<f64>> {
    hypotheses
        .par_iter()
        .zip(references.par_iter())
        .map(|(hypothesis_row, reference_row)| {
            chrf_aggregate(
                hypothesis_row,
                reference_row,
                char_order,
                beta,
                remove_whitespace,
                eps_smoothing,
            )
        })
        .collect()
}


fn chrf_aggregate(
    hypotheses: &[String],
    references: &[String],
    char_order: usize,
    beta: f64,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> Vec<f64> {
    let num_references = references.len() as u32;

    // Extract ngrams for all references and sum up counts over all references
    let ngrams_for_all_references: Vec<HashMap<String, u32>> = references
        .iter()
        .map(|reference| extract_all_char_ngrams(reference, char_order, remove_whitespace))
        .fold(vec![HashMap::new(); char_order], |mut acc, ngrams| {
            for n in 0..char_order {
                for (key, value) in ngrams[n].iter() {
                    *acc[n].entry(key.to_string()).or_insert(0) += value;
                }
            }
            acc
    });

    let eps = 1e-16;
    let factor = beta.powi(2);
    hypotheses.par_iter().map(|hypothesis| {
        let mut score = 0.0;
        let mut effective_order = 0;
        let mut avg_prec = 0.0;
        let mut avg_rec = 0.0;
        // Extract hypothesis ngrams and multiply counts by the number of references
        let hyp_ngrams = extract_all_char_ngrams(hypothesis, char_order, remove_whitespace)
            .into_iter().map(|mut ngram_map| {
                for value in ngram_map.values_mut() {
                    *value *= num_references;
                }
                ngram_map
            }).collect::<Vec<HashMap<String, u32>>>();
        for n in 0..char_order {
            let (n_hyp, n_ref, n_match) = get_match_statistics(&hyp_ngrams[n], &ngrams_for_all_references[n]);
            let prec = n_match as f64 / n_hyp as f64;
            let rec = n_match as f64 / n_ref as f64;
            let denom = factor * prec + rec;
            score += ((1.0 + factor) * prec * rec / denom).max(eps);
            if n_hyp > 0 && n_ref > 0 {
                avg_prec += prec;
                avg_rec += rec;
                effective_order += 1;
            }
        }
        if eps_smoothing {
            return 100.0 * score / char_order as f64;
        }
        if effective_order == 0 {
            avg_prec = 0.0;
            avg_rec = 0.0;
        } else {
            avg_prec /= effective_order as f64;
            avg_rec /= effective_order as f64;
        }
        if avg_prec + avg_rec > 0.0 {
            score = (1.0 + factor) * avg_prec * avg_rec;
            score /= (factor * avg_prec) + avg_rec;
            100.0 * score
        } else {
            0.0
        }
    }).collect()
}


/// Corresponds to sacrebleu.metrics.helpers.extract_all_char_ngrams
/// Returns a list of counters of character n-grams across all orders up to max_order
fn extract_all_char_ngrams(
    line: &str,
    max_order: usize,
    remove_whitespace: bool
) -> Vec<HashMap<String, u32>> {
    let processed_line: Vec<char> = if remove_whitespace {
        line.split_whitespace().flat_map(|s| s.chars()).collect()
    } else {
        line.chars().collect()
    };
    let mut counters = Vec::new();
    for order in 1..=max_order {
        let mut counts: HashMap<String, u32> = HashMap::new();
        for ngram in processed_line.windows(order) {
            let ngram_str = ngram.iter().collect::<String>();
            *counts.entry(ngram_str).or_insert(0) += 1;
        }
        counters.push(counts);
    }
    counters
}


/// Corresponds to sacrebleu.CHRF._get_match_statistics
/// Returns the number of ngrams in the hypothesis, the number of ngrams in the reference, and the number of matches
fn get_match_statistics(
    hyp_ngrams: &HashMap<String, u32>,
    ref_ngrams: &HashMap<String, u32>,
) -> (u32, u32, u32) {
    let ref_total: u32 = ref_ngrams.values().sum();
    if ref_total == 0 {
        return (0, 0, 0);
    }
    let hyp_total: u32 = hyp_ngrams.values().sum();
    let intersection: u32 = hyp_ngrams.iter().filter_map(|(key, &hyp_count)| {
        ref_ngrams.get(key).map(|&ref_count| {
            std::cmp::min(hyp_count, ref_count)
        })
    }).sum();
    (hyp_total, ref_total, intersection)
}


#[pymodule]
fn fastchrf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pairwise_chrf_py, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_chrf_py, m)?)?;
    Ok(())
}
