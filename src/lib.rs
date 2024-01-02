use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    hypotheses: Vec<Vec<String>>,
    references: Vec<Vec<String>>,
    char_order: usize,
    beta: f32,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> PyResult<Vec<Vec<Vec<f32>>>> {
    if hypotheses.len() == 0 || references.len() == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("hypotheses and references must be non-empty"));
    }
    Ok(chrf_pairwise_batched(hypotheses, references, char_order, beta, remove_whitespace, eps_smoothing))
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
    hypotheses: Vec<Vec<String>>,
    references: Vec<Vec<String>>,
    char_order: usize,
    beta: f32,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> PyResult<Vec<Vec<f32>>> {
    if hypotheses.len() == 0 || references.len() == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("hypotheses and references must be non-empty"));
    }
    Ok(chrf_aggregate_batched(hypotheses, references, char_order, beta, remove_whitespace, eps_smoothing))
}


fn chrf_pairwise_batched(
    hypotheses: Vec<Vec<String>>,
    references: Vec<Vec<String>>,
    char_order: usize,
    beta: f32,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> Vec<Vec<Vec<f32>>> {
    let batch_size = hypotheses.len();
    let num_hypotheses = hypotheses[0].len();
    let num_references = references[0].len();
    let metric_scores = Arc::new(Mutex::new(vec![vec![vec![0.0; num_references]; num_hypotheses]; batch_size]));
    hypotheses.par_iter().enumerate().for_each(|(i, row)| {
        let row_scores = chrf_pairwise(
            row.to_vec(),
            references[i].to_vec(),
            char_order,
            beta,
            remove_whitespace,
            eps_smoothing,
        );
        let mut scores = metric_scores.lock().unwrap();
        scores[i] = row_scores;
    });
    Arc::try_unwrap(metric_scores).unwrap().into_inner().unwrap()
}


fn chrf_pairwise(
    hypotheses: Vec<String>,
    references: Vec<String>,
    char_order: usize,
    beta: f32,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> Vec<Vec<f32>> {
    let num_hypotheses = hypotheses.len();
    let num_references = references.len();
    let mut metric_scores = vec![vec![0.0; num_references]; num_hypotheses];
    let ngrams_per_hypothesis: Vec<Vec<HashMap<String, u32>>> = hypotheses.iter().map(|hypothesis| {
        extract_all_char_ngrams(hypothesis, char_order, remove_whitespace)
    }).collect();
    let ngrams_per_reference: Vec<Vec<HashMap<String, u32>>> = references.iter().map(|reference| {
        extract_all_char_ngrams(reference, char_order, remove_whitespace)
    }).collect();
    let eps = 1e-16;
    let factor = beta.powi(2);
    for j in 0..num_hypotheses {
        for k in 0..num_references {
            let hyp_ngrams = &ngrams_per_hypothesis[j];
            let ref_ngrams = &ngrams_per_reference[k];
            let mut score = 0.0;
            let mut effective_order = 0;
            let mut avg_prec = 0.0;
            let mut avg_rec = 0.0;
            for n in 0..char_order {
                let (n_hyp, n_ref, n_match) = get_match_statistics(&hyp_ngrams[n], &ref_ngrams[n]);
                let prec = n_match as f32 / n_hyp as f32;
                let rec = n_match as f32 / n_ref as f32;
                let denom = factor * prec + rec;
                score += ((1.0 + factor) * prec * rec / denom).max(eps);
                if n_hyp > 0 && n_ref > 0 {
                    avg_prec += prec;
                    avg_rec += rec;
                    effective_order += 1;
                }
            }
            if eps_smoothing {
                metric_scores[j][k] = 100.0 * score / char_order as f32;
                continue;
            }
            if effective_order == 0 {
                avg_prec = 0.0;
                avg_rec = 0.0;
            } else {
                avg_prec /= effective_order as f32;
                avg_rec /= effective_order as f32;
            }
            if avg_prec + avg_rec > 0.0 {
                score = (1.0 + factor) * avg_prec * avg_rec;
                score /= (factor * avg_prec) + avg_rec;
                metric_scores[j][k] = 100.0 * score;
            }
        }
    }
    metric_scores
}


fn chrf_aggregate_batched(
    hypotheses: Vec<Vec<String>>,
    references: Vec<Vec<String>>,
    char_order: usize,
    beta: f32,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> Vec<Vec<f32>> {
    let batch_size = hypotheses.len();
    let num_hypotheses = hypotheses[0].len();
    let metric_scores = Arc::new(Mutex::new(vec![vec![0.0; num_hypotheses]; batch_size]));
    hypotheses.par_iter().enumerate().for_each(|(i, row)| {
        let row_scores = chrf_aggregate(
            row.to_vec(),
            references[i].to_vec(),
            char_order,
            beta,
            remove_whitespace,
            eps_smoothing,
        );
        let mut scores = metric_scores.lock().unwrap();
        scores[i] = row_scores;
    });
    Arc::try_unwrap(metric_scores).unwrap().into_inner().unwrap()
}


fn chrf_aggregate(
    hypotheses: Vec<String>,
    references: Vec<String>,
    char_order: usize,
    beta: f32,
    remove_whitespace: bool,
    eps_smoothing: bool,
) -> Vec<f32> {
    let num_hypotheses = hypotheses.len();
    let num_references = references.len() as u32;
    let metric_scores = Arc::new(Mutex::new(vec![0.0; num_hypotheses]));

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
    hypotheses.par_iter().enumerate().for_each(|(j, _)| {
        let mut score = 0.0;
        let mut effective_order = 0;
        let mut avg_prec = 0.0;
        let mut avg_rec = 0.0;
        // Extract hypothesis ngrams and multiply counts by the number of references
        let hyp_ngrams = extract_all_char_ngrams(&hypotheses[j], char_order, remove_whitespace)
            .into_iter().map(|mut ngram_map| {
                for value in ngram_map.values_mut() {
                    *value *= num_references;
                }
                ngram_map
            }).collect::<Vec<HashMap<String, u32>>>();
        for n in 0..char_order {
            let (n_hyp, n_ref, n_match) = get_match_statistics(&hyp_ngrams[n], &ngrams_for_all_references[n]);
            let prec = n_match as f32 / n_hyp as f32;
            let rec = n_match as f32 / n_ref as f32;
            let denom = factor * prec + rec;
            score += ((1.0 + factor) * prec * rec / denom).max(eps);
            if n_hyp > 0 && n_ref > 0 {
                avg_prec += prec;
                avg_rec += rec;
                effective_order += 1;
            }
        }
        if eps_smoothing {
            let mut scores = metric_scores.lock().unwrap();
            scores[j] = 100.0 * score / char_order as f32;
            return;
        }
        if effective_order == 0 {
            avg_prec = 0.0;
            avg_rec = 0.0;
        } else {
            avg_prec /= effective_order as f32;
            avg_rec /= effective_order as f32;
        }
        if avg_prec + avg_rec > 0.0 {
            score = (1.0 + factor) * avg_prec * avg_rec;
            score /= (factor * avg_prec) + avg_rec;
            let mut scores = metric_scores.lock().unwrap();
            scores[j] = 100.0 * score;
        }
    });
    Arc::try_unwrap(metric_scores).unwrap().into_inner().unwrap()
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
fn fastchrf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pairwise_chrf_py, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_chrf_py, m)?)?;
    Ok(())
}
