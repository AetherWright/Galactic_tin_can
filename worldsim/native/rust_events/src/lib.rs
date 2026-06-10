use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use rand::Rng;

#[derive(Default, Serialize, Deserialize)]
pub struct QTable(pub HashMap<String, HashMap<String, Vec<f64>>>);

fn load_qtable<P: AsRef<Path>>(path: P) -> io::Result<QTable> {
    if let Ok(mut f) = File::open(&path) {
        let mut data = String::new();
        f.read_to_string(&mut data)?;
        let table: QTable = serde_yaml::from_str(&data).unwrap_or_default();
        Ok(table)
    } else {
        Ok(QTable::default())
    }
}

fn save_qtable<P: AsRef<Path>>(path: P, table: &QTable) -> io::Result<()> {
    let data = serde_yaml::to_string(table).unwrap_or_default();
    fs::create_dir_all(path.as_ref().parent().unwrap_or(Path::new(".")))?;
    fs::write(path, data)
}

fn state_key(state: &[f64]) -> String {
    state
        .iter()
        .map(|v| ((v * 5.0).floor() / 5.0).to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub fn choose_event(
    path: &str,
    event: &str,
    options_len: usize,
    state: &[f64],
) -> io::Result<usize> {
    let mut table = load_qtable(path)?;
    let event_q = table.0.entry(event.to_string()).or_default();
    let key = state_key(state);
    let qvals = event_q.entry(key.clone()).or_insert_with(|| vec![0.0; options_len]);
    if qvals.len() != options_len {
        qvals.resize(options_len, 0.0);
    }
    let mut best = 0usize;
    for i in 1..options_len {
        if qvals[i] > qvals[best] {
            best = i;
        }
    }
    if qvals.iter().all(|&v| v == qvals[0]) {
        let mut rng = rand::thread_rng();
        best = rng.gen_range(0..options_len);
    }
    save_qtable(path, &table)?;
    Ok(best)
}

pub fn update_table(
    path: &str,
    event: &str,
    state: &[f64],
    action: usize,
    reward: f64,
    new_state: &[f64],
) -> io::Result<()> {
    let mut table = load_qtable(path)?;
    let event_q = table.0.entry(event.to_string()).or_default();
    let key = state_key(state);
    let nkey = state_key(new_state);
    let next_vals = event_q.get(&nkey).cloned();
    let qvals = event_q.entry(key).or_insert_with(|| vec![0.0; action + 1]);
    if qvals.len() <= action {
        qvals.resize(action + 1, 0.0);
    }
    let next_vec = next_vals.unwrap_or_else(|| vec![0.0; qvals.len()]);
    let max_next = next_vec.into_iter().fold(0.0_f64, |a, b| a.max(b));
    let lr = 0.1;
    let gamma = 0.9;
    qvals[action] += lr * (reward + gamma * max_next - qvals[action]);
    save_qtable(path, &table)
}
