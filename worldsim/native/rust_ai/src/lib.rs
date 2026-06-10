use rand::Rng;
use std::collections::HashMap;
use std::ffi::{c_double, CStr};
use std::os::raw::{c_char, c_int};
use std::fs::File;
use std::io::{Read, Write};

#[repr(C)]
pub struct PerceptronHandle {
    weights: Vec<f64>,
    bias: f64,
}

#[no_mangle]
pub extern "C" fn rp_create(n_inputs: i32) -> *mut PerceptronHandle {
    let mut rng = rand::thread_rng();
    let weights: Vec<f64> = (0..n_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let bias = rng.gen_range(-1.0..1.0);
    Box::into_raw(Box::new(PerceptronHandle { weights, bias }))
}

#[no_mangle]
pub extern "C" fn rp_destroy(ptr: *mut PerceptronHandle) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.01 * x }
}

fn gelu(x: f64) -> f64 {
    let k = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x.powi(3))).tanh())
}

#[no_mangle]
pub extern "C" fn rp_predict_prob(ptr: *const PerceptronHandle, inputs: *const c_double) -> c_double {
    assert!(!ptr.is_null());
    let p = unsafe { &*ptr };
    let slice = unsafe { std::slice::from_raw_parts(inputs, p.weights.len()) };
    let mut sum = p.bias;
    for (w, i) in p.weights.iter().zip(slice) {
        sum += w * i;
    }
    gelu(sum)
}

#[no_mangle]
pub extern "C" fn rp_predict(ptr: *const PerceptronHandle, inputs: *const c_double) -> i32 {
    if rp_predict_prob(ptr, inputs) >= 0.5 { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn rp_train(ptr: *mut PerceptronHandle, inputs: *const c_double, label: i32, lr: c_double) {
    assert!(!ptr.is_null());
    let p = unsafe { &mut *ptr };
    let slice = unsafe { std::slice::from_raw_parts(inputs, p.weights.len()) };
    let out = rp_predict_prob(ptr, inputs);
    let err = label as f64 - out;
    for (w, i) in p.weights.iter_mut().zip(slice) {
        *w += lr * err * i;
    }
    p.bias += lr * err;
}

#[repr(C)]
pub struct QLearningHandle {
    n_inputs: usize,
    n_actions: usize,
    lr: f64,
    gamma: f64,
    epsilon: f64,
    eps_decay: f64,
    bins: i32,
    table: HashMap<String, Vec<f64>>,
}

fn key_from_state(state: &[f64], bins: i32) -> String {
    let mut k = String::new();
    for s in state.iter() {
        let v = (s * bins as f64) as i32;
        k.push_str(&format!("{},", v));
    }
    k
}

#[no_mangle]
pub extern "C" fn rql_create(n_inputs: c_int, n_actions: c_int, lr: c_double, gamma: c_double) -> *mut QLearningHandle {
    let h = QLearningHandle {
        n_inputs: n_inputs as usize,
        n_actions: n_actions as usize,
        lr,
        gamma,
        epsilon: 0.1,
        eps_decay: 0.995,
        bins: 10,
        table: HashMap::new(),
    };
    Box::into_raw(Box::new(h))
}

#[no_mangle]
pub extern "C" fn rql_destroy(ptr: *mut QLearningHandle) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

fn ensure_state<'a>(h: &'a mut QLearningHandle, key: String) -> &'a mut Vec<f64> {
    h.table.entry(key).or_insert_with(|| vec![0.0; h.n_actions])
}

#[no_mangle]
pub extern "C" fn rql_choose_action(ptr: *mut QLearningHandle, state: *const c_double) -> c_int {
    assert!(!ptr.is_null());
    let h = unsafe { &mut *ptr };
    let n_actions = h.n_actions;
    let slice = unsafe { std::slice::from_raw_parts(state, h.n_inputs) };
    let key = key_from_state(slice, h.bins);
    let best = {
        let qvals = ensure_state(h, key);
        let mut b = 0usize;
        for i in 1..n_actions {
            if qvals[i] > qvals[b] { b = i; }
        }
        b
    };
    let mut rng = rand::thread_rng();
    let act = if rng.gen::<f64>() < h.epsilon { rng.gen_range(0..n_actions) } else { best };
    h.epsilon *= h.eps_decay;
    if h.epsilon < 0.01 { h.epsilon = 0.01; }
    act as c_int
}

#[no_mangle]
pub extern "C" fn rql_predict(ptr: *mut QLearningHandle, state: *const c_double, out: *mut c_double) {
    assert!(!ptr.is_null());
    let h = unsafe { &mut *ptr };
    let n_actions = h.n_actions;
    let slice = unsafe { std::slice::from_raw_parts(state, h.n_inputs) };
    let key = key_from_state(slice, h.bins);
    let values = {
        let qvals = ensure_state(h, key);
        qvals.clone()
    };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, n_actions) };
    for i in 0..n_actions {
        out_slice[i] = values[i];
    }
}

#[no_mangle]
pub extern "C" fn rql_train(ptr: *mut QLearningHandle, state: *const c_double, action: c_int, reward: c_double, next_state: *const c_double) {
    assert!(!ptr.is_null());
    let h = unsafe { &mut *ptr };
    let n_actions = h.n_actions;
    let lr = h.lr;
    let gamma = h.gamma;
    let s_slice = unsafe { std::slice::from_raw_parts(state, h.n_inputs) };
    let ns_slice = unsafe { std::slice::from_raw_parts(next_state, h.n_inputs) };
    let key = key_from_state(s_slice, h.bins);
    let next_key = key_from_state(ns_slice, h.bins);
    let max_next = {
        let next_vals = ensure_state(h, next_key);
        let mut m = next_vals[0];
        for i in 1..n_actions { if next_vals[i] > m { m = next_vals[i]; } }
        m
    };
    {
        let qvals = ensure_state(h, key);
        let act = action as usize;
        let old = qvals[act];
        qvals[act] = old + lr * (reward + gamma * max_next - old);
    }
}


#[no_mangle]
pub extern "C" fn rql_save(ptr: *const QLearningHandle, path: *const std::os::raw::c_char) {
    if ptr.is_null() || path.is_null() { return; }
    let h = unsafe { &*ptr };
    let cstr = unsafe { CStr::from_ptr(path) };
    let path_str = match cstr.to_str() { Ok(s) => s, Err(_) => return };
    if let Ok(mut f) = File::create(path_str) {
        for (k, v) in &h.table {
            let _ = writeln!(f, "{}:", k);
            for val in v {
                let _ = writeln!(f, "  - {}", val);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn rql_load(ptr: *mut QLearningHandle, path: *const std::os::raw::c_char) {
    if ptr.is_null() || path.is_null() { return; }
    let h = unsafe { &mut *ptr };
    let cstr = unsafe { CStr::from_ptr(path) };
    let path_str = match cstr.to_str() { Ok(s) => s, Err(_) => return };
    let mut data = String::new();
    if File::open(path_str).and_then(|mut f| f.read_to_string(&mut data)).is_err() { return; }
    h.table.clear();
    let mut key: Option<String> = None;
    for line in data.lines() {
        if line.trim().is_empty() { continue; }
        if line.ends_with(':') {
            if let Some(k) = key.take() {
                h.table.entry(k).or_insert(Vec::new());
            }
            key = Some(line.trim_end_matches(':').to_string());
        } else if let Some(idx) = line.find('-') {
            if let Some(ref k) = key {
                let val: f64 = line[idx + 1..].trim().parse().unwrap_or(0.0);
                h.table.entry(k.clone()).or_default().push(val);
            }
        }
    }
    if let Some(k) = key.take() {
        h.table.entry(k).or_insert(Vec::new());
    }
}

