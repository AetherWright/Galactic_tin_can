use rust_utils::{
    ru_logistic_growth, ru_logistic_growth_batch, ru_distance,
    ru_polygon_area, ru_polygon_centroid,
};
use serde_json::Value;
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let line = line.trim();
        if line == "quit" { break; }
        if line.is_empty() { continue; }
        let val: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => { writeln!(stdout, "null").unwrap(); continue; }
        };
        let cmd = val.get("cmd").and_then(|v| v.as_str()).unwrap_or("");
        let out = match cmd {
            "ping" => serde_json::json!("ok"),
            "logistic" => {
                let n = val.get("n").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let r = val.get("r").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let k = val.get("k").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let a = val.get("approx").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                serde_json::json!(ru_logistic_growth(n, r, k, a))
            }
            "logistic_batch" => {
                let vals = val.get("vals").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let caps = val.get("caps").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let r = val.get("r").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let a = val.get("approx").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let v_vec: Vec<f64> = vals.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
                let c_vec: Vec<f64> = caps.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
                let len = v_vec.len();
                let mut out_buf = vec![0.0f64; len];
                ru_logistic_growth_batch(
                    v_vec.as_ptr(),
                    c_vec.as_ptr(),
                    len as i32,
                    r,
                    a,
                    out_buf.as_mut_ptr(),
                );
                serde_json::json!(out_buf)
            }
            "distance" => {
                let p1 = val.get("p1").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let p2 = val.get("p2").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let v1: Vec<f64> = p1.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
                let v2: Vec<f64> = p2.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
                let len = v1.len();
                let d = ru_distance(v1.as_ptr(), v2.as_ptr(), len as i32);
                serde_json::json!(d)
            }
            "area" => {
                let coords = val.get("coords").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let mut flat: Vec<f64> = Vec::with_capacity(coords.len()*2);
                for c in &coords {
                    if let Some(arr) = c.as_array() {
                        if arr.len() >= 2 {
                            flat.push(arr[0].as_f64().unwrap_or(0.0));
                            flat.push(arr[1].as_f64().unwrap_or(0.0));
                        }
                    }
                }
                let area = ru_polygon_area(flat.as_ptr(), coords.len() as i32);
                serde_json::json!(area)
            }
            "centroid" => {
                let coords = val.get("coords").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let mut flat: Vec<f64> = Vec::with_capacity(coords.len()*2);
                for c in &coords {
                    if let Some(arr) = c.as_array() {
                        if arr.len() >= 2 {
                            flat.push(arr[0].as_f64().unwrap_or(0.0));
                            flat.push(arr[1].as_f64().unwrap_or(0.0));
                        }
                    }
                }
                let mut out_x = 0.0f64;
                let mut out_y = 0.0f64;
                ru_polygon_centroid(
                    flat.as_ptr(),
                    coords.len() as i32,
                    &mut out_x,
                    &mut out_y,
                );
                serde_json::json!([out_x, out_y])
            }
            _ => serde_json::json!(null),
        };
        writeln!(stdout, "{}", out.to_string()).unwrap();
    }
}
