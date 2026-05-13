use rust_events::{choose_event, update_table};
use serde_json::Value;
use std::io::{self, Read};
use std::env;

fn main() {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "help".to_string());
    let path = args.next().unwrap_or_else(|| "qtable.yml".to_string());
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let val: Value = if input.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&input).unwrap_or(Value::Null)
    };
    match mode.as_str() {
        "choose" => {
            let event = val.get("event").and_then(|v| v.as_str()).unwrap_or("");
            let opts = val.get("options").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
            let state: Vec<f64> = val
                .get("state")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
                .unwrap_or_default();
            match choose_event(&path, event, opts, &state) {
                Ok(idx) => println!("{}", idx),
                Err(_) => println!("0"),
            }
        }
        "update" => {
            let event = val.get("event").and_then(|v| v.as_str()).unwrap_or("");
            let action = val.get("action").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let reward = val.get("reward").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let state: Vec<f64> = val
                .get("state")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
                .unwrap_or_default();
            let next_state: Vec<f64> = val
                .get("new_state")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
                .unwrap_or_default();
            let _ = update_table(&path, event, &state, action, reward, &next_state);
            println!("ok");
        }
        "ping" => {
            println!("ok");
        }
        _ => {
            eprintln!("unknown mode");
        }
    }
}
