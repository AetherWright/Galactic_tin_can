use std::os::raw::{c_double, c_int};

#[no_mangle]
pub extern "C" fn ru_logistic_growth(n: c_double, r: c_double, k: c_double, approximate: c_int) -> c_double {
    if approximate != 0 {
        n + r * n
    } else {
        n + r * n * (1.0 - n / k)
    }
}

#[no_mangle]
pub extern "C" fn ru_distance(p1: *const c_double, p2: *const c_double, dim: c_int) -> c_double {
    if p1.is_null() || p2.is_null() {
        return 0.0;
    }
    let slice1 = unsafe { std::slice::from_raw_parts(p1, dim as usize) };
    let slice2 = unsafe { std::slice::from_raw_parts(p2, dim as usize) };
    let mut sum = 0.0;
    for i in 0..(dim as usize) {
        let d = slice1[i] - slice2[i];
        sum += d * d;
    }
    sum.sqrt()
}

#[no_mangle]
pub extern "C" fn ru_polygon_area(coords: *const c_double, n: c_int) -> c_double {
    if coords.is_null() || n < 3 {
        return 0.0;
    }
    let slice = unsafe { std::slice::from_raw_parts(coords, n as usize * 2) };
    let mut area = 0.0;
    for i in 0..(n as usize) {
        let xi = slice[2 * i];
        let yi = slice[2 * i + 1];
        let j = (i + 1) % (n as usize);
        let xj = slice[2 * j];
        let yj = slice[2 * j + 1];
        area += xi * yj - xj * yi;
    }
    area.abs() / 2.0
}

#[no_mangle]
pub extern "C" fn ru_polygon_centroid(coords: *const c_double, n: c_int, out_x: *mut c_double, out_y: *mut c_double) {
    if coords.is_null() || out_x.is_null() || out_y.is_null() || n == 0 {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts(coords, n as usize * 2) };
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut factor = 0.0;
    for i in 0..(n as usize) {
        let xi = slice[2 * i];
        let yi = slice[2 * i + 1];
        let j = (i + 1) % (n as usize);
        let xj = slice[2 * j];
        let yj = slice[2 * j + 1];
        let cross = xi * yj - xj * yi;
        cx += (xi + xj) * cross;
        cy += (yi + yj) * cross;
        factor += cross;
    }
    factor *= 3.0;
    if factor == 0.0 {
        unsafe {
            *out_x = 0.0;
            *out_y = 0.0;
        }
        return;
    }
    unsafe {
        *out_x = cx / factor;
        *out_y = cy / factor;
    }
}

#[no_mangle]
pub extern "C" fn ru_logistic_growth_batch(
    values: *const c_double,
    caps: *const c_double,
    n: c_int,
    r: c_double,
    approximate: c_int,
    out: *mut c_double,
) {
    if values.is_null() || caps.is_null() || out.is_null() || n <= 0 {
        return;
    }
    let len = n as usize;
    let vals = unsafe { std::slice::from_raw_parts(values, len) };
    let caps_slice = unsafe { std::slice::from_raw_parts(caps, len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, len) };
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let chunk = (len + workers - 1) / workers;
    let mut handles = Vec::new();
    let mut ranges = Vec::new();
    for start in (0..len).step_by(chunk) {
        let end = usize::min(start + chunk, len);
        let vals_chunk = vals[start..end].to_vec();
        let caps_chunk = caps_slice[start..end].to_vec();
        ranges.push((start, end));
        handles.push(std::thread::spawn(move || {
            let mut out_chunk = Vec::with_capacity(vals_chunk.len());
            for (n_val, k_val) in vals_chunk.iter().zip(&caps_chunk) {
                let res = if approximate != 0 {
                    n_val + r * n_val
                } else {
                    n_val + r * n_val * (1.0 - n_val / k_val)
                };
                out_chunk.push(res);
            }
            out_chunk
        }));
    }
    for ((start, end), h) in ranges.into_iter().zip(handles) {
        if let Ok(chunk_out) = h.join() {
            out_slice[start..end].copy_from_slice(&chunk_out);
        }
    }
}

