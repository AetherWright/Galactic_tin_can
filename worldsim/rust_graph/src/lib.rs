use std::collections::{BinaryHeap, HashMap};
use ordered_float::OrderedFloat;
use std::os::raw::{c_double, c_int};

#[repr(C)]
pub struct Graph {
    adj: HashMap<i32, Vec<(i32, f64)>>,
}

#[no_mangle]
pub extern "C" fn rg_create() -> *mut Graph {
    Box::into_raw(Box::new(Graph { adj: HashMap::new() }))
}

#[no_mangle]
pub extern "C" fn rg_free(ptr: *mut Graph) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

#[no_mangle]
pub extern "C" fn rg_add_node(ptr: *mut Graph, node: c_int) {
    if ptr.is_null() { return; }
    let g = unsafe { &mut *ptr };
    g.adj.entry(node).or_insert_with(Vec::new);
}

#[no_mangle]
pub extern "C" fn rg_add_edge(ptr: *mut Graph, a: c_int, b: c_int, w: c_double) {
    if ptr.is_null() { return; }
    let g = unsafe { &mut *ptr };
    g.adj.entry(a).or_insert_with(Vec::new).push((b, w));
    g.adj.entry(b).or_insert_with(Vec::new).push((a, w));
}

#[no_mangle]
pub extern "C" fn rg_shortest_paths(
    ptr: *const Graph,
    start: c_int,
    nodes: *const c_int,
    n: c_int,
    out: *mut c_double,
) {
    if ptr.is_null() { return; }
    let g = unsafe { &*ptr };
    let mut dist: HashMap<i32, f64> = HashMap::new();
    let mut heap: BinaryHeap<(OrderedFloat<f64>, i32)> = BinaryHeap::new();
    dist.insert(start, 0.0);
    heap.push((OrderedFloat(0.0), start));
    while let Some((OrderedFloat(d), node)) = heap.pop() {
        if let Some(neigh) = g.adj.get(&node) {
            for &(next, w) in neigh {
                let nd = d + w;
                if nd < *dist.get(&next).unwrap_or(&f64::INFINITY) {
                    dist.insert(next, nd);
                    heap.push((OrderedFloat(nd), next));
                }
            }
        }
    }
    let nodes_slice = unsafe { std::slice::from_raw_parts(nodes, n as usize) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, n as usize) };
    for (i, &n) in nodes_slice.iter().enumerate() {
        out_slice[i] = *dist.get(&n).unwrap_or(&f64::INFINITY);
    }
}

