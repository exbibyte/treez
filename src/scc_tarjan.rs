use std::collections::{HashMap, HashSet};

fn adj_list(rel: &[(usize, usize)]) -> HashMap<usize, HashSet<usize>> {
    let mut rev_adj = HashMap::new();
    for i in rel.iter() {
        rev_adj.entry(i.0).or_insert(HashSet::new()).insert(i.1);
    }
    rev_adj
}

fn explore(
    cur_node: usize,
    adj: &HashMap<usize, HashSet<usize>>,
    order_gen: &mut usize,
    ordering: &mut HashMap<usize, usize>,
    ordering_low: &mut HashMap<usize, usize>,
    assignment: &mut HashMap<usize, usize>,
    component: &mut usize,
    stack: &mut Vec<usize>,
    inprogress: &mut HashSet<usize>,
) {
    debug_assert_eq!(false, ordering.contains_key(&cur_node));
    debug_assert_eq!(false, ordering_low.contains_key(&cur_node));
    
    ordering.insert(cur_node, *order_gen);
    ordering_low.insert(cur_node, *order_gen);
    *order_gen += 1;
    stack.push(cur_node);
    inprogress.insert(cur_node);

    if let Some(x) = adj.get(&cur_node) {
        for i in x.iter() {
            if !ordering.contains_key(i) {
                explore(
                    *i,
                    adj,
                    order_gen,
                    ordering,
                    ordering_low,
                    assignment,
                    component,
                    stack,
                    inprogress,
                );
                let oo = ordering_low[i];
                let o = ordering_low[&cur_node];
                *ordering_low.get_mut(&cur_node).unwrap() = std::cmp::min(o, oo);
            } else if inprogress.contains(i) {
                let o = ordering_low[&cur_node];
                *ordering_low.get_mut(&cur_node).unwrap() = std::cmp::min(o, ordering[i]);
            }
        }
    }

    if ordering_low[&cur_node] == ordering[&cur_node] {
        //root of the scc found
        loop {
            let item = stack.pop().unwrap();
            inprogress.remove(&item);
            assignment.insert(item, *component);
            if item == cur_node {
                break;
            }
        }
        *component += 1;
    }
}

pub fn compute(num_nodes: usize, rel: &[(usize, usize)]) -> Vec<usize> {
    let mut out = vec![0; num_nodes];
    let adj = adj_list(&rel[..]);

    let mut order_gen = 0;
    let mut ordering = HashMap::new();
    let mut ordering_low = HashMap::new();
    let mut assignment = HashMap::new();
    let mut component = 0;
    let mut inprogress = HashSet::new();
    for i in 0..num_nodes {
        if !assignment.contains_key(&i) {
            let mut stack = vec![i];
            explore(
                i,
                &adj,
                &mut order_gen,
                &mut ordering,
                &mut ordering_low,
                &mut assignment,
                &mut component,
                &mut stack,
                &mut inprogress,
            );
            component += 1;
        }
    }

    for (k, v) in assignment {
        out[k] = v;
    }

    out
}

#[test]
fn test_scc_tarjan() {
    let num_nodes = 4usize;
    let rel = vec![(0, 1), (1, 2), (2, 0), (1, 3)];
    let out = compute(num_nodes, &rel[..]);
    println!("component map: {:?}", out);
    assert!(out[0] == out[1] && out[1] == out[2] && out[0] != out[3]);
}
