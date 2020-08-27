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
    path_stack: &mut Vec<usize>,
    unassigned_stack: &mut Vec<usize>,
    order_gen: &mut usize,
    ordering: &mut HashMap<usize, usize>,
    assignment: &mut HashMap<usize, usize>,
    component: &mut usize,
) -> bool {
    if ordering.contains_key(&cur_node) {
        return false;
    }

    ordering.insert(cur_node, *order_gen);
    *order_gen += 1;

    unassigned_stack.push(cur_node);
    path_stack.push(cur_node);

    if let Some(x) = adj.get(&cur_node) {
        for i in x.iter() {
            if explore(
                *i,
                adj,
                path_stack,
                unassigned_stack,
                order_gen,
                ordering,
                assignment,
                component,
            ) {
            } else if !assignment.contains_key(i) {
                let o = ordering.get(i).unwrap();
                while unassigned_stack.len() > 0 {
                    let item = *unassigned_stack.last().unwrap();
                    let oo = ordering.get(&item).unwrap();
                    if oo <= o {
                        break;
                    }
                    unassigned_stack.pop();
                }
            }
        }
    }

    if cur_node == *unassigned_stack.last().unwrap() {
        while path_stack.len() > 0 {
            let item = path_stack.pop().unwrap();
            assignment.insert(item, *component);
            if item == cur_node {
                break;
            }
        }
        *component += 1;
        unassigned_stack.pop();
    }
    true
}

pub fn compute(num_nodes: usize, rel: &[(usize, usize)]) -> Vec<usize> {
    let mut out = vec![0; num_nodes];
    let adj = adj_list(&rel[..]);
    let mut path_stack = vec![];
    let mut unassigned_stack = vec![];
    let mut order_gen = 0;
    let mut ordering = HashMap::new();
    let mut assignment = HashMap::new();
    let mut component = 0;

    for i in 0..num_nodes {
        explore(
            i,
            &adj,
            &mut path_stack,
            &mut unassigned_stack,
            &mut order_gen,
            &mut ordering,
            &mut assignment,
            &mut component,
        );
    }

    for (k, v) in assignment {
        out[k] = v;
    }

    out
}

#[test]
fn test_scc_path() {
    let num_nodes = 4usize;
    let rel = vec![(0, 1), (1, 2), (2, 0), (1, 3)];
    let out = compute(num_nodes, &rel[..]);
    println!("component map: {:?}", out);
    assert!(out[0] == out[1] && out[1] == out[2] && out[0] != out[3]);
}
