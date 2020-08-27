use std::collections::HashMap;
use std::collections::HashSet;

fn reverse_graph(rel: &[(usize, usize)], rel_rev: &mut [(usize, usize)]) {
    for (i, v) in rel.iter().enumerate() {
        rel_rev[i] = (v.1, v.0);
    }
}

fn adj_list(rel: &[(usize, usize)]) -> HashMap<usize, HashSet<usize>> {
    let mut rev_adj = HashMap::new();
    for i in rel.iter() {
        rev_adj.entry(i.0).or_insert(HashSet::new()).insert(i.1);
    }
    rev_adj
}

fn explore_post_order(
    cur: usize,
    adj: &HashMap<usize, HashSet<usize>>,
    order: &mut Vec<usize>,
    done: &mut HashSet<usize>,
) {
    if !done.contains(&cur) {
        done.insert(cur);
        if let Some(y) = adj.get(&cur) {
            for k in y.iter() {
                explore_post_order(*k, adj, order, done);
            }
        };
        order.push(cur);
    }
}
pub fn compute(num_nodes: usize, rel: &[(usize, usize)]) -> Vec<usize> {
    let mut out = vec![0; num_nodes];
    let adj = adj_list(&rel[..]);

    //get reverse of the post order of the dfs
    let mut visit_order = vec![];
    let mut done = HashSet::new();
    for i in 0..num_nodes {
        explore_post_order(i, &adj, &mut visit_order, &mut done);
    }
    visit_order.reverse();

    //get transpose graph
    let mut rel_rev = vec![(0usize, 0usize); rel.len()];
    reverse_graph(rel, &mut rel_rev[..]);
    let rev_adj = adj_list(&rel_rev[..]);

    //compute the ssc components
    let mut component_id = 0;
    let mut node_to_component = HashMap::new();
    for i in visit_order.iter() {
        if !node_to_component.contains_key(i) {
            component_id += 1;
            let mut queue = vec![i];
            loop {
                let n = match queue.last() {
                    Some(n) => *n,
                    _ => {
                        break;
                    }
                };
                let mut found = false;
                if !node_to_component.contains_key(n) {
                    node_to_component.insert(*n, component_id);
                    found = true;
                    if let Some(y) = rev_adj.get(n) {
                        for k in y.iter() {
                            queue.push(k);
                        }
                    };
                }
                if !found {
                    queue.pop();
                }
            }
        }
    }

    for (k, v) in node_to_component {
        out[k] = v;
    }
    out
}

#[test]
fn test_scc_kosaraju() {
    let num_nodes = 4usize;
    let rel = vec![(0, 1), (1, 2), (2, 0), (1, 3)];
    let out = compute(num_nodes, &rel[..]);
    println!("component map: {:?}", out);
    assert!(out[0] == out[1] && out[1] == out[2] && out[0] != out[3]);
}
