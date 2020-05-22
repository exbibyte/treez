use std::collections::HashMap;
use std::collections::HashSet;

pub fn reverse_graph(rel: &[(usize, usize)], rel_rev: &mut [(usize, usize)]) {
    for (i, v) in rel.iter().enumerate() {
        rel_rev[i] = (v.1, v.0);
    }
}

pub fn adj_list(rel: &[(usize, usize)]) -> HashMap<usize, HashSet<usize>> {
    let mut rev_adj = HashMap::new();
    for i in rel.iter() {
        let exists = rev_adj.contains_key(&i.0);
        if !exists {
            let mut new_set = HashSet::new();
            new_set.insert(i.1);
            rev_adj.insert(i.0, new_set);
        } else {
            match rev_adj.get_mut(&i.0) {
                Some(x) => {
                    x.insert(i.1);
                }
                None => {
                    // panic!("unexpected result");
                }
            };
        }
    }
    rev_adj
}

pub fn visit_post_order(num_nodes: usize, adj: &HashMap<usize, HashSet<usize>>) -> Vec<usize> {
    let mut visited = vec![false; num_nodes];
    let mut queue = vec![];
    let mut visit_order = vec![];
    for i in 0..num_nodes {
        if visited[i] == false {
            visited[i] = true;
            queue.push(i);
        }
        loop {
            let n = match queue.last() {
                Some(n) => *n,
                _ => {
                    break;
                }
            };
            let mut found = false;
            if let Some(y) = adj.get(&n) {
                for k in y.iter() {
                    if visited[*k] == false {
                        visited[*k] = true;
                        queue.push(*k);
                        found = true;
                    }
                }
            };
            if found {
                continue;
            }
            visit_order.push(n);
            queue.pop();
        }
    }
    visit_order
}

pub fn compute(num_nodes: usize, rel: &[(usize, usize)]) -> Vec<usize> {
    let mut out = vec![0; num_nodes];
    //computes strongly connected components using Kosaraju's algorithm
    let adj = adj_list(&rel[..]);
    // println!( "adj: {:?}", adj );

    //get reverse of the post order of the dfs
    let visit_order = {
        let mut a = visit_post_order(num_nodes, &adj);
        a.reverse();
        a
    };
    // println!( "visit order: {:?}", visit_order );

    //get transpose graph
    let mut rel_rev = vec![(0usize, 0usize); rel.len()];
    reverse_graph(rel, &mut rel_rev[..]);
    // println!( "reverse rel: {:?}", rel_rev );
    let rev_adj = adj_list(&rel_rev[..]);
    // println!( "reverse adj lists: {:?}", rev_adj );

    //compute the ssc components
    let mut component_id = 0;
    let mut node_to_component = HashMap::new();
    for i in visit_order.iter() {
        // println!( "visit: {}", i );
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

    //todo: add more test cases
}
