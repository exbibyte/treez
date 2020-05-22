extern crate num;

use std::collections::BTreeSet;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::Hash;
// use std::cmp;

struct Node<T, V>
where
    T: Ord + Default + Clone,
    V: Clone + Hash + Eq,
{
    _index: isize,
    _parent: isize,
    _child_left: isize,
    _child_right: isize,
    _segs: Vec<V>,
    _bound_l: T, //todo: use generic types?
    _bound_r: T,
}

impl<T, V> Default for Node<T, V>
where
    T: Ord + Default + Clone,
    V: Clone + Hash + Eq,
{
    fn default() -> Node<T, V> {
        Node {
            _index: -1isize,
            _parent: -1isize,
            _child_left: -1isize,
            _child_right: -1isize,
            _segs: vec![],
            _bound_l: Default::default(),
            _bound_r: Default::default(),
        }
    }
}

pub struct TreeSeg<T, V>
where
    T: Ord + Default + Clone,
    V: Clone + Hash + Eq,
{
    _intervals: Vec<Node<T, V>>,
    _root_index: isize,
}

impl<T, V> TreeSeg<T, V>
where
    T: Ord + Default + Clone,
    V: Clone + Hash + Eq,
{
    ///builds a tree using input segments
    pub fn init(input: &[(T, T, V)]) -> TreeSeg<T, V> {
        let mut intervals = BTreeSet::new();
        for i in input {
            intervals.insert(i.0.clone());
            intervals.insert(i.1.clone());
        }
        let mut buf = vec![];
        let mut queue = VecDeque::new();

        for i in &intervals {
            let n_index = buf.len();
            let n = Node {
                _index: n_index as isize,
                _bound_l: (*i).clone(),
                _bound_r: (*i).clone(),
                ..Default::default()
            };
            // println!( "elementary intervals created: [{},{}]", n._bound_l, n._bound_r );
            buf.push(n);
            queue.push_back(n_index);
        }
        while queue.len() > 1usize {
            let drained: Vec<usize> = queue.drain(..).collect();
            let num_parents = drained.len() / 2 + drained.len() % 2;
            for i in 0..num_parents {
                let n_index = buf.len();
                if (i == num_parents - 1) && ((drained.len() % 2) == 1) {
                    //odd one left, combine it with the last created node
                    let nr = i * 2;
                    // println!("left: {}", (buf.len()-1) );
                    // println!("right: {}", drained[nr] );
                    let n = Node {
                        _index: n_index as isize,
                        _child_left: (buf.len() - 1) as isize,
                        _child_right: drained[nr] as isize,
                        // _num_children: buf[buf.len()-1]._num_children + buf[drained[nr]]._num_children + 2,
                        // _height: cmp::max( buf[buf.len()-1]._height, buf[drained[nr]]._height ) + 1,
                        _bound_l: buf[buf.len() - 1]._bound_l.clone(),
                        _bound_r: buf[drained[nr]]._bound_r.clone(),
                        ..Default::default()
                    };
                    // println!( "bound created: [{},{}]", n._bound_l, n._bound_r );
                    queue.pop_back(); //discard the previously queued parent index
                    buf.push(n);
                } else {
                    let nl = i * 2;
                    let nr = i * 2 + 1;
                    // println!("left: {}", drained[nl] );
                    // println!("right: {}", drained[nr] );
                    buf[drained[nl]]._parent = n_index as isize;
                    buf[drained[nr]]._parent = n_index as isize;
                    let n = Node {
                        _index: n_index as isize,
                        _child_left: drained[nl] as isize,
                        _child_right: drained[nr] as isize,
                        // _num_children: buf[drained[nl]]._num_children + buf[drained[nr]]._num_children + 2,
                        // _height: cmp::max( buf[drained[nl]]._height, buf[drained[nr]]._height ) + 1,
                        _bound_l: buf[drained[nl]]._bound_l.clone(),
                        _bound_r: buf[drained[nr]]._bound_r.clone(),
                        ..Default::default()
                    };
                    // println!( "bound created: [{},{}]", n._bound_l, n._bound_r );
                    buf.push(n);
                }
                queue.push_back(n_index); //queue the parent index
            }
        }

        let buf_size = buf.len();
        let mut t = TreeSeg {
            _intervals: buf,
            _root_index: if buf_size > 0 {
                (buf_size - 1) as isize
            } else {
                -1isize
            },
        };

        //insert segments into the tree
        for i in input {
            let mut q = vec![];
            q.push(t._root_index);
            let left = i.0.clone();
            let right = i.1.clone();
            let id = i.2.clone();
            while q.len() > 0 {
                let index = q.pop().unwrap();
                if index != -1 {
                    let n = index as usize;
                    if left <= t._intervals[n]._bound_l && right >= t._intervals[n]._bound_r {
                        t._intervals[n]._segs.push(id.clone());
                    } else if left > t._intervals[n]._bound_r || right < t._intervals[n]._bound_l {
                        //do nothing
                    } else {
                        q.push(t._intervals[n]._child_left);
                        q.push(t._intervals[n]._child_right);
                    }
                }
            }
        }

        t
    }
    ///get total number of nodes in tree
    pub fn len_nodes(&self) -> usize {
        self._intervals.len()
    }
    ///get a list of segments that is contained in the bound
    pub fn get_segs_from_bound(&self, bound: (T, T)) -> Vec<V> {
        let l = bound.0;
        let r = bound.1;
        let mut hs = HashSet::new();
        let mut q = vec![];
        if self._root_index >= 0 {
            q.push(self._root_index);
            while q.len() > 0 {
                let index = q.pop().unwrap();
                if index != -1 {
                    let n = index as usize;
                    if l > self._intervals[n]._bound_r || r < self._intervals[n]._bound_l {
                        //nothing
                    } else {
                        for i in &self._intervals[n]._segs {
                            hs.insert(i);
                        }
                        q.push(self._intervals[n]._child_left);
                        q.push(self._intervals[n]._child_right);
                    }
                }
            }
        }
        let ret = hs.drain().cloned().collect();
        ret
    }
}

#[test]
fn test_seg() {
    let mut segments = vec![];
    for i in 0..10 {
        let n = (i * 5, 5 * i + 5, i);
        segments.push(n);
    }
    let t: TreeSeg<i32, i32> = TreeSeg::init(segments.as_slice());
    assert!(t.len_nodes() == 21);

    {
        let check: HashSet<_> = [1, 2, 3].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound((10, 15)).iter().cloned().collect();
        println!("query segs: {:?}", query_segs);
        assert!(check.intersection(&query_segs).count() == check.len());
    }
    {
        let check: HashSet<_> = [2, 3, 4].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound((15, 20)).iter().cloned().collect();
        println!("query segs: {:?}", query_segs);
        assert!(check.intersection(&query_segs).count() == check.len());
    }
    {
        //point query
        let check: HashSet<_> = [2].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound((12, 12)).iter().cloned().collect();
        println!("query segs: {:?}", query_segs);
        assert!(check.intersection(&query_segs).count() == check.len());
    }
    {
        //out of bound query
        let check: HashSet<_> = [].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound((-99, -5)).iter().cloned().collect();
        println!("query segs: {:?}", query_segs);
        assert!(check.intersection(&query_segs).count() == check.len());
    }
    {
        //out of bound query
        let check: HashSet<_> = [].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound((100, 200)).iter().cloned().collect();
        println!("query segs: {:?}", query_segs);
        assert!(check.intersection(&query_segs).count() == check.len());
    }
    {
        //collect all segs
        let check: HashSet<_> = (0..10).collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound((0, 50)).iter().cloned().collect();
        println!("query segs: {:?}", query_segs);
        assert!(check.intersection(&query_segs).count() == check.len());
    }
    {
        //collect all segs
        let check: HashSet<_> = (0..10).collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound((-9, 100)).iter().cloned().collect();
        println!("query segs: {:?}", query_segs);
        assert!(check.intersection(&query_segs).count() == check.len());
    }
}
