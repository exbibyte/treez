use std::collections::BTreeSet;
use std::collections::HashSet;
use std::collections::VecDeque;
// use std::cmp;

struct Node {
    _index: i64,
    _parent: i64,
    _child_left: i64,
    _child_right: i64,
    // _num_children: usize,
    _segs: Vec<i32>,
    // _height: usize,
    _bound_l: i32, //todo: use generic types?
    _bound_r: i32,
}


impl Default for Node {
    fn default() -> Node {
        Node {
            _index: -1i64,
            _parent: -1i64,
            _child_left: -1i64,
            _child_right: -1i64,
            // _num_children: 0usize,
            _segs: vec![],
            // _height: 0usize,
            _bound_l: 0i32,
            _bound_r: 0i32,
        }
    }
}

pub struct TreeSeg {
    _intervals: Vec<Node>,
    _root_index: i64,
    // _height: usize,
}

impl TreeSeg {
    ///builds a tree using input segments
    pub fn init( input: &[(i32,i32,i32)] ) -> TreeSeg {
        let mut intervals = BTreeSet::new();
        for i in input {
            intervals.insert( i.0 );
            intervals.insert( i.1 );
        }
        let mut buf = vec![];
        let mut queue = VecDeque::new();
        
        for i in &intervals {
            let n_index = buf.len();
            let n = Node {
                _index: n_index as i64,
                _bound_l: *i,
                _bound_r: *i,
                ..Default::default()
            };
            // println!( "elementary intervals created: [{},{}]", n._bound_l, n._bound_r );
            buf.push( n );
            queue.push_back(n_index);
        }
        while queue.len() > 1usize {
            let drained : Vec<usize> = queue.drain(..).collect();
            let num_parents = drained.len() / 2 + drained.len() % 2;
            for i in 0..num_parents {
                let n_index = buf.len();
                if (i == num_parents-1) && ((drained.len() % 2) == 1) {
                    //odd one left, combine it with the last created node
                    let nr = i*2;
                    // println!("left: {}", (buf.len()-1) );
                    // println!("right: {}", drained[nr] );
                    let n = Node {
                        _index: n_index as i64,
                        _child_left: (buf.len()-1) as i64,
                        _child_right: drained[nr] as i64,
                        // _num_children: buf[buf.len()-1]._num_children + buf[drained[nr]]._num_children + 2,
                        // _height: cmp::max( buf[buf.len()-1]._height, buf[drained[nr]]._height ) + 1,
                        _bound_l: buf[buf.len()-1]._bound_l,
                        _bound_r: buf[drained[nr]]._bound_r,
                        ..Default::default()
                    };
                    // println!( "bound created: [{},{}]", n._bound_l, n._bound_r );
                    queue.pop_back(); //discard the previously queued parent index
                    buf.push( n );
                } else {   
                    let nl = i*2;
                    let nr = i*2+1;
                    // println!("left: {}", drained[nl] );
                    // println!("right: {}", drained[nr] );
                    buf[drained[nl]]._parent = n_index as i64;
                    buf[drained[nr]]._parent = n_index as i64;
                    let n = Node {
                        _index: n_index as i64,
                        _child_left: drained[nl] as i64,
                        _child_right: drained[nr] as i64,
                        // _num_children: buf[drained[nl]]._num_children + buf[drained[nr]]._num_children + 2,
                        // _height: cmp::max( buf[drained[nl]]._height, buf[drained[nr]]._height ) + 1,
                        _bound_l: buf[drained[nl]]._bound_l,
                        _bound_r: buf[drained[nr]]._bound_r,
                        ..Default::default()
                    };
                    // println!( "bound created: [{},{}]", n._bound_l, n._bound_r );
                    buf.push( n );
                }
                queue.push_back( n_index ); //queue the parent index
            }
        }
        
        let buf_size = buf.len();
        let mut t = TreeSeg {
            _intervals: buf,
            _root_index: if buf_size > 0 { (buf_size-1) as i64 } else { -1i64 },
        };
        
        //insert segments into the tree
        for i in input {
            let mut q = vec![];
            q.push( t._root_index );
            let left = i.0;
            let right = i.1;
            let id = i.2;
            while q.len() > 0 {
                let index = q.pop().unwrap();
                if index != -1 {
                    let n = index as usize;
                    if left <= t._intervals[n]._bound_l && right >= t._intervals[n]._bound_r {
                        t._intervals[n]._segs.push( id );
                    } else if left > t._intervals[n]._bound_r || right < t._intervals[n]._bound_l {
                        //do nothing
                    } else {
                        q.push( t._intervals[n]._child_left );
                        q.push( t._intervals[n]._child_right );
                    }
                }
            }
        }
        
        t
    }
    ///get total number of nodes in tree
    pub fn len_nodes( & self ) -> usize {
        self._intervals.len()
    }
    ///get a list of segments that is contained in the bound
    pub fn get_segs_from_bound( & self, bound: (i32,i32) ) -> Vec<i32> {
        let l = bound.0;
        let r = bound.1;
        let mut hs = HashSet::new();
        let mut q = vec![];
        if self._root_index >= 0 {
            q.push( self._root_index );
            while q.len() > 0 {
                let index = q.pop().unwrap();
                if index != -1 {
                    let n = index as usize;
                    if l > self._intervals[n]._bound_r || r < self._intervals[n]._bound_l {
                        //nothing
                    } else {
                        for i in &self._intervals[n]._segs {
                            hs.insert( i );
                        }
                        q.push( self._intervals[n]._child_left );
                        q.push( self._intervals[n]._child_right );
                    }
                }
            }
        }
        let ret = hs.drain().cloned().collect();
        ret
    }
}
