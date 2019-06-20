use std::cmp;
use std::mem;
use std::collections::HashMap;

extern crate rand;
use self::rand::Rng;

#[derive(Default,Clone,Debug)]
pub struct Treap< T > where T: Clone + Default {
    pub keys: Vec<f32>,
    pub priorities: Vec<f32>,
    pub vals: Vec<T>,
    pub link_child: Vec<(Option<usize>,Option<usize>)>,
    pub link_parent: Vec<usize>,
    pub freelist: Vec<usize>,
    pub instances: HashMap<usize,Option<usize>>,
}

pub enum SearchResult<T> where T: Clone + Default{
    Exact((usize,f32,T)),
    Nearest((usize,f32,T)),
    Empty,
}

pub enum ChildBranch {
    Left,
    Right,
    NotApplicable,
}

impl <T> Treap<T> where T: Clone + Default {

    ///helper function
    fn new_slot( & mut self ) -> usize {
        let idx = match self.freelist.pop() {
            Some(x) => { x },
            _ => {
                let l = self.keys.len();
                
                self.keys.push( Default::default() );
                self.priorities.push( Default::default() );
                self.vals.push( Default::default() );
                self.link_child.push( Default::default() );
                self.link_parent.push( Default::default() );
                
                l
            },
        };

        idx
    }
    
    ///helper function
    pub fn key( & self, a: usize ) -> f32 {
        assert!( a < self.keys.len() );
        self.keys[a]
    }

    ///helper function
    pub fn val( & self, a: usize ) -> T {
        assert!( a < self.vals.len() );
        self.vals[a].clone()
    }
    pub fn prio( & self, a: usize ) -> f32 {
        assert!( a < self.priorities.len() );
        self.priorities[a]
    }
    
    ///helper function
    fn gen_priority_random() -> f32 {
        let mut rng = rand::thread_rng();
        let r = rng.gen_range( -1e-12, 1e12 );
        r
    }
    
    pub fn init() -> Self {
        Treap::default()
    }

    pub fn new_instance( & mut self ) -> usize {
        let l = self.instances.len();
        self.instances.insert( l, None );
        l
    }
    
    pub fn search( & self, instance: usize, k: f32 ) -> SearchResult<T> {
        let r : usize = match self.instances.get(&instance) {
            Some(x) if x.is_some() => {
                x.unwrap()
            },
            _ => {
                return SearchResult::Empty
            },
        };
        let mut n = r;
        let ret = loop {
            assert!( n < self.keys.len() );
            let node_key = self.key(n);
            if k < node_key {
                if self.link_child[n].0.is_none() {
                    break SearchResult::Nearest( (n, node_key, self.val(n)) );
                } else {
                    n = self.link_child[n].0.unwrap();
                }
            } else if k > node_key {
                if self.link_child[n].1.is_none() {
                    break SearchResult::Nearest( (n, node_key, self.val(n)) );
                } else {
                    n = self.link_child[n].1.unwrap();
                }
            } else {
                break SearchResult::Exact( (n, k, self.val(n)) );
            }
        };
        ret
        // }  
    
    }

    ///return position of an item if it has the same key, priority is updated for an existing item
    pub fn insert_with_priority( & mut self, instance: usize, k: f32, p: f32, val: T ) -> (bool, usize) {
        match self.search( instance, k ) {
            SearchResult::Empty => {
                let idx = self.new_slot();
                self.keys[idx] = k;
                self.priorities[idx] = p;
                self.vals[idx] = val;
                self.link_parent[idx] = idx;
                self.instances.insert( instance, Some(idx) );
                
                (false, idx)
            },
            SearchResult::Exact((n_pos,n_key,n_val)) => {
                //item with key already exists
                self.vals[n_pos] = val;
                self.priorities[n_pos] = p;
                self.fixup_priority( instance, n_pos);
                
                (true, n_pos)
            },
            SearchResult::Nearest((n_pos,n_key,n_val)) => {
                
                let idx = self.new_slot();
                self.keys[idx] = k;
                self.priorities[idx] = p;
                self.vals[idx] = val;
                
                self.link_parent[idx] = n_pos;
                
                match self.key(n_pos) {
                    x if k < x => {
                        self.link_child[n_pos].0 = Some(idx);

                    },
                    x => {
                        self.link_child[n_pos].1 = Some(idx);
                    },                                 
                }

                self.fixup_priority( instance, idx);
                
                (false, idx)
            },
        }
    }
    
    pub fn insert( & mut self, instance: usize, k: f32, val: T ) -> (bool,usize) {
        let priority = Self::gen_priority_random();
        self.insert_with_priority( instance, k, priority, val )
    }

    fn child_branch( & self, n: usize, parent: usize ) -> ChildBranch {
        match self.link_child[parent].0 {
            Some(x) if x == n => {
                return ChildBranch::Left
            },
            _ => {},
        }
        match self.link_child[parent].1 {
            Some(x) if x == n => {
                return ChildBranch::Right
            },
            _ => {},
        }
        ChildBranch::NotApplicable
    }

    fn fixup_priority( & mut self, instance: usize, mut n: usize ){
        
        //fix priority by rotating up the tree
        
        let mut r : usize = self.instances.get(&instance).unwrap().expect("root non-existent");
        
        let mut par = self.link_parent[n];
        
        while self.prio(par) > self.prio(n) && r != n {
            
            match self.child_branch( n, par ) {
                ChildBranch::Left => {
                    self.rot_left( instance, n, par );     
                },
                ChildBranch::Right => {
                    self.rot_right( instance, n, par );
                },
                _ => { panic!("child link error"); },
            }
            par = self.link_parent[n];
            r = self.instances.get(&instance).unwrap().expect("root non-existent");
        }
    }

    fn link_left( & mut self, parent: usize, child: Option<usize> ){
        match child {
            Some(x) => {
                self.link_parent[x] = parent;
                self.link_child[parent].0 = Some(x);
            },
            _ => {
                self.link_child[parent].0 = None;
            },
        }
    }
    
    fn link_right( & mut self, parent: usize, child: Option<usize> ){
        match child {
            Some(x) => {
                self.link_parent[x] = parent;
                self.link_child[parent].1 = Some(x);
            },
            _ => {
                self.link_child[parent].1 = None;
            },
        }
    }
    
    fn rot_left( & mut self, instance: usize, n: usize, parent: usize ) {

        // before:
        //          pp
        //          |
        //          p
        //         / \
        //        n   c
        //       / \
        //      a   b
        //
        // after:
        //          pp
        //          |
        //          n
        //         / \
        //        a   p
        //           / \
        //          b   c
        //

        let parent_is_root = match self.instances.get( &instance ).expect("instance non-existent"){
            Some(x) => {
                if *x == n {
                    return
                } else if *x == parent {
                    true
                } else {
                    false
                }
            },
            _ => {panic!("root does not exist");},
        };

        if !parent_is_root {
            let pp = self.link_parent[parent];
            
            match self.child_branch( parent, pp ){
                ChildBranch::Left => {
                    self.link_left( pp, Some(n) );
                },
                ChildBranch::Right => {
                    self.link_right( pp, Some(n) );
                },
                _ => { panic!(); },
            }
        } else {
            //update to new root
            self.instances.insert( instance, Some(n) );
            
            self.link_parent[n] = n;
        }

        self.link_left( parent, self.link_child[n].1 );

        self.link_right( n, Some(parent) );
    }

    fn rot_right( & mut self, instance: usize, n: usize, parent: usize ) {

        // before:
        //          pp
        //          |
        //          p
        //         / \
        //        c   n
        //           / \
        //          a   b
        //
        // after:
        //          pp
        //          |
        //          n
        //         / \
        //        p   b
        //       / \
        //      c   a
        //

        let parent_is_root = match self.instances.get( &instance ).expect("instance non-existent"){
            Some(x) => {
                if *x == n {
                    return
                } else if *x == parent {
                    true
                } else {
                    false
                }
            },
            _ => {panic!("root does not exist");},
        };

        if !parent_is_root {
            let pp = self.link_parent[parent];
            
            match self.child_branch( parent, pp ){
                ChildBranch::Left => {
                    self.link_left( pp, Some(n) );
                },
                ChildBranch::Right => {
                    self.link_right( pp, Some(n) );
                },
                _ => { panic!(); },
            }
        } else {
            //update to new root
            self.instances.insert( instance, Some(n) );

            self.link_parent[n] = n;
        }

        self.link_right( parent, self.link_child[n].0 );

        self.link_left( n, Some(parent) );
    }

    pub fn successor( & self, idx: usize ) -> Option<usize> {

        let mut choice1 = None;
        match self.link_child[idx].1 {
            Some(x) => {
                let mut cur = x;
                while let Some(y) = self.link_child[cur].0 {
                    cur = y;
                }
                choice1 = Some(cur);
            },
            _ => {},
        }

        if choice1.is_some() {
            
            choice1
                
        } else {

            let mut cur = idx;
            let mut p = self.link_parent[cur];

            loop {
                match self.child_branch( cur, p ){
                    ChildBranch::Left => {
                        return Some(p)
                    },
                    ChildBranch::Right => {
                        cur = p;
                        p = self.link_parent[p];
                    },
                    _ => { return None },
                }
            }
        }
    }

    pub fn predecessor( & self, idx: usize ) -> Option<usize> {
        
        let mut choice1 = None;
        match self.link_child[idx].0 {
            Some(x) => {
                let mut cur = x;
                while let Some(y) = self.link_child[cur].1 {
                    cur = y;
                }
                choice1 = Some(cur);
            },
            _ => {},
        }

        if choice1.is_some() {
            
            choice1
                
        } else {

            
            let mut cur = idx;
            let mut p = self.link_parent[cur];
            
            loop {
                match self.child_branch( cur, p ){
                    ChildBranch::Right => {
                        return Some(p)
                    },
                    ChildBranch::Left => {
                        cur = p;
                        p = self.link_parent[p];
                    },
                    _ => { return None },
                }
            }
        }        
    }


    /// get indices of items with key in [k_start,k_end)
    pub fn query_range( & mut self, instance: usize, k_start: f32, k_end: f32 ) -> Vec<usize> {
        
        let n_start = match self.search( instance, k_start ){
            SearchResult::Exact( (idx,key,val) ) => {
                Some(idx)
            },
            SearchResult::Nearest( (idx,key,val)) => {
                if key < k_start {
                    let mut n = self.successor(idx);
                    while let Some(x) = n {
                        if self.key(x) > k_start {
                            break;
                        }
                        n = self.successor(x);
                    }
                    match n {
                        Some(x) if self.key(x) > k_start => {
                            Some(x)
                        }
                        _ => { None },
                    }
                } else {
                    Some(idx)
                }
            },
            _ => { None }
        };
        
        match n_start {
            Some(start) if self.key(start) < k_end => {
                
                let mut cur = start;
                
                let mut ret = vec![ cur ];

                while let Some(x) = self.successor( cur ) {
                    
                    if !( self.key(x) < k_end ) {
                        break;
                    }

                    cur = x;
                    ret.push( cur );
                }

                ret
            },
            _ => { vec![] }
        }
    }
    
    pub fn remove_index( & mut self, instance: usize, idx: usize ){

        loop {
            let rot_index = match self.link_child[idx] {
                (Some(l),Some(r)) => {
                    if self.key(l) < self.key(r) {
                        l
                    } else {
                        r
                    }
                },
                (Some(l),None) => { l },
                (None,Some(r)) => { r },
                _ => { break; },
            };
            match self.child_branch( rot_index, idx ){
                ChildBranch::Left => {
                    self.rot_left( instance, rot_index, idx );
                },
                ChildBranch::Right => {
                    self.rot_right( instance, rot_index, idx );
                },
                _ => { panic!(); },
            }
        }

        let p = self.link_parent[idx];
        if p != idx {
            match self.child_branch( idx, p ){
                ChildBranch::Left => {
                    self.link_child[p].0 = None;
                },
                ChildBranch::Right => {
                    self.link_child[p].1 = None;
                },
                _ => { panic!(); },
            }            
        }

        self.link_child[idx].0 = None;
        self.link_child[idx].1 = None;
        self.link_parent[idx] = idx;

        match self.instances.get(&instance).expect("instance non-existent"){
            Some(x) if *x == idx => {
                self.instances.insert( instance, None );
            },
            _ => {},
        }
        
        self.freelist.push(idx);
    }

    /// removes items with key in range of [k_start, k_end)
    pub fn remove_key_range( & mut self, instance: usize, k_start: f32, k_end: f32 ){
        self.query_range( instance, k_start, k_end ).iter()
            .for_each(|x| self.remove_index( instance, *x));
    }

    /// split given treap instance into two instances: a:[ x | x.key < k ], b:[ x | x.key >= k ]
    /// returns instance handles to split treaps (a,b)
    pub fn split( & mut self, instance: usize, k: f32 ) -> ( usize, usize ) {
        fn equal_f32( a: f32, b: f32 ) -> bool {
            if a - 1e-4 < b || a + 1e-4 > b {
                true
            } else {
                false
            }
        }

        let (existing, idx) = self.insert_with_priority( instance, k, -1e-20, Default::default() );
        
        assert!( equal_f32( self.key(self.instances.get(&instance).unwrap().expect("root non-existent") ), -1e-20) );
        
        assert_eq!( idx, self.instances.get(&instance).unwrap().expect("root non-existent") );

        let l = self.link_child[idx].0;
        match l {
            Some(x) => {
                self.link_parent[x] = x;
            },
            _ => {},
        }
        
        self.link_child[idx].0 = None;
        
        //update root
        *self.instances.get_mut(&instance).unwrap() = l;
        
        let r = if existing {
            Some(idx)
        } else {
            let temp = self.link_child[idx].1;
            self.link_child[idx].1 = None;
            temp
        };

        match r {
            Some(x) => {
                self.link_parent[x] = x;
            },
            _ => {},
        }

        //update root
        let new_inst = self.instances.len();
        self.instances.insert( new_inst, r );

        ( instance, new_inst )
    }

    /// merges 2 trees and returns handle to a combined tree
    pub fn merge( & mut self, inst_a: usize, inst_b: usize ) -> usize {
        
        let root_a = *self.instances.get(&inst_a).expect("instance non-existent");
        let root_b = *self.instances.get(&inst_b).expect("instance non-existent");

        match (root_a,root_b) {
            (Some(l), Some(r)) => {
                    
                let idx = self.new_slot();
                
                *self.instances.get_mut(&inst_a).unwrap() = Some(idx);

                self.link_left( idx, Some(l) );
                self.link_right( idx, Some(r) );
                
                self.priorities[idx] = 1e20;
                
                self.remove_index( inst_a, idx );

                *self.instances.get_mut(&inst_b).unwrap() = None;

                inst_a
            },
            (Some(l),_) => {
                inst_a
            },
            (_,Some(r)) => {
                inst_b
            },
            _ => { panic!(); },
        }
    }
}

#[test]
fn test_treap_search() {
    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }
    // test tree
    //            6
    //         3     8
    //      1    5  7  10
    //        2
    let mut t = Treap::init();

    let inst = t.new_instance();

    match t.search( inst, 5. ) {
        SearchResult::Empty => {},
        _ => { panic!("search failure"); },
    }
    
    t.keys = vec![1.,2.,3.,5.,6.,7.,8.,10.];
    *t.instances.get_mut(&inst).unwrap() = Some(4);
    t.vals = t.keys.iter().enumerate().map(|x| x.0 as i32).collect();
    t.link_child.resize(t.keys.len(), (None,None) );
    t.link_child[0].1 = Some(1);
    t.link_child[2].0 = Some(0);
    t.link_child[2].1 = Some(3);
    t.link_child[4].0 = Some(2);
    t.link_child[4].1 = Some(6);
    t.link_child[6].0 = Some(5);
    t.link_child[6].1 = Some(7);
    
    for i in t.keys.iter().cloned(){
        match t.search( inst, i ) {
            SearchResult::Exact(_) => {},
            _ => { panic!("search failure"); },
        }
    }

    match t.search( inst, 0. ) {
        SearchResult::Nearest((0, k, 0)) if equal_f32(k,1.) => {},
        _ => { panic!("search failure"); },
    }

    match t.search( inst, 4. ) {
        SearchResult::Nearest((3, k, 3)) if equal_f32(k,5.) => {},
        _ => { panic!("search failure"); },
    }

    match t.search( inst, 8. ) {
        SearchResult::Exact((6, k, 6)) if equal_f32(k,8.) => {},
        _ => { panic!("search failure"); },
    }

    match t.search( inst, 99. ) {
        SearchResult::Nearest((7, k, 7)) if equal_f32(k,10.) => {},
        _ => { panic!("search failure"); },
    }
}

#[test]
fn test_treap_insert_with_priority() {
    // test tree
    //            6
    //         3     8
    //      1   
    let mut t = Treap::init();
    let inst = t.new_instance();
    assert_eq!( None, *t.instances.get(&inst).unwrap() );
    assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );
    assert_eq!( true, t.instances.get(&inst).unwrap().is_some() );
    
    assert_eq!( t.insert_with_priority( inst, 3., 50., 3 ).0, true );
    assert_eq!( t.insert_with_priority( inst, 3., 50., 3 ).1, 0 );

    assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );
}

#[test]
fn test_treap_insert() {
    
    let mut t = Treap::init();
    let inst = t.new_instance();
    assert_eq!( None, *t.instances.get(&inst).unwrap() );
    assert_eq!( t.insert( inst, 3., 33 ).0, false );
    assert_eq!( true, t.instances.get(&inst).unwrap().is_some() );
    assert_eq!( t.insert( inst, 3., 3 ).0, true );
    assert_eq!( t.insert( inst, 1., 1 ).0, false );
    assert_eq!( t.insert( inst, 8., 8 ).0, false );
    assert_eq!( t.insert( inst, 6., 6 ).0, false );
}


#[test]
fn test_treap_successor() {
    // test tree
    //            6
    //         3     8
    //      1   
    let mut t = Treap::init();
    let inst = t.new_instance();
    
    assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 3., 50., 3 ).0, true );
    assert_eq!( t.insert_with_priority( inst, 3., 50., 3 ).1, 0 );

    assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

    let mut n = t.successor( 1 );
    assert_eq!( 0, n.unwrap() );
    n = t.successor( n.unwrap() );
    assert_eq!( 3, n.unwrap() );
    n = t.successor( n.unwrap() );
    assert_eq!( 2, n.unwrap() );
    assert!( t.successor( n.unwrap() ).is_none() );
}

#[test]
fn test_treap_predecessor() {

    // test tree
    //            6
    //         3     8
    //      1      7
    
    let mut t = Treap::init();
    let inst = t.new_instance();
    
    assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 7., 0., 7 ).0, false );

    let mut n = t.predecessor( 2 );
    assert_eq!( 4, n.unwrap() );
    n = t.predecessor( n.unwrap() );
    assert_eq!( 3, n.unwrap() );
    n = t.predecessor( n.unwrap() );
    assert_eq!( 0, n.unwrap() );
    n = t.predecessor( n.unwrap() );
    assert_eq!( 1, n.unwrap() );
    assert!( t.predecessor( n.unwrap() ).is_none() );
}

#[test]
fn test_treap_query_range() {
    
    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }
    
    // test tree
    //            6
    //         3     8
    //      1      7
    
    let mut t = Treap::init();
    let inst = t.new_instance();
    
    assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 7., 0., 7 ).0, false );

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), 5 );
        let expected = [1.,3.,6.,7.,8.].to_vec();
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }
    {
        let ret = t.query_range( inst, 3., 8. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [3.,6.,7.].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }
    {
        let ret = t.query_range( inst, 8.5, 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }
    {
        let ret = t.query_range( inst, -10., 0. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }
    {
        let ret = t.query_range( inst, -10., 6. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [1.,3.].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }
    {
        let ret = t.query_range( inst, 7.99, 8.1 ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [8.].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }
}

#[test]
fn test_treap_remove(){

    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }
    
    // test tree
    //            6
    //         3     8
    //      1      7
    
    let mut t = Treap::init();
    let inst = t.new_instance();
    
    assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 7., 0., 7 ).0, false );

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), 5 );
        let expected = [1.,3.,6.,7.,8.].to_vec();
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }


    t.remove_index(inst, 1);

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [3.,6.,7.,8.].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    t.remove_index(inst, 0);

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [6.,7.,8.].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    t.remove_index(inst, 3);

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [7.,8.].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    t.remove_index(inst, 2);

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [7.].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
        assert_eq!( t.key(t.instances.get(&inst).unwrap().unwrap()), 7. );
    }

    t.remove_index(inst, 4);

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
        assert!( t.instances.get(&inst).unwrap().is_none() );
    }
    dbg!( &t );
}

#[test]
fn test_treap_remove_key_range(){
    
    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }
    
    // test tree
    //            6
    //         3     8
    //      1      7
    
    let mut t = Treap::init();
    let inst = t.new_instance();
    
    assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

    assert_eq!( t.insert_with_priority( inst, 7., 0., 7 ).0, false );

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), 5 );
        let expected = [1.,3.,6.,7.,8.].to_vec();
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }
    
    t.remove_key_range( inst, 2.5, 6.5 );

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [1.,7.,8.].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    t.remove_key_range( inst, 7.5, 10. );

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [1.,7.].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    t.remove_key_range( inst, 0., 10. );

    {
        let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        let expected = [].to_vec();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
        assert!( t.instances.get(&inst).unwrap().is_none() );
    }
}

#[test]
fn test_treap_insert_remove_loop(){
    
    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }
    
    // test tree
    //            6
    //         3     8
    //      1      7
    
    let mut t = Treap::init();
    
    let inst = t.new_instance();

    let expected = vec![1.,3.,6.,7.,8.];

    expected.iter().for_each(|x| {t.insert( inst, *x,*x as i32);} );

    {
        let ret = t.query_range( inst, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    use std::collections::HashSet;
    let mut hs = HashSet::new();
    expected.iter().for_each(|x| {hs.insert( *x as i32 );} );

    while hs.len() != 0 {
        let remain = hs.iter().cloned().collect::<Vec<_>>();
        let mut rng = rand::thread_rng();
        let select = rng.gen_range(0, remain.len());
        let key = remain[select];

        let k_start = key as f32 - 1e-3;
        let k_end = key as f32 + 1e-3;
        
        t.remove_key_range( inst, k_start, k_end );
        
        {
            let ret = t.query_range( inst, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
            assert_eq!( ret.len(), remain.len()-1 );
            
            ret.iter().for_each(|x| {assert!(remain.iter().any(|y| equal_f32(*y as f32,*x) ));} );
        }

        hs.remove(&key);
        if hs.len() > 0 {
            assert!( t.instances.get(&inst).unwrap().is_some() );
        }
    }
    
    assert!( t.instances.get(&inst).unwrap().is_none() );
}

#[test]
fn test_treap_split1(){
    
    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }
    
    // test tree
    //            6
    //         3     8
    //      1      7
    
    let mut t = Treap::init();
    
    let inst = t.new_instance();

    let expected = vec![1.,3.,6.,7.,8.];

    expected.iter().for_each(|x| {t.insert( inst, *x,*x as i32);} );

    {
        let ret = t.query_range( inst, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    let (inst1, inst2) = t.split( inst, 6.5 );

    {
        let ret = t.query_range( inst1, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), 3 );
        let check = expected.iter().take(3).zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    {
        let ret = t.query_range( inst2, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), 2 );
        let check = expected.iter().skip(3).zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    let (inst3, inst4) = t.split( inst1, 3. );

    {
        let ret = t.query_range( inst3, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), 1 );
        let check = expected.iter().take(1).zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    {
        let ret = t.query_range( inst4, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), 2 );
        let check = expected.iter().skip(1).zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }
}

#[test]
fn test_treap_split2(){
    
    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }
    
    // test tree
    //            6
    //         3     8
    //      1      7
    
    let mut t = Treap::init();
    
    let inst = t.new_instance();

    let expected = vec![1.,3.,6.,7.,8.];

    expected.iter().for_each(|x| {t.insert( inst, *x,*x as i32);} );

    {
        let ret = t.query_range( inst, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), expected.len() );
        let check = expected.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    let (inst1, inst2) = t.split( inst, 3. );

    {
        let ret = t.query_range( inst1, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), 1 );
        let check = expected.iter().take(1).zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }

    {
        let ret = t.query_range( inst2, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), 4 );
        let check = expected.iter().skip(1).zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
    }    
}

#[test]
fn test_treap_merge(){
    
    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }
    
    // test tree 1 vals: [1,3,4,11]

    // test tree 2 vals: [12,13,17,18]
    
    let mut t = Treap::init();
    
    let inst1 = t.new_instance();
    let inst2 = t.new_instance();
    
    let expected1 = vec![1,3,4,11];
    let expected2 = vec![12,13,17,18];

    expected1.iter().for_each(|x| {t.insert( inst1, *x as f32,*x as i32);} );
    expected2.iter().for_each(|x| {t.insert( inst2, *x as f32,*x as i32);} );

    {
        let ret = t.query_range( inst1, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), expected1.len() );
        let check = expected1.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a as f32, *b ) );} );
    }

    {
        let ret = t.query_range( inst2, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), expected2.len() );
        let check = expected2.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a as f32, *b ) );} );
    }

    let inst3 = t.merge( inst1, inst2 );

    assert_eq!( inst3, inst1 );
    assert!( t.instances.get(&inst2).unwrap().is_none() );
    
    let expected_merged = expected1.iter().cloned().chain(expected2.iter().cloned()).collect::<Vec<_>>();
    {
        let ret = t.query_range( inst3, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), expected_merged.len() );
        let check = expected_merged.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a as f32, *b ) );} );
        
    }
}

#[test]
fn test_treap_split_merge(){
    
    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }
    
    // test tree vals: [1,3,4,11,12,13,17,18]
    
    let mut t = Treap::init();
    
    let inst1 = t.new_instance();
    
    let expected1 = vec![1,17,11,13,18,3,4,12];

    expected1.iter().for_each(|x| {t.insert( inst1, *x as f32,*x as i32);} );

    {
        let ret = t.query_range( inst1, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), expected1.len() );
        let check = expected1.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( *a as f32, *b ) );} );
    }

    let (inst2, inst3) = t.split( inst1, 9. );

    let mut split_expected1 = expected1.iter().cloned().filter(|x| (*x as f32) < 9. ).collect::<Vec<_>>();
    split_expected1.sort();

    let mut split_expected2 = expected1.iter().cloned().filter(|x| (*x as f32) > 9. ).collect::<Vec<_>>();
    split_expected2.sort();
    
    {
        let ret = t.query_range( inst2, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), split_expected1.len() );
        let check = split_expected1.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( (*a as f32), *b ) );} );
    }

    {
        let ret = t.query_range( inst3, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), split_expected2.len() );
        let check = split_expected2.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( (*a as f32), *b ) );} );
    }
    
    let inst4 = t.merge( inst2, inst3 );
    
    let mut expected_merged = expected1.clone();
    expected_merged.sort();
    
    {
        let ret = t.query_range( inst4, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
        assert_eq!( ret.len(), expected_merged.len() );
        let check = expected_merged.iter().zip( ret.iter() );
        check.for_each(|(a,b)| {assert!( equal_f32( (*a as f32), *b ) );} );
    }
}
