use std::cmp;
use std::mem;
use std::collections::{HashSet,HashMap};
use std::rc::{Rc,Weak};
use std::ops::{Deref,DerefMut};
use std::cell::{RefCell,RefMut};
// use std::borrow::{Borrow,BorrowMut};
extern crate rand;
use self::rand::Rng;

extern crate chrono;
use self::chrono::prelude::*;

#[derive(Default,Clone,Debug)]
pub struct Node<T> where T: Clone + Default {
    pub key: f32,
    pub prio: f32,
    pub val: T,
    pub parent: NodePtrWk<T>,
    pub children: (Option<NodePtr<T>>, Option<NodePtr<T>>),
    pub invalid: bool,
}

#[derive(Default,Clone,Debug)]
pub struct NodePtr<T>(pub Rc<RefCell<Node<T>>>) where T: Clone + Default;

impl<T> From<Node<T>> for NodePtr<T> where T: Clone + Default {
    fn from( n: Node<T> ) -> Self {
        Self(Rc::new(RefCell::new(n)))
    }
}

#[derive(Default,Clone,Debug)]
pub struct NodePtrWk<T>(pub Weak<RefCell<Node<T>>>) where T: Clone + Default;

impl<T> From<&NodePtr<T>> for NodePtrWk<T> where T: Clone + Default {
    fn from( n: &NodePtr<T> ) -> Self {
        Self(Rc::downgrade(&n.0))
    }
}

// impl<T> Deref for NodePtr<T> where T: Clone + Default {
//     type Target = Rc<RefCell<Node<T>>>;

//     fn deref<'a>(&'a self) -> &'a Self::Target {
//         &self.0
//     }
// }

// pub trait Borrow<Borrowed> where Borrowed: ?Sized {
//     fn borrow(&self) -> &Borrowed;
// }

// pub trait BorrowMut<Borrowed>: Borrow<Borrowed> where Borrowed: ?Sized {
//     fn borrow_mut(&mut self) -> &mut Borrowed;
// }

// impl<T> Borrow<Node<T>> for NodePtr<T> where T: Clone + Default {
//     fn borrow(&self) -> &Node<T> {
//         self.0.borrow()
//     }
// }

// impl<T> BorrowMut<Node<T>> for NodePtr<T> where T: Clone + Default {
//     fn borrow_mut(&mut self) -> &mut Node<T> {
//         & mut self.0.deref().borrow_mut()
//     }
// }

pub enum SearchResult<T> where T: Clone + Default{
    Exact( NodePtr<T> ),
    Nearest( NodePtr<T> ),
    Empty,
}

pub enum ChildBranch {
    Left,
    Right,
    NotApplicable,
}

impl <T> NodePtr<T> where T: Clone + Default {

    ///helper function
    fn gen_priority_random() -> f32 {
        let mut rng = rand::thread_rng();
        let r = rng.gen_range( -1e-12, 1e12 );
        r
    }

    fn child_l(&self) -> Option<NodePtr<T>> {
        self.0.borrow().children.0.clone()
    }

    fn child_r(&self) -> Option<NodePtr<T>> {
        self.0.borrow().children.1.clone()
    }

    fn par(&self) -> NodePtrWk<T> {
        self.0.borrow().parent.clone()
    }

    fn link_left( & self, child: & Option<NodePtr<T>> ){
        match child {
            Some(x) => {
                x.0.borrow_mut().parent = NodePtrWk::from( self );
                self.0.borrow_mut().children.0 = child.clone();
            },
            _ =>{
                self.0.borrow_mut().children.0 = None;
            },
        }
    }
    
    fn link_right( & self, child: & Option<NodePtr<T>> ){
        match child {
            Some(x) => {
                x.0.borrow_mut().parent = NodePtrWk::from( self );
                self.0.borrow_mut().children.1 = child.clone();
            },
            _ =>{
                self.0.borrow_mut().children.1 = None;
            },
        }
    }
    
    ///helper function
    fn child_branch( & self, child: & Option<NodePtr<T>> ) -> ChildBranch {
        match child {
            Some(x) => {
                match ( self.0.borrow().children.0.as_ref(), x.0.borrow().parent.0.upgrade() ) {
                    (Some(ref a),Some(ref b))
                        if Rc::ptr_eq( &a.0, &x.0 ) &&
                           Rc::ptr_eq( &self.0, &b ) => {
                        return ChildBranch::Left
                    },
                    _ => {},
                }
                match ( self.0.borrow().children.1.as_ref(), x.0.borrow().parent.0.upgrade() ) {
                    (Some(ref a),Some(ref b))
                        if Rc::ptr_eq( &a.0, &x.0 ) &&
                           Rc::ptr_eq( &self.0, &b ) => {
                        return ChildBranch::Right
                    },
                    _ => {},
                }
            },
            _ => {},
        }
        ChildBranch::NotApplicable
    }

    pub fn init() -> Self {
        let mut n = Node::default();
        n.invalid = true;
        NodePtr::from(n)
    }
    
    pub fn search( & mut self, k: f32 ) -> SearchResult<T> {
        if self.0.borrow().invalid {
            SearchResult::Empty
        } else if k < (*self.0).borrow().key {
            match (*self.0).borrow_mut().children.0 {
                Some(ref mut x) => { x.search( k ) },
                _ => { SearchResult::Nearest( self.clone() ) },
            }
        } else if k > (*self.0).borrow().key {
            match (*self.0).borrow_mut().children.1 {
                Some(ref mut x) => { x.search( k ) },
                _ => { SearchResult::Nearest( self.clone() ) },
            }
        } else {
            SearchResult::Exact( self.clone() )
        }
    }

    pub fn get_root( & self ) -> NodePtr<T> {
        
        let mut n = self.clone();

        loop {
            let m = match n.0.borrow().parent.0.upgrade() {
                Some(x) => {
                    x
                },
                _ => { break; },
            };
            n = NodePtr(m);
        }
        
        n
    }
    
    pub fn insert_with_priority( & mut self, k: f32, val: T, priority: f32 ) -> NodePtr<T> {
        
        match self.search( k ){
            SearchResult::Exact(x) => {
                {
                    let mut n = x.0.borrow_mut();
                    n.val = val;
                    n.prio = priority;
                }
                x.fixup_priority();
                self.get_root()
            },
            SearchResult::Nearest(x) => {
                
                let n = Node {
                    key: k,
                    prio: priority,
                    val: val,
                    parent: Default::default(),
                    children: (None,None),
                    invalid: false,
                };
                
                let child = Some( NodePtr::from(n) );
                
                if k < x.0.borrow().key {
                    x.link_left( &child );
                } else {
                    x.link_right( &child );
                }
                x.fixup_priority();
                self.get_root()
            },
            Empty => {
                //create new node using current stale node
                self.0.borrow_mut().invalid = false;
                self.0.borrow_mut().key = k;
                self.0.borrow_mut().val = val;
                self.0.borrow_mut().prio = priority;
                self.0.borrow_mut().children = (None,None);
                self.0.borrow_mut().parent = Default::default();
                self.clone()
            },
        }
    }

    /// rotates current node up
    pub fn rot_up_left( & self ) {
        unimplemented!();
    }

    /// rotates current node up
    pub fn rot_up_right( & self ) {
        unimplemented!();
    }

    /// perform rotations to preserve heap property using priority
    pub fn fixup_priority( & self ) {
        unimplemented!();
    }

    /// removes current node and returns the root
    pub fn remove( & self ) -> Self {
        unimplemented!();
    }

    /// returns the node with the next highest key
    pub fn successor( & self ) -> Self {
        unimplemented!();
    }

    /// returns the node with the next lowest key
    pub fn predecessor( & self ) -> Self {
        unimplemented!();
    }

    /// return [x| x.key >= k_l && k.key < k_r ]
    pub fn query_range( & self, k_l: f32, k_r: f32 ) -> Vec<Self> {
        unimplemented!();
    }
    /// returns a, b pair such that a: [x| x.key<k], b: [x| x.key >=k]
    pub fn split( & self,k : f32 ) -> (Self, Self) {
        unimplemented!();
    }

    /// assumes a.merge(b) is such that keys of a < keys of b
    pub fn merge( & self, other: Self ) -> Self {
        unimplemented!();
    }
    
    pub fn union( & self, other: Self ) -> Self {
        unimplemented!();
    }
    
    /// returns min, max, avg depths
    pub fn dbg_depth( & self ) -> (f32,f32,f32) {
        unimplemented!();
    }
}


#[test]
fn test_search(){

    //         n0(5)
    //        /  \
    //     (2)n1  n2(7)
    //      / \
    //(-1)n3  n4(3)
    
    let mut n0 = Node {
        key: 5.,
        prio: 0.,
        val: 5,
        parent: Default::default(),
        children: ( None, None ),
        invalid: false,
    };

    let mut n1 = Node {
        key: 2.,
        prio: 0.,
        val: 2,
        parent: Default::default(),
        children: ( None, None ),
        invalid: false,
    };

    let mut n2 = Node {
        key: 7.,
        prio: 0.,
        val: 7,
        parent: Default::default(),
        children: ( None, None ),
        invalid: false,
    };

    let mut n3 = Node {
        key: -1.,
        prio: 0.,
        val: -1,
        parent: Default::default(),
        children: ( None, None ),
        invalid: false,
    };
    
    let mut n4 = Node {
        key: 3.,
        prio: 0.,
        val: 3,
        parent: Default::default(),
        children: ( None, None ),
        invalid: false,
    };
    
    let r3 = NodePtr(Rc::new( RefCell::new(n3) ));
    let r4 = NodePtr(Rc::new( RefCell::new(n4) ));
    n1.children.0 = Some(r3);
    n1.children.1 = Some(r4);
    let r1 = NodePtr(Rc::new( RefCell::new(n1) ));
    {
        let weak1 = NodePtrWk::from(&r1);
        let weak2 = weak1.clone();
        r1.child_l().as_ref().unwrap().0.borrow_mut().parent = weak1;
        r1.child_r().as_ref().unwrap().0.borrow_mut().parent = weak2;
    }
    let r2 = NodePtr(Rc::new( RefCell::new(n2) ));
    n0.children.0 = Some(r1);
    n0.children.1 = Some(r2);
    let mut r0 = NodePtr(Rc::new( RefCell::new(n0) ));
    
    {
        let weak1 = NodePtrWk::from(&r0);
        let weak2 = weak1.clone();
        
        r0.child_l().as_ref().unwrap().0.borrow_mut().parent = weak1;
        r0.child_r().as_ref().unwrap().0.borrow_mut().parent = weak2;
    }
    
    match r0.child_branch( &r0.child_l() ){
        ChildBranch::Left => {},
        _ => {panic!("incorrect child branch");},
    }
    
    fn equal_f32( a: f32, b: f32 ) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }

    }
    
    match r0.search( 3. ) {
        SearchResult::Exact(x) => {
            assert!( equal_f32( x.0.borrow().key, 4. ) );

            assert!( equal_f32( x.get_root().0.borrow().key, 5. ) );
        },
        _ => {
            panic!();
        }
    }

    match r0.search( 4. ) {
        SearchResult::Nearest(x) => {
            assert!( equal_f32( x.0.borrow().key, 4. ) );
        },
        _ => {
            panic!();
        }
    }

    match r0.search( 2. ) {
        SearchResult::Exact(x) => {
            assert!( equal_f32( x.0.borrow().key, 2. ) );

            // dbg!( &x.child_r().unwrap().0.borrow().key );
            
            match x.child_branch( &x.child_r() ){
                ChildBranch::Right => {
                    assert!( equal_f32( x.child_r().unwrap().0.borrow().key, 3. ) );
                },
                _ => {panic!("incorrect child branch");},
            }       
        },
        _ => {
            panic!();
        }
    }
    
    match r0.search( -1. ) {
        SearchResult::Exact(x) => {
            assert!( equal_f32( x.0.borrow().key, -1. ) );
        },
        _ => {
            panic!();
        }
    }
    match r0.search( -10. ) {
        SearchResult::Nearest(x) => {
            assert!( equal_f32( x.0.borrow().key, -1. ) );
        },
        _ => {
            panic!();
        }
    }
    match r0.search( 5. ) {
        SearchResult::Exact(x) => {
            assert!( equal_f32( x.0.borrow().key, 5. ) );
        },
        _ => {
            panic!();
        }
    }

    match r0.search( 6. ) {
        SearchResult::Nearest(x) => {
            assert!( equal_f32( x.0.borrow().key, 5. ) );
        },
        _ => {
            panic!();
        }
    }
    
    match r0.search( 10. ) {
        SearchResult::Nearest(x) => {
            assert!( equal_f32( x.0.borrow().key, 7. ) );
        },
        _ => {
            panic!();
        }
    }

    match r0.search( 2. ) {
        SearchResult::Exact(mut x) => {
            match x.search(-10.) {
                SearchResult::Nearest(y) => {
                    assert!( equal_f32( y.0.borrow().key, -1. ) );
                    assert_eq!( y.0.borrow().val, -1 );
                    y.0.borrow_mut().val = 100;
                },
                _ => { panic!(); },
            }
        },
        _ => {
            panic!();
        }
    }

    match r0.search( -1. ) {
        SearchResult::Exact(x) => {
            assert!( equal_f32( x.0.borrow().key, -1. ) );
            assert_eq!( x.0.borrow().val, 100 );

            let parent_key = x.0.borrow().parent.0.upgrade().expect("parent invalid").borrow().key;
            assert!( equal_f32( parent_key, 2. ) );
        },
        _ => {
            panic!();
        }
    }
}

//     ///return position of an item if it has the same key, priority is updated for an existing item
//     pub fn insert_with_priority( & mut self, instance: usize, k: f32, p: f32, val: T ) -> (bool, usize) {
//         match self.search( instance, k ) {
//             SearchResult::Empty => {
//                 let idx = self.new_slot();
//                 self.keys[idx] = k;
//                 self.priorities[idx] = p;
//                 self.vals[idx] = val;
//                 self.link_parent[idx] = idx;
//                 self.instances.insert( instance, Some(idx) );
                
//                 (false, idx)
//             },
//             SearchResult::Exact((n_pos,n_key,n_val)) => {
//                 //item with key already exists
//                 self.vals[n_pos] = val;
//                 self.priorities[n_pos] = p;
//                 self.fixup_priority( instance, n_pos);
                
//                 (true, n_pos)
//             },
//             SearchResult::Nearest((n_pos,n_key,n_val)) => {
                
//                 let idx = self.new_slot();
//                 self.keys[idx] = k;
//                 self.priorities[idx] = p;
//                 self.vals[idx] = val;
                
//                 self.link_parent[idx] = n_pos;
                
//                 match self.key(n_pos) {
//                     x if k < x => {
//                         self.link_child[n_pos].0 = Some(idx);

//                     },
//                     x => {
//                         self.link_child[n_pos].1 = Some(idx);
//                     },                                 
//                 }

//                 self.fixup_priority( instance, idx);
                
//                 (false, idx)
//             },
//         }
//     }

//     fn fixup_priority( & mut self, instance: usize, mut n: usize ){
        
//         //fix priority by rotating up the tree
        
//         let mut r : usize = self.instances.get(&instance).unwrap().expect("root non-existent");
        
//         let mut par = self.link_parent[n];
        
//         while self.prio(par) > self.prio(n) && r != n {
            
//             match self.child_branch( n, par ) {
//                 ChildBranch::Left => {
//                     self.rot_left( instance, n, par );     
//                 },
//                 ChildBranch::Right => {
//                     self.rot_right( instance, n, par );
//                 },
//                 _ => { panic!("child link error"); },
//             }
//             par = self.link_parent[n];
//             r = self.instances.get(&instance).unwrap().expect("root non-existent");
//         }
//     }

//     fn rot_left( & mut self, instance: usize, n: usize, parent: usize ) {

//         // before:
//         //          pp
//         //          |
//         //          p
//         //         / \
//         //        n   c
//         //       / \
//         //      a   b
//         //
//         // after:
//         //          pp
//         //          |
//         //          n
//         //         / \
//         //        a   p
//         //           / \
//         //          b   c
//         //

//         let parent_is_root = match self.instances.get( &instance ).expect("instance non-existent"){
//             Some(x) => {
//                 if *x == n {
//                     return
//                 } else if *x == parent {
//                     true
//                 } else {
//                     false
//                 }
//             },
//             _ => {panic!("root does not exist");},
//         };

//         if !parent_is_root {
//             let pp = self.link_parent[parent];
            
//             match self.child_branch( parent, pp ){
//                 ChildBranch::Left => {
//                     self.link_left( pp, Some(n) );
//                 },
//                 ChildBranch::Right => {
//                     self.link_right( pp, Some(n) );
//                 },
//                 _ => { panic!(); },
//             }
//         } else {
//             //update to new root
//             self.instances.insert( instance, Some(n) );
            
//             self.link_parent[n] = n;
//         }

//         self.link_left( parent, self.link_child[n].1 );

//         self.link_right( n, Some(parent) );
//     }

//     fn rot_right( & mut self, instance: usize, n: usize, parent: usize ) {

//         // before:
//         //          pp
//         //          |
//         //          p
//         //         / \
//         //        c   n
//         //           / \
//         //          a   b
//         //
//         // after:
//         //          pp
//         //          |
//         //          n
//         //         / \
//         //        p   b
//         //       / \
//         //      c   a
//         //

//         let parent_is_root = match self.instances.get( &instance ).expect("instance non-existent"){
//             Some(x) => {
//                 if *x == n {
//                     return
//                 } else if *x == parent {
//                     true
//                 } else {
//                     false
//                 }
//             },
//             _ => {panic!("root does not exist");},
//         };

//         if !parent_is_root {
//             let pp = self.link_parent[parent];
            
//             match self.child_branch( parent, pp ){
//                 ChildBranch::Left => {
//                     self.link_left( pp, Some(n) );
//                 },
//                 ChildBranch::Right => {
//                     self.link_right( pp, Some(n) );
//                 },
//                 _ => { panic!(); },
//             }
//         } else {
//             //update to new root
//             self.instances.insert( instance, Some(n) );

//             self.link_parent[n] = n;
//         }

//         self.link_right( parent, self.link_child[n].0 );

//         self.link_left( n, Some(parent) );
//     }

//     pub fn successor( & self, idx: usize ) -> Option<usize> {

//         let mut choice1 = None;
//         match self.link_child[idx].1 {
//             Some(x) => {
//                 let mut cur = x;
//                 while let Some(y) = self.link_child[cur].0 {
//                     cur = y;
//                 }
//                 choice1 = Some(cur);
//             },
//             _ => {},
//         }

//         if choice1.is_some() {
            
//             choice1
                
//         } else {

//             let mut cur = idx;
//             let mut p = self.link_parent[cur];

//             loop {
//                 match self.child_branch( cur, p ){
//                     ChildBranch::Left => {
//                         return Some(p)
//                     },
//                     ChildBranch::Right => {
//                         cur = p;
//                         p = self.link_parent[p];
//                     },
//                     _ => { return None },
//                 }
//             }
//         }
//     }

//     pub fn predecessor( & self, idx: usize ) -> Option<usize> {
        
//         let mut choice1 = None;
//         match self.link_child[idx].0 {
//             Some(x) => {
//                 let mut cur = x;
//                 while let Some(y) = self.link_child[cur].1 {
//                     cur = y;
//                 }
//                 choice1 = Some(cur);
//             },
//             _ => {},
//         }

//         if choice1.is_some() {
            
//             choice1
                
//         } else {

            
//             let mut cur = idx;
//             let mut p = self.link_parent[cur];
            
//             loop {
//                 match self.child_branch( cur, p ){
//                     ChildBranch::Right => {
//                         return Some(p)
//                     },
//                     ChildBranch::Left => {
//                         cur = p;
//                         p = self.link_parent[p];
//                     },
//                     _ => { return None },
//                 }
//             }
//         }        
//     }


//     /// get indices of items with key in [k_start,k_end)
//     pub fn query_range( & mut self, instance: usize, k_start: f32, k_end: f32 ) -> Vec<usize> {
        
//         let n_start = match self.search( instance, k_start ){
//             SearchResult::Exact( (idx,key,val) ) => {
//                 Some(idx)
//             },
//             SearchResult::Nearest( (idx,key,val)) => {
//                 if key < k_start {
//                     let mut n = self.successor(idx);
//                     while let Some(x) = n {
//                         if self.key(x) > k_start {
//                             break;
//                         }
//                         n = self.successor(x);
//                     }
//                     match n {
//                         Some(x) if self.key(x) > k_start => {
//                             Some(x)
//                         }
//                         _ => { None },
//                     }
//                 } else {
//                     Some(idx)
//                 }
//             },
//             _ => { None }
//         };
        
//         match n_start {
//             Some(start) if self.key(start) < k_end => {
                
//                 let mut cur = start;
                
//                 let mut ret = vec![ cur ];

//                 while let Some(x) = self.successor( cur ) {
                    
//                     if !( self.key(x) < k_end ) {
//                         break;
//                     }

//                     cur = x;
//                     ret.push( cur );
//                 }

//                 ret
//             },
//             _ => { vec![] }
//         }
//     }
    
//     pub fn remove_index( & mut self, instance: usize, idx: usize ){

//         loop {
//             let rot_index = match self.link_child[idx] {
//                 (Some(l),Some(r)) => {
//                     if self.key(l) < self.key(r) {
//                         l
//                     } else {
//                         r
//                     }
//                 },
//                 (Some(l),None) => { l },
//                 (None,Some(r)) => { r },
//                 _ => { break; },
//             };
//             match self.child_branch( rot_index, idx ){
//                 ChildBranch::Left => {
//                     self.rot_left( instance, rot_index, idx );
//                 },
//                 ChildBranch::Right => {
//                     self.rot_right( instance, rot_index, idx );
//                 },
//                 _ => { panic!(); },
//             }
//         }

//         let p = self.link_parent[idx];
//         if p != idx {
//             match self.child_branch( idx, p ){
//                 ChildBranch::Left => {
//                     self.link_child[p].0 = None;
//                 },
//                 ChildBranch::Right => {
//                     self.link_child[p].1 = None;
//                 },
//                 _ => { panic!(); },
//             }            
//         }

//         self.link_child[idx].0 = None;
//         self.link_child[idx].1 = None;
//         self.link_parent[idx] = idx;

//         match self.instances.get(&instance).expect("instance non-existent"){
//             Some(x) if *x == idx => {
//                 self.instances.insert( instance, None );
//             },
//             _ => {},
//         }
        
//         self.freelist.push(idx);
//     }

//     /// removes items with key in range of [k_start, k_end)
//     pub fn remove_key_range( & mut self, instance: usize, k_start: f32, k_end: f32 ){
//         self.query_range( instance, k_start, k_end ).iter()
//             .for_each(|x| self.remove_index( instance, *x));
//     }

//     /// split given treap instance into two instances: a:[ x | x.key < k ], b:[ x | x.key >= k ]
//     /// returns instance handles to split treaps (a,b)
//     pub fn split( & mut self, instance: usize, k: f32 ) -> ( usize, usize ) {
//         fn equal_f32( a: f32, b: f32 ) -> bool {
//             if a - 1e-4 < b || a + 1e-4 > b {
//                 true
//             } else {
//                 false
//             }
//         }

//         let (existing, idx) = self.insert_with_priority( instance, k, -1e-20, Default::default() );
        
//         assert!( equal_f32( self.key(self.instances.get(&instance).unwrap().expect("root non-existent") ), -1e-20) );
        
//         assert_eq!( idx, self.instances.get(&instance).unwrap().expect("root non-existent") );

//         let l = self.link_child[idx].0;
//         match l {
//             Some(x) => {
//                 self.link_parent[x] = x;
//             },
//             _ => {},
//         }
        
//         self.link_child[idx].0 = None;
        
//         //update root
//         *self.instances.get_mut(&instance).unwrap() = l;
        
//         let r = if existing {
//             Some(idx)
//         } else {
//             let temp = self.link_child[idx].1;
//             self.link_child[idx].1 = None;
//             temp
//         };

//         match r {
//             Some(x) => {
//                 self.link_parent[x] = x;
//             },
//             _ => {},
//         }

//         //update root
//         let new_inst = self.instances.len();
//         self.instances.insert( new_inst, r );

//         ( instance, new_inst )
//     }

//     /// merges 2 trees and returns handle to a combined tree
//     pub fn merge( & mut self, inst_a: usize, inst_b: usize ) -> usize {
        
//         let root_a = *self.instances.get(&inst_a).expect("instance non-existent");
//         let root_b = *self.instances.get(&inst_b).expect("instance non-existent");

//         match (root_a,root_b) {
//             (Some(l), Some(r)) => {
                    
//                 let idx = self.new_slot();
                
//                 *self.instances.get_mut(&inst_a).unwrap() = Some(idx);

//                 self.link_left( idx, Some(l) );
//                 self.link_right( idx, Some(r) );
                
//                 self.priorities[idx] = 1e20;
                
//                 self.remove_index( inst_a, idx );

//                 *self.instances.get_mut(&inst_b).unwrap() = None;

//                 inst_a
//             },
//             (Some(l),_) => {
//                 inst_a
//             },
//             (_,Some(r)) => {
//                 inst_b
//             },
//             _ => { panic!(); },
//         }
//     }

//     pub fn union( & mut self, inst_a: usize, inst_b: usize ) -> usize {
//         unimplemented!();
//     }

//     pub fn dbg_depth( & self, instance: usize ) -> f32 {
        
//         let r : usize = match self.instances.get(&instance) {
//             Some(x) if x.is_some() => {
//                 x.unwrap()
//             },
//             _ => {
//                 return 0.
//             },
//         };
        
//         let mut hm = HashSet::new();
//         let mut q = vec![r];
        
//         let mut leaf_depths = vec![];
        
//         while !q.is_empty() {
//             let l = q.len();
//             let cur = q.pop().unwrap();
//             match hm.get(&cur) {
//                 Some(x) => {
//                     leaf_depths.push( l );
//                 },
//                 _ => {
//                     hm.insert(cur);
//                     q.push(cur);
//                     if let Some(x) = self.link_child[cur].0 {
//                         q.push(x);
//                     }
//                     if let Some(x) = self.link_child[cur].1 {
//                         q.push(x);
//                     }                    
//                 },
//             }
//         }

//         let total = leaf_depths.iter().fold(0,|acc,val| acc + *val );
//         let avg = total as f32 / leaf_depths.len() as f32;
//         avg
//     }
// }

// #[test]
// fn test_treap_search() {
//     fn equal_f32( a: f32, b: f32 ) -> bool {
//         if a - 1e-4 < b || a + 1e-4 > b {
//             true
//         } else {
//             false
//         }
//     }
//     // test tree
//     //            6
//     //         3     8
//     //      1    5  7  10
//     //        2
//     let mut t = Treap::init();

//     let inst = t.new_instance();

//     match t.search( inst, 5. ) {
//         SearchResult::Empty => {},
//         _ => { panic!("search failure"); },
//     }
    
//     t.keys = vec![1.,2.,3.,5.,6.,7.,8.,10.];
//     *t.instances.get_mut(&inst).unwrap() = Some(4);
//     t.vals = t.keys.iter().enumerate().map(|x| x.0 as i32).collect();
//     t.link_child.resize(t.keys.len(), (None,None) );
//     t.link_child[0].1 = Some(1);
//     t.link_child[2].0 = Some(0);
//     t.link_child[2].1 = Some(3);
//     t.link_child[4].0 = Some(2);
//     t.link_child[4].1 = Some(6);
//     t.link_child[6].0 = Some(5);
//     t.link_child[6].1 = Some(7);
    
//     for i in t.keys.iter().cloned(){
//         match t.search( inst, i ) {
//             SearchResult::Exact(_) => {},
//             _ => { panic!("search failure"); },
//         }
//     }

//     match t.search( inst, 0. ) {
//         SearchResult::Nearest((0, k, 0)) if equal_f32(k,1.) => {},
//         _ => { panic!("search failure"); },
//     }

//     match t.search( inst, 4. ) {
//         SearchResult::Nearest((3, k, 3)) if equal_f32(k,5.) => {},
//         _ => { panic!("search failure"); },
//     }

//     match t.search( inst, 8. ) {
//         SearchResult::Exact((6, k, 6)) if equal_f32(k,8.) => {},
//         _ => { panic!("search failure"); },
//     }

//     match t.search( inst, 99. ) {
//         SearchResult::Nearest((7, k, 7)) if equal_f32(k,10.) => {},
//         _ => { panic!("search failure"); },
//     }
// }

// #[test]
// fn test_treap_insert_with_priority() {
//     // test tree
//     //            6
//     //         3     8
//     //      1   
//     let mut t = Treap::init();
//     let inst = t.new_instance();
//     assert_eq!( None, *t.instances.get(&inst).unwrap() );
//     assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );
//     assert_eq!( true, t.instances.get(&inst).unwrap().is_some() );
    
//     assert_eq!( t.insert_with_priority( inst, 3., 50., 3 ).0, true );
//     assert_eq!( t.insert_with_priority( inst, 3., 50., 3 ).1, 0 );

//     assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );
// }

// #[test]
// fn test_treap_insert() {
    
//     let mut t = Treap::init();
//     let inst = t.new_instance();
//     assert_eq!( None, *t.instances.get(&inst).unwrap() );
//     assert_eq!( t.insert( inst, 3., 33 ).0, false );
//     assert_eq!( true, t.instances.get(&inst).unwrap().is_some() );
//     assert_eq!( t.insert( inst, 3., 3 ).0, true );
//     assert_eq!( t.insert( inst, 1., 1 ).0, false );
//     assert_eq!( t.insert( inst, 8., 8 ).0, false );
//     assert_eq!( t.insert( inst, 6., 6 ).0, false );
// }


// #[test]
// fn test_treap_successor() {
//     // test tree
//     //            6
//     //         3     8
//     //      1   
//     let mut t = Treap::init();
//     let inst = t.new_instance();
    
//     assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 3., 50., 3 ).0, true );
//     assert_eq!( t.insert_with_priority( inst, 3., 50., 3 ).1, 0 );

//     assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

//     let mut n = t.successor( 1 );
//     assert_eq!( 0, n.unwrap() );
//     n = t.successor( n.unwrap() );
//     assert_eq!( 3, n.unwrap() );
//     n = t.successor( n.unwrap() );
//     assert_eq!( 2, n.unwrap() );
//     assert!( t.successor( n.unwrap() ).is_none() );
// }

// #[test]
// fn test_treap_predecessor() {

//     // test tree
//     //            6
//     //         3     8
//     //      1      7
    
//     let mut t = Treap::init();
//     let inst = t.new_instance();
    
//     assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 7., 0., 7 ).0, false );

//     let mut n = t.predecessor( 2 );
//     assert_eq!( 4, n.unwrap() );
//     n = t.predecessor( n.unwrap() );
//     assert_eq!( 3, n.unwrap() );
//     n = t.predecessor( n.unwrap() );
//     assert_eq!( 0, n.unwrap() );
//     n = t.predecessor( n.unwrap() );
//     assert_eq!( 1, n.unwrap() );
//     assert!( t.predecessor( n.unwrap() ).is_none() );
// }

// #[test]
// fn test_treap_query_range() {
    
//     fn equal_f32( a: f32, b: f32 ) -> bool {
//         if a - 1e-4 < b || a + 1e-4 > b {
//             true
//         } else {
//             false
//         }
//     }
    
//     // test tree
//     //            6
//     //         3     8
//     //      1      7
    
//     let mut t = Treap::init();
//     let inst = t.new_instance();
    
//     assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 7., 0., 7 ).0, false );

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 5 );
//         let expected = [1.,3.,6.,7.,8.].to_vec();
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }
//     {
//         let ret = t.query_range( inst, 3., 8. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [3.,6.,7.].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }
//     {
//         let ret = t.query_range( inst, 8.5, 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }
//     {
//         let ret = t.query_range( inst, -10., 0. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }
//     {
//         let ret = t.query_range( inst, -10., 6. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [1.,3.].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }
//     {
//         let ret = t.query_range( inst, 7.99, 8.1 ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [8.].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }
// }

// #[test]
// fn test_treap_remove(){

//     fn equal_f32( a: f32, b: f32 ) -> bool {
//         if a - 1e-4 < b || a + 1e-4 > b {
//             true
//         } else {
//             false
//         }
//     }
    
//     // test tree
//     //            6
//     //         3     8
//     //      1      7
    
//     let mut t = Treap::init();
//     let inst = t.new_instance();
    
//     assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 7., 0., 7 ).0, false );

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 5 );
//         let expected = [1.,3.,6.,7.,8.].to_vec();
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }


//     t.remove_index(inst, 1);

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [3.,6.,7.,8.].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     t.remove_index(inst, 0);

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [6.,7.,8.].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     t.remove_index(inst, 3);

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [7.,8.].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     t.remove_index(inst, 2);

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [7.].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//         assert_eq!( t.key(t.instances.get(&inst).unwrap().unwrap()), 7. );
//     }

//     t.remove_index(inst, 4);

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//         assert!( t.instances.get(&inst).unwrap().is_none() );
//     }
//     // dbg!( &t );
// }

// #[test]
// fn test_treap_remove_key_range(){
    
//     fn equal_f32( a: f32, b: f32 ) -> bool {
//         if a - 1e-4 < b || a + 1e-4 > b {
//             true
//         } else {
//             false
//         }
//     }
    
//     // test tree
//     //            6
//     //         3     8
//     //      1      7
    
//     let mut t = Treap::init();
//     let inst = t.new_instance();
    
//     assert_eq!( t.insert_with_priority( inst, 3., 50., 33 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 1., 75., 1 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 8., 30., 8 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 6., -20., 6 ).0, false );

//     assert_eq!( t.insert_with_priority( inst, 7., 0., 7 ).0, false );

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 5 );
//         let expected = [1.,3.,6.,7.,8.].to_vec();
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }
    
//     t.remove_key_range( inst, 2.5, 6.5 );

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [1.,7.,8.].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     t.remove_key_range( inst, 7.5, 10. );

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [1.,7.].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     t.remove_key_range( inst, 0., 10. );

//     {
//         let ret = t.query_range( inst, 0., 10. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         let expected = [].to_vec();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//         assert!( t.instances.get(&inst).unwrap().is_none() );
//     }
// }

// #[test]
// fn test_treap_insert_remove_loop(){
    
//     fn equal_f32( a: f32, b: f32 ) -> bool {
//         if a - 1e-4 < b || a + 1e-4 > b {
//             true
//         } else {
//             false
//         }
//     }
    
//     // test tree
//     //            6
//     //         3     8
//     //      1      7
    
//     let mut t = Treap::init();
    
//     let inst = t.new_instance();

//     let expected = vec![1.,3.,6.,7.,8.];

//     expected.iter().for_each(|x| {t.insert( inst, *x,*x as i32);} );

//     {
//         let ret = t.query_range( inst, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     use std::collections::HashSet;
//     let mut hs = HashSet::new();
//     expected.iter().for_each(|x| {hs.insert( *x as i32 );} );

//     while hs.len() != 0 {
//         let remain = hs.iter().cloned().collect::<Vec<_>>();
//         let mut rng = rand::thread_rng();
//         let select = rng.gen_range(0, remain.len());
//         let key = remain[select];

//         let k_start = key as f32 - 1e-3;
//         let k_end = key as f32 + 1e-3;
        
//         t.remove_key_range( inst, k_start, k_end );
        
//         {
//             let ret = t.query_range( inst, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//             assert_eq!( ret.len(), remain.len()-1 );
            
//             ret.iter().for_each(|x| {assert!(remain.iter().any(|y| equal_f32(*y as f32,*x) ));} );
//         }

//         hs.remove(&key);
//         if hs.len() > 0 {
//             assert!( t.instances.get(&inst).unwrap().is_some() );
//         }
//     }
    
//     assert!( t.instances.get(&inst).unwrap().is_none() );
// }

// #[test]
// fn test_treap_split1(){
    
//     fn equal_f32( a: f32, b: f32 ) -> bool {
//         if a - 1e-4 < b || a + 1e-4 > b {
//             true
//         } else {
//             false
//         }
//     }
    
//     // test tree
//     //            6
//     //         3     8
//     //      1      7
    
//     let mut t = Treap::init();
    
//     let inst = t.new_instance();

//     let expected = vec![1.,3.,6.,7.,8.];

//     expected.iter().for_each(|x| {t.insert( inst, *x,*x as i32);} );

//     {
//         let ret = t.query_range( inst, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     let (inst1, inst2) = t.split( inst, 6.5 );

//     {
//         let ret = t.query_range( inst1, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 3 );
//         let check = expected.iter().take(3).zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     {
//         let ret = t.query_range( inst2, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 2 );
//         let check = expected.iter().skip(3).zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     let (inst3, inst4) = t.split( inst1, 3. );

//     {
//         let ret = t.query_range( inst3, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 1 );
//         let check = expected.iter().take(1).zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     {
//         let ret = t.query_range( inst4, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 2 );
//         let check = expected.iter().skip(1).zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }
// }

// #[test]
// fn test_treap_split2(){
    
//     fn equal_f32( a: f32, b: f32 ) -> bool {
//         if a - 1e-4 < b || a + 1e-4 > b {
//             true
//         } else {
//             false
//         }
//     }
    
//     // test tree
//     //            6
//     //         3     8
//     //      1      7
    
//     let mut t = Treap::init();
    
//     let inst = t.new_instance();

//     let expected = vec![1.,3.,6.,7.,8.];

//     expected.iter().for_each(|x| {t.insert( inst, *x,*x as i32);} );

//     {
//         let ret = t.query_range( inst, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), expected.len() );
//         let check = expected.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     let (inst1, inst2) = t.split( inst, 3. );

//     {
//         let ret = t.query_range( inst1, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 1 );
//         let check = expected.iter().take(1).zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }

//     {
//         let ret = t.query_range( inst2, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 4 );
//         let check = expected.iter().skip(1).zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a, *b ) );} );
//     }    
// }

// #[test]
// fn test_treap_merge(){
    
//     fn equal_f32( a: f32, b: f32 ) -> bool {
//         if a - 1e-4 < b || a + 1e-4 > b {
//             true
//         } else {
//             false
//         }
//     }
    
//     // test tree 1 vals: [1,3,4,11]

//     // test tree 2 vals: [12,13,17,18]
    
//     let mut t = Treap::init();
    
//     let inst1 = t.new_instance();
//     let inst2 = t.new_instance();
    
//     let expected1 = vec![1,3,4,11];
//     let expected2 = vec![12,13,17,18];

//     expected1.iter().for_each(|x| {t.insert( inst1, *x as f32,*x as i32);} );
//     expected2.iter().for_each(|x| {t.insert( inst2, *x as f32,*x as i32);} );

//     {
//         let ret = t.query_range( inst1, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), expected1.len() );
//         let check = expected1.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a as f32, *b ) );} );
//     }

//     {
//         let ret = t.query_range( inst2, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), expected2.len() );
//         let check = expected2.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a as f32, *b ) );} );
//     }

//     let inst3 = t.merge( inst1, inst2 );

//     assert_eq!( inst3, inst1 );
//     assert!( t.instances.get(&inst2).unwrap().is_none() );
    
//     let expected_merged = expected1.iter().cloned().chain(expected2.iter().cloned()).collect::<Vec<_>>();
//     {
//         let ret = t.query_range( inst3, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), expected_merged.len() );
//         let check = expected_merged.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a as f32, *b ) );} );
        
//     }
// }

// #[test]
// fn test_treap_split_merge(){
    
//     fn equal_f32( a: f32, b: f32 ) -> bool {
//         if a - 1e-4 < b || a + 1e-4 > b {
//             true
//         } else {
//             false
//         }
//     }
    
//     // test tree vals: [1,3,4,11,12,13,17,18]
    
//     let mut t = Treap::init();
    
//     let inst1 = t.new_instance();
    
//     let expected1 = vec![1,17,11,13,18,3,4,12];

//     expected1.iter().for_each(|x| {t.insert( inst1, *x as f32,*x as i32);} );

//     {
//         let ret = t.query_range( inst1, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), expected1.len() );
//         let check = expected1.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( *a as f32, *b ) );} );
//     }

//     let (inst2, inst3) = t.split( inst1, 9. );

//     let mut split_expected1 = expected1.iter().cloned().filter(|x| (*x as f32) < 9. ).collect::<Vec<_>>();
//     split_expected1.sort();

//     let mut split_expected2 = expected1.iter().cloned().filter(|x| (*x as f32) > 9. ).collect::<Vec<_>>();
//     split_expected2.sort();
    
//     {
//         let ret = t.query_range( inst2, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), split_expected1.len() );
//         let check = split_expected1.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( (*a as f32), *b ) );} );
//     }

//     {
//         let ret = t.query_range( inst3, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), split_expected2.len() );
//         let check = split_expected2.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( (*a as f32), *b ) );} );
//     }
    
//     let inst4 = t.merge( inst2, inst3 );
    
//     let mut expected_merged = expected1.clone();
//     expected_merged.sort();
    
//     {
//         let ret = t.query_range( inst4, 0., 20. ).iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), expected_merged.len() );
//         let check = expected_merged.iter().zip( ret.iter() );
//         check.for_each(|(a,b)| {assert!( equal_f32( (*a as f32), *b ) );} );
//     }
// }

// #[test]
// fn test_treap_stress(){

//     let count = 100_000;
//     let v = (0..count).map(|x| x as f32 ).collect::<Vec<_>>();
//     let mut t = Treap::init();
//     let inst = t.new_instance();

//     let t0 = Local::now();

//     for i in v.iter() {
//         t.insert( inst, *i, *i as i32 );
//     }
    
//     let t1 = Local::now();
        
//     {
//         let ret = t.query_range( inst, -1e-20, 1e20 )
//             .iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), count );
//     }

//     let avg_depth = t.dbg_depth( inst );
//     dbg!( (count as f32).log2()+1., &avg_depth );
    
//     assert!( 5. * ((count as f32).log2()+1.) > avg_depth );
    
//     let v2 = v.iter().cloned().map(|x| (x, x+0.5) ).collect::<Vec<_>>();

//     let t2 = Local::now();
    
//     for i in v2.iter() {
//         t.remove_key_range( inst, i.0, i.1 );
//     }
    
//     let t3 = Local::now();

//     {
//         let ret = t.query_range( inst, -1e-20, 1e20 )
//             .iter().map(|x| t.key(*x)).collect::<Vec<_>>();
//         assert_eq!( ret.len(), 0 );
//     }
    
//     let t_ins = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
//     let t_rem = t3.signed_duration_since(t2).num_microseconds().unwrap() as f64;
    
//     println!( "{} us / insert", t_ins / count as f64 );
//     println!( "{} us / removal", t_rem / count as f64 );
// }
