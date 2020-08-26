//! treap implementation based from various references such as:
//! http://e-maxx.ru/algo/treap
//! Fast Set Operations Using Treaps: http://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/treaps-spaa98.html

use std::cell::RefCell;
#[cfg(test)]
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f32;
use std::fmt::Debug;
use std::rc::{Rc, Weak};

extern crate rand;
use self::rand::Rng;

#[cfg(test)]
extern crate chrono;
#[cfg(test)]
use self::chrono::prelude::*;

#[derive(Default, Clone, Debug)]
pub struct Node<K, T>
where
    T: Clone + Default + Debug,
    K: PartialOrd + Clone + Copy + Default + Debug,
{
    pub key: K,
    pub prio: f32,
    pub val: T,
    pub parent: NodePtrWk<K, T>,
    pub children: (Option<NodePtr<K, T>>, Option<NodePtr<K, T>>),
    pub invalid: bool,
}

#[derive(Default, Clone, Debug)]
pub struct NodePtr<K, T>(pub Rc<RefCell<Node<K, T>>>)
where
    T: Clone + Default + Debug,
    K: PartialOrd + Clone + Copy + Default + Debug;

impl<K, T> From<Node<K, T>> for NodePtr<K, T>
where
    T: Clone + Default + Debug,
    K: PartialOrd + Clone + Copy + Default + Debug,
{
    fn from(n: Node<K, T>) -> Self {
        Self(Rc::new(RefCell::new(n)))
    }
}

#[derive(Default, Clone, Debug)]
pub struct NodePtrWk<K, T>(pub Weak<RefCell<Node<K, T>>>)
where
    T: Clone + Default + Debug,
    K: PartialOrd + Clone + Copy + Default + Debug;

impl<K, T> From<&NodePtr<K, T>> for NodePtrWk<K, T>
where
    T: Clone + Default + Debug,
    K: PartialOrd + Clone + Copy + Default + Debug,
{
    fn from(n: &NodePtr<K, T>) -> Self {
        Self(Rc::downgrade(&n.0))
    }
}

pub enum SearchResult<K, T>
where
    T: Clone + Default + Debug,
    K: PartialOrd + Clone + Copy + Default + Debug,
{
    Exact(NodePtr<K, T>),
    Nearest(NodePtr<K, T>),
    Empty,
}

pub enum ChildBranch {
    Left,
    Right,
    NotApplicable,
}

impl<K, T> NodePtr<K, T>
where
    T: Clone + Default + Debug,
    K: PartialOrd + Clone + Copy + Default + Debug,
{
    ///helper function
    fn gen_priority_random() -> f32 {
        let mut rng = rand::thread_rng();
        let r = rng.gen_range(-1e30_f32, 1e30_f32);
        r
    }

    pub fn child_l(&self) -> Option<NodePtr<K, T>> {
        self.0.borrow().children.0.clone()
    }

    pub fn child_r(&self) -> Option<NodePtr<K, T>> {
        self.0.borrow().children.1.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.0.borrow().invalid
    }

    pub fn is_leaf(&self) -> bool {
        self.child_l().is_none() && self.child_r().is_none()
    }

    pub fn par(&self) -> NodePtrWk<K, T> {
        self.0.borrow().parent.clone()
    }

    pub fn key(&self) -> K {
        self.0.borrow().key.clone()
    }

    pub fn val(&self) -> T {
        self.0.borrow().val.clone()
    }

    pub fn prio(&self) -> f32 {
        self.0.borrow().prio
    }

    fn link_left(&self, child: &Option<NodePtr<K, T>>) {
        match child {
            Some(x) => {
                x.0.borrow_mut().parent = NodePtrWk::from(self);
                self.0.borrow_mut().children.0 = child.clone();
            }
            _ => {
                self.0.borrow_mut().children.0 = None;
            }
        }
    }

    fn link_right(&self, child: &Option<NodePtr<K, T>>) {
        match child {
            Some(x) => {
                x.0.borrow_mut().parent = NodePtrWk::from(self);
                self.0.borrow_mut().children.1 = child.clone();
            }
            _ => {
                self.0.borrow_mut().children.1 = None;
            }
        }
    }

    ///helper function
    fn child_branch(&self, child: &Option<NodePtr<K, T>>) -> ChildBranch {
        match child {
            Some(x) => {
                match (
                    self.0.borrow().children.0.as_ref(),
                    x.0.borrow().parent.0.upgrade(),
                ) {
                    (Some(ref a), Some(ref b))
                        if Rc::ptr_eq(&a.0, &x.0) && Rc::ptr_eq(&self.0, &b) =>
                    {
                        return ChildBranch::Left
                    }
                    _ => {}
                }
                match (
                    self.0.borrow().children.1.as_ref(),
                    x.0.borrow().parent.0.upgrade(),
                ) {
                    (Some(ref a), Some(ref b))
                        if Rc::ptr_eq(&a.0, &x.0) && Rc::ptr_eq(&self.0, &b) =>
                    {
                        return ChildBranch::Right
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        ChildBranch::NotApplicable
    }

    pub fn new() -> Self {
        let mut n = Node::default();
        n.invalid = true;
        NodePtr::from(n)
    }

    pub fn search(&self, k: K) -> SearchResult<K, T> {
        if self.0.borrow().invalid {
            SearchResult::Empty
        } else if k < (*self.0).borrow().key {
            match (*self.0).borrow_mut().children.0 {
                Some(ref mut x) => x.search(k),
                _ => SearchResult::Nearest(self.clone()),
            }
        } else if k > (*self.0).borrow().key {
            match (*self.0).borrow_mut().children.1 {
                Some(ref mut x) => x.search(k),
                _ => SearchResult::Nearest(self.clone()),
            }
        } else {
            SearchResult::Exact(self.clone())
        }
    }

    pub fn get_root(&self) -> NodePtr<K, T> {
        let mut n = self.clone();

        loop {
            let m = match n.0.borrow().parent.0.upgrade() {
                Some(x) => x,
                _ => {
                    break;
                }
            };
            n = NodePtr(m);
        }

        n
    }

    ///inserts a node and returns ( root, already exists )
    ///existing node value and priority is updated
    pub fn insert_with_priority(&self, k: K, val: T, priority: f32) -> (NodePtr<K, T>, bool) {
        match self.search(k) {
            SearchResult::Exact(x) => {
                {
                    let mut n = x.0.borrow_mut();
                    n.val = val;
                    n.prio = priority;
                }

                //replacing priority require fixup

                //fix upward pass
                let _root = x.fixup_priority();

                //fix downward pass
                let _root2 = x.fixdown_priority();

                (self.get_root(), true)
            }
            SearchResult::Nearest(x) => {
                let n = Node {
                    key: k,
                    prio: priority,
                    val: val,
                    parent: Default::default(),
                    children: (None, None),
                    invalid: false,
                };

                let child = Some(NodePtr::from(n));

                if k < x.0.borrow().key {
                    x.link_left(&child);
                } else {
                    x.link_right(&child);
                }

                let _root = child.as_ref().unwrap().fixup_priority();

                (self.get_root(), false)
            }
            SearchResult::Empty => {
                //create new node using current stale node
                self.0.borrow_mut().invalid = false;
                self.0.borrow_mut().key = k;
                self.0.borrow_mut().val = val;
                self.0.borrow_mut().prio = priority;
                self.0.borrow_mut().children = (None, None);
                self.0.borrow_mut().parent = Default::default();

                (self.get_root(), false)
            }
        }
    }

    ///inserts a node and returns ( root, already_exists )
    pub fn insert(&self, k: K, val: T) -> (NodePtr<K, T>, bool) {
        let prio = Self::gen_priority_random();
        self.insert_with_priority(k, val, prio)
    }

    pub fn link_grandparent(&self) {
        let p = match self.0.borrow().parent.0.upgrade() {
            Some(x) => NodePtr(x),
            _ => return,
        };

        match p.0.clone().borrow().parent.0.upgrade() {
            Some(x) => {
                let pp = NodePtr(x);
                match pp.child_branch(&Some(p.clone())) {
                    ChildBranch::Left => {
                        pp.link_left(&Some(self.clone()));
                    }
                    ChildBranch::Right => {
                        pp.link_right(&Some(self.clone()));
                    }
                    _ => {
                        panic!();
                    }
                }
            }
            _ => {
                self.0.borrow_mut().parent = NodePtrWk(Weak::new());
            }
        }
    }

    /// rotates current node up
    pub fn rot_up_left(&self) -> Self {
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

        let p = match self.0.borrow().parent.0.upgrade() {
            Some(x) => NodePtr(x),
            _ => {
                panic!();
            }
        };

        self.link_grandparent();

        let temp = self.0.borrow().children.1.clone();

        self.link_right(&None);

        p.link_left(&temp);

        self.link_right(&Some(p));

        self.clone()
    }

    /// rotates current node up and returns itself
    pub fn rot_up_right(&self) -> Self {
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

        let p = match self.0.borrow().parent.0.upgrade() {
            Some(x) => NodePtr(x),
            _ => {
                panic!();
            }
        };

        self.link_grandparent();

        let temp = self.0.borrow().children.0.clone();

        self.link_left(&None);

        p.link_right(&temp);

        self.link_left(&Some(p));

        self.clone()
    }

    /// perform rotations upward to preserve heap property using priority
    /// this moves a node with low priority upward
    /// return a node if it is the new root
    pub fn fixup_priority(&self) -> Option<Self> {
        let prio = self.0.borrow().prio;

        let mut p = self.par().0.upgrade();

        loop {
            match p {
                Some(ref x) => {
                    let parent = NodePtr(x.clone());

                    if parent.0.borrow().prio < prio {
                        break;
                    } else {
                        match parent.child_branch(&Some(self.clone())) {
                            ChildBranch::Left => {
                                self.rot_up_left();
                            }
                            ChildBranch::Right => {
                                self.rot_up_right();
                            }
                            _ => {
                                panic!();
                            }
                        }
                        p = self.par().0.upgrade();
                    }
                }
                _ => {
                    break;
                }
            }
        }

        match self.par().0.upgrade().as_ref() {
            None => Some(self.clone()),
            _ => None,
        }
    }

    /// fixes the priority in a downward pass
    /// this moves a node with high priority downward
    /// return a node if it is the new root
    pub fn fixdown_priority(&self) -> Option<Self> {
        let x = self.clone();

        let mut root = None;

        loop {
            match (x.child_l().as_ref(), x.child_r().as_ref()) {
                (Some(l), Some(r)) => {
                    if l.0.borrow().prio < r.0.borrow().prio
                        && l.0.borrow().prio < x.0.borrow().prio
                    {
                        let n = l.rot_up_left();
                        match n.clone().0.borrow().parent.0.upgrade().as_ref() {
                            None => root = Some(n),
                            _ => {}
                        }
                    } else if l.0.borrow().prio > r.0.borrow().prio
                        && r.0.borrow().prio < x.0.borrow().prio
                    {
                        let n = r.rot_up_right();

                        match n.clone().0.borrow().parent.0.upgrade().as_ref() {
                            None => root = Some(n),
                            _ => {}
                        }
                    } else {
                        break;
                    }
                }
                (Some(l), None) => {
                    if l.0.borrow().prio < x.0.borrow().prio {
                        let n = l.rot_up_left();

                        match n.clone().0.borrow().parent.0.upgrade().as_ref() {
                            None => root = Some(n),
                            _ => {}
                        }
                    } else {
                        break;
                    }
                }
                (None, Some(r)) => {
                    if r.0.borrow().prio < x.0.borrow().prio {
                        let n = r.rot_up_right();

                        match n.clone().0.borrow().parent.0.upgrade().as_ref() {
                            None => root = Some(n),
                            _ => {}
                        }
                    } else {
                        break;
                    }
                }
                _ => {
                    break;
                }
            }
        }
        root
    }

    /// removes current node and returns the root
    pub fn remove(&self) -> Self {
        let mut n = self.get_root();

        {
            self.0.borrow_mut().prio = f32::INFINITY;

            //fix downward pass
            match self.fixdown_priority() {
                Some(new_root) => {
                    n = new_root;
                }
                _ => {}
            }
        }

        if Rc::ptr_eq(&self.0, &n.0) {
            //leave node as sentil for empty tree by marking the invalid bit
            self.0.borrow_mut().invalid = true;
        } else {
            //remove node

            let ref_count = Rc::strong_count(&self.0);

            let p = NodePtr(self.par().0.upgrade().expect("parent non-existent"));
            match p.child_branch(&Some(self.clone())) {
                ChildBranch::Left => {
                    p.0.borrow_mut().children.0 = None;
                }
                ChildBranch::Right => {
                    p.0.borrow_mut().children.1 = None;
                }
                _ => {
                    panic!();
                }
            }

            //there should be 1 less strong ref count for the current node
            debug_assert_eq!(1, ref_count - Rc::strong_count(&self.0));
        }

        n
    }

    /// removes a node and returns the root
    pub fn remove_by_key(&self, k: K) -> Self {
        match self.search(k) {
            SearchResult::Exact(x) => x.remove(),
            _ => self.clone(),
        }
    }

    /// removes nodes with keys k in [k_l, k_r) and returns the root
    pub fn remove_by_key_range(&self, k_l: K, k_r: K) -> Self {
        let mut root = self.get_root();
        for i in self.query_key_range(k_l, k_r) {
            root = i.remove();
        }
        root
    }

    /// returns the node with the next highest key
    pub fn successor(&self) -> Option<Self> {
        match self.child_r().as_ref() {
            Some(x) => {
                let mut cur = x.clone();
                while let Some(y) = cur.child_l().as_ref() {
                    cur = y.clone();
                }
                Some(cur)
            }
            _ => {
                let mut cur = self.clone();

                while let Some(y) = cur.par().0.upgrade() {
                    let p = NodePtr(y);
                    match p.child_branch(&Some(cur.clone())) {
                        ChildBranch::Left => return Some(p),
                        ChildBranch::Right => {
                            cur = p;
                        }
                        _ => {
                            panic!("parent child link non-existent");
                        }
                    }
                }
                None
            }
        }
    }

    /// returns the node with the next lowest key
    pub fn predecessor(&self) -> Option<Self> {
        match self.child_l().as_ref() {
            Some(x) => {
                let mut cur = x.clone();
                while let Some(y) = cur.child_r().as_ref() {
                    cur = y.clone();
                }
                Some(cur)
            }
            _ => {
                let mut cur = self.clone();

                while let Some(y) = cur.par().0.upgrade() {
                    let p = NodePtr(y);
                    match p.child_branch(&Some(cur.clone())) {
                        ChildBranch::Left => {
                            cur = p;
                        }
                        ChildBranch::Right => return Some(p),
                        _ => {
                            panic!("parent child link non-existent");
                        }
                    }
                }
                None
            }
        }
    }

    /// return [x| x.key >= k_l && k.key < k_r ]
    pub fn query_key_range(&self, k_l: K, k_r: K) -> Vec<Self> {
        let start = match self.search(k_l) {
            SearchResult::Exact(x) => {
                if x.key() < k_r {
                    Some(x)
                } else {
                    None
                }
            }
            SearchResult::Nearest(x) => {
                if x.key() < k_l {
                    let mut cur = x;
                    while let Some(y) = cur.successor() {
                        cur = y;
                        if !(cur.key() < k_l) {
                            break;
                        }
                    }
                    if !(cur.key() < k_l) && cur.key() < k_r {
                        Some(cur)
                    } else {
                        None
                    }
                } else if x.key() < k_r {
                    Some(x)
                } else {
                    None
                }
            }
            SearchResult::Empty => None,
        };

        if start.is_none() {
            vec![]
        } else {
            let mut cur = start.unwrap();
            let mut ret = vec![cur.clone()];
            while let Some(n) = cur.successor() {
                if n.key() < k_r {
                    ret.push(n.clone());
                    cur = n;
                } else {
                    break;
                }
            }
            ret
        }
    }
    /// returns ((a, b), c) such that a: [x| x.key<k], b: [x| x.key>k]
    /// and c is present if c.key == k
    pub fn split_by_key(&self, k: K) -> ((Self, Self), Option<Self>) {
        //insert node with a sentil lowest priority so it's at the root

        let (root, exists) = self.insert_with_priority(k, Default::default(), f32::NEG_INFINITY);

        //remove the root node and return 2 child nodes
        let l = root.child_l();
        let r = root.child_r();

        let t_l = match l {
            Some(x) => {
                x.0.borrow_mut().parent = NodePtrWk(Weak::new());
                root.0.borrow_mut().children.0 = None;
                x
            }
            None => NodePtr::new(),
        };

        let t_r = match r {
            Some(x) => {
                x.0.borrow_mut().parent = NodePtrWk(Weak::new());
                root.0.borrow_mut().children.1 = None;
                x
            }
            None => NodePtr::new(),
        };

        if exists {
            ((t_l, t_r), Some(root.clone()))
        } else {
            ((t_l, t_r), None)
        }
    }

    /// assumes a.merge(b) is such that keys of a < keys of b and returns merged tree
    pub fn merge_contiguous(&self, other: Self) -> Self {
        if self.is_empty() {
            other
        } else if other.is_empty() {
            self.clone()
        } else {
            let n = NodePtr::new();
            {
                n.0.borrow_mut().prio = f32::INFINITY;
            }

            n.link_left(&Some(self.clone()));
            n.link_right(&Some(other));

            match n.fixdown_priority() {
                Some(_new_root) => n.remove(),
                _ => {
                    panic!("unexpected root");
                }
            }
        }
    }

    /// returns the union of 2 trees
    pub fn union(&self, other: Self) -> Self {
        if self.is_empty() && other.is_empty() {
            return self.clone();
        } else if self.is_empty() {
            return other;
        } else if other.is_empty() {
            return self.clone();
        }

        let (a, b) = if self.prio() < other.prio() {
            (self.clone(), other.clone())
        } else {
            (other.clone(), self.clone())
        };

        b.0.borrow_mut().parent = NodePtrWk(Weak::new());

        let k = a.key();
        let ((t1, t2), _) = b.split_by_key(k);

        let l2 = if t1.is_empty() { None } else { Some(t1) };
        let r2 = if t2.is_empty() { None } else { Some(t2) };

        let l = a.child_l();
        let r = a.child_r();

        match (&l, &l2) {
            (Some(x), Some(y)) => {
                let ll = x.union(y.clone());
                if ll.is_empty() {
                    a.link_left(&None);
                } else {
                    a.link_left(&Some(ll));
                }
            }
            (Some(x), None) => {
                a.link_left(&Some(x.clone()));
            }
            (None, Some(y)) => {
                a.link_left(&Some(y.clone()));
            }
            (None, None) => {
                a.link_left(&None);
            }
        }

        match (&r, &r2) {
            (Some(x), Some(y)) => {
                let rr = x.union(y.clone());
                if rr.is_empty() {
                    a.link_right(&None);
                } else {
                    a.link_right(&Some(rr));
                }
            }
            (Some(x), None) => {
                a.link_right(&Some(x.clone()));
            }
            (None, Some(y)) => {
                a.link_right(&Some(y.clone()));
            }
            (None, None) => {
                a.link_right(&None);
            }
        }

        a
    }

    /// returns the intersection of 2 trees
    pub fn intersect(&self, other: Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return NodePtr::new();
        }

        let (mut a, b) = if self.prio() < other.prio() {
            (self.clone(), other.clone())
        } else {
            (other.clone(), self.clone())
        };

        b.0.borrow_mut().parent = NodePtrWk(Weak::new());

        let k = a.key();
        let ((t1, t2), exists) = b.split_by_key(k);

        let l2 = if t1.is_empty() { None } else { Some(t1) };
        let r2 = if t2.is_empty() { None } else { Some(t2) };

        let l = a.child_l();
        let r = a.child_r();

        match exists {
            Some(m) => {
                //intersection of current node is non-empty

                a = m.clone();

                match (&l, &l2) {
                    (Some(x), Some(y)) => {
                        let ll = x.intersect(y.clone());

                        if ll.is_empty() {
                            a.link_left(&None);
                        } else {
                            a.link_left(&Some(ll));
                        }
                    }
                    _ => {
                        a.link_left(&None);
                    }
                }

                match (&r, &r2) {
                    (Some(x), Some(y)) => {
                        let rr = x.intersect(y.clone());

                        if rr.is_empty() {
                            a.link_right(&None);
                        } else {
                            a.link_right(&Some(rr));
                        }
                    }
                    _ => {
                        a.link_right(&None);
                    }
                }

                a
            }
            _ => {
                //intersection of current node is empty

                let left_branch = match (&l, &l2) {
                    (Some(x), Some(y)) => {
                        let ll = x.intersect(y.clone());
                        if ll.is_empty() {
                            None
                        } else {
                            Some(ll)
                        }
                    }
                    _ => None,
                };

                let right_branch = match (&r, &r2) {
                    (Some(x), Some(y)) => {
                        let rr = x.intersect(y.clone());
                        if rr.is_empty() {
                            None
                        } else {
                            Some(rr)
                        }
                    }
                    _ => None,
                };

                match (left_branch, right_branch) {
                    (Some(l), Some(r)) => l.merge_contiguous(r.clone()),
                    (Some(l), None) => l,
                    (None, Some(r)) => r,
                    (None, None) => NodePtr::new(),
                }
            }
        }
    }

    /// returns min, max, avg depths
    pub fn dbg_depth(&self) -> (f32, f32, f32) {
        if self.0.borrow().invalid {
            return (0., 0., 0.);
        }

        let mut q = vec![self.clone()];

        let mut hm: HashMap<usize, usize> = HashMap::new();

        let mut depth_prev = 0;

        let mut rec = vec![];

        while !q.is_empty() {
            let n = q.pop().unwrap();
            let key = Rc::into_raw(n.0.clone()) as usize;

            if !hm.contains_key(&key) {
                hm.insert(key, depth_prev + 1);
                depth_prev += 1;
                q.push(n.clone());
                match n.child_l() {
                    Some(x) => {
                        q.push(x);
                    }
                    _ => {}
                }
                match n.child_r() {
                    Some(x) => {
                        q.push(x);
                    }
                    _ => {}
                }
            } else {
                match hm.get_mut(&key) {
                    Some(x) => {
                        if n.is_leaf() {
                            rec.push(*x);
                        }
                        depth_prev = *x;
                        depth_prev -= 1;
                    }
                    _ => {}
                }
            }
        }

        let avg = rec.iter().fold(0, |acc, x| acc + x) as f32 / rec.len() as f32;
        (
            *rec.iter().min().unwrap() as f32,
            *rec.iter().max().unwrap() as f32,
            avg,
        )
    }
}

#[test]
fn test_treap_search() {
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
        children: (None, None),
        invalid: false,
    };

    let mut n1 = Node {
        key: 2.,
        prio: 0.,
        val: 2,
        parent: Default::default(),
        children: (None, None),
        invalid: false,
    };

    let n2 = Node {
        key: 7.,
        prio: 0.,
        val: 7,
        parent: Default::default(),
        children: (None, None),
        invalid: false,
    };

    let n3 = Node {
        key: -1.,
        prio: 0.,
        val: -1,
        parent: Default::default(),
        children: (None, None),
        invalid: false,
    };

    let n4 = Node {
        key: 3.,
        prio: 0.,
        val: 3,
        parent: Default::default(),
        children: (None, None),
        invalid: false,
    };

    let r3 = NodePtr(Rc::new(RefCell::new(n3)));
    let r4 = NodePtr(Rc::new(RefCell::new(n4)));
    n1.children.0 = Some(r3);
    n1.children.1 = Some(r4);
    let r1 = NodePtr(Rc::new(RefCell::new(n1)));
    {
        let weak1 = NodePtrWk::from(&r1);
        let weak2 = weak1.clone();
        r1.child_l().as_ref().unwrap().0.borrow_mut().parent = weak1;
        r1.child_r().as_ref().unwrap().0.borrow_mut().parent = weak2;
    }
    let r2 = NodePtr(Rc::new(RefCell::new(n2)));
    n0.children.0 = Some(r1);
    n0.children.1 = Some(r2);
    let r0 = NodePtr(Rc::new(RefCell::new(n0)));

    {
        let weak1 = NodePtrWk::from(&r0);
        let weak2 = weak1.clone();

        r0.child_l().as_ref().unwrap().0.borrow_mut().parent = weak1;
        r0.child_r().as_ref().unwrap().0.borrow_mut().parent = weak2;
    }

    match r0.child_branch(&r0.child_l()) {
        ChildBranch::Left => {}
        _ => {
            panic!("incorrect child branch");
        }
    }

    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b || a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    match r0.search(3.) {
        SearchResult::Exact(x) => {
            assert!(equal_f32(x.0.borrow().key, 4.));

            assert!(equal_f32(x.get_root().0.borrow().key, 5.));
        }
        _ => {
            panic!();
        }
    }

    match r0.search(4.) {
        SearchResult::Nearest(x) => {
            assert!(equal_f32(x.0.borrow().key, 4.));
        }
        _ => {
            panic!();
        }
    }

    match r0.search(2.) {
        SearchResult::Exact(x) => {
            assert!(equal_f32(x.0.borrow().key, 2.));

            match x.child_branch(&x.child_r()) {
                ChildBranch::Right => {
                    assert!(equal_f32(x.child_r().unwrap().0.borrow().key, 3.));
                }
                _ => {
                    panic!("incorrect child branch");
                }
            }
        }
        _ => {
            panic!();
        }
    }

    match r0.search(-1.) {
        SearchResult::Exact(x) => {
            assert!(equal_f32(x.0.borrow().key, -1.));
        }
        _ => {
            panic!();
        }
    }
    match r0.search(-10.) {
        SearchResult::Nearest(x) => {
            assert!(equal_f32(x.0.borrow().key, -1.));
        }
        _ => {
            panic!();
        }
    }
    match r0.search(5.) {
        SearchResult::Exact(x) => {
            assert!(equal_f32(x.0.borrow().key, 5.));
        }
        _ => {
            panic!();
        }
    }

    match r0.search(6.) {
        SearchResult::Nearest(x) => {
            assert!(equal_f32(x.0.borrow().key, 5.));
        }
        _ => {
            panic!();
        }
    }

    match r0.search(10.) {
        SearchResult::Nearest(x) => {
            assert!(equal_f32(x.0.borrow().key, 7.));
        }
        _ => {
            panic!();
        }
    }

    match r0.search(2.) {
        SearchResult::Exact(x) => match x.search(-10.) {
            SearchResult::Nearest(y) => {
                assert!(equal_f32(y.0.borrow().key, -1.));
                assert_eq!(y.0.borrow().val, -1);
                y.0.borrow_mut().val = 100;
            }
            _ => {
                panic!();
            }
        },
        _ => {
            panic!();
        }
    }

    match r0.search(-1.) {
        SearchResult::Exact(x) => {
            assert!(equal_f32(x.0.borrow().key, -1.));
            assert_eq!(x.0.borrow().val, 100);

            let parent_key =
                x.0.borrow()
                    .parent
                    .0
                    .upgrade()
                    .expect("parent invalid")
                    .borrow()
                    .key;
            assert!(equal_f32(parent_key, 2.));
        }
        _ => {
            panic!();
        }
    }
}

#[test]
fn test_treap_rotate_0() {
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
        children: (None, None),
        invalid: false,
    };

    let mut n1 = Node {
        key: 2.,
        prio: 0.,
        val: 2,
        parent: Default::default(),
        children: (None, None),
        invalid: false,
    };

    let n2 = Node {
        key: 7.,
        prio: 0.,
        val: 7,
        parent: Default::default(),
        children: (None, None),
        invalid: false,
    };

    let n3 = Node {
        key: -1.,
        prio: 0.,
        val: -1,
        parent: Default::default(),
        children: (None, None),
        invalid: false,
    };

    let n4 = Node {
        key: 3.,
        prio: 0.,
        val: 3,
        parent: Default::default(),
        children: (None, None),
        invalid: false,
    };

    let r3 = NodePtr(Rc::new(RefCell::new(n3)));
    let r4 = NodePtr(Rc::new(RefCell::new(n4)));
    n1.children.0 = Some(r3);
    n1.children.1 = Some(r4);
    let r1 = NodePtr(Rc::new(RefCell::new(n1)));
    {
        let weak1 = NodePtrWk::from(&r1);
        let weak2 = weak1.clone();
        r1.child_l().as_ref().unwrap().0.borrow_mut().parent = weak1;
        r1.child_r().as_ref().unwrap().0.borrow_mut().parent = weak2;
    }
    let r2 = NodePtr(Rc::new(RefCell::new(n2)));
    n0.children.0 = Some(r1);
    n0.children.1 = Some(r2);
    let r0 = NodePtr(Rc::new(RefCell::new(n0)));

    {
        let weak1 = NodePtrWk::from(&r0);
        let weak2 = weak1.clone();

        r0.child_l().as_ref().unwrap().0.borrow_mut().parent = weak1;
        r0.child_r().as_ref().unwrap().0.borrow_mut().parent = weak2;
    }

    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    //left rotate up node with key=2
    match r0.search(2.) {
        SearchResult::Exact(x) => {
            assert!(equal_f32(x.0.borrow().key, 2.));

            let parent_key =
                x.0.borrow()
                    .parent
                    .0
                    .upgrade()
                    .expect("parent invalid")
                    .borrow()
                    .key;
            assert!(equal_f32(parent_key, 5.));

            x.rot_up_left();

            assert!(x.0.borrow().parent.0.upgrade().is_none());

            assert!(equal_f32(
                x.0.borrow().children.1.as_ref().unwrap().0.borrow().key,
                5.
            ));

            let x_r_l_key =
                x.0.borrow()
                    .children
                    .1
                    .as_ref()
                    .unwrap()
                    .0
                    .borrow()
                    .children
                    .0
                    .as_ref()
                    .unwrap()
                    .0
                    .borrow()
                    .key;

            assert!(equal_f32(x_r_l_key, 3.));
        }
        _ => {
            panic!();
        }
    }
}

#[test]
fn test_treap_rotate_1() {
    //         n0(5)
    //        /  \
    //     (2)n1  n2(7)
    //      / \
    //(-1)n3  n4(3)

    let r0 = {
        let mut n0 = Node {
            key: 5.,
            prio: 0.,
            val: 5,
            parent: Default::default(),
            children: (None, None),
            invalid: false,
        };

        let mut n1 = Node {
            key: 2.,
            prio: 0.,
            val: 2,
            parent: Default::default(),
            children: (None, None),
            invalid: false,
        };

        let n2 = Node {
            key: 7.,
            prio: 0.,
            val: 7,
            parent: Default::default(),
            children: (None, None),
            invalid: false,
        };

        let n3 = Node {
            key: -1.,
            prio: 0.,
            val: -1,
            parent: Default::default(),
            children: (None, None),
            invalid: false,
        };

        let n4 = Node {
            key: 3.,
            prio: 0.,
            val: 3,
            parent: Default::default(),
            children: (None, None),
            invalid: false,
        };

        let r3 = NodePtr(Rc::new(RefCell::new(n3)));
        let r4 = NodePtr(Rc::new(RefCell::new(n4)));
        n1.children.0 = Some(r3);
        n1.children.1 = Some(r4);
        let r1 = NodePtr(Rc::new(RefCell::new(n1)));
        {
            let weak1 = NodePtrWk::from(&r1);
            let weak2 = weak1.clone();
            r1.child_l().as_ref().unwrap().0.borrow_mut().parent = weak1;
            r1.child_r().as_ref().unwrap().0.borrow_mut().parent = weak2;
        }
        let r2 = NodePtr(Rc::new(RefCell::new(n2)));
        n0.children.0 = Some(r1);
        n0.children.1 = Some(r2);

        // let mut r0 = NodePtr(Rc::new( RefCell::new(n0) ));
        NodePtr(Rc::new(RefCell::new(n0)))
    };

    {
        let weak1 = NodePtrWk::from(&r0);
        let weak2 = weak1.clone();

        r0.child_l().as_ref().unwrap().0.borrow_mut().parent = weak1;
        r0.child_r().as_ref().unwrap().0.borrow_mut().parent = weak2;
    }

    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    //left rotate up node with key=-1
    match r0.search(-1.) {
        SearchResult::Exact(x) => {
            assert!(equal_f32(x.0.borrow().key, -1.));

            let parent_key =
                x.0.borrow()
                    .parent
                    .0
                    .upgrade()
                    .expect("parent invalid")
                    .borrow()
                    .key;
            assert!(equal_f32(parent_key, 2.));

            x.rot_up_left();

            assert!(equal_f32(
                x.0.borrow()
                    .parent
                    .0
                    .upgrade()
                    .expect("parent after rotate")
                    .borrow()
                    .key,
                5.
            ));

            let x_r_key = x.0.borrow().children.1.as_ref().unwrap().0.borrow().key;

            assert!(equal_f32(x_r_key, 2.));
        }
        _ => {
            panic!();
        }
    }
}

#[test]
fn test_treap_insert_with_priority() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    // test sequence

    //            3

    //            3
    //         1

    //            8
    //         3
    //      1

    //            6
    //         3     8
    //      1

    let t = NodePtr::new();

    let (t1, _) = t.insert_with_priority(3., 3, 50.);

    assert!(equal_f32(t1.0.borrow().key, 3.));

    let (t2, _) = t1.insert_with_priority(1., 1, 75.);

    assert!(equal_f32(t2.0.borrow().key, 3.));

    let (t3, _) = t2.insert_with_priority(8., 8, 30.);

    assert!(equal_f32(t3.0.borrow().key, 8.));

    let (t4, _) = t3.insert_with_priority(6., 6, -20.);

    assert!(equal_f32(t4.0.borrow().key, 6.));
}

#[test]
fn test_treap_insert_with_priority_replacement() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    // test sequence

    //            3

    //            3
    //         1

    //            8
    //         3
    //      1

    //            6
    //         3     8
    //      1

    //after priority replacement for 6
    //            8
    //         3
    //      1    6

    //after priority replacement for 1
    //            1
    //               8
    //            3
    //         6

    let t = NodePtr::new();

    let (t1, _) = t.insert_with_priority(3., 3, 50.);

    assert!(equal_f32(t1.0.borrow().key, 3.));

    let (t2, _) = t1.insert_with_priority(1., 1, 75.);

    assert!(equal_f32(t2.0.borrow().key, 3.));

    let (t3, _) = t2.insert_with_priority(1., 1, 60.);

    assert!(equal_f32(t3.0.borrow().key, 3.));

    let (t4, _) = t3.insert_with_priority(8., 8, 30.);

    assert!(equal_f32(t4.0.borrow().key, 8.));

    let (t5, _) = t4.insert_with_priority(6., 6, -20.);

    assert!(equal_f32(t5.0.borrow().key, 6.));

    let (t6, _) = t5.insert_with_priority(6., 6, 100.);

    assert!(equal_f32(t6.0.borrow().key, 8.));

    let (t7, _) = t6.insert_with_priority(1., 1, 0.);

    assert!(equal_f32(t7.0.borrow().key, 1.));
}

#[test]
fn test_treap_successor() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    // test sequence

    //            3

    //            3
    //         1

    //            8
    //         3
    //      1

    //            6
    //         3     8
    //      1

    let t = NodePtr::new();

    let (t1, _) = t.insert_with_priority(3., 3, 50.);
    let (t2, _) = t1.insert_with_priority(1., 1, 75.);
    let (t3, _) = t2.insert_with_priority(8., 8, 30.);
    let (t4, _) = t3.insert_with_priority(6., 6, -20.);

    match t4.search(1.) {
        SearchResult::Exact(x) => {
            assert!(equal_f32(x.0.borrow().key, 1.));
            let mut keys = vec![x.key()];

            let mut cur = x;
            while let Some(y) = cur.successor() {
                keys.push(y.key());
                cur = y;
            }

            let expected = vec![1, 3, 6, 8];

            assert_eq!(expected.len(), keys.len());
            expected.iter().zip(keys.iter()).for_each(|(a, b)| {
                assert!(equal_f32(*a as f32, *b));
            });
        }
        _ => {
            panic!();
        }
    }
}

#[test]
fn test_treap_predecessor() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    //            6
    //         3     8
    //      1       7

    let t = NodePtr::new();

    let (t1, _) = t.insert_with_priority(3., 3, 50.);
    let (t2, _) = t1.insert_with_priority(1., 1, 75.);
    let (t3, _) = t2.insert_with_priority(8., 8, 30.);
    let (t4, _) = t3.insert_with_priority(6., 6, -20.);
    let (t5, _) = t4.insert_with_priority(7., 7, 100.);

    match t5.search(8.) {
        SearchResult::Exact(x) => {
            assert!(equal_f32(x.0.borrow().key, 8.));
            let mut keys = vec![x.key()];

            let mut cur = x;
            while let Some(y) = cur.predecessor() {
                keys.push(y.key());
                cur = y;
            }

            let expected = vec![8, 7, 6, 3, 1];

            assert_eq!(expected.len(), keys.len());
            expected.iter().zip(keys.iter()).for_each(|(a, b)| {
                assert!(equal_f32(*a as f32, *b));
            });
        }
        _ => {
            panic!();
        }
    }
}

#[test]
fn test_treap_query_key_range() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    let items = vec![1, 3, 6, 7, 8];

    //            6
    //         3     8
    //      1       7

    let t = NodePtr::new();

    let (t1, _) = t.insert_with_priority(3., 3, 50.);
    let (t2, _) = t1.insert_with_priority(1., 1, 75.);
    let (t3, _) = t2.insert_with_priority(8., 8, 30.);
    let (t4, _) = t3.insert_with_priority(6., 6, -20.);
    let (t5, _) = t4.insert_with_priority(7., 7, 100.);

    {
        let v = t5
            .query_key_range(-10., 10.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), items.len());
        items
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }

    {
        let v = t5
            .query_key_range(1., 2.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        let expected = items
            .iter()
            .cloned()
            .filter(|x| (*x as f32) < 2. && !((*x as f32) < 1.))
            .collect::<Vec<_>>();

        assert_eq!(v.len(), expected.len());

        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }

    {
        let v = t5
            .query_key_range(1., 1.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), 0);
    }

    {
        let v = t5
            .query_key_range(2., 7.5)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        let expected = items
            .iter()
            .cloned()
            .filter(|x| !((*x as f32) < 2.) && (*x as f32) < 7.5)
            .collect::<Vec<_>>();

        assert_eq!(v.len(), expected.len());

        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }
}

#[test]
fn test_treap_remove() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    let items = vec![1, 3, 6, 7, 8];

    //            6
    //         3     8
    //      1       7

    // remove 6
    //            8
    //         3     7
    //      1

    // remove 8
    //            3
    //         1     7

    let t5 = {
        let t = NodePtr::new();
        let (t1, _) = t.insert_with_priority(3., 3, 50.);
        let (t2, _) = t1.insert_with_priority(1., 1, 75.);
        let (t3, _) = t2.insert_with_priority(8., 8, 30.);
        let (t4, _) = t3.insert_with_priority(6., 6, -20.);
        t4.insert_with_priority(7., 7, 100.).0
    };

    assert!(equal_f32(t5.key(), 6.));

    {
        let v = t5
            .query_key_range(-10., 10.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), items.len());
        items
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }

    let t6 = match t5.search(6.) {
        SearchResult::Exact(x) => x.remove(),
        _ => {
            panic!("search unexpected result");
        }
    };

    assert!(equal_f32(t6.key(), 8.));

    {
        let v = t6
            .query_key_range(-10., 10.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        let expected = items
            .iter()
            .cloned()
            .filter(|x| *x != 6)
            .collect::<Vec<_>>();
        assert_eq!(v.len(), expected.len());
        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }

    let t7 = t6.remove();

    assert!(equal_f32(t7.key(), 3.));

    let t8 = t7.remove();

    assert!(equal_f32(t8.key(), 1.));

    let t9 = t8.remove();

    assert!(equal_f32(t9.key(), 7.));

    let t10 = t9.remove();

    assert_eq!(true, t10.0.borrow().invalid);

    //query on empty tree
    {
        match t10.search(0.) {
            SearchResult::Empty => {}
            _ => {
                panic!("unexpected search result");
            }
        }
        let v = t10
            .query_key_range(-10., 10.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), 0);
    }
}

#[test]
fn test_treap_insert_remove_by_key() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    let mut t = NodePtr::new();

    {
        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), 0);
    }

    let items = vec![56, -45, 1, 6, 9, -30, 7, -9, 12, 77, -25];
    for i in items.iter() {
        t = t.insert(*i as f32, *i).0;
    }

    let mut expected = items.clone();
    expected.sort();

    {
        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), expected.len());

        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }

    let l = items.len() / 2;
    for i in expected.iter().take(l) {
        t = t.remove_by_key(*i as f32);
    }

    {
        let f = expected.iter().skip(l).cloned().collect::<Vec<_>>();

        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), f.len());

        f.iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }
}

#[test]
fn test_treap_remove_key_range() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    let mut t = NodePtr::new();

    {
        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), 0);
    }

    let items = vec![56, -45, 1, 6, 9, -30, 7, -9, 12, 77, -25];
    for i in items.iter() {
        t = t.insert(*i as f32, *i).0;
    }

    let mut expected = items.clone();
    expected.sort();

    {
        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), expected.len());

        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }

    t = t.remove_by_key_range(-9., 56.);

    {
        let f = expected
            .iter()
            .cloned()
            .filter(|x| *x < -9 || *x >= 56)
            .collect::<Vec<_>>();

        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), f.len());

        f.iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }
}

#[test]
fn test_treap_split_by_key() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    let mut t = NodePtr::new();

    {
        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), 0);
    }

    let items = vec![56, -45, 1, 6, 9, -30, 7, -9, 12, 77, -25];
    for i in items.iter() {
        t = t.insert(*i as f32, *i).0;
    }

    let mut expected = items.clone();
    expected.sort();

    {
        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), expected.len());

        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }

    let ((t1, t2), _) = t.split_by_key(0.);

    {
        let f = expected
            .iter()
            .cloned()
            .filter(|x| *x < 0)
            .collect::<Vec<_>>();

        let v = t1
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), f.len());

        f.iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }

    {
        let f = expected
            .iter()
            .cloned()
            .filter(|x| *x > 0)
            .collect::<Vec<_>>();

        let v = t2
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), f.len());

        f.iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }
}

#[test]
fn test_treap_split_merge() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    let mut t = NodePtr::new();

    {
        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), 0);
    }

    let items = vec![56, -45, 1, 6, 9, -30, 7, -9, 12, 77, -25];
    for i in items.iter() {
        t = t.insert(*i as f32, *i).0;
    }

    let mut expected = items.clone();
    expected.sort();

    {
        let v = t
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), expected.len());

        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }

    let ((t1, t2), _) = t.split_by_key(0.);

    let t3 = t1.merge_contiguous(t2);

    {
        let v = t3
            .query_key_range(-100., 100.)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), expected.len());

        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
    }
}

#[test]
fn test_treap_depth() {
    let count = 100;
    let nums = (0..count)
        .map(|_| {
            let mut rng = rand::thread_rng();
            let n: f32 = rng.gen_range(-10_000_000., 10_000_000.);
            n
        })
        .collect::<Vec<_>>();

    let mut t = NodePtr::new();

    for i in nums.iter() {
        t = t.insert(*i, 0i32).0;
    }

    let (d_min, d_max, d_avg) = t.dbg_depth();

    println!("dmin: {:?}, dmax: {:?}, davg: {:?}", &d_min, &d_max, &d_avg);

    println!("expected depth: {}", (count as f32).log2() + 1.);

    assert!(5. * ((count as f32).log2() + 1.) > d_avg);
}

#[test]
fn test_treap_insert_remove_stress() {
    let count = 10_000;
    let nums = (0..count)
        .map(|_| {
            let mut rng = rand::thread_rng();
            let n: f32 = rng.gen_range(-10_000_000., 10_000_000.);
            n
        })
        .collect::<Vec<_>>();

    let mut t = NodePtr::new();

    let t0 = Local::now();

    for i in nums.iter() {
        t = t.insert(*i, 0i32).0;
    }

    let t1 = Local::now();

    for i in nums.iter() {
        t = t.remove_by_key(*i);
    }

    let t2 = Local::now();

    assert!(t.is_empty());

    let t_ins = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
    let t_del = t2.signed_duration_since(t1).num_microseconds().unwrap() as f64;

    println!("{} us/ins", t_ins as f32 / count as f32);
    println!("{} us/del", t_del as f32 / count as f32);
}

#[test]
fn test_treap_insert_remove_range_stress() {
    let count = 10_000;
    let nums = (0..count)
        .map(|_| {
            let mut rng = rand::thread_rng();
            let n: f32 = rng.gen_range(-10_000_000., 10_000_000.);
            n
        })
        .collect::<Vec<_>>();

    let mut t = NodePtr::new();

    let t0 = Local::now();

    for i in nums.iter() {
        t = t.insert(*i, 0i32).0;
    }

    let t1 = Local::now();

    t = t.remove_by_key_range(-1e10, 1e10);

    let t2 = Local::now();

    assert!(t.is_empty());

    let t_ins = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
    let t_del = t2.signed_duration_since(t1).num_microseconds().unwrap() as f64;

    println!("{} us/ins", t_ins as f32 / count as f32);
    println!("{} us/del", t_del as f32 / count as f32);
}

#[test]
fn test_treap_union_empty() {
    fn equal_f32(a: f32, b: f32) -> bool {
        if a - 1e-4 < b && a + 1e-4 > b {
            true
        } else {
            false
        }
    }

    {
        let t1: NodePtr<f32, i32> = NodePtr::new();
        let t2 = NodePtr::new();
        let t3 = t1.union(t2);
        assert!(t3.is_empty());
    }

    {
        let count = 10;

        let va = (0..count).map(|x| x * 2).collect::<Vec<_>>();

        let mut t1 = NodePtr::new();
        let t2 = NodePtr::new();

        for i in va.iter() {
            t1 = t1.insert(*i as f32, *i).0;
        }

        let t3 = t1.union(t2);

        {
            let v = t3
                .query_key_range(-1e10, 1e10)
                .iter()
                .map(|x| x.key())
                .collect::<Vec<_>>();

            assert_eq!(v.len(), va.len());

            va.iter()
                .zip(v.iter())
                .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
        }
    }

    {
        let count = 10;

        let vb = (0..count).map(|x| x * 2 + 1).collect::<Vec<_>>();

        let t1 = NodePtr::new();
        let mut t2 = NodePtr::new();

        for i in vb.iter() {
            t2 = t2.insert(*i as f32, *i).0;
        }

        let t3 = t1.union(t2);

        {
            let v = t3
                .query_key_range(-1e10, 1e10)
                .iter()
                .map(|x| x.key())
                .collect::<Vec<_>>();

            assert_eq!(v.len(), vb.len());

            vb.iter()
                .zip(v.iter())
                .for_each(|(a, b)| assert!(equal_f32(*a as f32, *b)));
        }
    }
}

#[test]
fn test_treap_union_nonempty() {
    let count = 1000;

    let va = (0..count).map(|x| (x * 2)).collect::<Vec<i32>>();
    let vb = (0..count).map(|x| (x * 2 + 1)).collect::<Vec<i32>>();

    let mut t1: NodePtr<i32, i32> = NodePtr::new();
    let mut t2: NodePtr<i32, i32> = NodePtr::new();

    for i in va.iter() {
        t1 = t1.insert(*i, *i).0;
    }

    for i in vb.iter() {
        t2 = t2.insert(*i, *i).0;
    }

    {
        let v = t1
            .query_key_range(-10_000_000, 10_000_000)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), va.len());

        va.iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert_eq!(*a, *b));
    }

    {
        let v = t2
            .query_key_range(-10_000_000, 10_000_000)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), vb.len());

        vb.iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert_eq!(*a, *b));
    }

    let ck1 = Local::now();

    let t3 = t2.union(t1);

    let ck2 = Local::now();

    {
        let v = t3
            .query_key_range(-10_000_000, 10_000_000)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        let mut combined = va.iter().chain(vb.iter()).cloned().collect::<Vec<_>>();

        combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        assert_eq!(v.len(), combined.len());

        combined
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert_eq!(*a, *b));
    }

    let t_union = ck2.signed_duration_since(ck1).num_microseconds().unwrap() as f64;
    println!("union of sizes({},{}): {} us", count, count, t_union);
}

#[test]
fn test_treap_intersect_empty() {
    {
        let t1: NodePtr<f32, i32> = NodePtr::new();
        let t2 = NodePtr::new();
        let t3 = t1.intersect(t2);
        assert!(t3.is_empty());
    }

    {
        let count = 10;

        let va = (0..count).map(|x| x * 2).collect::<Vec<_>>();

        let mut t1 = NodePtr::new();
        let t2 = NodePtr::new();

        for i in va.iter() {
            t1 = t1.insert(*i as f32, *i).0;
        }

        let t3 = t1.intersect(t2);

        {
            let v = t3
                .query_key_range(-1e10, 1e10)
                .iter()
                .map(|x| x.key())
                .collect::<Vec<_>>();

            assert_eq!(v.len(), 0);
        }
    }

    {
        let count = 10;

        let vb = (0..count).map(|x| x * 2 + 1).collect::<Vec<_>>();

        let t1 = NodePtr::new();
        let mut t2 = NodePtr::new();

        for i in vb.iter() {
            t2 = t2.insert(*i as f32, *i).0;
        }

        let t3 = t1.intersect(t2);

        {
            let v = t3
                .query_key_range(-1e10, 1e10)
                .iter()
                .map(|x| x.key())
                .collect::<Vec<_>>();

            assert_eq!(v.len(), 0);
        }
    }
}

#[test]
fn test_treap_intersect_nonempty() {
    let count = 20;

    let va = (0..count).map(|x| (x)).collect::<Vec<i32>>();
    let vb = (0..count / 2).rev().map(|x| (x)).collect::<Vec<i32>>();

    let mut t1: NodePtr<i32, i32> = NodePtr::new();
    let mut t2: NodePtr<i32, i32> = NodePtr::new();

    for i in va.iter() {
        t1 = t1.insert(*i, *i).0;
    }

    for i in vb.iter() {
        t2 = t2.insert(*i, *i).0;
    }

    {
        let v = t1
            .query_key_range(-10_000_000, 10_000_000)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), va.len());

        va.iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert_eq!(*a, *b));
    }

    {
        let v = t2
            .query_key_range(-10_000_000, 10_000_000)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), vb.len());

        let mut vb_expected = vb.clone();
        vb_expected.sort();

        vb_expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert_eq!(*a, *b));
    }

    let ck1 = Local::now();

    let t3 = t2.intersect(t1);

    let ck2 = Local::now();

    {
        let v = t3
            .query_key_range(-10_000_000, 10_000_000)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        let mut expected = vb.clone();
        expected.sort();

        assert_eq!(v.len(), expected.len());

        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert_eq!(*a, *b));
    }

    let t_elapse = ck2.signed_duration_since(ck1).num_microseconds().unwrap() as f64;
    println!("intersect of sizes({},{}): {} us", count, count, t_elapse);
}

#[test]
fn test_treap_intersect_nonempty_2() {
    let count = 20;

    let va = (0..count).map(|x| (x)).collect::<Vec<i32>>();
    let vb = [100, -50, 75, 45, 15, -10];

    let mut t1: NodePtr<i32, i32> = NodePtr::new();
    let mut t2: NodePtr<i32, i32> = NodePtr::new();

    for i in va.iter() {
        t1 = t1.insert(*i, *i).0;
    }

    for i in vb.iter() {
        t2 = t2.insert(*i, *i).0;
    }

    {
        let v = t1
            .query_key_range(-10_000_000, 10_000_000)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), va.len());

        va.iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert_eq!(*a, *b));
    }

    {
        let v = t2
            .query_key_range(-10_000_000, 10_000_000)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        assert_eq!(v.len(), vb.len());

        let mut vb_expected = vb.clone();
        vb_expected.sort();

        vb_expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert_eq!(*a, *b));
    }

    let ck1 = Local::now();

    let t3 = t2.intersect(t1);

    let ck2 = Local::now();

    {
        let v = t3
            .query_key_range(-10_000_000, 10_000_000)
            .iter()
            .map(|x| x.key())
            .collect::<Vec<_>>();

        let mut expected = vb
            .iter()
            .cloned()
            .filter(|x| *x >= 0 && *x < 20)
            .collect::<Vec<_>>();
        expected.sort();

        assert_eq!(v.len(), expected.len());

        expected
            .iter()
            .zip(v.iter())
            .for_each(|(a, b)| assert_eq!(*a, *b));
    }

    let t_elapse = ck2.signed_duration_since(ck1).num_microseconds().unwrap() as f64;
    println!("intersect of sizes({},{}): {} us", count, count, t_elapse);
}
