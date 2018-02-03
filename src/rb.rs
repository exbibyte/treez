///array based red black tree implementation

extern crate num;

use std::isize;
use std::collections::HashMap;

use self::num::Bounded;

#[allow(dead_code)]
#[derive(Debug,Copy,Clone)]
enum Colour {
    Red,
    Black,
}
///internal tree node
#[derive(Debug,Copy,Clone)]
struct Node < K, V > where K: Ord + Default + Bounded + Clone, V : Default + Clone {
    _key: K,
    _colour: Colour,
    _parent: isize,
    _child_l: isize,
    _child_r: isize,
    _val: V,
    _index: isize,
}

impl < K, V > Default for Node< K, V > where K: Ord + Default + Bounded + Clone, V : Default + Clone {
    fn default() -> Node< K, V > {
        Node {
            _key: Default::default(),
            _colour: Colour::Red,
            _parent: -1isize,
            _child_l: -1isize,
            _child_r: -1isize,
            _val: Default::default(),
            _index: -1isize,
        }
    }
}
///vector indexed red-black tree implementation
pub struct TreeRb< K, V > where K: Ord + Default + Bounded + Clone, V : Default + Clone {
    _root: isize,
    _buf: Vec< Node< K, V > >,
    _sentinil: Node< K, V >,
    _freelist: Vec<isize>,
    _leaf_remove_index: isize, //dummy leaf for fixup operation
}

impl < K, V > TreeRb< K, V > where K: Ord + Default + Bounded + Clone, V : Default + Clone {
    pub fn new() -> TreeRb< K, V > {
        TreeRb {
            _root: -1isize,
            _buf: vec![],
            _sentinil: Node {
                _colour: Colour::Black,
                _parent: -1isize,
                ..Default::default()
            },
            _freelist: vec![],
            _leaf_remove_index: -1isize,
        }
    }
    pub fn len( & self ) -> usize {
        self._buf.len() - self._freelist.len()
    }
    pub fn len_freelist( & self ) -> usize {
        self._freelist.len()
    }
    pub fn is_empty( & self ) -> bool {
        self._buf.len() - self._freelist.len() == 0
    }
    pub fn insert( & mut self, key: K, val: V ) -> Option< V > {
        let mut x = self._root;
        let mut prev = -1isize;
        while x != -1 {
            prev = x;
            if key < self._buf[x as usize]._key {
                x = self._buf[x as usize]._child_l;
            } else if key > self._buf[x as usize]._key {
                x = self._buf[x as usize]._child_r;
            } else {
                //found equal key, then replace existing val of the node, no need to fixup
                let val_prev = self._buf[prev as usize]._val.clone();
                self._buf[prev as usize]._val = val;
                return Some( val_prev )
            }
        }
        let n_index = self._buf.len();
        let n = Node  {
            _key: key.clone(),
            _colour: Colour::Red,
            _parent: prev,
            _val: val,
            _index: n_index as isize,
            ..Default::default()
        };
        if prev == -1 {
            self._buf.push( n );
            self._root = n_index as isize;
            self.fixup_insert( n_index as isize );
            None
        } else if key < self._buf[prev as usize]._key {
            self._buf.push( n );
            self.connect_left( prev as isize, n_index as isize);
            self.fixup_insert( n_index as isize);
            None
        } else {
            self._buf.push( n );
            self.connect_right( prev as isize, n_index as isize);
            self.fixup_insert( n_index as isize);
            None
        }
    }
    ///returns the value of the removed item, otherwise return None
    pub fn remove( & mut self, key: & K ) -> Option< V > {
        if let Some(z) = self.get_index( key ) {
            let val = self.get_node(z)._val.clone();
            // println!("remove node {}, val {}", z, val );
            #[allow(unused_assignments)]
            let mut x = -1;
            #[allow(unused_assignments)]
            let mut x_p = -1;
            let mut y = z;
            let mut y_colour_orig = self.get_node(y)._colour;

            //create a special leaf node to handle edge case during fixup if necessary
            self._leaf_remove_index = self._buf.len() as isize;
            let mut leaf_dummy = Node {
                _key: Bounded::max_value(),
                _colour: Colour::Black,
                _index: self._leaf_remove_index,
                ..Default::default()
            };
            self._buf.push( leaf_dummy );
            
            if self.get_node(z)._child_l == -1 {
                x = self.get_node(z)._child_r;
                x_p = self.get_node(z)._parent;
                if x == -1 {
                    let a = self._leaf_remove_index;
                    self.transplant( z, a );    
                } else {
                    self.transplant( z, x );
                }

            } else if self.get_node(z)._child_r == -1 {
                x = self.get_node(z)._child_l;
                x_p = self.get_node(z)._parent;
                if x == -1 {
                    let a = self._leaf_remove_index;
                    self.transplant( z, a );
                } else {
                    self.transplant( z, x );
                }
            } else {
                let z_r = self.get_node(z)._child_r;
                y = self.get_subtree_leftmost( z_r );
                y_colour_orig = self.get_node(y)._colour;
                x = self.get_node(y)._child_r;
                x_p = y;
                if x == -1 {
                    let a = self._leaf_remove_index;
                    self.connect_right( y, a );
                }
                if self.get_node(y)._parent == z {
                    self.get_node_mut(x)._parent = y;
                } else {
                    if x == -1 {
                        let a = self._leaf_remove_index;
                        self.transplant( y, a );
                    } else {
                        self.transplant( y, x );
                    }
                    self.connect_right( y, z_r );
                }
                self.transplant( z, y );
                let z_l = self.get_node(z)._child_l;
                self.connect_left( y, z_l );
                self.get_node_mut(y)._colour = self.get_node(z)._colour;
            }
            match y_colour_orig {
                Colour::Black => {
                    if x == -1 {
                        x = self._leaf_remove_index;
                    }else{
                        //leaf handling not necessary
                        self._leaf_remove_index = -1;
                        self._buf.pop();
                    }
                    self.fixup_remove( x );
                    if self._leaf_remove_index != -1 {
                        // println!("removing dummy leaf node after fixup process");
                        // println!( "buf size: {}, leaf dummy index: {}", self._buf.len(), self._leaf_remove_index );
                        assert!( self._buf.len() as isize == self._leaf_remove_index + 1, "leaf dummy node not at back of buffer" );
                        let leaf_p = self._buf[ self._leaf_remove_index as usize ]._parent;
                        let leaf_p_l = self.get_node(leaf_p)._child_l;
                        if leaf_p_l == self._leaf_remove_index {
                            // println!("reset root node child left");
                            self.get_node_mut(leaf_p)._child_l = -1;
                        }
                        let leaf_p_r = self.get_node(leaf_p)._child_r;
                        if leaf_p_r == self._leaf_remove_index {
                            // println!("reset root node child right");
                            self.get_node_mut(leaf_p)._child_r = -1;
                        }

                        let leaf_x_p_l = self.get_node(x_p)._child_l;
                        let leaf_x_p_r = self.get_node(x_p)._child_r;
                        if leaf_x_p_l == self._leaf_remove_index {
                            // println!("reset leaf node parent left: {}", leaf_p );
                            self.get_node_mut(leaf_p)._child_l = -1;
                        }
                        if leaf_x_p_r == self._leaf_remove_index {
                            // println!("reset leaf node parent right: {}", leaf_p );
                            self.get_node_mut(leaf_p)._child_r = -1;
                        }

                        let h = self._leaf_remove_index;
                        if self.get_node(h)._parent == -1 && self._root == h {
                            self._root = -1;
                        }
                        
                        self._leaf_remove_index = -1;
                        self._buf.pop();                        

                        // println!( "buf size: {} after dummy removal", self._buf.len() );
                    }
                }
                _ => {
                    //no fixup
                    if x == -1 {
                        //clean up dummy leaf node
                        let leaf_p = self._buf[ self._leaf_remove_index as usize ]._parent;
                        let leaf_p_l = self.get_node(leaf_p)._child_l;
                        if leaf_p_l == self._leaf_remove_index {
                            self.get_node_mut(leaf_p)._child_l = -1;
                        }
                        let leaf_p_r = self.get_node(leaf_p)._child_r;
                        if leaf_p_r == self._leaf_remove_index {
                            self.get_node_mut(leaf_p)._child_r = -1;
                        }

                        let leaf_x_p_l = self.get_node(x_p)._child_l;
                        let leaf_x_p_r = self.get_node(x_p)._child_r;
                        if leaf_x_p_l == self._leaf_remove_index {
                            self.get_node_mut(leaf_p)._child_l = -1;
                        }
                        if leaf_x_p_r == self._leaf_remove_index {
                            self.get_node_mut(leaf_p)._child_r = -1;
                        }

                        let h = self._leaf_remove_index;
                        if self.get_node(h)._parent == -1 && self._root == h {
                            self._root = -1;
                        }
                    }
                    self._leaf_remove_index = -1;
                    self._buf.pop();
                    // println!( "buf size: {} after dummy removal", self._buf.len() );
                },
            }
            
            self._freelist.push( z );
            if self._freelist.len() > self._buf.len() * 7 / 8 { //todo: adjust compacting threshold
                self.compact();
            }

            Some( val.clone() )
        } else {
            None
        }
    }
    ///check to see if an item with the input key exists
    pub fn contains_key( & self, key: K ) -> bool {
        let mut x = self._root;
        while x != -1 {
            let k = & self._buf[x as usize]._key;
            if &key == k {
                return true
            } else if &key < k {
                x = self._buf[x as usize]._child_l;
            } else {
                x = self._buf[x as usize]._child_r;
            }
        }
        false
    }
    ///get the value of the item with the input key, otherwise return None
    pub fn get( & self, key: K ) -> Option< V > {
        let mut x = self._root;
        while x != -1 {
            let k = & self._buf[x as usize]._key;
            if &key == k {
                return Some( self._buf[x as usize]._val.clone() )
            } else if &key < k {
                x = self._buf[x as usize]._child_l;
            } else {
                x = self._buf[x as usize]._child_r;
            }
        }
        None
    }
    ///get the index of the node with the input key, otherwise return None
    fn get_index( & self, key: & K ) -> Option< isize > {
        let mut x = self._root;
        // println!("get_index root index: {}", x);
        while x != -1 {
            let k = & self._buf[x as usize]._key;
            if key == k {
                return Some( x )
            } else if key < k {
                x = self._buf[x as usize]._child_l;
            } else {
                x = self._buf[x as usize]._child_r;
            }
        }
        None
    }
    pub fn clear( & mut self ){
        self._root = -1isize;
        self._buf.clear();
    }
    fn fixup_insert( & mut self, node: isize ){
        // println!("fixup_insert enter node {}", node);
        // self.print();
        assert!( node >= 0 && node < self._buf.len() as isize );
        let mut n = node;
        loop {
            if self.get_node(n)._parent == -1 {
                // println!("break 1");
                break;
            }
            let mut n_p = self.get_node(n)._parent;
            match self.get_node(n_p)._colour {
                Colour::Black => {
                    // println!("break 2");
                    break;
                },
                _ => (),
            }
            // println!("fixup_insert loop");
            // println!("fixup_insert npp before");
            let mut n_p_p = self.get_node(n_p)._parent;
            if n_p_p == -1 {
                self.get_node_mut(n)._colour = Colour::Red;
                break;
            }
            // println!("fixup_insert npp after");
            if n_p == self.get_node(n_p_p)._child_l {
                // println!("fixup_insert left case");
                let y = self.get_node(n_p_p)._child_r;
                match self.get_node(y)._colour {
                    Colour::Red => {
                        //case 1
                        self.get_node_mut(n_p)._colour = Colour::Black;
                        self.get_node_mut(y)._colour = Colour::Black;
                        self.get_node_mut(n_p_p)._colour = Colour::Red;
                        n = n_p_p;
                    },
                    _ => {
                        // println!("node parent: {}", n_p);
                        if n == self.get_node(n_p)._child_r {
                            //case 2
                            // println!("case 2 rot left");
                            n = n_p;
                            self.rotate_left( n );
                        }
                        //case 3
                        n_p = self.get_node(n)._parent;
                        n_p_p = self.get_node(n_p)._parent;
                        self.get_node_mut(n_p)._colour = Colour::Black;
                        self.get_node_mut(n_p_p)._colour = Colour::Red;
                        self.rotate_right( n_p_p );
                    },
                }
            } else {
                // println!("fixup_insert right case");
                let y = self.get_node(n_p_p)._child_l;
                // println!(".p.p.l: {}", y );
                match self.get_node(y)._colour {
                    Colour::Red => {
                        //case 1
                        // println!(".p.p.l: {} case 1", y );
                        self.get_node_mut(n_p)._colour = Colour::Black;
                        self.get_node_mut(y)._colour = Colour::Black;
                        self.get_node_mut(n_p_p)._colour = Colour::Red;
                        n = n_p_p;
                    },
                    _ => {
                        // println!(".p.p.l: {} case 2/3", y );
                        if n == self.get_node(n_p)._child_l {
                            //case 2
                            // println!("case 2 rot left");
                            n = n_p;
                            self.rotate_right( n );    
                        }
                        //case 3
                        n_p = self.get_node(n)._parent;
                        n_p_p = self.get_node(n_p)._parent;
                        self.get_node_mut(n_p)._colour = Colour::Black;
                        self.get_node_mut(n_p_p)._colour = Colour::Red;
                        self.rotate_left( n_p_p );
                    },
                }
            }
        }
        let n_root = self._root;
        self.get_node_mut(n_root)._colour = Colour::Black;
        // println!("fixup_insert exit");
    }
    fn fixup_remove( & mut self, node: isize ){
        let mut x = node;
        loop {
            // println!("fixup remove node {}", x );
            if x == -1 {
                // println!("fixup remove sentinil break");
                break;
            }
            let mut x_p = self.get_node(x)._parent;
            if x_p == -1 {
                // println!("fixup remove root node break");
                break;
            }
            match self.get_node(x)._colour {
                Colour::Red => {
                    // println!("fixup remove node {} is red, break", x );
                    break;
                },
                _ => {},
            }
            //node x is black at this point and x is not root
            if x == self.get_node(x_p)._child_l {
                // println!("fixup remove left case");
                //left case
                let mut w = self.get_node(x_p)._child_r;
                let w_colour = self.get_node(w)._colour;
                match w_colour {
                    Colour::Red => {
                        // println!("fixup remove left case 1, node w: {}", w);
                        //case 1
                        self.get_node_mut(w)._colour = Colour::Black;
                        self.get_node_mut(x_p)._colour = Colour::Red;
                        self.rotate_left( x_p );
                        w = self.get_node(x_p)._child_r;
                    },
                    _ => {},
                }
                let w_left = self.get_node(w)._child_l;
                let w_left_colour = self.get_node(w_left)._colour;
                let w_right_colour = {
                    let w_right = self.get_node(w)._child_r;
                    self.get_node(w_right)._colour
                };
                match ( w_left_colour, w_right_colour ) {
                    ( Colour::Black, Colour::Black ) => {
                        // println!("fixup remove left case 2, node w: {}", w);
                        //case 2
                        self.get_node_mut(w)._colour = Colour::Red;
                        x = self.get_node(x)._parent;
                    },
                    _ => {
                        match w_right_colour {
                            Colour::Black => {
                                // println!("fixup remove left case 3, node w: {}", w);
                                //case 3
                                self.get_node_mut(w_left)._colour = Colour::Black;
                                self.get_node_mut(w)._colour = Colour::Red;
                                self.rotate_right( w );
                                x_p = self.get_node(x)._parent;
                                w = self.get_node(x_p)._child_r;
                            },
                            _ => {},
                        }
                        // println!("fixup remove left case 4, node w: {}", w);
                        //case 4
                        x_p = self.get_node_mut(x)._parent;
                        self.get_node_mut(w)._colour = self.get_node(x_p)._colour;
                        self.get_node_mut(x_p)._colour = Colour::Black;
                        let w_right = self.get_node(w)._child_r;
                        self.get_node_mut(w_right)._colour = Colour::Black;
                        self.rotate_left( x_p );
                        x = self._root;
                    },
                }
            } else {
                // println!("fixup remove right case");
                //right case
                let mut w = self.get_node(x_p)._child_l;
                // println!("hreer!!!! {} ", w );
                let w_colour = self.get_node(w)._colour;
                match w_colour {
                    Colour::Red => {
                        // println!("fixup remove right case 1, node w: {}", w);
                        //case 1
                        self.get_node_mut(w)._colour = Colour::Black;
                        self.get_node_mut(x_p)._colour = Colour::Red;
                        self.rotate_right( x_p );
                        w = self.get_node(x_p)._child_l;
                    },
                    _ => {},
                }
                // println!("probe point");
                let w_left_colour = {
                    let w_left = self.get_node(w)._child_l;
                    self.get_node(w_left)._colour
                };
                let w_right = self.get_node(w)._child_r;
                let w_right_colour = self.get_node(w_right)._colour;

                // println!("probe point 2");
                match ( w_left_colour, w_right_colour ) {
                    ( Colour::Black, Colour::Black ) => {
                        // println!("fixup remove right case 2, node w: {}", w);
                        //case 2
                        self.get_node_mut(w)._colour = Colour::Red;
                        x = self.get_node(x)._parent;
                    },
                    _ => {
                        // println!("probe point 4");
                        match w_left_colour {
                            Colour::Black => {
                                //case 3
                                // println!("fixup remove right case 3, node w: {}", w);
                                self.get_node_mut(w_right)._colour = Colour::Black;
                                self.get_node_mut(w)._colour = Colour::Red;
                                self.rotate_left( w );
                                x_p = self.get_node(x)._parent;
                                w = self.get_node(x_p)._child_l;
                            },
                            _ => {},
                        }
                        //case 4
                        // println!("fixup remove right case 4, node w: {}", w);
                        x_p = self.get_node_mut(x)._parent;
                        self.get_node_mut(w)._colour = self.get_node(x_p)._colour;
                        self.get_node_mut(x_p)._colour = Colour::Black;
                        let w_left = self.get_node(w)._child_l;
                        self.get_node_mut(w_left)._colour = Colour::Black;
                        self.rotate_right( x_p );
                        x = self._root;
                    },
                }
            }
        }
        self.get_node_mut(x)._colour = Colour::Black;
    }
    fn get_node( & mut self, node: isize ) -> &Node< K, V > {        
        // println!( "get_node index: {}, buf len: {}", node, self._buf.len() );
        assert!( node >= -1 && node < self._buf.len() as isize );
        if node == -1 {
            self._sentinil._parent = -1;
            &self._sentinil
        } else {
            assert!( node >= 0 && node < self._buf.len() as isize );
            &self._buf[ node as usize ]
        }
    }
    fn get_node_mut( & mut self, node: isize ) -> & mut Node< K, V > {
        assert!( node >= -1 && node < self._buf.len() as isize );
        if node == -1 {
            & mut self._sentinil
        } else {
            assert!( node >= 0 && node < self._buf.len() as isize );
            & mut self._buf[ node as usize ]
        }
    }
    #[allow(dead_code)]
    fn get_parent_left( & self, node: isize ) -> isize {
        assert!( node >= 0 && node < self._buf.len() as isize );
        let mut n = node;
        #[allow(unused_assignments)]
        let mut prev = -1isize;
        loop {
            prev = n;
            n = self._buf[n as usize]._parent;
            if n == -1 {
                break;
            }
            if prev == self._buf[n as usize]._child_r {
                break;
            }
        }
        if n == -1 {
            //root case, no left parent exists, return itself
            node
        } else {
            prev
        }
    }
    #[allow(dead_code)]
    fn get_parent_right( & self, node: isize ) -> isize {
        assert!( node >= 0 && node < self._buf.len() as isize );
        let mut n = node;
        #[allow(unused_assignments)]
        let mut prev = -1isize;
        loop {
            prev = n;
            n = self._buf[n as usize]._parent;
            if n == -1 {
                break;
            }
            if prev == self._buf[n as usize]._child_l {
                break;
            }
        }
        if n == -1 {
            //root case, no right parent exists, return itself
            node
        } else {
            prev
        }
    }
    #[allow(dead_code)]
    fn get_subtree_leftmost( & self, node: isize ) -> isize {
        assert!( node >= 0 && node < self._buf.len() as isize );
        let mut n = node;
        let mut prev = -1isize;
        while n != -1 {
            prev = n;
            n = self._buf[n as usize]._child_l;
        }
        prev
    }
    #[allow(dead_code)]
    fn get_subtree_rightmost( & self, node: isize ) -> isize {
        assert!( node >= 0 && node < self._buf.len() as isize );
        let mut n = node;
        let mut prev = -1isize;
        while n != -1 {
            prev = n;
            n = self._buf[n as usize]._child_r;
        }
        prev
    }
    ///replaces node_dest with node_src
    fn transplant( & mut self, node_dest: isize, node_src: isize ) {
        if self.get_node(node_dest)._parent == -1 {
            // println!("transplant set root: {}", node_src );
            self._root = node_src;
            if node_src != -1 {
                self.get_node_mut(node_src)._parent = -1;
            }
        } else {
            let n_p = self.get_node(node_dest)._parent;
            if node_dest == self.get_node(n_p)._child_l {
                self.connect_left( n_p, node_src );
            } else {
                self.connect_right( n_p, node_src );
            }
        }
    }
    ///connects as left child
    fn connect_left( & mut self, node_parent: isize, node_child: isize ){
        if node_parent != -1 {
            self._buf[node_parent as usize]._child_l = node_child;
        } else {
            self._root = node_child;
        }
        if node_child != -1 {
            self._buf[node_child as usize]._parent = node_parent;
        }
    }
    ///connects as right child
    fn connect_right( & mut self, node_parent: isize, node_child: isize ){
        if node_parent != -1 {
            self._buf[node_parent as usize]._child_r = node_child;
        } else {
            self._root = node_child;
        }
        if node_child != -1 {
            self._buf[node_child as usize]._parent = node_parent;
        }
    }
    ///left rotates and returns the id of the new node
    fn rotate_left( & mut self, node: isize ) -> Option< isize > {
        // println!("rot left node {}", node );
        if node >= 0 && node < self._buf.len() as isize {
            let n_p = self.get_node(node)._parent;
            let y = self.get_node(node)._child_r;
            let y_l = self.get_node(y)._child_l;

            self.connect_right( node, y_l );
            self.get_node_mut(y)._parent = n_p;
            if n_p == -1 {
                self._root = y;
            } else if node == self.get_node(n_p)._child_l {
                //left child case
                self.connect_left( n_p, y );
            } else {
                //right child case
                self.connect_right( n_p, y );
            }
            self.connect_left( y, node );
            Some( y )
        } else {
            None
        }
    }
    ///right rotates and returns the id of the new node
    fn rotate_right( & mut self, node: isize ) -> Option< isize > {
        // println!("rot right node {}", node );
        if node >= 0 && node < self._buf.len() as isize {
            let n_p = self.get_node(node)._parent;
            let y = self.get_node(node)._child_l;
            let y_r = self.get_node(y)._child_r;

            self.connect_left( node, y_r );
            self.get_node_mut(y)._parent = n_p;
            if n_p == -1 {
                self._root = y;
            } else if node == self.get_node(n_p)._child_l {
                //left child case
                self.connect_left( n_p, y );
            } else {
                //right child case
                self.connect_right( n_p, y );
            }
            self.connect_right( y, node );
            Some( y )
        } else {
            None
        }
    }
    ///compacts up unused slots in node array
    pub fn compact( & mut self ){
        // println!("start of compaction: {:?}", self._buf );
        // self.print();
        self._freelist.sort_unstable();
        // println!("freelist: {:?}", self._freelist );
        let mut f = 0;
        let mut f_rev = self._freelist.len();
        let mut n = self._buf.len();
        loop {
            // println!( "n: {}, f: {}, f_rev: {}", n,f,f_rev );
            if f >= f_rev || self._freelist[f] >= n as isize {
                // println!("compact break: n: {}", n );
                break;
            }
            //find linked parent and children
            if (n as isize - 1) == self._freelist[f_rev as usize - 1] {
                // println!("remove garbage at end");
                f_rev -= 1;
                n -= 1;
                continue;
            }
            
            let n_p = self.get_node(n as isize -1)._parent;
            let n_l = self.get_node(n as isize -1)._child_l;
            let n_r = self.get_node(n as isize -1)._child_r;

            let f_index = self._freelist[f];
            // println!("compacting {} to {}", n-1, f_index);
            // println!("compacting node parent index: {}", n_p);
            self._buf[ f_index as usize ] = self._buf[ n -1 ].clone();
            self._buf[ f_index as usize ]._index = f_index;
            self.connect_left( f_index, n_l );
            self.connect_right( f_index, n_r );
            if self.get_node(n_p)._child_l == n as isize - 1 {
                self.connect_left( n_p, f_index );
            } else if self.get_node(n_p)._child_r == n as isize - 1 {
                self.connect_right( n_p, f_index );
            }
            if n_p == -1 {
                self._root = f_index;
            }

            n -= 1;
            f += 1;
        }
        self._buf.resize( n, Default::default() );
        self._freelist.clear();

        // println!("end of compaction: {:?}", self._buf );
    }
    pub fn print( & mut self ){
        let x = self._root;
        let mut v = vec![];
        v.push(x);
        println!("tree root: {}", x );
        println!("tree print: ");
        while v.len() > 0 {
            let n = v.pop().unwrap();
            if n != -1 {
                // println!("{:?}", self.get_node(n) );
                v.push( self.get_node(n)._child_l );
                v.push( self.get_node(n)._child_r );
            }
        }
    }
    pub fn check_nodes( & self ){
        let mut hm = HashMap::new(); //stores number of black nodes from node down to leave
        let mut leaves = vec![];
        let x = self._root;
        let mut v = vec![x];
        //collect all nodes that are leaves
        while v.len() > 0 {
            let & n = v.last().unwrap();
            v.pop();
            if n != -1 {
                let nl = self._buf[n as usize]._child_l;
                let nr = self._buf[n as usize]._child_r;
                if (nl,nr) == (-1,-1) {
                    leaves.push(n);
                } else {
                    v.push(nl);
                    v.push(nr);
                }
            }
        }
        // println!("check_nodes leaves: {:?}", leaves );
        //follow all leaves up to root and accumulate number of black nodes upward
        for i in leaves {
            let mut n = i;
            let mut count = 0;
            while n != -1 {
                match hm.insert( n, count ) {
                    Some(v) => { assert!( v == count ); break; },
                    _ => {},
                }
                let c = self._buf[n as usize]._colour;
                match c {
                    Colour::Black => { count += 1; },
                    _ => {},
                }
                n = self._buf[n as usize]._parent;
            }
        }
        if x != -1 {
            match self._buf[x as usize]._colour {
                Colour::Red => { panic!("root colour incorrect"); },
                _ => {},
            }
        }
        // for (k,v) in hm.iter() {
        //     println!( "node {}: count black to leaf: {}", k, v );
        // }
    }
}
