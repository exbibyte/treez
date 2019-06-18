use std::io::{self, Read};
use std::collections::{HashSet,HashMap};
use std::cmp;
    
#[derive(Debug,Clone,Copy)]
pub struct S {
    pub id: usize,
    pub parent: usize,
    pub ssize: u32,
}

impl S {
    pub fn init( id: usize ) -> Self {
        Self { id: id, parent: id, ssize: 1 }
    }
}

pub struct Dsu( pub Vec<S> );

impl Dsu {

    pub fn init( len: usize ) -> Self {
        Dsu( (0..len).map(|x| S::init(x) ).collect::<Vec<_>>() )
    }
    
    pub fn compress_path( & mut self, mut idx: usize ) -> usize {
        
        debug_assert!( idx < self.0.len() );
        
        let mut buf = vec![];
        
        while self.0[idx].parent != self.0[idx].id {
            buf.push(idx);
            idx = self.0[idx].parent;
        }
        
        for i in buf {
            self.0[i].parent = idx;
        }

        idx
    }
    
    pub fn merge( & mut self, mut a: usize, mut b: usize ) {

        if a != b {

            let a_parent = self.compress_path( a );
            let b_parent = self.compress_path( b );
            
            if a_parent != b_parent { // different sets check
                let (small,big) = if self.0[a_parent].ssize < self.0[b_parent].ssize {
                    (a_parent, b_parent)
                } else {
                    (b_parent, a_parent)
                };
                self.0[big].ssize += self.0[small].ssize;
                self.0[small].parent = big;
            }
        }
    }

    pub fn get_sets_repr( & self ) -> Vec<usize>{
        self.0.iter().enumerate().filter(|x| x.1.id == x.1.parent ).map(|x| x.0).collect()
    }
}

#[test]
fn test_dsu() {
    
    let mut v = Dsu::init(10);

    //1, 3, 5, 7 ,9
    for i in 0..5 {
        let j = i*2+1;
        v.merge( j, j-1 );
    }

    let ret = v.get_sets_repr();
    assert_eq!( ret.len(), 5 );

    v.merge(5,9);

    assert_eq!( v.get_sets_repr().len(), 4 );
    
    v.merge(7,2);

    assert_eq!( v.get_sets_repr().len(), 3 );

    v.merge(4,3);

    assert_eq!( v.get_sets_repr().len(), 2 );

    v.merge(0,5);

    assert_eq!( v.get_sets_repr().len(), 1 );
}
