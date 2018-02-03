use std::mem;
use std::ops::Add;
use std::ops::Sub;

pub struct TreePrefix < V > where V: Add< Output = V > + Sub< Output = V > + Default + Clone {
    _buf: Vec< V >,
}

impl < V > TreePrefix < V > where V: Add< Output = V > + Sub< Output = V > + Default + Clone {
    pub fn init( num: usize ) -> TreePrefix< V > {
        TreePrefix {
            _buf: vec![ Default::default(); num + 1]
        }
    }
    pub fn add( & mut self, index: usize, val: V ){
        assert!( index < self._buf.len() );
        let mut i = index;
        while i < self._buf.len() {
            let v = self._buf[i].clone();
            self._buf[i] = v + val.clone();
            i += self.lsb(i+1);
        }
    }
    pub fn set( & mut self, index: usize, val: V ){
        assert!( index < self._buf.len() );
        let v = self.get( index );
        self.add( index, val - v );
    }
    pub fn get( & self, index: usize ) -> V {
        assert!( index < self._buf.len() );
        self.get_interval( index, index + 1 )
    }
    pub fn get_interval( & self, index: usize, index_end: usize ) -> V {
        assert!( index < self._buf.len() );
        assert!( index_end < self._buf.len() );

        let mut s = Default::default();
        let mut i = index_end;
        let mut j = index;
        if j > i {
            mem::swap( & mut i, & mut j );
        }
        while i > j {
            // println!("i:{}", i );
            s = s + self._buf[i-1].clone();
            i = i - self.lsb(i);
        }
        while j > i {
            // println!("j:{}", j );
            s = s - self._buf[j-1].clone();
            j = j - self.lsb(j);
        }
        s
    }
    pub fn get_interval_start( & self, index: usize ) -> V {
        let mut i = index;
        let mut s = Default::default();
        while i > 0 {
            s = s + self._buf[i-1].clone();
            i = i - self.lsb(i);
        }
        s
    }
    pub fn get_len( & mut self ) -> usize {
        self._buf.len() - 1
    }
    fn lsb( & self, v: usize ) -> usize {
        let a = !v + 1;
        let b = a & v;
        // println!( "v: {:0>8b}, a: {:0>8b}, b:{:0>8b}", v, a, b );
        b
    }
}
