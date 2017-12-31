use std::mem;

pub struct TreePrefix {
    _buf: Vec<isize>,
}

impl TreePrefix {
    pub fn init( num: usize ) -> TreePrefix {
        TreePrefix {
            _buf: vec![ 0isize; num + 1]
        }
    }
    pub fn add( & mut self, index: usize, val: isize ){
        assert!( index < self._buf.len() );
        let mut i = index;
        while i < self._buf.len() {
            self._buf[i] += val;
            i += self.lsb(i+1);
        }
    }
    pub fn set( & mut self, index: usize, val: isize ){
        assert!( index < self._buf.len() );
        let v = self.get( index );
        self.add( index, val - v );
    }
    pub fn get( & self, index: usize ) -> isize {
        assert!( index < self._buf.len() );
        self.get_interval( index, index + 1 )
    }
    pub fn get_interval( & self, index: usize, index_end: usize ) -> isize {
        assert!( index < self._buf.len() );
        assert!( index_end < self._buf.len() );

        let mut s = 0isize;
        let mut i = index_end;
        let mut j = index;
        if j > i {
            mem::swap( & mut i, & mut j );
        }
        while i > j {
            // println!("i:{}", i );
            s += self._buf[i-1];
            i -= self.lsb(i);
        }
        while j > i {
            // println!("j:{}", j );
            s -= self._buf[j-1];
            j -= self.lsb(j);
        }
        s
    }
    pub fn get_interval_start( & self, index: usize ) -> isize {
        let mut i = index;
        let mut s = 0isize;
        while i > 0 {
            s += self._buf[i-1];
            i -= self.lsb(i);
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
