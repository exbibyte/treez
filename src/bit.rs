use std::ops::{AddAssign, Sub};

pub struct Bit<T>(pub Vec<T>)
where
    T: Default + AddAssign + Sub<Output = T> + Clone;

impl<T> Bit<T>
where
    T: Default + AddAssign + Sub<Output = T> + Clone,
{
    ///init with default values
    pub fn init(len: usize) -> Self {
        //set bound to next power of 2
        let mut adjust = 1;
        while adjust <= len {
            adjust = adjust << 1;
        }
        Self(vec![T::default(); adjust + 1])
    }

    ///init with values
    pub fn init_with(vals: &[T]) -> Self {
        let mut v = Self::init(vals.len());
        for (idx, val) in vals.iter().enumerate() {
            v.set(idx, val.clone());
        }
        v
    }

    /// a[index] += val, index is 0-indexed
    pub fn add(&mut self, index: usize, val: T) {
        assert!(index < self.0.len());
        let mut a = index + 1;
        while a < self.0.len() {
            self.0[a] += val.clone();
            a += a & (!a + 1); //least significant high bit
        }
    }

    /// for i in [0..index) sum a[i], index is 0-indexed
    pub fn prefix_sum(&mut self, index: usize) -> T {
        assert!(index < self.0.len());
        let mut a = index;
        let mut s = T::default();
        while a >= 1 {
            s += self.0[a].clone();
            a -= a & (!a + 1); //least significant high bit
        }
        s
    }

    /// for i in [l..index) sum a[i], 0-indexed
    pub fn range_sum(&mut self, l: usize, r: usize) -> T {
        assert!(l <= r);
        self.prefix_sum(r)
            - if l == 0 {
                T::default()
            } else {
                self.prefix_sum(l)
            }
    }

    /// a[index] = val, index is 0-indexed
    pub fn set(&mut self, index: usize, val: T) {
        let old = self.range_sum(index, index + 1);
        let extra = val - old;
        self.add(index, extra);
    }
}

#[test]
fn test_bit() {
    {
        let mut v = Bit::init(10);
        assert_eq!(v.0.len(), 17);
        for i in 0..10 {
            v.add(i, 5);
        }

        assert_eq!(v.prefix_sum(10), 50);

        assert_eq!(v.range_sum(1, 2), 5);
        assert_eq!(v.range_sum(1, 4), 15);

        assert_eq!(v.prefix_sum(16), 50);

        v.add(5, -10);

        assert_eq!(v.prefix_sum(16), 40);

        assert_eq!(v.range_sum(5, 6), -5);

        v.add(6, 20);

        assert_eq!(v.range_sum(5, 7), -5 + 25);

        assert_eq!(v.range_sum(0, 16), 60);

        assert_eq!(v.range_sum(5, 6), -5);

        v.set(5, 50);

        assert_eq!(v.range_sum(5, 6), 50);
        assert_eq!(v.prefix_sum(16), 115);

        assert_eq!(v.range_sum(5, 7), 50 + 25);
    }
    {
        let w = (0..10).collect::<Vec<_>>();
        let mut v = Bit::init_with(&w[..]);
        assert_eq!(v.range_sum(0, 10), 9 * 5);
        assert_eq!(v.range_sum(5, 7), 11);
    }
}
