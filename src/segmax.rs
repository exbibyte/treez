#[cfg(test)]
use rand::distributions::{Distribution, Uniform};
#[cfg(test)]
use rand::thread_rng;

use std::cmp::*;
#[cfg(test)]
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Add;

pub trait Min {
    fn min() -> Self;
}

pub trait Max {
    fn max() -> Self;
}

#[derive(Debug, Clone, Copy)]
enum Change<T>
where
    T: Add<Output = T> + Ord + Debug + Clone + Min,
{
    Abs(T),
    Rel(T),
}

#[derive(Debug, Clone)]
struct N<T>
where
    T: Add<Output = T> + Ord + Debug + Clone + Min,
{
    l: i64,
    r: i64,
    s: T,
    left: Option<Box<N<T>>>,
    right: Option<Box<N<T>>>,
    mark: Option<Change<T>>, //lazy
}

impl<T> N<T>
where
    T: Add<Output = T> + Ord + Debug + Clone + Min,
{
    pub fn new(l: i64, r: i64) -> Self {
        Self {
            l: l,
            r: r,
            s: Min::min(),
            left: None,
            right: None,
            mark: None,
        }
    }
    fn set_lazy_abs(&mut self, val: T) {
        self.push_down();
        self.s = max(self.range_val(), val.clone());
        self.mark = Some(Change::Abs(val));
    }
    fn set_lazy_rel(&mut self, val: T) {
        match &self.mark {
            Some(Change::Abs(_)) => {
                self.push_down_abs();
                assert!(self.mark.is_none());
            }
            _ => {}
        }
        match &mut self.mark {
            Some(Change::Rel(ref mut v)) => {
                *v = v.clone() + val;
            }
            _ => {
                self.mark = Some(Change::Rel(val));
            }
        }
    }
    pub fn range_val(&self) -> T {
        match &self.mark {
            Some(Change::Abs(v)) => self.s.clone(),
            Some(Change::Rel(v)) => self.s.clone() + v.clone(),
            _ => self.s.clone(),
        }
    }
    fn extend(&mut self) {
        if self.l + 1 >= self.r {
            return;
        }
        self.extend_left();
        self.extend_right();
    }
    fn extend_left(&mut self) {
        let m = (self.l + self.r) / 2;
        if self.l < self.r && self.l < m {
            match self.left {
                None => {
                    self.left = Some(Box::new(N::new(self.l, m)));
                }
                _ => {}
            }
        }
    }
    fn extend_right(&mut self) {
        let m = (self.l + self.r) / 2;
        if self.l < self.r && m < self.r {
            match self.right {
                None => {
                    self.right = Some(Box::new(N::new(m, self.r)));
                }
                _ => {}
            }
        }
    }
    fn push_down_abs(&mut self) {
        match &mut self.mark {
            Some(Change::Abs(_)) => match self.mark.take() {
                Some(Change::Abs(x)) => {
                    self.extend();
                    if let Some(ref mut y) = &mut self.left {
                        y.set_lazy_abs(x.clone());
                    }
                    if let Some(ref mut y) = &mut self.right {
                        y.set_lazy_abs(x.clone());
                    }
                }
                _ => {
                    panic!();
                }
            },
            _ => {}
        }
    }
    fn push_down_rel(&mut self) {
        match &mut self.mark {
            Some(Change::Rel(_)) => match self.mark.take() {
                Some(Change::Rel(x)) => {
                    self.extend();
                    if let Some(ref mut y) = &mut self.left {
                        y.set_lazy_rel(x.clone());
                    }
                    if let Some(ref mut y) = &mut self.right {
                        y.set_lazy_rel(x.clone());
                    }
                    self.s = self.s.clone() + x;
                }
                _ => {
                    panic!();
                }
            },
            _ => {}
        }
    }
    fn push_down(&mut self) {
        if let Some(ref x) = &self.mark {
            match x {
                Change::Abs(_) => {
                    self.push_down_abs();
                }
                _ => {
                    self.push_down_rel();
                }
            }
        }
    }
    pub fn update(&mut self, ll: i64, rr: i64, val: &T) {
        if ll <= self.l && self.r <= rr {
            self.set_lazy_abs(val.clone());
        } else {
            self.push_down();
            let m = (self.l + self.r) / 2;
            if ll < m {
                self.extend_left();
                match &mut self.left {
                    Some(ref mut x) => {
                        x.update(ll, min(m, rr), val);
                    }
                    _ => {}
                }
            }
            if m < rr {
                self.extend_right();
                match &mut self.right {
                    Some(ref mut x) => {
                        x.update(max(m, ll), rr, val);
                    }
                    _ => {}
                }
            }
            self.s = max(
                self.left.as_mut().map_or(Min::min(), |x| x.range_val()),
                self.right.as_mut().map_or(Min::min(), |x| x.range_val()),
            );
            debug_assert!(self.mark.is_none());
        }
    }
    pub fn add(&mut self, ll: i64, rr: i64, delta: &T) {
        if ll <= self.l && self.r <= rr {
            self.set_lazy_rel(delta.clone());
        } else {
            self.push_down();
            let m = (self.l + self.r) / 2;
            if ll < m {
                self.extend();
                if let Some(ref mut x) = &mut self.left {
                    x.add(ll, min(m, rr), delta);
                }
            }
            if m < rr {
                self.extend();
                if let Some(ref mut x) = &mut self.right {
                    x.add(max(m, ll), rr, delta);
                }
            }
            self.s = max(
                self.left.as_mut().map_or(Min::min(), |x| x.range_val()),
                self.right.as_mut().map_or(Min::min(), |x| x.range_val()),
            );
            debug_assert!(self.mark.is_none());
        }
    }
    pub fn query(&mut self, ll: i64, rr: i64) -> T {
        if ll >= rr || max(ll, self.l) >= min(rr, self.r) {
            Min::min()
        } else if ll <= self.l && self.r <= rr {
            self.range_val()
        } else {
            self.push_down();
            // let mut ret: T = Default::default();
            let m = (self.l + self.r) / 2;
            let a = match &mut self.left {
                Some(ref mut x) if ll < m => x.query(ll, min(m, rr)),
                _ => Min::min(),
            };
            let b = match &mut self.right {
                Some(ref mut x) if m < rr => x.query(max(m, ll), rr),
                _ => Min::min(),
            };
            max(a, b)
        }
    }
    #[cfg(test)]
    pub fn dbg(&self, d_parent: i64, depths: &mut HashMap<i64, i64>) -> usize {
        *depths.entry(d_parent).or_default() += 1;
        let mut count = 0;
        if let Some(ref x) = &self.left {
            count += x.dbg(d_parent + 1, depths);
        }
        println!(
            "[{},{}), mark: {:?}, s: {:?}",
            self.l, self.r, self.mark, self.s
        );
        if let Some(ref x) = &self.right {
            count += x.dbg(d_parent + 1, depths);
        }
        count + 1
    }
}

#[derive(Debug, Clone)]
pub struct SegMax<T>
where
    T: Add<Output = T> + Ord + Debug + Clone + Min,
{
    lim_l: i64,
    lim_r: i64,
    r: N<T>,
}

impl<T> SegMax<T>
where
    T: Add<Output = T> + Ord + Debug + Clone + Min,
{
    pub fn new(l: i64, r: i64) -> Self {
        Self {
            lim_l: l,
            lim_r: r,
            r: N::new(l, r),
        }
    }
    pub fn add(&mut self, l: i64, r: i64, delta: &T) {
        if l < r {
            assert!(self.lim_l <= l);
            assert!(r <= self.lim_r);
            self.r.add(l, r, delta);
        }
    }
    pub fn query_range(&mut self, l: i64, r: i64) -> T {
        assert!(self.lim_l <= l);
        assert!(r <= self.lim_r);
        self.r.query(l, r)
    }
    pub fn update(&mut self, l: i64, r: i64, val: &T) {
        if l < r {
            assert!(self.lim_l <= l);
            assert!(r <= self.lim_r);
            self.r.update(l, r, val);
        }
    }
    #[cfg(test)]
    pub fn dbg(&self) {
        let mut count = 0;
        let mut depths = HashMap::new();
        self.r.dbg(0, &mut depths);
        let mut d_avg = 0.;
        for (depth, freq) in depths {
            println!("depth: {}, count: {}", depth, freq);
            d_avg += (depth * freq) as f64;
            count += freq;
        }
        d_avg /= count as f64;
        println!("avg depth per node: {}", d_avg);
        println!("count nodes: {}", count);
    }
}

#[cfg(test)]
impl Min for i64 {
    fn min() -> i64 {
        i64::MIN
    }
}
#[test]
fn test_seg() {
    const m: usize = 64;
    let mut seg = SegMax::new(0, m as i64);
    let mut g = thread_rng();
    let distr = Uniform::from(0..m as i64);
    let distr2 = Uniform::from(0..m as i64 + 1);
    let distr3 = Uniform::from(-10..11);

    let mut reference = vec![0; m];
    seg.update(0, m as i64, &0);

    for _ in 0..1000 {
        if distr.sample(&mut g) % 2 == 0 {
            let mut a: i64 = distr.sample(&mut g);
            let mut b: i64 = max(distr2.sample(&mut g), a);
            while a >= b {
                a = distr.sample(&mut g);
                b = distr2.sample(&mut g);
            }
            let delta: i64 = distr3.sample(&mut g);
            println!("add: [{},{}), {}", a, b, delta);
            seg.add(a as _, b as _, &delta);
            for j in a..b {
                reference[j as usize] += delta;
            }
        } else {
            let mut a: i64 = distr.sample(&mut g);
            let mut b: i64 = max(distr2.sample(&mut g), a);
            while a >= b {
                a = distr.sample(&mut g);
                b = distr2.sample(&mut g);
            }
            let val: i64 = distr3.sample(&mut g);
            seg.update(a as i64, b as i64, &val);
            for h in a..b {
                reference[h as usize] = max(reference[h as usize], val);
            }
            println!("set max: [{},{}), {}", a, b, val);
        }
    }

    seg.dbg();

    for i in 1..m {
        for j in 1..m + 1 {
            let v = seg.query_range(i as i64, j as i64);
            let mut expect = i64::MIN;
            for k in i..j {
                expect = max(expect, reference[k]);
            }
            assert_eq!(
                expect, v,
                "not matching: [{},{}), {}, {:?}",
                i, j, expect, v
            );
        }
    }
}
