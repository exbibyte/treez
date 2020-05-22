use std::cmp;

#[cfg(test)]
extern crate rand;

#[cfg(test)]
use self::rand::distributions::{Distribution, Uniform};

/// find first index where arr[idx] >= v; assume arr is sorted
pub fn lower_bound<T>(arr: &[T], v: &T) -> usize where T: cmp::Ord {
    let mut l = 0;
    let mut r = arr.len();
    while l<r {
        let m = (l+r)/2;
        let vm = & arr[m];
        match vm.cmp(v) {
            cmp::Ordering::Less => { l = m+1; },
            cmp::Ordering::Greater => { r = m; },
            cmp::Ordering::Equal => { r = m; }
        }
    }
    assert_eq!(l,r);
    l
}

#[test]
fn test_lower_bound() {

    let distr = Uniform::from(0..20);
    let mut rng = rand::thread_rng();
    
    let mut arr = (0..1000).map(|_| distr.sample(&mut rng) ).collect::<Vec<_>>();
    arr.sort();
    let val = distr.sample(&mut rng);
    
    let idx = lower_bound(&arr[..], &val);

    let mut check = 0;
    for (i,v) in arr.iter().enumerate(){
        check = i;
        if *v == val {
            break;
        }
    }
    assert_eq!(check, idx);
}

#[test]
fn test_lower_bound_end() {

    let distr = Uniform::from(0..20);
    let mut rng = rand::thread_rng();
    
    let mut arr = (0..1000).map(|_| distr.sample(&mut rng) ).collect::<Vec<_>>();
    arr.sort();
    let val = 99;
    
    let idx = lower_bound(&arr[..], &val);

    assert_eq!(arr.len(), idx);
}
