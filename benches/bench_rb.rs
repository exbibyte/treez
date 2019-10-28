// extern crate treez;
extern crate chrono;
extern crate criterion;
extern crate num;
extern crate rand;

use self::rand::distributions::{IndependentSample, Range};
use self::rand::Rng;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn stress_insert_remove(t: &mut treez::rb::TreeRb<i32, i32>, nums: &[i32]) {
    for i in 0..nums.len() {
        let r = nums[i];
        t.insert(r, i as i32);
    }
    for i in 0..nums.len() {
        let r = nums[i];
        let v = t.remove(&r).expect("remove unsuccessful");
    }
}

fn benchmark_rb(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let bounds = Range::new(-1000000, 1000000);
    let mut nums: Vec<i32> = (0..10000)
        .enumerate()
        .map(|(_, _)| bounds.ind_sample(&mut rng))
        .collect();
    nums.sort_unstable();
    nums.dedup();
    rng.shuffle(&mut nums);

    let mut t: treez::rb::TreeRb<i32, i32> = treez::rb::TreeRb::new();

    println!("number of insert-remove pairs: {}", nums.len());

    c.bench_function("red-black tree insert remove", |b| {
        b.iter(|| stress_insert_remove(&mut t, black_box(&nums[..])))
    });

    assert_eq!( t.len(), 0 );
}

criterion_group!(benches, benchmark_rb);
criterion_main!(benches);
