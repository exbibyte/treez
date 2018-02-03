extern crate treez;
extern crate rand;
extern crate chrono;
extern crate num;

use self::chrono::prelude::*;
use std::collections::HashMap;
use self::rand::distributions::{IndependentSample, Range};
use self::rand::{Rng};
use std::collections::BTreeMap;

#[test]
fn insert() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..10 {
        t.insert( i, i );
    }
    t.check_nodes();
}
#[test]
fn contains_key() {
    {
        let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
        for i in 0..10 {
            t.insert( i, i );
        }
        for i in 0..10 {
            assert!( t.contains_key( i ) );
        }
        for i in 10..15 {
            assert!( !t.contains_key( i ) );
        }
    }
    {
        let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
        for i in (0..10).rev() {
            t.insert( i, i );
        }
        for i in 0..10 {
            assert!( t.contains_key( i ) );
        }
        for i in 10..15 {
            assert!( !t.contains_key( i ) );
        }
    }
}
#[test]
fn get() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..10 {
        t.insert( i, i );
    }
    for i in 0..10 {
        let n = t.get( i ).expect( "get() unsuccessful" );
        assert!( n == i );
    }
    for i in 10..15 {
        match  t.get( i ) {
            Some(_) => { panic!( "get() unsuccessfil" ); },
            None => (),
        }
    }
}
#[test]
fn len() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    assert!( t.len() == 0 );
    assert!( t.is_empty() );
    for i in 0..10 {
        t.insert( i, i );
    }
    assert!( t.len() == 10 );
    assert!( !t.is_empty() );
}
#[test]
fn clear() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..10 {
        t.insert( i, i );
    }
    assert!( t.len() == 10 );
    t.clear();
    assert!( t.len() == 0 );
}
#[test]
fn remove_compact() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..3 {
        t.insert( i, i );
    }
    // t.print();
    t.check_nodes();
    let t_size = t.len();
    for i in 0..3 {
        let r = t.remove( &i ).expect( "remove unsuccessful" );
        assert!( r == i );
        // println!( "t size: {}", t.len() );
        assert!( t.len() == t_size - i as usize - 1 );
    }
    // t.print();
    t.check_nodes();
}
#[test]
fn remove_leaf_black() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..10 {
        t.insert( i, i );
    }
    // t.print();
    t.check_nodes();
    let t_size = t.len();
    let mut count_remove = 0;
    for i in 8..9 {
        let r = t.remove( &i ).expect( "remove unsuccessful" );
        assert!( r == i );
        // println!( "t size: {}", t.len() );
        count_remove += 1;
        assert!( t.len() == t_size - count_remove );
    }
    // t.print();
    t.check_nodes();
}

#[test]
fn remove_leaf_red() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..10 {
        t.insert( i, i );
    }
    // t.print();
    t.check_nodes();
    let t_size = t.len();
    let mut count_remove = 0;
    for i in 4..5 {
        let r = t.remove( &i ).expect( "remove unsuccessful" );
        assert!( r == i );
        // println!( "t size: {}", t.len() );
        count_remove += 1;
        assert!( t.len() == t_size - count_remove );
    }
    // t.print();
    t.check_nodes();
}

#[test]
fn remove_internal_red() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..10 {
        t.insert( i, i );
    }
    // t.print();
    t.check_nodes();
    let t_size = t.len();
    let mut count_remove = 0;
    for i in 7..8 {
        let r = t.remove( &i ).expect( "remove unsuccessful" );
        assert!( r == i );
        // println!( "t size: {}", t.len() );
        count_remove += 1;
        assert!( t.len() == t_size - count_remove );
    }
    // t.print();
    t.check_nodes();
}

#[test]
fn remove_internal_black() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..10 {
        t.insert( i, i );
    }
    // t.print();
    t.check_nodes();
    let t_size = t.len();
    let mut count_remove = 0;
    for i in 1..2 {
        let r = t.remove( &i ).expect( "remove unsuccessful" );
        assert!( r == i );
        // println!( "t size: {}", t.len() );
        count_remove += 1;
        assert!( t.len() == t_size - count_remove );
    }
    // t.print();
    t.check_nodes();
}

#[test]
fn remove_internal_black_2() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..10 {
        t.insert( i, i );
    }
    // t.print();
    t.check_nodes();
    let t_size = t.len();
    let mut count_remove = 0;
    for i in 5..6 {
        let r = t.remove( &i ).expect( "remove unsuccessful" );
        assert!( r == i );
        // println!( "t size: {}", t.len() );
        count_remove += 1;
        assert!( t.len() == t_size - count_remove );
    }
    // t.print();
    t.check_nodes();
}

#[test]
fn insert_remove_rand() {
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();

    let bounds = Range::new( -300, 300 );
    let mut rng = rand::thread_rng();

    let mut hm = HashMap::new();
    for i in 0..10000 {
        let r = bounds.ind_sample( & mut rng );
        t.insert( r, i );
        hm.insert( r, i );
    }
    // t.print();
    // println!( "t len: {}, hm len: {}", t.len(), hm.len() );
    t.check_nodes();
    assert!( t.len() == hm.len() );

    for _ in 0..30000 {
        let r = bounds.ind_sample( & mut rng );
        // println!("removing: {}", r );
        match hm.remove( &r ) {
            Some( v_check ) => {
                // println!("v_check: {:}", v_check);
                // println!("hm: {:?}", hm);
                // t.print();
                let v = t.remove( &r ).expect( "remove unsuccessful" );
                assert!( v == v_check );
                assert!( hm.len() == t.len() );
                // println!("remove after");
                // t.print();
                t.check_nodes();
            },
            _ => {
                match t.remove( &r ) {
                    Some( _v ) => { panic!( "remove unsuccessful" ); },
                    _ => {},
                }
            },
        }
    }
}

#[test]
fn perf(){

    let mut rng = rand::thread_rng();
    let bounds = Range::new( -1000000, 1000000 );
    let mut nums : Vec<isize> = (0..1000000).enumerate().map( |(_,_)| bounds.ind_sample( & mut rng ) ).collect();
    nums.sort_unstable();
    nums.dedup();
    rng.shuffle( & mut nums );

    {
        println!("rbtree performance test:");
        let mut verify = vec![0;nums.len()];

        let t0 = Local::now();

        let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
        for i in 0..nums.len() {
            let r = nums[i];
            t.insert( r, i as isize );
        }
        
        let t1 = Local::now();
        
        for i in 0..nums.len() {
            let r = nums[i];
            let v = t.remove( &r ).expect( "remove unsuccessful" );
            verify[v as usize] = r;
        }

        let t2 = Local::now();

        assert!(t.len() == 0);
        assert_eq!( verify.as_slice(), nums.as_slice() );
        
        let t_insert = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
        let t_remove = t2.signed_duration_since(t1).num_microseconds().unwrap() as f64;
        println!("insertion: count: {}, time: {}s, rate: {:.6} inserts/s, {:.6} us/insert", nums.len(), t_insert, nums.len() as f64 * 1000000f64 / t_insert, t_insert / nums.len() as f64 );
        println!("insertion: count: {}, time: {}s, rate: {:.6} removes/s, {:.6} us/remove", nums.len(), t_remove, nums.len() as f64 * 1000000f64 / t_remove, t_remove / nums.len() as f64 );
    }

    {
        println!("reference std::collections::BTreeMap performance test:");
        let mut verify = vec![0;nums.len()];

        let t0 = Local::now();

        let mut t = BTreeMap::new();
        for i in 0..nums.len() {
            let r = nums[i];
            t.insert( r, i as isize );
        }
        
        let t1 = Local::now();
        
        for i in 0..nums.len() {
            let r = nums[i];
            let v = t.remove( &r ).expect( "remove unsuccessful" );
            verify[v as usize] = r;
        }

        let t2 = Local::now();

        assert!(t.len() == 0);
        assert_eq!( verify.as_slice(), nums.as_slice() );
        
        let t_insert = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
        let t_remove = t2.signed_duration_since(t1).num_microseconds().unwrap() as f64;
        println!("insertion: count: {}, time: {}s, rate: {:.6} inserts/s, {:.6} us/insert", nums.len(), t_insert, nums.len() as f64 * 1000000f64 / t_insert, t_insert / nums.len() as f64 );
        println!("insertion: count: {}, time: {}s, rate: {:.6} removes/s, {:.6} us/remove", nums.len(), t_remove, nums.len() as f64 * 1000000f64 / t_remove, t_remove / nums.len() as f64 );
    }

    {
        println!("reference std::collections::HashMap performance test:");
        let mut verify = vec![0;nums.len()];

        let t0 = Local::now();

        let mut t = HashMap::new();
        for i in 0..nums.len() {
            let r = nums[i];
            t.insert( r, i as isize );
        }
        
        let t1 = Local::now();
        
        for i in 0..nums.len() {
            let r = nums[i];
            let v = t.remove( &r ).expect( "remove unsuccessful" );
            verify[v as usize] = r;
        }

        let t2 = Local::now();

        assert!(t.len() == 0);
        assert_eq!( verify.as_slice(), nums.as_slice() );
        
        let t_insert = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
        let t_remove = t2.signed_duration_since(t1).num_microseconds().unwrap() as f64;
        println!("insertion: count: {}, time: {}s, rate: {:.6} inserts/s, {:.6} us/insert", nums.len(), t_insert, nums.len() as f64 * 1000000f64 / t_insert, t_insert / nums.len() as f64 );
        println!("insertion: count: {}, time: {}s, rate: {:.6} removes/s, {:.6} us/remove", nums.len(), t_remove, nums.len() as f64 * 1000000f64 / t_remove, t_remove / nums.len() as f64 );
    }    
}
