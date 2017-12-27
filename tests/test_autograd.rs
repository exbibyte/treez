// use std::collections::HashSet;

extern crate treez;

use self::treez::autograd;

#[test]
fn init() {
    let mut x : autograd::Link = Default::default();
    let mut y : autograd::Link = Default::default();
    x._val = 6f64;
    y._val = 7f64;
    let mut z : autograd::Link = Default::default();
    z._op = Box::new( autograd::OpMul{} );

    let mut buf = vec![ x, y, z ];
    buf[2].set_precedent( &[ 0, 1 ] );
    buf[0].set_descendent( &[ 2 ] );
    buf[1].set_descendent( &[ 2 ] );

    let fwd_order = autograd::check_links( buf.as_mut_slice() ).unwrap();

    println!( "fwd_order: {:?}", fwd_order );
    assert!(fwd_order.len() == 3 );

    assert!( buf[2]._val == 42f64 );

    let mut rev_order = fwd_order.clone();
    rev_order.reverse();

    println!( "rev_order: {:?}", rev_order );

    autograd::compute_grad( buf.as_mut_slice(), rev_order.as_slice() ).is_ok();

    println!( "buf: {:?}", buf );
    assert!( buf[2]._grad == 1f64 );
    assert!( buf[0]._grad == 7f64 );
    assert!( buf[1]._grad == 6f64 );

}
