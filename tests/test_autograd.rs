// use std::collections::HashSet;

extern crate treez;

use self::treez::autograd;

#[test]
fn init() {

    //context for id generation
    let mut c : autograd::Context = Default::default();

    //setup variables
    let mut x = autograd::init_var( & mut c, 6f64 );
    let mut y = autograd::init_var( & mut c, 7f64 );
    let mut z = autograd::init_op( & mut c, autograd::OpType::Mul, & mut [ & mut x, & mut y ] );
    let mut a = autograd::init_var( & mut c, 3f64 );
    let mut b = autograd::init_op( & mut c, autograd::OpType::Add, & mut [ & mut z, & mut a ] );
    
    //do a forward calc and compute gradients back to each variable
    {
        let buf = & mut [ & mut x, & mut y, & mut z, & mut a, & mut b ];

        let ( id_map, rev_order ) = autograd::check_links( & mut buf[..] ).unwrap();
        
        assert!(rev_order.len() == 5 );
        println!( "rev_order: {:?}", rev_order );

        autograd::compute_grad( & id_map, & mut buf[..], rev_order.as_slice() ).is_ok();
    }
    
    assert!( b._val == 45f64 );
    assert!( a._val == 3f64 );
    assert!( z._val == 42f64 );
    assert!( x._val == 6f64 );
    assert!( y._val == 7f64 );

    assert!( b._grad == 1f64 );
    assert!( a._grad == 1f64 );
    assert!( z._grad == 1f64 );
    assert!( x._grad == 7f64 );
    assert!( y._grad == 6f64 );

}
