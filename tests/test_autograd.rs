use std::collections::HashMap;

extern crate treez;

use self::treez::autograd;

#[test]
fn init() {

    let mut c : autograd::Context = Default::default();

    //setup variables
    let mut buf = {
        let mut x = autograd::init_var( & mut c, 6f64 );
        let mut y = autograd::init_var( & mut c, 7f64 );
        let mut z = autograd::init_op( & mut c, autograd::OpType::Mul, & mut [ & mut x, & mut y ] );
        let mut a = autograd::init_var( & mut c, 3f64 );
        let b = autograd::init_op( & mut c, autograd::OpType::Add, & mut [ & mut z, & mut a ] );
        vec![ x, y, z, a, b ]
    };

    let var_ids = autograd::fwd_pass( & mut c, & mut buf ).unwrap();
    
    let mut var_map = HashMap::new();
    for i in [ "x", "y", "z", "a", "b" ].iter().zip( var_ids ) {
        var_map.insert( i.0, i.1 );
    }

    //compute gradient of b with respect to every other variable
    {
        let mut var_grad = HashMap::new();

        let b_index = *var_map.get(&"b").unwrap();
        for i in var_map.iter() {
            let grad = autograd::compute_grad( & mut c, b_index, *i.1 ).unwrap();
            var_grad.insert( *i.0, grad );
        }

        assert!( c.get_var(*var_map.get(&"b").unwrap()).unwrap()._val == 45f64 );
        assert!( c.get_var(*var_map.get(&"a").unwrap()).unwrap()._val == 3f64 );
        assert!( c.get_var(*var_map.get(&"z").unwrap()).unwrap()._val == 42f64 );
        assert!( c.get_var(*var_map.get(&"x").unwrap()).unwrap()._val == 6f64 );
        assert!( c.get_var(*var_map.get(&"y").unwrap()).unwrap()._val == 7f64 );

        assert!( *var_grad.get(&"b").unwrap() == 1f64 );
        assert!( *var_grad.get(&"a").unwrap() == 1f64 );
        assert!( *var_grad.get(&"z").unwrap() == 1f64 );
        assert!( *var_grad.get(&"x").unwrap() == 7f64 );
        assert!( *var_grad.get(&"y").unwrap() == 6f64 );
    }
    //compute gradient of z with respect to a
    {
        let z_id = *var_map.get(&"z").unwrap();
        let a_id = *var_map.get(&"a").unwrap();
        let grad = autograd::compute_grad( & mut c, z_id, a_id ).unwrap();
        assert!( grad == 0f64 );
    }
}
