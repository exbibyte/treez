use std::collections::HashMap;

extern crate treez;

use self::treez::autograd;

#[test]
fn autograd() {

    let mut c: autograd::Context = Default::default();

    //setup variables
    let buf = {
        let mut x = c.init_var( &[6f64, 5f64] );
        let mut y = c.init_var( &[7f64, 3f64] );
        let mut z = c.init_op( autograd::OpType::Mul, & mut [ & mut x, & mut y ] );
        let mut a = c.init_var( &[3f64, 8f64] );
        let mut b = c.init_op( autograd::OpType::AddParallel, & mut [ & mut z, & mut a ] );
        vec![ x, y, z, a, b ]
    };

    let var_ids = c.fwd_pass( buf ).unwrap();
    
    let mut var_map = HashMap::new();
    for i in [ "x", "y", "z", "a", "b" ].iter().zip( var_ids ) {
        var_map.insert( i.0, i.1 );
    }

    //compute gradient of b with respect to every other variable
    {
        let mut var_grad = HashMap::new();

        let b_id = *var_map.get(&"b").unwrap();
        for i in var_map.iter() {
            let grad = c.compute_grad( b_id, *i.1 ).unwrap();
            var_grad.insert( *i.0, grad );
        }

        assert_eq!( c.get_var(*var_map.get(&"z").unwrap()).unwrap()._val, &[ 42f64, 15f64 ] );
        assert_eq!( c.get_var(*var_map.get(&"x").unwrap()).unwrap()._val, &[ 6f64,  5f64  ] );
        assert_eq!( c.get_var(*var_map.get(&"y").unwrap()).unwrap()._val, &[ 7f64,  3f64  ] );
        assert_eq!( c.get_var(*var_map.get(&"b").unwrap()).unwrap()._val, &[ 45f64, 23f64 ] );
        assert_eq!( c.get_var(*var_map.get(&"a").unwrap()).unwrap()._val, &[ 3f64,  8f64  ] );


        assert_eq!( var_grad.get(&"z").unwrap(), &[ 1f64, 1f64 ] );
        assert_eq!( var_grad.get(&"x").unwrap(), &[ 7f64, 3f64 ] );
        assert_eq!( var_grad.get(&"y").unwrap(), &[ 6f64, 5f64 ] );
        assert_eq!( var_grad.get(&"b").unwrap(), &[ 1f64, 1f64 ] );
        assert_eq!( var_grad.get(&"a").unwrap(), &[ 1f64, 1f64 ] );
    }

    //compute gradient of z with respect to a
    {
        let z_id = *var_map.get(&"z").unwrap();
        let a_id = *var_map.get(&"a").unwrap();
        let grad = c.compute_grad( z_id, a_id ).unwrap();
        assert_eq!( &grad[..], &[ 0f64, 0f64 ] );
    }
}


#[test]
fn autograd_scalar_vector_op() {

    let mut c : autograd::Context = Default::default();

    //setup variables
    let buf = {
        let mut x = c.init_var( &[ 6f64 ] );
        let mut y = c.init_var( &[ 7f64, 3f64 ] );
        let mut z = c.init_op( autograd::OpType::Mul, & mut [ & mut x, & mut y ] );
        let mut a = c.init_var( &[ 3f64, 8f64 ] );
        let mut b = c.init_op( autograd::OpType::AddParallel, & mut [ & mut z, & mut a ] );
        vec![ x, y, z, a, b ]
    };

    let var_ids = c.fwd_pass( buf ).unwrap(); //returns ids for input ops and vars
    
    let mut var_map = HashMap::new();
    for i in [ "x", "y", "z", "a", "b" ].iter().zip( var_ids ) {
        var_map.insert( i.0, i.1 );
    }

    //compute gradient of b with respect to every other variable
    {
        let mut var_grad = HashMap::new();

        let b_id = *var_map.get(&"b").unwrap();
        for i in var_map.iter() {
            let grad = c.compute_grad( b_id, *i.1 ).unwrap();
            var_grad.insert( *i.0, grad );
        }

        //var x reshaped?
        assert_eq!( c.get_var(*var_map.get(&"x").unwrap()).unwrap()._val.len(), 2usize );
        assert_eq!( c.get_var(*var_map.get(&"x").unwrap()).unwrap()._grad.len(), 2usize );
        
        assert_eq!( c.get_var(*var_map.get(&"z").unwrap()).unwrap()._val, &[ 42f64, 18f64 ] );
        assert_eq!( c.get_var(*var_map.get(&"x").unwrap()).unwrap()._val, &[ 6f64,  6f64  ] );
        assert_eq!( c.get_var(*var_map.get(&"y").unwrap()).unwrap()._val, &[ 7f64,  3f64  ] );
        assert_eq!( c.get_var(*var_map.get(&"b").unwrap()).unwrap()._val, &[ 45f64, 26f64 ] );
        assert_eq!( c.get_var(*var_map.get(&"a").unwrap()).unwrap()._val, &[ 3f64,  8f64  ] );


        assert_eq!( var_grad.get(&"z").unwrap(), &[ 1f64, 1f64 ] );
        assert_eq!( var_grad.get(&"x").unwrap(), &[ 7f64, 3f64 ] );
        assert_eq!( var_grad.get(&"y").unwrap(), &[ 6f64, 6f64 ] );
        assert_eq!( var_grad.get(&"b").unwrap(), &[ 1f64, 1f64 ] );
        assert_eq!( var_grad.get(&"a").unwrap(), &[ 1f64, 1f64 ] );
    }

    //compute gradient of z with respect to a
    {
        let z_id = *var_map.get(&"z").unwrap();
        let a_id = *var_map.get(&"a").unwrap();
        let grad = c.compute_grad( z_id, a_id ).unwrap();
        assert_eq!( &grad[..], &[ 0f64, 0f64 ] );
    }
}

#[test]
fn autograd_concat_op() {

    let mut c : autograd::Context = Default::default();

    //setup variables
    let buf = {
        let mut x = c.init_var( &[ 6f64 ] );
        let mut y = c.init_var( &[ 7f64, 3f64 ] );
        let mut z = c.init_var( &[ 8f64, 2f64 ] );

        let mut a = c.init_op( autograd::OpType::Concat, & mut [ & mut x, & mut y, & mut z ] );

        let mut b = c.init_var( &[ 6f64 ] );
        
        let mut d = c.init_op( autograd::OpType::Mul, & mut [ & mut a, & mut b ] );
        vec![ x, y, z, a, b, d ]
    };

    let var_ids = c.fwd_pass( buf ).unwrap(); //returns ids for input ops and vars
    
    let mut var_map = HashMap::new();
    for i in [ "x", "y", "z", "a", "b", "d" ].iter().zip( var_ids ) {
        var_map.insert( i.0, i.1 );
    }

    //compute gradient of d with respect to every other variable
    {
        let mut var_grad = HashMap::new();

        let d_id = *var_map.get(&"d").unwrap();
        for i in var_map.iter() {
            let grad = c.compute_grad( d_id, *i.1 ).unwrap();
            var_grad.insert( *i.0, grad );
        }

        assert_eq!( c.get_var(*var_map.get(&"a").unwrap()).unwrap()._val, &[ 6., 7., 3., 8., 2. ] );
        assert_eq!( c.get_var(*var_map.get(&"d").unwrap()).unwrap()._val, &[ 36., 42., 18., 48., 12.  ] );        

        assert_eq!( var_grad.get(&"a").unwrap(), &[ 6.; 5 ] );
        assert_eq!( var_grad.get(&"b").unwrap(), &[ 6., 7., 3., 8., 2. ] );
        assert_eq!( var_grad.get(&"x").unwrap(), &[ 6f64 ] );
        assert_eq!( var_grad.get(&"y").unwrap(), &[ 6f64, 6f64 ] );
        assert_eq!( var_grad.get(&"z").unwrap(), &[ 6f64, 6f64 ] );

    }
}

#[test]
fn autograd_add_all_op() {

    let mut c : autograd::Context = Default::default();

    //setup variables
    let buf = {
        let mut x = c.init_var( &[ 6f64, 5f64, 7f64, 3f64 ] );
        let mut y = c.init_op( autograd::OpType::AddAll, & mut [ & mut x ] );
        let mut z = c.init_var( &[ 3f64 ] );
        let mut d = c.init_op( autograd::OpType::Mul, & mut [ & mut y, & mut z ] );
        vec![ x, y, z, d ]
    };

    let var_ids = c.fwd_pass( buf ).unwrap(); //returns ids for input ops and vars
    
    let mut var_map = HashMap::new();
    for i in [ "x", "y", "z", "d" ].iter().zip( var_ids ) {
        var_map.insert( i.0, i.1 );
    }

    //compute gradient of d with respect to every other variable
    {
        let mut var_grad = HashMap::new();

        let d_id = *var_map.get(&"d").unwrap();
        for i in var_map.iter() {
            let grad = c.compute_grad( d_id, *i.1 ).unwrap();
            var_grad.insert( *i.0, grad );
        }
        assert_eq!( var_grad.get(&"x").unwrap(), &[ 3.; 4 ] );
        assert_eq!( var_grad.get(&"y").unwrap(), &[ 3. ] );
        assert_eq!( var_grad.get(&"z").unwrap(), &[ 6. + 5. + 7. + 3. ] );
    }
}
