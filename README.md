# treez

## A collection of useful data structures

### segment tree  
#### implementation: array based  
#### todo: generic type  
#### notes: for static use after initialization  
```rust

let mut segments = vec![];
for i in 0..10 {
    let n = (i*5, 5*i+5, i); //(left_bound,right_bound,segment_id); inclusive bounds
    segments.push( n );
}

let t : treez::seg::TreeSeg = treez::seg::TreeSeg::init( segments.as_slice() );
let query_segs: HashSet<_> = t.get_segs_from_bound( (15,20) ).iter().cloned().collect();

let check: HashSet<_> = [ 2, 3, 4 ].iter().cloned().collect();
println!( "query segs: {:?}", query_segs );
assert!( check.intersection(&query_segs).count() == check.len() );

```

### red black tree  
#### implementation: array based, threshold compaction, minimal heap allocation  
#### todo: optimize internal representation and operations, generic type  
#### notes: comparable performance to BTreeMap  

```rust

let mut t = treez::rb::TreeRb::new();
for i in 0..nums.len() {
    let r = nums[i];
    t.insert( r, i as isize );
}

for i in 0..nums.len() {
    let r = nums[i];
    let v = t.remove( &r ).expect( "remove unsuccessful" );
}

```
	 
### reverse automatic gradient differentiation  
#### implementation: array based, scalar variable  
#### todo: add more test coverage, tweek to more ergonomic interface, interval optimization

```rust

    let mut c : autograd::Context = Default::default();

    //setup variables
    let mut buf = {
        let mut x = autograd::init_var( & mut c, &[ 6f64 ] );
        let mut y = autograd::init_var( & mut c, &[ 7f64, 3f64 ] );
        let mut z = autograd::init_op( & mut c, autograd::OpType::Mul, & mut [ & mut x, & mut y ] );
        let mut a = autograd::init_var( & mut c, &[ 3f64, 8f64 ] );
        let mut b = autograd::init_op( & mut c, autograd::OpType::Add, & mut [ & mut z, & mut a ] );
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

        let b_id = *var_map.get(&"b").unwrap();
        for i in var_map.iter() {
            let grad = autograd::compute_grad( & mut c, b_id, *i.1 ).unwrap();
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
        let grad = autograd::compute_grad( & mut c, z_id, a_id ).unwrap();
        assert_eq!( &grad[..], &[ 0f64, 0f64 ] );
    }
```

### prefix sum tree
#### implementation: array based
#### todo: support generic commutative operation

```rust

    let mut t = treez::prefix::TreePrefix::init(16);
    t.set(0, 5);
    t.set(1, 7);
    t.set(10, 4);
    assert_eq!( t.get_interval(0, 16), 16isize );
    assert_eq!( t.get_interval(10, 11), 4isize );
    assert_eq!( t.get_interval(1, 11), 11isize );

    t.set(1, 9);
    assert_eq!( t.get_interval(1, 2), 9isize );
    assert_eq!( t.get_interval(1, 11), 13isize );
    assert_eq!( t.get_interval_start( 2 ), 14isize );
    assert_eq!( t.get_interval_start( 11 ), 18isize );

    t.add( 0, 1);
    assert_eq!( t.get_interval_start( 2 ), 15isize );
    assert_eq!( t.get_interval_start( 11 ), 19isize );
```