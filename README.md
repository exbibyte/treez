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
#### todo: vectorize operations instead of scalars, add more test coverage, tweek to more ergonomic interface  

```rust

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
```
