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
	 
	 
      	
