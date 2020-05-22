# treez

## A collection of useful data structures and algorithms

### implementations:
#### monotone queue
#### segment tree
#### rb tree
#### prefix sum
#### treap/cartesian tree
#### disjoint set
#### strongly connected components
#### backtracking

### monotone queue
```rust
    using treez::queue_monotone::QueueMonotone;
    
    let mut q : QueueMonotone<i32> = QueueMonotone::new();
    
    const window : usize = 20;

    q.set_auto_len(window);
    
    let mut rng = rand::thread_rng();

    let arr : Vec<i32> = (0..100).map(|x| rng.gen_range(-1000,1000)).collect();
    
    for (i,v) in arr.iter().enumerate() {
        
        q.push(*v);
       
        let bound_left = std::cmp::max((i+1).saturating_sub(window),0);

        let m = *q.max().expect("max");
        
        assert_eq!(m, *arr[bound_left..=i].iter().max().unwrap());
    }
```

### segment tree
#### notes: for static use after initialization
```rust
    let mut segments = vec![];
    for i in 0..10 {
        let n = (i*5, 5*i+5, i); //(left_bound,right_bound,segment_id); inclusive bounds
        segments.push( n );
    }

    let t : treez::seg::TreeSeg< i32, i32 > = treez::seg::TreeSeg::init( segments.as_slice() );
    let query_segs: HashSet<_> = t.get_segs_from_bound( (15,20) ).iter().cloned().collect();
    
    let check: HashSet<_> = [ 2, 3, 4 ].iter().cloned().collect();
    println!( "query segs: {:?}", query_segs );
    assert!( check.intersection(&query_segs).count() == check.len() );
```

### red black tree
```rust
    let mut t : treez::rb::TreeRb< isize, isize > = treez::rb::TreeRb::new();
    for i in 0..nums.len() {
        let r = nums[i];
        t.insert( r, i as isize );
    }

    for i in 0..nums.len() {
        let r = nums[i];
        let v = t.remove( &r ).expect( "remove unsuccessful" );
    }
```
        
### prefix sum
```rust
    let mut t = treez::prefix::TreePrefix< isize >::init(16);
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

### treap
#### implementation: insert, search, query_key_range( [low,high) ), split_by_key, merge_contiguous( a.keys < b.keys ), union, intersect, remove_by_key, remove_by_key_range( [low,high) )
```rust
    let mut t = treap::NodePtr::new();
    
    {
        let v = t.query_key_range( -100., 100. ).iter().
            map(|x| x.key()).collect::<Vec<_>>();
        
        assert_eq!( v.len(), 0 );
    }

    let items = vec![ 56, -45, 1, 6, 9, -30, 7, -9, 12, 77, -25 ];
    for i in items.iter() {
        t = t.insert( *i as f32, *i ).0;
    }
    
    t = t.remove_by_key_range( 5., 10. );
    
    let mut expected = items.iter().cloned().filter(|x| *x < 5 || *x >= 10 ).collect::<Vec<_>>();
    expected.sort();

    {
        let v = t.query_key_range( -100., 100. ).iter().
            map(|x| x.key()).collect::<Vec<_>>();
        
        assert_eq!( v.len(), expected.len() );

        expected.iter().zip( v.iter() )
            .for_each(|(a,b)| assert!(equal_f32( (*a as f32), *b ) ) );
    }

    let ((t1, t2), node_with_key_0 ) = t.split_by_key(0.);
    
    assert!( node_with_key_0.is_some() );
    
    let t3 = t1.merge_contiguous( t2 );

    {
        let v = t3.query_key_range( -100., 100. ).iter().
            map(|x| x.key()).collect::<Vec<_>>();
        
        assert_eq!( v.len(), expected.len() );

        expected.iter().zip( v.iter() )
            .for_each(|(a,b)| assert!(equal_f32( (*a as f32), *b ) ) );
    }
    
    let va = (100..200).map(|x| (x*2) ).collect::<Vec<i32>>();
    
    let mut t4 = treap::NodePtr::new();

    for i in va.iter() {
        t4 = t4.insert( (*i as f32), *i ).0;
    }

    let t5 = t3.union(t4);
    
    let vc = (50..70).map(|x| (x*2) ).collect::<Vec<i32>>();

    let mut t6 = treap::NodePtr::new();

    for i in vc.iter() {
        t6 = t6.insert( (*i as f32), *i ).0;
    }
    
    let t7 = t5.intersect( t6 );    
```

### disjoint set
```rust
    let mut v = Dsu::init(10);

    //1, 3, 5, 7 ,9
    for i in 0..5 {
        let j = i*2+1;
        v.merge( j, j-1 );
    }

    let ret = v.get_sets_repr();
    assert_eq!( ret.len(), 5 );

    v.merge(5,9);

    assert_eq!( v.get_sets_repr().len(), 4 );
```

### lower_bound, upper_bound
#### same logic as C++ lower/upper_bound; requires item type to have cmp::Ord trait
```rust
    let mut arr = ...
    arr.sort();
    let val = ...
    let idx = bound::upper_bound(&arr[..], &val);
    //idx in [0, arr.size]
    
    let mut arr = ...
    arr.sort();
    let val = ...
    let idx = bound::lower_bound(&arr[..], &val);
```
