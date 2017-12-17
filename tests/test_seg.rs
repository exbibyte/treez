use std::collections::HashSet;

extern crate treez;

#[test]
fn init() {
    
    let mut segments = vec![];
    for i in 0..10 {
        let n = (i*5, 5*i+5, i);
        segments.push( n );
    }
    let t : treez::seg::TreeSeg = treez::seg::TreeSeg::init( segments.as_slice() );
    assert!( t.len_nodes() == 21 );

    {
        let check: HashSet<_> = [ 1, 2, 3 ].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound( (10,15) ).iter().cloned().collect();
        println!( "query segs: {:?}", query_segs );
        assert!( check.intersection(&query_segs).count() == check.len() );
    }
    {
        let check: HashSet<_> = [ 2, 3, 4 ].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound( (15,20) ).iter().cloned().collect();
        println!( "query segs: {:?}", query_segs );
        assert!( check.intersection(&query_segs).count() == check.len() );
    }
    {
        //point query
        let check: HashSet<_> = [ 2 ].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound( (12,12) ).iter().cloned().collect();
        println!( "query segs: {:?}", query_segs );
        assert!( check.intersection(&query_segs).count() == check.len() );
    }
    {
        //out of bound query
        let check: HashSet<_> = [ ].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound( (-99,-5) ).iter().cloned().collect();
        println!( "query segs: {:?}", query_segs );
        assert!( check.intersection(&query_segs).count() == check.len() );
    }
    {
        //out of bound query
        let check: HashSet<_> = [ ].iter().cloned().collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound( (100,200) ).iter().cloned().collect();
        println!( "query segs: {:?}", query_segs );
        assert!( check.intersection(&query_segs).count() == check.len() );
    }
    {
        //collect all segs
        let check: HashSet<_> = (0..10).collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound( (0,50) ).iter().cloned().collect();
        println!( "query segs: {:?}", query_segs );
        assert!( check.intersection(&query_segs).count() == check.len() );
    }
    {
        //collect all segs
        let check: HashSet<_> = (0..10).collect();
        let query_segs: HashSet<_> = t.get_segs_from_bound( (-9,100) ).iter().cloned().collect();
        println!( "query segs: {:?}", query_segs );
        assert!( check.intersection(&query_segs).count() == check.len() );
    }
}
