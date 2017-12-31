//use std::collections::HashSet;

extern crate treez;

#[test]
fn prefix() {
    let mut t = treez::prefix::TreePrefix::init(16);
    assert_eq!( t.get_interval(0, 15), 0isize );
    t.set(0, 5);
    assert_eq!( t.get_interval(0, 16), 5isize );
    assert_eq!( t.get_interval(0, 1), 5isize );
    t.set(1, 7);
    assert_eq!( t.get_interval(0, 16), 12isize );
    assert_eq!( t.get_interval(0, 1), 5isize );
    assert_eq!( t.get_interval(1, 2), 7isize );
    assert_eq!( t.get_interval(0, 2), 12isize );
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
}
