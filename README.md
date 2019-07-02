# treez

## A collection of useful data structures and algorithms

### current implementations:
#### segment tree
#### rb tree
#### indexed tree
#### treap/cartesian tree
#### disjoint union set
#### strongly connected components
#### backtracking
#### sarsa Q-Learning

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

let t : treez::seg::TreeSeg< i32, i32 > = treez::seg::TreeSeg::init( segments.as_slice() );
let query_segs: HashSet<_> = t.get_segs_from_bound( (15,20) ).iter().cloned().collect();

let check: HashSet<_> = [ 2, 3, 4 ].iter().cloned().collect();
println!( "query segs: {:?}", query_segs );
assert!( check.intersection(&query_segs).count() == check.len() );

```

### red black tree  
#### implementation: array based, threshold compaction, minimal heap allocation  

#### todo: optimize internal representation and operations  

#### notes: comparable performance to BTreeMap  

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

### prefix sum tree
#### implementation: array based  

#### todo: support generic commutative operation

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

### sarsa policy search
#### implementation: using eligibility trace, configurable reward decay and rollout factors, SARSA, basic thread parallel implementation  

#### notes: This is an implementation attempt based on readings from various sources such as Reinforcement Learning by Sutton et al.

#### todo: switch to fine grained parallelism

```rust

use self::treez::sarsa;
use std::f64;

//define and implement interfaces to application specific logic
#[derive(Clone)]
pub struct GameGridWorld {
    _dim: ( usize, usize ),
    _start: ( usize, usize ),
    _end: ( usize, usize ),
    _obstacles: Vec< (usize,usize) >,
    _play_path: Vec< ( State, Action ) >,
}

#[derive(Eq, PartialEq, Clone, Hash, Debug)]
pub enum Action {
    UP,
    DOWN,
    LEFT,
    RIGHT,
    NONE,
}

#[derive(Eq, PartialEq, Clone, Hash, Debug)]
pub struct State( usize, usize );

impl sarsa::Game< State, Action > for GameGridWorld {
    fn gen_initial_state( & mut self ) -> State {
        self._play_path.clear();
        let s = State( self._start.0, self._start.1 );
        s
    }
    fn gen_possible_actions( & mut self, s: & State ) -> Vec< Action > {
        let mut actions = [ true; 4 ];
        for x in self._obstacles.iter() {
            if ( s.0 >= self._dim.0 - 1 ) || ( s.0 + 1 == x.0 ) && ( s.1 == x.1 ) {
                actions[0] = false;
            }
            if ( s.0 == 0 ) || ( ( s.0 - 1 == x.0 ) && ( s.1 == x.1 ) ) {
                actions[1] = false;
            }
            if ( s.1 >= self._dim.1 - 1 ) || ( ( s.0 == x.0 ) && ( s.1 + 1 == x.1 ) ) {
                actions[2] = false;
            }
            if ( s.1 == 0 ) || ( ( s.0 == x.0 ) && ( s.1 - 1 == x.1 ) ) {
                actions[3] = false;
            }
        }
        if s.0 >= self._dim.0 - 1 {
            actions[0] = false;
        }
        if s.0 == 0 {
            actions[1] = false;
        }
        if s.1 >= self._dim.1 - 1 {
            actions[2] = false;
        }
        if s.1 == 0 {
            actions[3] = false;
        }
        
        let v : Vec< Action > = actions.iter().zip( &[ Action::RIGHT, Action::LEFT, Action::UP, Action::DOWN ] ).filter_map( |x| if *x.0 == true { Some( x.1.clone() ) } else { None } ).collect();
        v
    }

    ///select action at current state and transition to new state with reward
    fn do_action( & mut self, s: & State, a: & Action ) -> ( sarsa::Reward, State ) {
        let s_next = match *a {
            Action::LEFT => {
                (s.0 - 1, s.1)
            },
            Action::RIGHT => {
                (s.0 + 1, s.1)
            },
            Action::UP => {
                (s.0, s.1 + 1)
            },
            Action::DOWN => {
                (s.0, s.1 - 1)
            },
            _ => { panic!{}; },
        };
        self._play_path.push( ( s.clone(), a.clone() ) );
        let r = if s_next == self._end {
            sarsa::Reward(1.)
        } else {
            sarsa::Reward(0.)
        };
        ( r, State( s_next.0, s_next.1 ) )
    }
    
    fn is_state_terminal( & mut self, s: & State ) -> bool {
        self._end.0 == s.0 && self._end.1 == s.1
    }
    fn get_state_history( & self ) -> Vec< ( State, Action ) > {
        self._play_path.clone()
    }
    fn set_state_history( & mut self, h: & [ (State, Action) ] ) {
        self._play_path = h.to_vec();
    }
}

fn main() {

    //initialize game world
    let mut game = GameGridWorld {
        _dim: (5,5),
        _start: (4, 0),
        _end: (0, 0),
         _obstacles: vec![ (1, 0), (1, 1), (1, 2) ],
        _play_path: vec![],
    };
    

    //setup search criteria and run
    let sc = sarsa::SearchCriteria {
        _lambda: 0.99,
        _gamma: 0.9,
        _alpha: 0.03,
        // _stop_limit: sarsa::StopCondition::EpisodeIter(100), //number of episodes
        _stop_limit: sarsa::StopCondition::TimeMicro( 10_000_000.0 ), //time allotted to search
        // _policy_select_method: sarsa::PolicySelectMethod::EpsilonGreedy( 0.4 ),
        _policy_select_method: sarsa::PolicySelectMethod::Softmax,
    };

    let ( _policy_map, policy_normalized, expectation, iter ) = sarsa::search( & sc, & mut game ).unwrap();

    //display results

    for i in (0..game._dim.1).rev() {
        let mut v = vec![];
        for j in 0..game._dim.0 {
            let actions_percentage = match policy_normalized.get( &State( j, i ) ) {
                Some( x ) => { x.clone() },
                _ => { vec![] },
            };
            let best = actions_percentage.iter().
                fold( (Action::NONE, f64::MIN), |accum, x| if x.1 > accum.1 { x.clone() } else { accum } );

            let a_text = {
                if (j,i) == game._end {
                    '🏁'
                } else if (j,i) == game._start {
                    '🚶'
                } else {
                    match best {
                        ( Action::UP, _ ) => { '↑' },
                        ( Action::DOWN , _) => { '↓' },
                        ( Action::LEFT, _ ) => { '←' },
                        ( Action::RIGHT, _ ) => { '→' },
                        _ => { ' ' },
                    }
                }
            };
            v.push( a_text );
        }
        println!( "{:?}", v );
    }

    println!( "linear normalized optimal policy value" );
    for i in (0..game._dim.1).rev() {
        let mut v = vec![];
        for j in 0..game._dim.0 {
            let actions_percentage = match policy_normalized.get( &State( j, i ) ) {
                Some( x ) => { x.clone() },
                _ => { vec![] },
            };
            let best = actions_percentage.iter().
                fold( (Action::NONE, f64::MIN), |accum, x| if x.1 > accum.1 { x.clone() } else { accum } );
            let val = match best {
                ( Action::NONE, _ ) => { 0. },
                ( _, x ) => { x },
            };
            v.push( val );
        }
        println!( "{:?}", v );
    }

    println!( "policy expectation value map" );
    for i in (0..game._dim.1).rev() {
        for j in 0..game._dim.0 {
            let a = expectation.get( &State( j, i ) ).unwrap_or( & 0. );
            print!( "{:.5} ", a );
        }
        print!( "\n" );
    }
}
```

### treap
#### implementation: insert, search, query_key_range( [low,high) ), split_by_key, merge_contiguous( a.keys < b.keys ), union, intersect, remove_by_key, remove_by_key_range( [low,high) )
#### todo: optimize

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
