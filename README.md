# treez

## A collection of useful data structures  
current implementations: segment tree, rb tree, autograd, indexed tree, sarsa search

work in progress: variants of mcts-related search

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

#### todo: optimize internal representation and operations, generic type

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

### sarsa search

```rust

use self::treez::sarsa;
use std::f64;

//define and implement interfaces to application specific logic
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

    let ( _policy_map, policy_normalized, expectation ) = sarsa::search( & sc, & mut game ).unwrap();

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