extern crate treez;

use self::treez::sarsa;
use std::f64;

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

#[test]
fn sarsa_grid_world() {

    //initialize game world
    let mut game = GameGridWorld {
        _dim: (5,5),
        _start: (4, 0),
        _end: (0, 0),
         _obstacles: vec![ (1, 0), (1, 1), (1, 2) ],
        _play_path: vec![],
    };
    
    
    let sc = sarsa::SearchCriteria {
        _lambda: 0.99,
        _gamma: 0.9,
        _alpha: 0.03,
        _stop_limit: sarsa::StopCondition::EpisodeIter(20), //number of episodes
        // _stop_limit: sarsa::StopCondition::TimeMicro( 1_000_000.0 ), //time allotted to search
        // _policy_select_method: sarsa::PolicySelectMethod::EpsilonGreedy( 0.4 ),
        _policy_select_method: sarsa::PolicySelectMethod::Softmax,
    };

    let ( _policy_map, policy_normalized, expectation, iter ) = sarsa::search( & sc, & mut game ).unwrap();

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

    println!( "iterations: {}", iter );
}
