use std::collections::HashMap;
use std::hash;
use std::cmp;

extern crate treez;

use self::treez::sarsa;

pub struct GameGridWorld {
    _dim: ( usize, usize ),
    _start: ( usize, usize ),
    _end: ( usize, usize ),
    _obstacles: Vec< (usize,usize) >,
    // _play_path: Vec< ( Action, State ) >,
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
        let s = State( self._start.0, self._start.1 );
        // self._play_path = vec![ ( Action::NONE, s.clone() ) ];
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
        let r = if s_next == self._end {
            sarsa::Reward(1.)
        } else {
            sarsa::Reward(0.)
        };
        // self._play_path.push( ( a.clone(), State( s_next.0, s_next.1 ) ) );
        ( r, State( s_next.0, s_next.1 ) )
    }

    // ///undo previous action and along with it the state
    // fn undo_previous( & mut self ){
    //     unimplemented!();
    // }
    
    // ///return the state action chain from the beginning of play
    // fn get_state_action_chain( & mut self ) -> Vec< ( Action, State ) > {
    //     self._play_path.clone()
    // }
    
    // fn set_state( & mut self, s: & State ) -> Result< (), & 'static str > {
        
    // }
    
    fn is_state_terminal( & mut self, s: & State ) -> bool {
        self._end.0 == s.0 && self._end.1 == s.1
    }
}

#[test]
fn sarsa_grid_world() {

    //initialize game world
    let mut game = GameGridWorld {
        _dim: (10,10),
        _start: (0,0),
        _end: (5, 5),
        // _obstacles: vec![ (14, 19), (14, 18), (14, 17) ],
        _obstacles: vec![],
        // _play_path: vec![],
    };
    
    
    let sc = sarsa::SearchCriteria {
        _lambda: 0.9,
        _gamma: 0.9,
        _e: 0.1,
        _alpha: 0.001,
        _stop_limit: sarsa::StopCondition::EpisodeIter(20),
    };

    let policy_map = sarsa::search( & sc, & mut game ).unwrap();

    println!( "post game play policy: {:?}", policy_map );
}
