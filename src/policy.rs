extern crate chrono;
extern crate rand;

use std::fmt;
use std::f64;
use std::hash;
use std::collections::HashMap;
use std::cmp;
use self::rand::distributions::{IndependentSample, Range};

use softmax;

///for epsilon proportion of the time, explore using a random action, otherwise use greedy action
pub fn e_greedy_select< Action: Clone >( epsilon: f64,
                                         actions_possible: & [ Action ],
                                         action_greedy: & Option<Action> )
                                         -> Action where Action: cmp::PartialEq {
    let bounds = Range::new( 0., 1. );
    let mut rng = rand::thread_rng();    
    let r = bounds.ind_sample( & mut rng );

    let force_random = match *action_greedy {
        None => { true },
        Some(_) => {
            let a = action_greedy.clone().unwrap().clone();
            !actions_possible.iter().any( |x| *x == a )
        },
    };

    if r <= epsilon || force_random {
        let bounds_array = Range::new( 0, actions_possible.len() );
        let i = bounds_array.ind_sample( & mut rng );
        assert!( i < actions_possible.len() );
        actions_possible[ i ].clone()
    } else {
        action_greedy.clone().unwrap().clone()
    }
}

///get best(highest) action at input state
pub fn get_greedy_action_at_state< State, Action >( hm: & HashMap< (State,Action), f64 >,
                                                s: & State )
                                                -> Option< Action >
    where State: Clone + cmp::Eq + hash::Hash, Action: Clone + cmp::Eq + hash::Hash {

    let v: Vec< ((State, Action), f64) > = hm.iter().filter( |x| (x.0).0 == *s ).map(|x| ( x.0.clone(), x.1.clone() ) ).collect();
    let p = f64::MIN;
    let index = v.iter().enumerate().fold(0, |i, x| if (x.1).1 > p { x.0 } else { i } );
    if index >= v.len() {
        None
    } else {
        Some( (v[index].0).1.clone() )
    }
}

///select a policy using softmax
pub fn softmax_select< Action: Clone >( actions_possible: & [ Action ], distr: & softmax::Distr )
                                        -> Action where Action: cmp::PartialEq + fmt::Debug {
    assert!( actions_possible.len() == distr.0.len() );
    let cdf = distr.eval_cdf();
    assert!( cdf.len() > 0 );
    let val_max = cdf.last().unwrap();
    let bounds = Range::new( 0., *val_max );
    let mut rng = rand::thread_rng();    
    let r = bounds.ind_sample( & mut rng );
    let select_index = {
        let mut j = 0;
        for i in 0..cdf.len() {
            if r <= cdf[i] {
                j = i;
                break;
            }
        }
        j
    };
    assert!( select_index < cdf.len() );
    actions_possible[ select_index ].clone()
}
