///Implementation of model-free SARSA uing eligibility trace update with replacement update
///specializable to full rollout as in monte carlo,
///non full rollout combinatorial DP, non full rollout TD(0),
///or somewhere in between the spectrum.
///

extern crate chrono;
extern crate rand;

use std::fmt;
use std::f64;
use std::hash;
use std::collections::HashMap;
use std::cmp;
use self::chrono::prelude::*;
use self::rand::{Rng};
use self::rand::distributions::{IndependentSample, Range};

///Input search parameters:
///lambda: rollout factor
///gamma: discount factor
///e: epsilon greedy policy proportion
///alpha: correction step size
///search stop condition
#[derive(Debug,Clone)]
pub struct SearchCriteria {
    pub _lambda: f64,
    pub _gamma: f64,
    pub _e: f64,
    pub _alpha: f64,
    pub _stop_limit: StopCondition,
}

impl SearchCriteria {
    pub fn check( & self ) -> Result< (), & 'static str > {
        if self._lambda < 0. || self._lambda > 1. ||
           self._gamma < 0. || self._gamma > 1. ||
           self._e < 0. || self._e > 1. ||
           self._alpha <= 0. {
            Err( "search criteria out of range" )
        } else {
            Ok( () )
        }
    }
}

#[derive(Debug,Clone)]
pub enum StopCondition {
    TimeMicro(f64), //time allotted to search
    EpisodeIter(usize), //max iterations allotted to search
}

pub struct Reward(pub f64);

///extensible interface to be defined by a specific game of interest
pub trait Game< State, Action > where State: Clone + cmp::Eq + hash::Hash, Action: Clone + cmp::Eq + hash::Hash {
    fn gen_initial_state( & mut self ) -> State;
    fn gen_possible_actions( & mut self, & State ) -> Vec< Action >;

    ///select action at current state and transition to new state with reward
    fn do_action( & mut self, & State, & Action ) -> (Reward, State);

    // ///undo previous action and along with it the state
    // fn undo_previous( & mut self );
    
    // ///return the state action chain from the beginning of play
    // fn get_state_action_chain( & mut self ) -> Vec< ( Action, State ) >;
    // fn set_state( & mut self, s: & State ) -> Result< (), & 'static str >;
    fn is_state_terminal( & mut self, s: & State ) -> bool;
}

///get best(highest) action at input state
fn get_greedy_action_at_state< State, Action >( hm: & HashMap< (State,Action), f64 >,
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

///for epsilon proportion of the time, explore using a random action, otherwise use greedy action
fn e_greedy_select< Action: Clone >( epsilon: f64,
                                     actions_possible: & [ Action ],
                                     action_greedy: & Option<Action> )
                                     -> Action where Action: cmp::PartialEq {
    let bounds = Range::new( 0., 1. );
    let mut rng = rand::thread_rng();    
    let r = bounds.ind_sample( & mut rng );

    let force_random = match *action_greedy {
        None => { true },
        Some(ref x) => {
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

// impl fmt::Display for HashMap< (State, Action), f64 > {
//     fn fmt( & self, f: & mut fmt::Formatter ) -> fmt::Result {
//         Ok( () )
//     }
// }

///main entry point for search
pub fn search< G, State, Action >( criteria: & SearchCriteria,
                                   g: & mut G )
                                   -> Result< HashMap< State, Action >, & 'static str >
    where G: Game< State, Action >, State: Clone + cmp::Eq + hash::Hash + fmt::Debug, Action: Clone + cmp::Eq + hash::Hash + fmt::Debug {

    criteria.check()?;

    //policy map: (state,[actions]) -> action
    // let mut policy_map : HashMap< (State, = HashMap::new();

    //init (state,action) -> value map
    let mut policy_values : HashMap< (State, Action), f64 > = HashMap::new();

    //init eligibility trace for state value estimation
    let mut eligibility_trace : HashMap< (State, Action), f64 > = HashMap::new();

    //init eligibility trace for (state, action) value estimation
                                   
    let t0 = Local::now();
    let mut iter = 0;

    //init state to some set value of interest
    let state_init = g.gen_initial_state(); //save this state to be reset at the start of an episode
    
    loop { //per episode

        println!( "iter: {}", iter );
        //reset state
        let mut state_episode = g.gen_initial_state();
        // g.set_state( & state_episode );

        if g.is_state_terminal( & state_episode ) {
            println!( "trivial terminal state detected");
            break;
        }
        
        //init action
        let mut action : Action = {
            let possible_actions = g.gen_possible_actions( & state_init );
            // println!( "possible_acitons: {:?}", possible_actions.len() );
            let action_greedy = get_greedy_action_at_state( & policy_values, & state_episode );
            e_greedy_select( criteria._e, possible_actions.as_slice(), & action_greedy )
        };

        // let mut step = 0;
        loop { //per step in episode
            // step += 1;
            // println!( "step: {}", step );
            if g.is_state_terminal( & state_episode ) {
                println!( "terminal state detected");
                break;
            }

            // println!( "state: {:?}, do_action: {:?}", state_episode, action );
            let ( reward, state_next ) = g.do_action( & state_episode, & action );
            //choose action using e-greedy policy selection
            let action_next : Action = {
                let possible_actions = g.gen_possible_actions( & state_next );
                // println!( "possible_acitons: {:?}", possible_actions );
                let action_greedy = get_greedy_action_at_state( & policy_values, & state_next );
                e_greedy_select( criteria._e, possible_actions.as_slice(), & action_greedy )
            };
            // println!( "checkpoint0");
            let error = {
                let q_next = policy_values.get( &( state_next.clone(), action_next.clone() ) ).unwrap_or(&0.);
                let q = policy_values.get( &( state_episode.clone(), action.clone() ) ).unwrap_or(&0.);
                reward.0 + criteria._gamma * q_next - q
            };
            // println!( "checkpoint");
            {
                let eligibility = eligibility_trace.entry( ( state_episode.clone(), action.clone() ) ).or_insert( 0. );
                *eligibility += 1.;
            }
            for (k,v) in eligibility_trace.iter_mut() {   
                //update policy values in eligibility trace
                let qq = *policy_values.get( &( k.0.clone(), k.1.clone() ) ).unwrap_or(& 0.);
                let n = qq + criteria._alpha * error * *v;
                // if n > 0.000001 { //arbitrary threshold
                    policy_values.insert( ( k.0.clone(), k.1.clone() ), n );                    
                // }
                //eligibility trace decay
                *v = criteria._gamma * criteria._lambda * *v;
            }
            // println!( "checkpoint2");
            //remove all eligibility trace below certain threshold
            eligibility_trace.retain( |k,v| *v > 0.00000001 );

            //save state and action
            state_episode = state_next;
            action = action_next;
        }
        //stopping condition check
        let t1 = Local::now();
        match criteria._stop_limit {
            StopCondition::TimeMicro(t) => {
                let t_delta = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
                if t_delta >= t {
                    println!("end condition");
                    break;
                }
            },
            StopCondition::EpisodeIter(n) => {
                if iter >= n {
                    println!("end condition");
                    break;
                }
            },
        }
        iter += 1;
    }
    
    Ok( get_optimal_policy( & policy_values ) )
}

fn get_optimal_policy< State, Action >( policy_map: & HashMap< (State, Action), f64 > ) -> HashMap< State, Action > where State: Clone + cmp::Eq + hash::Hash, Action: Clone + cmp::Eq + hash::Hash {
    let mut h : HashMap< State, ( Action, f64 ) > = HashMap::new();
    for i in policy_map.iter() {
        let v = h.entry( (i.0).0.clone() ).or_insert( ( (i.0).1.clone(), *i.1 ) );
        if v.1 < *i.1 {
            v.0 == (i.0).1.clone();
            v.1 == *i.1;
        }
    }
    let mut r = HashMap::new();
    for i in h.iter() {
        r.insert( i.0.clone(), (i.1).0.clone() );
    }
    r
}
