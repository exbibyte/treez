///Implementation of SARSA actor critic algorithm,
///--- work in progress ---
///specializable to full rollout as in monte carlo,
///non full rollout combinatorial DP, non full rollout TD(0),
///or somewhere in between the spectrum.
///
///The implementation uses backward updates to the points in the algorithm's
///path using eligibility trace and replacing traces
///

extern crate chrono;
extern crate rand;

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
    _lambda: f64,
    _gamma: f64,
    _e: f64,
    _alpha: f64,
    _stop_limit: StopCondition,
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

pub struct Reward(f64);

///extensible interface to be defined by a specific game of interest
pub trait Game< State, Action > where State: Clone + cmp::Eq + hash::Hash, Action: Clone + cmp::Eq + hash::Hash {
    fn gen_initial_state( & mut self ) -> State;
    fn gen_possible_actions( & mut self, & State ) -> Vec< Action >;
    ///select action at current state and transition to new state with reward
    fn do_action( & mut self, & State, & Action ) -> (Reward, State);
    ///undo previous action and along with it the state
    fn undo_previous( & mut self );
    ///return the state action chain from the beginning of play
    fn get_state_action_chain( & mut self ) -> Vec< ( State, Action ) >;
    fn set_state( & mut self, s: & State ) -> Result< (), & 'static str >;
    fn is_state_terminal( & mut self, s: & State ) -> bool;
}

///get best(highest) action at input state
fn get_greedy_action_at_state< State, Action >( hm: & HashMap< (State,Action), f64 >,
                                                s: & State )
                                                -> Action
    where State: Clone + cmp::Eq + hash::Hash, Action: Clone + cmp::Eq + hash::Hash {

    let v: Vec< ((State, Action), f64) > = hm.iter().filter( |x| (x.0).0 == *s ).map(|x| ( x.0.clone(), x.1.clone() ) ).collect();
    let p = f64::MIN;
    let index = v.iter().enumerate().fold(0, |i, x| if (x.1).1 > p { x.0 } else { i } );
    (v[index].0).1.clone()
}

///for epsilon proportion of the time, explore using a random action, otherwise use greedy action
fn e_greedy_select< Action: Clone >( epsilon: f64,
                                     actions_possible: & [ Action ],
                                     action_greedy: & Action )
                                     -> Action {
    let bounds = Range::new( 0., 1. );
    let mut rng = rand::thread_rng();    
    let r = bounds.ind_sample( & mut rng );
    if r <= epsilon {
        let bounds_array = Range::new( 0, actions_possible.len() );
        let i = bounds_array.ind_sample( & mut rng );
        actions_possible[ i ].clone()
    } else {
        action_greedy.clone()
    }
}

///main entry point for search
pub fn search< G, State, Action >( criteria: & SearchCriteria,
                                   g: & mut G )
                                   -> Result< (/*policy mapping*/), & 'static str >
    where G: Game< State, Action >, State: Clone + cmp::Eq + hash::Hash, Action: Clone + cmp::Eq + hash::Hash {

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

        //reset state
        let mut state_episode = state_init.clone();
        g.set_state( & state_episode );

        if !g.is_state_terminal( & state_episode ) { break; }
        
        //init action
        let mut action : Action = {
            let possible_actions = g.gen_possible_actions( & state_init );
            let action_greedy = get_greedy_action_at_state( & policy_values, & state_episode );
            e_greedy_select( criteria._e, possible_actions.as_slice(), & action_greedy )
        };
        
        loop { //per step in episode
            if !g.is_state_terminal( & state_episode ) { break; }

            let ( reward, state_next ) = g.do_action( & state_episode, & action );
            
            //choose action using e-greedy policy selection
            let action_next : Action = {
                let possible_actions = g.gen_possible_actions( & state_next );
                let action_greedy = get_greedy_action_at_state( & policy_values, & state_next );
                e_greedy_select( criteria._e, possible_actions.as_slice(), & action_greedy )
            };
            let error = {
                let q_next = policy_values.get( &( state_next.clone(), action_next.clone() ) ).unwrap_or(&0.);
                let q = policy_values.get( &( state_episode.clone(), action.clone() ) ).unwrap_or(&0.);
                reward.0 + criteria._gamma * q_next - q
            };
            {
                let eligibility = eligibility_trace.entry( ( state_episode.clone(), action.clone() ) ).or_insert( 0. );
                *eligibility += 1.;
            }
            for (k,v) in eligibility_trace.iter_mut() {   
                //update policy values in eligibility trace
                let qq = *policy_values.get( &( k.0.clone(), k.1.clone() ) ).unwrap_or(& 0.);
                let n = qq + criteria._alpha * error * *v;
                if n > 0.000001 { //arbitrary threshold
                    policy_values.insert( ( k.0.clone(), k.1.clone() ), n );                    
                }

                //eligibility trace decay
                *v = criteria._gamma * criteria._lambda * *v;
            }
            //remove all eligibility trace below certain threshold
            eligibility_trace.retain( |k,v| *v > 0.000001 );

            //save state and action
            state_episode = state_next;
            action = action_next;
        }
        //stopping condition check
        let t1 = Local::now();
        match criteria._stop_limit {
            StopCondition::TimeMicro(t) => {
                let t_delta = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
                if t_delta >= t { break; }
            },
            StopCondition::EpisodeIter(n) => {
                if iter >= n { break; }
            },
        }
        iter += 1;
    }
    
    Ok( () )
}
