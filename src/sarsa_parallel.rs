///Implementation of sarsa Q-learning
///using eligibility trace update with replacement update,
///rollout factor, e-greedy or softmax policy selection

extern crate chrono;
extern crate rand;
extern crate crossbeam;

use std::fmt;
use std::f64;
use std::hash;
use std::collections::HashMap;
use std::collections::HashSet;
use std::cmp;

use std::sync::{Arc,Mutex};
use self::chrono::prelude::*;

use policy;
use softmax;

use std::ops::Deref;
    
///Input search parameters
#[derive(Debug,Clone)]
pub struct SearchCriteria {
    ///lambda: rollout factor
    pub _lambda: f64,
    ///gamma: discount factor
    pub _gamma: f64,
    ///alpha: correction step size
    pub _alpha: f64,
    ///search stop condition
    pub _stop_limit: StopCondition,
    ///policy selection
    pub _policy_select_method: PolicySelectMethod,
}

impl SearchCriteria {
    pub fn check( & self ) -> Result< (), & 'static str > {
        if self._lambda < 0. || self._lambda > 1. ||
           self._gamma < 0. || self._gamma > 1. ||
           self._alpha <= 0.
        {
            Err( "search criteria out of range" )
        } else {
            match self._policy_select_method {
                PolicySelectMethod::EpsilonGreedy( x ) => { if x < 0. || x > 1.0 { return Err( "search criteria out of range" ) } },
                _ => {},
            }
            Ok( () )
        }
    }
}

#[derive(Debug, Clone)]
pub enum StopCondition {
    TimeMicro(f64), //time allotted to search
    EpisodeIter(u64), //max iterations allotted to search
}

#[derive(Debug, Clone)]
pub enum PolicySelectMethod {
    EpsilonGreedy( f64 ),
    Softmax,
}

pub struct Reward(pub f64);

///extensible interface to be defined by a specific game of interest
pub trait Game< State, Action > where State: Clone + cmp::Eq + hash::Hash, Action: Clone + cmp::Eq + hash::Hash {
    ///initial state to start with
    fn gen_initial_state( & mut self ) -> State;

    ///given state, give all possible actions
    fn gen_possible_actions( & mut self, & State ) -> Vec< Action >;

    ///select action at current state and transition to new state with reward
    fn do_action( & mut self, & State, & Action ) -> (Reward, State);

    fn is_state_terminal( & mut self, s: & State ) -> bool;

    fn get_state_history( & self ) -> Vec< ( State, Action ) >;

    fn set_state_history( & mut self, h: & [ (State, Action) ] );
}

///main entry point for search
pub fn search< G, State, Action >( criteria: & SearchCriteria,
                                   g_orig: & mut G )
                                   -> Result< ( HashMap< ( State, Action ), f64 >,
                                                HashMap< State, Vec< ( Action, f64 ) > >,
                                                HashMap< State, f64 >,
                                                u64 ), & 'static str >
    where G: Game< State, Action > + Send + Clone, State: Clone + cmp::Eq + hash::Hash + fmt::Debug + Send + Sync + 'static, Action: Clone + cmp::Eq + hash::Hash + fmt::Debug + Send {

    criteria.check()?;

    //init (state,action) -> value map
    let policy_values : Arc<Mutex<HashMap<(State,Action),f64>>> = Arc::new( Mutex::new( HashMap::new() ) ); // HashMap< (State, Action), f64 >

    let t0 = Local::now();
    let mut iter = 0u64;

    //init state to some set value of interest
    let g = Arc::new( Mutex::new( g_orig ) );
    let state_init = {
        let g_arc = g.clone();
        let mut g_access = g_arc.lock().unwrap();
        Arc::new( g_access.gen_initial_state().clone() ) //save this state to be reset at the start of an episode
    };

    let num_threads = 4;

    if { let g_arc = g.clone();
         let mut g_access = g_arc.lock().unwrap();
         !g_access.is_state_terminal( & state_init ) } {

        let mut arr_data = vec![];
        
        for _ in 0..num_threads {

            let s : &State = state_init.deref();
            
            let g_arc = g.clone();
            let g_access = g_arc.lock().unwrap();
            let g_copy : G = g_access.clone();

            arr_data.push( ( s.clone(), criteria.clone(), g_copy, g_arc.clone(), policy_values.clone() ) );
        }

        loop {
            crossbeam::scope( |scope| {
                
                for (_k, data) in arr_data.iter_mut().enumerate() {
                    
                    scope.spawn( move || {

                        let mut eligibility_trace : HashMap< (State, Action), f64 > = HashMap::new();
                        
                        // println!("thread {}", _k );

                        //initialization prior to rollout
                        let s_orig : & mut State = & mut data.0;
                        let mut s_context : State = s_orig.clone();
                        
                        let cri_context: & SearchCriteria = & mut data.1;
                        let g_context : & mut G = & mut data.2;
                        let g_global = & mut data.3;
                        *g_context = {
                            let g_global : & G = &g_global.lock().unwrap();
                            g_global.clone()
                        };
                        let policy_global : & mut Arc<Mutex<HashMap<(State,Action),f64>>> = & mut data.4;

                        //choose initial action
                        let mut action : Action = {
                            let possible_actions = g_context.gen_possible_actions( & s_context );
                            
                            match cri_context._policy_select_method {
                                PolicySelectMethod::EpsilonGreedy( epsilon ) => {
                                    let action_greedy = policy::get_greedy_action_at_state( & policy_global.lock().unwrap(), & s_context );
                                    policy::e_greedy_select( epsilon, possible_actions.as_slice(), & action_greedy )
                                },
                                PolicySelectMethod::Softmax => {
                                    //obtain policy values for currently available actions at current state
                                    let mut vals = softmax::Distr(vec![]);
                                    for (_k,i) in possible_actions.iter().enumerate() {
                                        let val = match policy_global.lock().unwrap().get( &( s_context.clone(), i.clone() ) ) {
                                            Some( x ) => *x,
                                            None => 0.,
                                        };
                                        vals.0.push(val);
                                    }
                                    policy::softmax_select( possible_actions.as_slice(), & vals )
                                },
                            }
                        };

                        // let mut inner_iter = 0u64;
                        //perform rollout 
                        loop {
                            if g_context.is_state_terminal( & s_context ) {
                                break;
                            }
                            
                            let ( reward, state_next ) = g_context.do_action( & s_context, & action );

                            //choose action using e-greedy policy selection
                            let action_next : Action = {
                                let possible_actions = g_context.gen_possible_actions( & state_next );

                                match cri_context._policy_select_method {
                                    PolicySelectMethod::EpsilonGreedy( epsilon ) => {
                                        let action_greedy = policy::get_greedy_action_at_state( & policy_global.lock().unwrap(), & state_next );
                                        policy::e_greedy_select( epsilon, possible_actions.as_slice(), & action_greedy )
                                    },
                                    PolicySelectMethod::Softmax => {
                                        //obtain policy values for currently available actions at current state
                                        let mut vals = softmax::Distr(vec![]);
                                        for (_k,i) in possible_actions.iter().enumerate() {
                                            let val = match policy_global.lock().unwrap().get( &( state_next.clone(), i.clone() ) ) {
                                                Some( x ) => *x,
                                                None => 0.,
                                            };
                                            vals.0.push(val);
                                        }
                                        policy::softmax_select( possible_actions.as_slice(), & vals )
                                    },
                                }
                            };

                            let td_error = {
                                
                                let q_next = { policy_global.lock().unwrap().get( &( state_next.clone(), action_next.clone() ) ).unwrap_or(&0.).clone() };
                                let q = { policy_global.lock().unwrap().get( &( s_context.clone(), action.clone() ) ).unwrap_or(&0.).clone() };       
                                reward.0 + criteria._gamma * q_next - q
                            };

                            //update eligibility trace
                            {
                                let eligibility = eligibility_trace.entry( ( s_context.clone(), action.clone() ) ).or_insert( 0. );
                                *eligibility = 1.;
                            }

                            //remove loops and zero out eligibility values for items in loops
                            let mut loop_detector = HashMap::new();
                            let mut items_in_path = HashSet::new();
                            let mut items_in_loops = HashSet::new();
                            let trace = g_context.get_state_history();
                            
                            for i in 0..trace.len() {
                                let t =  & trace[i];
                                let exists = match loop_detector.get( t ) {
                                    None => { false },
                                    Some(_) => { true },
                                };
                                if exists {
                                    let index = *loop_detector.get( t ).unwrap();
                                    for j in index..i {
                                        items_in_path.remove( &j );
                                        items_in_loops.insert( j );
                                    }
                                    loop_detector.insert( t.clone(), i );
                                    items_in_path.insert( i );
                                    items_in_loops.remove( &i );
                                } else {
                                    loop_detector.insert( t.clone(), i );
                                    items_in_path.insert( i );
                                    items_in_loops.remove( &i );
                                }
                            }

                            // for i in items_in_loops.iter() {
                            //     let t = & trace[ *i ];                
                            //     let v = eligibility_trace.get_mut( t ).unwrap();
                            //     //eligibility trace decay
                            //     *v = criteria._gamma * criteria._lambda * *v;
                            // }

                            let normalized_policies = normalized_policy_actions( & policy_global.lock().unwrap() );

                            //todo: consider possibly only updating a limited number of policy values close to the current state
                            for i in items_in_path.iter().take(2) {
                                let t = & trace[ *i ];

                                //update policy value
                                let qq = *policy_global.lock().unwrap().get( t ).unwrap_or(& 0.);
                                
                                let v = eligibility_trace.get_mut( t ).unwrap();

                                let alpha_adjust = match normalized_policies.get( t ) {
                                    Some(x) => { ( 1. - if x.is_nan() { 0. } else { *x } ) * criteria._alpha },
                                    None => { criteria._alpha },
                                };
                                let n = qq + alpha_adjust * td_error * *v / num_threads as f64;
                                
                                policy_global.lock().unwrap().insert( ( t.0.clone(), t.1.clone() ), n );

                                //eligibility trace decay
                                *v = criteria._gamma * criteria._lambda * *v;
                            }

                            //filter out loops in trace history
                            let mut sorted_index = items_in_path.iter().cloned().collect::<Vec<usize> >();
                            sorted_index.sort();
                            let history_filtered : Vec< (State, Action) > = sorted_index.iter().map( |x| trace[*x].clone() ).collect();
                            g_context.set_state_history( history_filtered.as_slice() );

                            //save state and action
                            s_context = state_next;
                            action = action_next;

                            //stopping condition check
                            let t1 = Local::now();
                            match criteria._stop_limit {
                                StopCondition::TimeMicro(t) => {
                                    let t_delta = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
                                    if t_delta >= t {
                                        break;
                                    }
                                },
                                _ => {},
                            }

                            // if inner_iter >= 100 {
                            //     break;
                            // }
                            // inner_iter += 1;
                            // println!("thread: {}, iter: {}", _k, inner_iter );
                        }

                    } );
                }
            });

            //stopping condition check
            let t1 = Local::now();
            match criteria._stop_limit {
                StopCondition::TimeMicro(t) => {
                    let t_delta = t1.signed_duration_since(t0).num_microseconds().unwrap() as f64;
                    if t_delta >= t {
                        break;
                    }
                },
                StopCondition::EpisodeIter(n) => {
                    if iter >= n {
                        break;
                    }
                },
            }
            iter += 1;
            // println!("outer iter: {}", iter );
        }
    }
    
    {
        let policy_values_raw = policy_values.lock().unwrap();
        let policy_normalized = normalized_policy_actions_array( & policy_values_raw );
        let expect = get_expectation_policy( & policy_values_raw );
        let pv = policy_values_raw.deref() as &HashMap<(State,Action),f64>;
        Ok( ( pv.clone(), policy_normalized, expect, iter ) )
    }
}

fn get_expectation_policy< State, Action >( policy_map: & HashMap< (State, Action), f64 > )
                                            -> HashMap< State, f64 >
    where State: Clone + cmp::Eq + hash::Hash + fmt::Debug,
          Action: Clone + cmp::Eq + hash::Hash + fmt::Debug {
    
    let mut h : HashMap< State, f64 > = HashMap::new();
    for i in policy_map.iter() {
        let x = h.entry( (i.0).0.clone() ).or_insert( 0. );
        *x += *i.1;
    }
    h
}

fn normalized_policy_actions< State, Action >( policy_map: & HashMap< (State, Action), f64 > ) -> HashMap< (State, Action), f64 >
    where State: Clone + cmp::Eq + hash::Hash + fmt::Debug,
          Action: Clone + cmp::Eq + hash::Hash + fmt::Debug {

    let mut h : HashMap< State, Vec< ( Action, f64 ) > > = HashMap::new();
    for i in policy_map.iter() {
        let v = h.entry( (i.0).0.clone() ).or_insert( vec![] );
        v.push( ( (i.0).1.clone(), *i.1 ) );
    }
    
    for i in h.iter_mut() {
        let bounds = i.1.iter()
            .fold( ( f64::MAX, f64::MIN ), |accum, x| {
                let mut b_l = accum.0;
                let mut b_h = accum.1;
                if x.1 <= b_l { b_l = x.1 }
                if x.1 >= b_h { b_h = x.1 }
                ( b_l, b_h )
            } );
        let total = i.1.iter().fold( 0., |accum, x| {
            accum + x.1 - bounds.0
        } );
        *i.1 = i.1.iter().cloned().map( |x| ( x.0, ( x.1 - bounds.0 ) / total ) ).collect();
    }
    let mut ret = HashMap::new();
    for i in h.iter() {
        for j in i.1.iter() {
            ret.insert( ( i.0.clone(), j.0.clone() ), j.1.clone() );
        }
    }
    ret
}

fn normalized_policy_actions_array< State, Action >( policy_map: & HashMap< (State, Action), f64 > ) -> HashMap< State, Vec< ( Action, f64 ) > >
    where State: Clone + cmp::Eq + hash::Hash + fmt::Debug,
          Action: Clone + cmp::Eq + hash::Hash + fmt::Debug {

    let mut h : HashMap< State, Vec< ( Action, f64 ) > > = HashMap::new();
    for i in policy_map.iter() {
        let v = h.entry( (i.0).0.clone() ).or_insert( vec![] );
        v.push( ( (i.0).1.clone(), *i.1 ) );
    }
    
    for i in h.iter_mut() {
        let bounds = i.1.iter()
            .fold( ( f64::MAX, f64::MIN ), |accum, x| {
                let mut b_l = accum.0;
                let mut b_h = accum.1;
                if x.1 <= b_l { b_l = x.1 }
                if x.1 >= b_h { b_h = x.1 }
                ( b_l, b_h )
            } );
        let total = i.1.iter().fold( 0., |accum, x| {
            accum + x.1 - bounds.0
        } );
        *i.1 = i.1.iter().cloned().map( |x| ( x.0, ( x.1 - bounds.0 ) / total ) ).collect();
    }
    h
}
