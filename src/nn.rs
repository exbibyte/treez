/// work in progress

use std::collections::{HashMap,HashSet};

extern crate rand;
use self::rand::Rng;
use self::rand::distributions::{Distribution, Uniform};

use autograd;

pub enum Connectivity {
    Full,
    ///between (0,1) to indicate fractional randomized connectivity
    Partial(f32),
}

pub struct NeuralNetwork {
    _context: autograd::Context,
    _link_buf: Vec< autograd::Link >,
    _input_idx: usize,
    _output_idx: usize,
    _weight_ids: HashMap<usize,usize>, //map link id to index in buffer
    _output_ids: HashMap<usize,usize>,
}

#[derive(PartialEq, Eq, PartialOrd)]
struct NodeGroupId(usize);

#[derive(PartialEq, Eq, PartialOrd)]
pub enum LayerSize {
    Manual(u32),
    SameAsInput,
    Auto,
}

impl NeuralNetwork {

    /// work in progress
    fn auto_generate_classification_layers( & mut self, input_size: usize, num_classes: usize ) -> Result<(), &'static str> {
        
        let num_input_nodes = input_size;
        let num_fully_connected = num_classes*2;
        
        let mut rng = rand::thread_rng();

        //generate weights for each input and additional weights for bias per fully connected node
        let w = (0..num_input_nodes + num_fully_connected).map(|x| {
            let weight :f64 = rng.gen_range(-0.01, 0.01);
            weight
        }).collect::<Vec<f64>>();

        let mut link_weights = self._context.init_var( &w[..] );

        //allocate space for input defaulted to 0 and bias defaulted to 1
        let mut temp = vec![0.; num_input_nodes];
        temp.extend_from_slice( &vec![1.; num_fully_connected] );
        
        let mut link_inputs = self._context.init_var( & mut temp );
        
        let mut link_wi = self._context.init_op( autograd::OpType::Mul, & mut [ & mut link_inputs, & mut link_weights ] );
        
        let mut link_fcs = (0..num_fully_connected).map(|x|{
            let fc = self._context.init_op( autograd::OpType::AddAll, & mut [ & mut link_wi ] );
            fc
        }).collect::<Vec<_>>();
            
        self._link_buf.push(link_weights);

        let idx_link_input_start = self._link_buf.len();
        self._link_buf.push(link_inputs);
        self._input_idx = idx_link_input_start;
        
        self._link_buf.push(link_wi);
        
        let mut link_nls = link_fcs.iter_mut().map(|x|{
            let nl = self._context.init_op( autograd::OpType::Sigmoid, & mut [ x ] );
            nl
        }).collect::<Vec<_>>();

        let idx_link_nls_start = self._link_buf.len();
        for i in link_nls {
            self._link_buf.push(i);
        }

        let mut link_nls_concat = {
            let mut link_nls_refs : Vec<& mut autograd::Link> = self._link_buf.iter_mut().skip(idx_link_nls_start).map(|x| x).collect();
            self._context.init_op( autograd::OpType::Concat, & mut link_nls_refs[..])
        };
        
        for i in link_fcs {
            self._link_buf.push(i);
        }
        
        let mut link_softmax_denom = self._context.init_op( autograd::OpType::AddAll, & mut [ & mut link_nls_concat ] );
        
        let mut link_softmax = self._context.init_op( autograd::OpType::Div, & mut [ & mut link_nls_concat, & mut link_softmax_denom ] );

        self._link_buf.push( link_nls_concat );
        self._link_buf.push( link_softmax_denom );
        self._output_idx = self._link_buf.len();
        self._link_buf.push( link_softmax );
        
        Ok( () )
    }

    /// work in progress

    fn load_input( & mut self, input: &[f64] ) -> Result< (), &'static str > {

        let mut input_link_ref = self._link_buf.iter_mut().nth(self._input_idx).expect("input link index error");

        //set input values which lies in the first section
        let val : & mut [f64] = input_link_ref.get_val_mut();
        val.iter_mut().take(input.len()).zip( input.iter() ).for_each(|(x,y)| *x=*y);
        
        Ok( () )
    }

    /// work in progress
    fn train_cross_entropy( & mut self, sample_train: &[&[f64]], sample_validation: &[&[f64]] ) {
        unimplemented!();
    }
}
