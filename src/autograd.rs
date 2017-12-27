// use std::collections::HashMap;
// use std::collections::VecDeque;
use std::fmt;

///implementation of reverse automatic differentiation

#[derive(Clone, Debug)]
pub struct Link {
    //incoming nodes in the forward computation graph
    pub _precedent: Vec< usize >,
    //outgoing nodes in the forward computation graph
    pub _descendent: Vec< usize >,
    pub _op: Box< Op >,
    pub _val: f64,
    pub _grad: f64,
    
}

impl Clone for Box< Op > {
    fn clone( &self ) -> Box< Op > {
        self.box_clone()
    }
}

impl Default for Link {
    fn default() -> Link {
        Link {
            _precedent: vec![],
            _descendent: vec![],
            _op: Box::new( OpLeaf{} ),
            _val: 0f64,
            _grad: 0f64,
        }
    }
}
impl Link {
    pub fn get_precedent( & self ) -> &[usize] {
        self._precedent.as_slice()
    }
    pub fn get_descendent( & self ) -> &[usize] {
        self._descendent.as_slice()
    }
    pub fn set_precedent(  & mut self, input: &[usize] ) {
        for i in input {
            self._precedent.push( *i );
        }
    }
    pub fn set_descendent( & mut self, input: &[usize] ) {
        for i in input {
            self._descendent.push( *i );
        }
    }
    pub fn clear_precedent( & mut self ) {
        self._precedent.clear();
    }
    pub fn clear_descendent( & mut self ) {
        self._descendent.clear();
    }
    pub fn check( & self ) -> Result< (), &'static str > {
        if self._op.get_arity() != self._precedent.len() {
            return Err( "op arity not match" )
        }
        Ok( () )
    }
}

///checker for link validity, computes forward values, and outputs forward pass order
pub fn check_links( link: & mut [Link] ) -> Result< Vec<usize>, &'static str > {
    //collect all leaf links that have no incoming dependencies

    let mut l : Vec< usize > = link.iter().enumerate().filter_map( |(e,x)| if x._precedent.len() == 0 { Some(e) } else { None } ).collect();
    // println!("collected leaves: {:?}", l );
    let mut checked = vec![ false; link.len() ];
    let mut temp : Vec< usize > = vec![];
    let mut fwd_order = vec![];
    // println!("link.len: {:?}", link.len() );
    while l.len() > 0 || temp.len() > 0 {
        // println!("l.len: {:?}", l.len() );
        for i in l.iter() {
            // println!("checking: {}", i );
            link[*i].check()?;
            let ret = {
                //get values from precendent and compute forward val
                let mut params = vec![];
                for j in link[*i].get_precedent() {
                    let ref v = link[*j]._val;
                    params.push(v);
                }
                link[*i]._op.exec( params.as_slice() )
            };
            if ret.len() > 0 {
                //store forward val
                link[*i]._val = ret[0];
            }
            //queue descendents
            for j in link[*i].get_descendent() {
                if checked[*j] == false {
                    checked[*j] = true;
                    temp.push(*j);
                }
            }
            fwd_order.push(*i);
        }
        l = temp.drain(..).collect();
    }
    Ok( fwd_order )
}

pub fn compute_grad( link: & mut [Link], eval_order: &[usize] ) -> Result< (), &'static str > {
    if eval_order.len() > 0 {
        link[*eval_order.first().unwrap()]._grad = 1f64;
    }
    for i in eval_order {
        //get values from precendent
        let mut params = vec![];
        for j in link[*i].get_precedent() {
            let ref v = link[*j]._val;
            params.push(v);
        }
        //compute and accumulate backward gradient
        let g = link[*i]._op.get_grad( params.as_slice() );
        // println!("i: {}", *i );
        // println!( "g: {:?}", g );
        // println!( "prece: {:?}", link[*i].get_precedent().len() );
        assert!( g.len() == link[*i].get_precedent().len() );
        let mut index = 0;
        let v =  { link[*i].get_precedent().iter().cloned().collect::<Vec< usize > >() };
        for j in v {
            link[j]._grad += g[index] * link[*i]._grad;
            index += 1;
        }
    }
    Ok( () )
}

///forward Op and gradient interface
pub trait Op : fmt::Debug {
    fn box_clone( & self ) -> Box< Op >;
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result;
    fn exec( & self, & [ & f64 ] ) -> Vec< f64 >;
    fn get_grad( & self, & [ & f64 ] ) -> Vec< f64 >;
    fn get_arity( & self ) -> usize;
}
    
///y = constant; y' = 0;
#[derive(Clone, Debug)]
struct OpLeaf {}
impl Op for OpLeaf {
    fn box_clone( & self ) -> Box< Op > {
        Box::new( (*self).clone() )
    }
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self )
    }
    fn get_grad( &self, input: &[ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 0 );
        vec![]
    }
    fn get_arity( &self ) -> usize {
        0
    }
    fn exec( & self, _input: & [ & f64 ] ) -> Vec< f64 > {
        vec![]
    }
}

///y = a*b; dy/da = b; dy/db = a
#[derive(Clone, Debug)]
pub struct OpMul {}
impl Op for OpMul {
    fn box_clone( & self ) -> Box< Op > {
        Box::new( (*self).clone() )
    }
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self )
    }
    fn get_grad( &self, input: &[ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        vec![ *input[1],
              *input[0] ]
    }
    fn get_arity( &self ) -> usize {
        2
    }
    fn exec( & self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        vec![ input[0] * input[1] ]
    }
}

///y = a + b; dy/da = 1; dy/db = 1;
#[derive(Clone, Debug)]
pub struct OpAdd {}
impl Op for OpAdd {
    fn box_clone( & self ) -> Box< Op > {
        Box::new( (*self).clone() )
    }
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self )
    }
    fn get_grad( &self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        vec![ *input[0],
              *input[1] ]
    }
    fn get_arity( &self ) -> usize {
        2
    }
    fn exec( & self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        vec![ input[0] + input[1] ]
    }
}

// ///grad(sin(x)) -> cos(x)
// struct OpSin {}
// impl Op for OpSin {
//     fn get_grad( &self, input: &[f64] ) -> f64 {
//         assert!( input.len() > 0 );
//         input[0].cos()
//     }
//     fn get_type( &self ) -> OpType {
//         OpType::SIN
//     }
//     fn get_arity( &self ) -> usize {
//         1
//     }
// }

// ///grad(cos(x)) -> -sin(x)
// struct OpCos {}
// impl Op for OpCos {
//     fn get_grad( &self, input: &[f64] ) -> f64 {
//         assert!( input.len() > 0 );
//         -input[0].sin()
//     }
//     fn get_type( &self ) -> OpType {
//         OpType::COS
//     }
//     fn get_arity( &self ) -> usize {
//         1
//     }
// }


// ///grad(tan(x)) -> 1/(cos(x))^2
// struct OpTan {}
// impl Op for OpTan {
//     fn get_grad( &self, input: &[f64] ) -> f64 {
//         assert!( input.len() > 0 );
//         let a = input[0].cos();
//         1f64/(a*a)
//     }
//     fn get_type( &self ) -> OpType {
//         OpType::TAN
//     }
//     fn get_arity( &self ) -> usize {
//         1
//     }
// }

// ///grad(a^x) -> ln(a)*a^x
// struct OpExponential {}
// impl Op for OpExponential {
//     fn get_grad( &self, input: &[f64] ) -> f64 {
//         assert!( input.len() >= 2 );
//         let a = input[0];
//         let x = input[1];
//         a.ln() * a.powf( x )
//     }
//     fn get_type( &self ) -> OpType {
//         OpType::EXPONENTIAL
//     }
//     fn get_arity( &self ) -> usize {
//         2
//     }
// }

// ///grad(log_base(x)) -> 1/(x*ln(base))
// struct OpLog {}
// impl Op for OpLog {
//     fn get_grad( &self, input: &[f64] ) -> f64 {
//         assert!( input.len() >= 2 );
//         let base = input[0];
//         let x = input[1];
//         1f64 / ( x * base.ln() )
//     }
//     fn get_type( &self ) -> OpType {
//         OpType::LOG
//     }
//     fn get_arity( &self ) -> usize {
//         2
//     }
// }
