use std::collections::HashMap;
// use std::collections::VecDeque;
use std::fmt;
use std::cell::Cell;

///implementation of reverse automatic differentiation
pub struct Context {
    _buf: Vec< Link >,
    _id: Cell< usize >,
}

#[derive(Clone, Debug)]
pub enum OpType {
    Mul,
    Add,
    Sin,
    Cos,
    Tan,
    Exponential,
    Log,
}

impl Default for Context {
    fn default() -> Context {
        Context {
            _buf: vec![],
            _id: Cell::new( 0usize ),
        }
    }
}

impl Context {
    fn gen_id( & mut self ) -> usize {
        let a = self._id.get();
        *self._id.get_mut() = a + 1usize;
        a + 1
    }
}


#[derive(Clone, Debug)]
pub struct Link {
    //incoming nodes in the forward computation graph
    pub _precedent: Vec< usize >,
    //outgoing nodes in the forward computation graph
    pub _descendent: Vec< usize >,
    pub _op: Box< Op >,
    pub _val: f64,
    pub _grad: f64,
    pub _id: usize,
    
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
            _id: 0usize,
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

pub fn init( c: & mut Context ) -> Link {
    let a : usize = c.gen_id();
    let mut l : Link = Default::default();
    l._id = a;
    l
}
pub fn init_var( c: & mut Context, v: f64 ) -> Link {
    let a : usize = c.gen_id();
    let mut l : Link = Default::default();
    l._id = a;
    l._val = v;
    l
}
pub fn init_op( c: & mut Context, op: OpType, args: & mut [ & mut Link ] ) -> Link {
    let a : usize = c.gen_id();
    let mut l : Link = Default::default();
    l._id = a;
    let b : Box< Op > = match op {
        OpType::Mul => { Box::new( OpMul{} ) },
        OpType::Add => { Box::new( OpAdd{} ) },
        OpType::Sin => { Box::new( OpSin{} ) },
        OpType::Cos => { Box::new( OpCos{} ) },
        OpType::Tan => { Box::new( OpTan{} ) },
        OpType::Exponential => { Box::new( OpExponential{} ) },
        OpType::Log => { Box::new( OpLog{} ) },
    };
    l._op = b;
    let arg_ids : Vec< usize > = args.iter().map( |x| x._id ).collect();
    l.set_precedent( arg_ids.as_slice() );
    for i in args {
        (*i).set_descendent( & [ a ] );
    }
    l
}

///checker for link validity, computes forward values, and outputs forward pass order
pub fn check_links( link: & mut [ & mut Link] ) -> Result< ( HashMap<usize,usize>, Vec<usize> ), &'static str > {
    //collect all leaf links that have no incoming dependencies


    let mut l : Vec< usize > = link.iter().enumerate().filter_map( |(_,x)| if x._precedent.len() == 0 { Some(x._id) } else { None } ).collect();
    // println!("collected leaves: {:?}", l );
    let mut checked = vec![ false; link.len() ];
    let mut temp : Vec< usize > = vec![];
    let mut eval_order = vec![];
    //map id to index in vec
    let mut id_map = HashMap::new();
    for (e,i) in link.iter().enumerate() {
        id_map.insert( i._id, e );
    }
    
    // println!("link.len: {:?}", link.len() );
    while l.len() > 0 || temp.len() > 0 {
        // println!("l.len: {:?}", l.len() );
        for i in l.iter() {
            // println!("checking: {}", i );

            let index_i = *id_map.get(i).unwrap();
            
            link[index_i].check()?;
            let ret = {
                //get values from precendent and compute forward val
                let mut params = vec![];
                for j in link[index_i].get_precedent() {
                    let index_j = *id_map.get(j).unwrap();
                    let ref v = link[index_j]._val;
                    params.push(v);
                }
                link[index_i]._op.exec( params.as_slice() )
            };
            if ret.len() > 0 {
                //store forward val
                link[index_i]._val = ret[0];
            }
            //queue descendents
            for j in link[index_i].get_descendent() {
                let index_j = *id_map.get(j).unwrap();
                if checked[index_j] == false {
                    checked[index_j] = true;
                    temp.push(*j);
                }
            }
            eval_order.push(index_i);
        }
        l = temp.drain(..).collect();
    }
    eval_order.reverse();
    //output the forward order in terms of index of the input link
    Ok( ( id_map, eval_order ) )
}

pub fn compute_grad( id_map: & HashMap< usize, usize >, link: & mut [ & mut Link], eval_order: &[usize] ) -> Result< (), &'static str > {
    if eval_order.len() > 0 {
        link[*eval_order.first().unwrap()]._grad = 1f64;
    }
    for i in eval_order {
        //get values from precendent
        let mut params = vec![];
        for j in link[*i].get_precedent() {
            let index_j = *id_map.get(j).unwrap();
            let ref v = link[index_j]._val;
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
            let index_j = *id_map.get(&j).unwrap();
            link[index_j]._grad += g[index] * link[*i]._grad;
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
        vec![ 1f64,
              1f64 ]
    }
    fn get_arity( &self ) -> usize {
        2
    }
    fn exec( & self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        vec![ input[0] + input[1] ]
    }
}

///y = sin(x); dy/dx = cos(x)
#[derive(Clone, Debug)]
pub struct OpSin {}
impl Op for OpSin {
    fn box_clone( & self ) -> Box< Op > {
        Box::new( (*self).clone() )
    }
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self )
    }
    fn get_grad( &self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 1 );
        vec![ (*input[0]).cos() ]
    }
    fn get_arity( &self ) -> usize {
        1
    }
    fn exec( & self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 1 );
        vec![ (*input[0]).sin() ]
    }
}

///y = cos(x); dy/dx = -sin(x)
#[derive(Clone, Debug)]
pub struct OpCos {}
impl Op for OpCos {
    fn box_clone( & self ) -> Box< Op > {
        Box::new( (*self).clone() )
    }
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self )
    }
    fn get_grad( &self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 1 );
        vec![ -(*input[0]).sin() ]
    }
    fn get_arity( &self ) -> usize {
        1
    }
    fn exec( & self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 1 );
        vec![ (*input[0]).cos() ]
    }
}


///y = tan(x); dy/dx =  1/(cos(x))^2
#[derive(Clone, Debug)]
pub struct OpTan {}
impl Op for OpTan {
    fn box_clone( & self ) -> Box< Op > {
        Box::new( (*self).clone() )
    }
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self )
    }
    fn get_grad( &self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 1 );
        vec![ 1f64 / ( ( *input[0] ).cos().powf( 2f64 ) ) ]
    }
    fn get_arity( &self ) -> usize {
        1
    }
    fn exec( & self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 1 );
        vec![ (*input[0]).tan() ]
    }
}

///y = a^x; dy/dx = ln(a) * a^x
#[derive(Clone, Debug)]
pub struct OpExponential {}
impl Op for OpExponential {
    fn box_clone( & self ) -> Box< Op > {
        Box::new( (*self).clone() )
    }
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self )
    }
    fn get_grad( &self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        vec![ (*input[0]).ln() * (*input[0]).powf( *input[1] ) ]
    }
    fn get_arity( &self ) -> usize {
        2
    }
    fn exec( & self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        vec![ (*input[0]).powf( *input[1] ) ]
    }
}

///y = log_base(x); dy/dx = 1/(x*ln(base))
#[derive(Clone, Debug)]
pub struct OpLog {}
impl Op for OpLog {
    fn box_clone( & self ) -> Box< Op > {
        Box::new( (*self).clone() )
    }
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self )
    }
    fn get_grad( &self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        vec![ 1f64 / ( (*input[1]) * (*input[0]).ln() ) ]
    }
    fn get_arity( &self ) -> usize {
        2
    }
    fn exec( & self, input: & [ & f64 ] ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        vec![ (*input[1]).log( *input[0] ) ]
    }
}
