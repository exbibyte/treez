use std::collections::HashMap;
// use std::collections::VecDeque;
use std::fmt;
use std::cell::Cell;
use std::cmp;

///implementation of reverse automatic differentiation
pub struct Context {
    _id: Cell< usize >,
    _id_map: HashMap< usize, usize >,
    _eval_order: Vec< usize >,
    _is_evaluated: usize,
    _buf: Vec< Link >,
    _eval_order_map: HashMap< usize, usize >,
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
            _id: Cell::new( 0usize ),
            _id_map: HashMap::new(),
            _eval_order: vec![],
            _is_evaluated: <usize>::max_value(),
            _buf: vec![],
            _eval_order_map: HashMap::new(),
        }
    }
}

impl Context {
    fn gen_id( & mut self ) -> usize {
        let a = self._id.get();
        *self._id.get_mut() = a + 1usize;
        a + 1
    }
    pub fn get_var( & mut self, id: usize ) -> Option< & Link > {
        match self._id_map.get( &id ) {
            Some( &i ) => Some( & self._buf[i] ),
            _ => None,
        }
    }
}


#[derive(Clone, Debug)]
pub struct Link {
    //incoming nodes in the forward computation graph
    pub _precedent: Vec< usize >,
    //outgoing nodes in the forward computation graph
    pub _descendent: Vec< usize >,
    pub _op: Box< Op >,
    pub _val: Vec< f64 >,
    pub _grad: Vec< f64 >,
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
            _val: vec![],
            _grad: vec![],
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
pub fn init_var( c: & mut Context, v: & [ f64 ] ) -> Link {
    let a : usize = c.gen_id();
    let mut l : Link = Default::default();
    l._id = a;
    l._val = v.to_vec();
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
        // _ => { panic!( "unsupported op" ); },
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
pub fn fwd_pass( c: & mut Context, link: & mut Vec< Link > ) -> Result< Vec< usize >, &'static str > {
    c._is_evaluated = <usize>::max_value();

    //collect all leaf links that have no incoming dependencies
    let mut l : Vec< usize > = link.iter().enumerate().filter_map( |(_,x)| if x._precedent.len() == 0 { Some(x._id) } else { None } ).collect();
    // println!("collected leaves: {:?}", l );
    let mut checked = vec![ false; link.len() ];
    let mut temp : Vec< usize > = vec![];
    let mut eval_order = vec![];

    let mut ids = vec![];

    //map id to index in vec
    c._id_map.clear();
    c._eval_order.clear();
    c._eval_order_map.clear();

    for (e,i) in link.iter_mut().enumerate() {
        c._id_map.insert( i._id, e );
        ids.push( i._id );
    }

    //initilize gradient vector of leaf variables
    for i in l.iter() {
        let index_i = *c._id_map.get(&i).unwrap();
        link[ index_i ]._grad = vec![ 0f64; link[index_i]._val.len() ];
        // println!("init grad: {:?}", link[ index_i ]._grad );
    }
    
    // println!("link.len: {:?}", link.len() );
    while l.len() > 0 || temp.len() > 0 {
        // println!("l.len: {:?}", l.len() );
        for i in l.iter() {
            // println!("checking: {}", i );

            let index_i = *c._id_map.get(i).unwrap();
            
            link[index_i].check()?;
            let ret = {

                let mut max_param_size = 0usize;
                let mut min_param_size = <usize>::max_value();
                let mut precedent_val_len = vec![];

                //presweep to determine if any scalar variable needs to be reshaped
                for j in link[index_i].get_precedent() {
                    let index_j = *c._id_map.get(j).unwrap();
                    let param_len = link[index_j]._val.len();
                    max_param_size = cmp::max( max_param_size, param_len );
                    min_param_size = cmp::min( min_param_size, param_len );
                    precedent_val_len.push( ( index_j, param_len ) );
                }

                if min_param_size != max_param_size {
                    for j in precedent_val_len {
                        let index = j.0;
                        let current_len = j.1;
                        if current_len == max_param_size {
                            continue;
                        } else if current_len == 1 {
                            //reshape this scalar to a vector
                            link[ index ]._val = {
                                let v = link[index]._val[0];
                                vec![ v; max_param_size ]
                            };
                            link[ index ]._grad = {
                                let v = link[index]._grad[0];
                                vec![ v; max_param_size ]
                            };
                        } else {
                            panic!( "variable length not consistent" )
                        }
                    }
                }

                //get values from precendent and compute forward val
                let mut params = vec![];
                for j in link[index_i].get_precedent() {
                    let index_j = *c._id_map.get(j).unwrap();
                    let v = link[index_j]._val.as_slice();
                    params.push(v);
                }
                link[index_i]._op.exec( params )
            };

            if ret.len() > 0 {
                //store forward val
                link[index_i]._val = ret;
                //initilize gradient vector of non-leaf variables
                if link[index_i]._grad.len() != link[index_i]._val.len() {
                    link[index_i]._grad = vec![ 0f64; link[index_i]._val.len() ];
                }
                // println!("init graident vector for node: {}, {:?}", index_i, link[index_i]._grad );
            }
            
            //queue descendents
            for j in link[index_i].get_descendent() {
                let index_j = *c._id_map.get(j).unwrap();
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
    
    c._buf = link.drain(..).collect();

    //save the forward order in terms of index of the input link
    c._eval_order = eval_order;

    for (e,v) in c._eval_order.iter().enumerate() {
        c._eval_order_map.insert(*v,e);
    }

    Ok( ids )
}

///computes dy/dx and other variables as well back propagating from y
pub fn compute_grad( c: & mut Context, y: usize, x: usize ) -> Result< Vec< f64 >, &'static str > {

    let index_y = *c._id_map.get(&y).unwrap();
    let index_x = *c._id_map.get(&x).unwrap();
    if c._is_evaluated ==  index_y {
        return Ok( c._buf[ index_x ]._grad.clone() )
    }

    //reset and do gradient compute starting at y
    c._is_evaluated = <usize>::max_value();

    // println!("eval order: {:?}", c._eval_order );
    
    let index_y = *c._id_map.get(&y).unwrap();

    // println!("y: {:?}", index_y );
    
    assert!( index_y < c._eval_order.len() );

    let ref mut link = & mut c._buf[..];

    //reset gradients of all variables
    for i in link.iter_mut() {
        for j in i._grad.iter_mut() {
            *j = 0f64;
        }
    }

    let index_y_order = *c._eval_order_map.get( & index_y ).unwrap();
    
    if c._eval_order.len() > 0 {
        for i in link[ c._eval_order[ index_y_order ] ]._grad.iter_mut() {
            *i = 1f64;
        }
    }
    for i in c._eval_order.iter() {
        //get values from precendent
        let mut params = vec![];
        for j in link[*i].get_precedent() {
            let index_j = *c._id_map.get(j).unwrap();
            let v = &link[index_j]._val[..];
            params.push( v );
        }

        //compute backward gradient
        let g = link[*i]._op.get_grad( params.as_slice() );
        //g is a vector of vector of computed graidents of precedents

        assert!( g.len() == link[*i].get_precedent().len() );
        let mut index = 0;
        let v =  { link[*i].get_precedent().iter().cloned().collect::<Vec< usize > >() };

        //accumulate gradients backward for precedents
        for j in v { //for each precedent

            let index_j = *c._id_map.get(&j).unwrap();

            assert!( g[index].len() == link[*i]._grad.len() );
            assert!( g[index].len() == link[index_j]._grad.len() );

            for n in 0..g[index].len() { //for each scalar in gradient vector
                link[index_j]._grad[n] += g[index][n] * link[*i]._grad[n];
            }
            index += 1;
        }
    }

    c._is_evaluated = index_y;

    let ans = link[ index_x ]._grad.clone();
    Ok( ans )
}

///forward Op and gradient interface
pub trait Op : fmt::Debug {
    fn box_clone( & self ) -> Box< Op >;
    fn box_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result;
    fn exec( & self, Vec< & [ f64 ] > ) -> Vec< f64 >;
    fn get_grad( & self, & [ & [ f64 ] ] ) -> Vec< Vec< f64 > >;
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
    fn get_grad( &self, input: & [ & [ f64 ] ] ) -> Vec< Vec< f64 > > {
        assert!( input.len() == 0 );
        vec![]
    }
    fn get_arity( &self ) -> usize {
        0
    }
    fn exec( & self, _input: Vec< & [ f64 ] > ) -> Vec< f64 > {
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
    fn get_grad( &self, input: &[ & [ f64 ] ] ) -> Vec< Vec< f64 > > {
        assert!( input.len() == 2 );
        if input[0].len() == input[1].len() {
            vec![
                (*input[1]).to_vec(),
                (*input[0]).to_vec()
            ]
        } else if input[0].len() == 1 {
            vec![
                (*input[1]).to_vec(),
                vec![ input[0][0]; input[1].len() ]
            ]
        } else if input[1].len() == 1 {
            vec![
                vec![ input[1][0]; input[0].len() ],
                (*input[0]).to_vec(),
            ]
        } else {
            panic!( "argument size invalid" );
        }
    }
    fn get_arity( &self ) -> usize {
        2
    }
    fn exec( & self, input: Vec< & [ f64 ] > ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        assert!( input[0].len() == input[1].len() );
        (*input[0]).iter().zip( (*input[1]).iter() ).map( |x| x.0 * x.1 ).collect()
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
    fn get_grad( &self, input: & [ & [ f64 ] ] ) -> Vec< Vec< f64 > > {
        assert!( input.len() == 2 );
        assert!( input[0].len() == input[1].len() );
        vec![ vec![ 1f64; (*input[1]).len() ],
              vec![ 1f64; (*input[0]).len() ] ]
    }
    fn get_arity( &self ) -> usize {
        2
    }
    fn exec( & self, input: Vec< & [ f64 ] > ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        assert!( input[0].len() == input[1].len() );
        (*input[0]).iter().zip( (*input[1]).iter() ).map( |x| x.0 + x.1 ).collect()
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
    fn get_grad( &self, input: & [ & [ f64 ] ] ) -> Vec< Vec< f64 > > {
        assert!( input.len() == 1 );
        vec![ (*input[0]).iter().map( |x| x.cos() ).collect() ]
    }
    fn get_arity( &self ) -> usize {
        1
    }
    fn exec( & self, input: Vec< & [ f64 ] > ) -> Vec< f64 > {
        assert!( input.len() == 1 );
        (*input[0]).iter().map( |x| x.sin() ).collect()
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
    fn get_grad( &self, input: & [ & [ f64 ] ] ) -> Vec< Vec< f64 > > {
        assert!( input.len() == 1 );
        vec![ (*input[0]).iter().map( |x| -x.cos() ).collect() ]
    }
    fn get_arity( &self ) -> usize {
        1
    }
    fn exec( & self, input: Vec< & [ f64 ] > ) -> Vec< f64 > {
        assert!( input.len() == 1 );
        (*input[0]).iter().map( |x| x.cos() ).collect()
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
    fn get_grad( &self, input: & [ & [ f64 ] ] ) -> Vec< Vec< f64 > > {
        assert!( input.len() == 1 );
        vec![ (*input[0]).iter().map( |x| 1f64 / ( x.cos().powf( 2f64 ) ) ).collect() ]
    }
    fn get_arity( &self ) -> usize {
        1
    }
    fn exec( & self, input: Vec< & [ f64 ] > ) -> Vec< f64 > {
        assert!( input.len() == 1 );
        (*input[0]).iter().map( |x| x.tan() ).collect()
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
    ///input[0]: bases, input[1]: exponents
    fn get_grad( &self, input: & [ & [ f64 ] ] ) -> Vec< Vec< f64 > > {
        assert!( input.len() == 2 );
        assert!( input[0].len() == input[1].len() );
        vec![ vec![ 0f64; input[0].len()],
              (*input[0])
                .iter()
                .zip( (*input[1]).iter() )
                .map( |(base,exp)|
                        (*base).ln() * (*base).powf( *exp ) )
                .collect()
        ]
    }
    fn get_arity( &self ) -> usize {
        2
    }
    ///input[0]: bases, input[1]: exponents
    fn exec( & self, input: Vec< & [ f64 ] > ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        assert!( input[0].len() == input[1].len() );
        (*input[0]).iter().zip( (*input[1]).iter() ).map( |(base,exp)| (*base).powf(*exp) ).collect()
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
    ///input[0]: bases, input[1]: nums
    fn get_grad( &self, input: & [ & [ f64 ] ] ) -> Vec< Vec< f64 > > {
        assert!( input.len() == 2 );
        // vec![ 1f64 / ( (*input[1]) * (*input[0]).ln() ) ]
        vec![ vec![ 0f64; input[0].len()],
              (*input[0])
                .iter()
                .zip( (*input[1]).iter() )
                .map( |(base,num)|
                        1f64 / ( (*num) * (*base).ln() ) )
                .collect()
        ]
    }
    fn get_arity( &self ) -> usize {
        2
    }
    ///input[0]: bases, input[1]: nums
    fn exec( & self, input: Vec< & [ f64 ] > ) -> Vec< f64 > {
        assert!( input.len() == 2 );
        assert!( input[0].len() == input[1].len() );
        (*input[0]).iter().zip( (*input[1]).iter() ).map( |(base,num)| (*num).log(*base) ).collect()
    }
}
