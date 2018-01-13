pub struct Distr( pub Vec< f64 > );

impl Distr {
    ///initialize a uniform unnormalized distribution of n bins
    pub fn init( n: usize ) -> Distr {
        assert!( n > 0 );
        let d = 1f64;
        Distr( (0..n).map(|_| d ).collect() )
    }
    ///computes cumulative distribution
    pub fn eval_cdf( & self ) -> Vec< f64 > {
        let total = (self.0).iter().fold( 0., |accum,x| accum + x.exp() );
        let mut accum = 0.;
        (self.0).iter().map(|x| { accum += x.exp(); accum / total } ).collect()
    }
}



