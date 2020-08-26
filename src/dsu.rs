use std::sync::{Arc, RwLock, Weak};

#[derive(Clone, Debug)]
pub struct Inner<T>
where
    T: Copy + Default,
{
    pub parent: Weak<RwLock<Inner<T>>>,
    pub ssize: usize,
    pub data: T,
}

impl<T> Inner<T>
where
    T: Copy + Default,
{
    pub fn new(d: T) -> Inner<T> {
        Inner {
            parent: Weak::new(),
            ssize: 1,
            data: d,
        }
    }
}

#[derive(Clone, Debug)]
struct AInner<T: Copy + Default>(pub Arc<RwLock<Inner<T>>>);

impl<T> AInner<T>
where
    T: Copy + Default,
{
    pub fn new(data: T) -> Self {
        AInner(Arc::new(RwLock::new(Inner::new(data))))
    }
    pub fn compress(&mut self) -> AInner<T> {
        let mut access = self.0.write().unwrap();
        match access.parent.upgrade() {
            Some(x) => {
                let p = AInner(x).compress();
                access.parent = Arc::downgrade(&p.0);
                p
            }
            _ => self.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Dsu<T>
where
    T: Copy + Default,
{
    inner: AInner<T>,
}

impl<T> Default for Dsu<T>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self {
            inner: AInner::new(T::default()),
        }
    }
}

impl<T> Dsu<T>
where
    T: Copy + Default,
{
    pub fn new(data: T) -> Self {
        Self {
            inner: AInner::new(data),
        }
    }
    fn compress(&mut self) -> AInner<T> {
        self.inner.compress()
    }
    fn len(&self) -> usize {
        self.inner.0.read().unwrap().ssize
    }
    fn get_data(&self) -> T {
        self.inner.0.read().unwrap().data
    }
    fn is_same_set(&mut self, other: &mut Self) -> bool {
        let a = self.compress();
        let b = other.compress();
        Arc::ptr_eq(&a.0, &b.0)
    }
    pub fn merge(&mut self, other: &mut Dsu<T>) {
        let a = self.compress();
        let b = other.compress();
        if !Arc::ptr_eq(&a.0, &b.0) {
            let sa = a.0.read().unwrap().ssize;
            let sb = b.0.read().unwrap().ssize;
            if sa >= sb {
                a.0.write().unwrap().ssize += sb;
                b.0.write().unwrap().parent = Arc::downgrade(&a.0);
            } else {
                b.0.write().unwrap().ssize += sa;
                a.0.write().unwrap().parent = Arc::downgrade(&b.0);
            }
        }
    }
}

#[test]
fn test_dsu_default() {
    let mut v = Dsu::<()>::default();
    let mut arr = vec![];
    for _ in 0..5 {
        let mut node_new = Dsu::default();
        v.merge(&mut node_new);
        arr.push(node_new);
    }
    assert_eq!(v.len(), 6);
    for i in arr.iter_mut() {
        assert!(i.is_same_set(&mut v));
    }
}

#[test]
fn test_dsu_0() {
    let mut v = Dsu::new(10);
    let mut arr = vec![];
    for i in 0..5 {
        let mut node_new = Dsu::new(i);
        v.merge(&mut node_new);
        arr.push(node_new);
    }
    assert_eq!(v.len(), 6);
    for i in arr.iter_mut() {
        assert!(i.is_same_set(&mut v));
    }
}

#[test]
fn test_dsu_1() {
    let mut arr = vec![];
    for i in 0..6 {
        let node_new = Dsu::new(i);
        arr.push(node_new);
    }

    for (idx, i) in arr.iter_mut().enumerate() {
        assert_eq!(i.len(), 1);
        assert_eq!(i.get_data(), idx);
    }
}

#[test]
fn test_dsu_2() {
    let mut arr = vec![];
    for i in 0..6 {
        let node_new = Dsu::new(i);
        arr.push(node_new);
    }

    {
        let (l, r) = arr[..].split_at_mut(2);
        for i in l.iter_mut() {
            i.merge(&mut r[3]);
        }
    }

    {
        let (ll, rr) = arr[2..].split_at_mut(3);
        for i in ll.iter_mut() {
            i.merge(&mut rr[0]);
        }
    }
    {
        let (l, r) = arr.split_at_mut(5);
        l[0].merge(&mut r[0]);
    }

    assert_eq!(6, arr.iter().map(|x| x.len()).max().unwrap());
}

#[test]
fn test_dsu_3() {
    let mut arr = vec![];
    for i in 0..100 {
        let node_new = Dsu::new(i);
        arr.push(node_new);
    }

    use rand::distributions::{Distribution, Uniform};
    use std::collections::HashSet;

    let mut hs = HashSet::new();
    let mut rng = rand::thread_rng();
    let distr = Uniform::from(0..75);
    let distr2 = Uniform::from(25..100);
    let mut count = 0;
    while hs.len() != 100 || count < 10000 {
        count += 1;
        let mut j = distr.sample(&mut rng);
        let mut i = distr2.sample(&mut rng);
        let (l, r) = if i > j {
            std::mem::swap(&mut i, &mut j);
            arr[..].split_at_mut(j)
        } else {
            arr[..].split_at_mut(j)
        };
        if i != j {
            l[i].merge(&mut r[0]);
            hs.insert(l[i].get_data());
            hs.insert(r[0].get_data());
        }
    }
    assert_eq!(100, arr.iter().map(|x| x.len()).max().unwrap());
}
