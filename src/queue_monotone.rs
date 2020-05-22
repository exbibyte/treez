//! monotonic queue
//! items are ordered such that all elements are non-decreasing
//! compared using cmp::Ordering::Less (eg: e_i >= e_j, for all i > j)

use std::cmp::Ordering;
use std::collections::VecDeque;

#[cfg(test)]
extern crate rand;
#[cfg(test)]
use self::rand::{thread_rng, Rng};

pub struct QueueMonotone<T>
where
    T: Ord + Clone,
{
    /// (value of element, number of smaller elements between current element and previous element)
    data: VecDeque<(T, usize)>,

    /// logical length of data
    length: usize,

    /// this sets a moving window for data storage and computation
    auto_length: Option<usize>,
}

impl<T> QueueMonotone<T>
where
    T: Ord + Clone,
{
    fn new() -> QueueMonotone<T> {
        QueueMonotone {
            data: VecDeque::new(),
            length: 0,
            auto_length: None,
        }
    }
    /// item is pushed in the back and compressed for items that are < current item
    fn push(&mut self, i: T) {
        let mut num_elements = 0;

        let mut count = 0;

        for item in self.data.iter().rev() {
            if item.0.cmp(&i) == Ordering::Less {
                num_elements += item.1 + 1;
                count += 1;
            } else {
                break;
            }
        }

        self.data.truncate(self.data.len() - count);

        self.data.push_back((i, num_elements));
        self.length += 1;

        match self.auto_length {
            Some(x) if self.length > x => {
                self.pop();
            }
            _ => {}
        }
    }
    /// item is popped from the front
    fn pop(&mut self) {
        match self.data.front_mut() {
            Some(x) => {
                if x.1 > 0 {
                    x.1 -= 1;
                } else {
                    self.data.pop_front();
                }
                self.length -= 1;
            }
            _ => {}
        }
    }
    /// query max in queue
    fn max(&self) -> Option<&T> {
        match self.data.front() {
            Some(x) => Some(&x.0),
            _ => None,
        }
    }
    fn len(&self) -> usize {
        self.length
    }
    fn set_auto_len(&mut self, l: usize) {
        self.auto_length = Some(l);
    }
}

#[test]
fn test0() {
    let mut q: QueueMonotone<i32> = QueueMonotone::new();
    for i in 0..10 {
        q.push(i);
        assert_eq!(q.max().expect("max"), &i);
        assert_eq!(q.len(), i as usize + 1);
    }
}
#[test]
fn test1() {
    let mut q: QueueMonotone<i32> = QueueMonotone::new();
    for i in 0..10 {
        q.push(i);
    }
    for i in 0..10 {
        q.pop();
        if i != 9 {
            assert_eq!(q.max().expect("max"), &9);
        } else {
            assert_eq!(q.max().is_none(), true);
        }
        assert_eq!(q.len(), 10 - i as usize - 1);
    }
    q.pop();
    assert_eq!(q.max().is_none(), true);
}
#[test]
fn test2() {
    let mut q: QueueMonotone<i32> = QueueMonotone::new();

    const window: usize = 20;

    q.set_auto_len(window);

    let mut rng = rand::thread_rng();

    let arr: Vec<i32> = (0..100).map(|x| rng.gen_range(-1000, 1000)).collect();

    for (i, v) in arr.iter().enumerate() {
        q.push(*v);

        assert!(q.len() <= window);

        let bound_left = std::cmp::max((i + 1).saturating_sub(window), 0);

        let m = *q.max().expect("max");

        assert_eq!(m, *arr[bound_left..=i].iter().max().unwrap());
    }
}
