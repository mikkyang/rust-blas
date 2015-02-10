// Copyright 2014 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use attribute::{
    Order,
    Transpose,
};

#[stable]
pub mod ll;
pub mod ops;

pub trait Matrix<T> {
    fn lead_dim(&self) -> i32 { self.rows() }
    fn order(&self) -> Order { Order::RowMajor }
    fn transpose(&self) -> Transpose { Transpose::NoTrans }
    fn rows(&self) -> i32;
    fn cols(&self) -> i32;
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait BandMatrix<T>: Matrix<T> {
    fn sub_diagonals(&self) -> i32;
    fn sup_diagonals(&self) -> i32;
}

#[derive(Clone, PartialEq)]
pub struct Mat<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize
}

impl<T> Mat<T> {
    fn new(n: usize, m: usize) -> Mat<T> {
        let mut data = Vec::<T>::with_capacity(n*m);
        unsafe{ data.set_len(n*m); }
        Mat::<T> { data: data, rows: n, cols: m }
    }

    fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

use std::ops::Index;
use std::ops::IndexMut;

impl<T> Index<usize> for Mat<T> {
    type Output = [T];

    fn index(&self, idx: &usize) -> &[T] {
        let beg = *idx * self.rows;
        let end = beg+self.cols;
        &self.data[beg..end]
    }
}

impl<T> IndexMut<usize> for Mat<T> {

    fn index_mut(&mut self, idx: &usize) -> &mut [T] {
        let beg = *idx * self.rows;
        let end = beg+self.cols;
        &mut self.data[beg..end]
    }
}

impl<T> Matrix<T> for Mat<T> {
    fn rows(&self) -> i32 { self.rows as i32 }
    fn cols(&self) -> i32 { self.cols as i32 }
    fn as_ptr(&self) -> *const T { self.data.as_ptr() }
    fn as_mut_ptr(&mut self) -> *mut T { self.data.as_mut_ptr() }
}

#[cfg(test)]
mod test_struct {
    use matrix::Matrix;

    impl<T> Matrix<T> for (i32, i32, Vec<T>) {
        fn rows(&self) -> i32 {
            self.0
        }

        fn cols(&self) -> i32 {
            self.1
        }

        #[inline]
        fn as_ptr(&self) -> *const T {
            self.2.as_slice().as_ptr()
        }

        #[inline]
        fn as_mut_ptr(&mut self) -> *mut T {
            self.2.as_mut_slice().as_mut_ptr()
        }
    }
}
