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
    fn new(m: usize, n: usize) -> Mat<T> {
        let len = m*n;
        let mut data = Vec::<T>::with_capacity(len);
        unsafe{ data.set_len(len); }
        Mat::<T> { data: data, rows: m, cols: n }
    }

    fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

use std::ops::{Index, IndexMut};

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

#[test]
fn mat_test() {
    let mut A = Mat::<f64>::new(3,3);
    //fill matrix
    for i in range(0,3) {
        for j in range(0,3) {
            A[i][j] = ((i*3+j) as f64)
        }

    }
    //test clone and eq
    let B = A.clone();
    assert!(A == B);
    A[0][0] = 10f64;
    assert!(A != B);
    //check if indexing worked and row of A is as expected
    let row0 = vec![10f64, 1f64, 2f64];
    assert_eq!(row0.as_slice(), &A[0]);
    //check B did not change
    for i in range(0,3) {
        for j in range(0,3) {
            assert_eq!(B[i][j], ((i*3+j) as f64));
        }
    }
}