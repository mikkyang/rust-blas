// Copyright 2015 Michael Yang. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

use std::ops::{
    Mul,
};
use attribute::Transpose;
use default::Default;
use vector::Vector;
use matrix_vector::ops::*;
use matrix::Matrix;
use math::Trans;
use math::Mat;

impl<'a, T> Mul<&'a Vector<T>> for &'a Matrix<T>
    where T: Default + Copy + Gemv
{
    type Output = Vec<T>;

    fn mul(self, x: &Vector<T>) -> Vec<T> {
        let n = self.rows() as usize;
        let mut result = Vec::with_capacity(n);
        unsafe { result.set_len(n); }
        let scale = Default::one();
        let clear = Default::zero();
        let t = Transpose::NoTrans;

        Gemv::gemv(t, &scale, self, x, &clear, &mut result);
        result
    }
}

impl<'a, T> Mul<Trans<&'a Vector<T>>> for &'a Vector<T>
    where T: Default + Ger + Gerc + Clone,
{
    type Output = Mat<T>;

    fn mul(self, x: Trans<&Vector<T>>) -> Mat<T> {
        let n = self.len() as usize;
        let m = (*x).len() as usize;
        let mut result = Mat::fill(Default::zero(), n, m);
        let scale = Default::one();

        match x {
            Trans::T(v) => Ger::ger(&scale, self, v, &mut result),
            Trans::H(v) => Gerc::gerc(&scale, self, v, &mut result),
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use Vector;
    use Matrix;
    use math::Mat;
    use math::Marker::T;

    #[test]
    fn mul() {
        let a = mat![2f32, -2.0; 2.0, -4.0];
        let x = vec![2f32, 1.0];

        let y = {
            let ar = &a as &Matrix<f32>;
            let xr = &x as &Vector<f32>;
            ar * xr
        };

        assert_eq!(y, vec![2.0, 0.0]);
    }

    #[test]
    fn outer() {
        let x = vec![2.0, 1.0, 4.0];
        let y = vec![3.0, 6.0, -1.0];

        let a = {
            let xr = &x as &Vector<_>;
            let yr = &y as &Vector<_>;

            xr * (yr ^ T)
        };

        let result = mat![  6.0, 12.0, -2.0;
                            3.0, 6.0, -1.0;
                            12.0, 24.0, -4.0];
        assert_eq!(a, result);
    }
}
