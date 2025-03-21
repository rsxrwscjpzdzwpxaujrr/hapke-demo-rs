use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Index, Mul, Neg};

#[derive(Default, Debug, Copy, Clone)]
pub(crate) struct Vec3<F: Copy + Mul<F, Output = F> + Add<F, Output = F>> {
    x: F,
    y: F,
    z: F,
}

// impl From<[f32; 3]> for Point3f {
//     fn from(value: [f32; 3]) -> Self {
//         unsafe { std::mem::transmute(value) }
//     }
// }

// impl From<Array1<f32>> for Point3f {
//     fn from(value: Array1<f32>) -> Self {
//         if value.len() != 3 {
//             panic!("Incompatible array conversion")
//         }
//         
//         Point3f {
//             x: value[0],
//             y: value[1],
//             z: value[2],
//         }
//     }
// }

impl<F: Copy + Mul<F, Output = F> + Add<F, Output = F>, T: Index<usize, Output = F>> From<T> for Vec3<F> {
    fn from(value: T) -> Self {
        Vec3 {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

// impl From<Point3f> for Array1<f32> {
//     fn from(value: Point3f) -> Self {
//         arr1(&[value.x, value.y, value.z])
//     }
// }

impl<F: Copy + Mul<F, Output = F> + Add<F, Output = F>> Mul<F> for Vec3<F> {
    type Output = Vec3<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<F: Copy + Mul<F, Output = F> + Add<F, Output = F>, T: Into<Vec3<F>>> Add<T> for Vec3<F> {
    type Output = Vec3<F>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs: Self = rhs.into();

        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<F: Copy + Mul<F, Output = F> + Add<F, Output = F>> Vec3<F> {
    pub(crate) fn scale<T: Into<Self>>(self, factor: T) -> Self {
        let factor: Vec3<F> = factor.into();

        Vec3 {
            x: self.x * factor.x,
            y: self.y * factor.y,
            z: self.z * factor.z,
        }
    }

    pub(crate) fn dot(self, other: &Self) -> F {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<F: Copy + Mul<F, Output = F> + Add<F, Output = F> + Neg<Output = F>> Neg for Vec3<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Display for Vec3<f32> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        (self.x, self.y, self.z).fmt(f)
    }
}