//! Canonical orientation groups for meshes (edges, triangles, quads, ...)
//! with small, copyable representations.

use core::fmt::{Debug, Formatter};

use crate::topology::sieve::oriented::Orientation;

/// 1-bit flip (edge reversal in 1D); group C₂.
/// Compose = XOR; inverse = self.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
#[repr(transparent)]
pub struct BitFlip(pub bool);
impl Debug for BitFlip {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("BitFlip").field(&self.0).finish()
    }
}
impl Orientation for BitFlip {
    #[inline]
    fn compose(a: Self, b: Self) -> Self {
        BitFlip(a.0 ^ b.0)
    }
    #[inline]
    fn inverse(a: Self) -> Self {
        a
    }
}

/// Pure rotation group C_N (polygon rotations); stored as u8 mod N.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Rot<const N: u8>(pub u8);
impl<const N: u8> Default for Rot<N> {
    fn default() -> Self {
        Rot(0)
    }
}
impl<const N: u8> Debug for Rot<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Rot").field(&self.0).finish()
    }
}
impl<const N: u8> Orientation for Rot<N> {
    #[inline]
    fn compose(a: Self, b: Self) -> Self {
        Rot::<N>((a.0 + b.0) % N)
    }
    #[inline]
    fn inverse(a: Self) -> Self {
        Rot::<N>((N - (a.0 % N)) % N)
    }
}

/// Dihedral group D_N (rotations + reflections); covers triangles (N=3) & quads (N=4).
/// Element = r^k * s^f, with k∈[0,N), f∈{0,1}; law: (k,f)*(k',f') = (k + (-1)^f k' mod N, f xor f')
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Dihedral<const N: u8> {
    pub rot: u8,
    pub flip: bool,
}
impl<const N: u8> Default for Dihedral<N> {
    fn default() -> Self {
        Self {
            rot: 0,
            flip: false,
        }
    }
}
impl<const N: u8> Debug for Dihedral<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Dihedral")
            .field("rot", &self.rot)
            .field("flip", &self.flip)
            .finish()
    }
}
impl<const N: u8> Orientation for Dihedral<N> {
    #[inline]
    fn compose(a: Self, b: Self) -> Self {
        let add = if a.flip {
            (N - (b.rot % N)) % N
        } else {
            b.rot % N
        };
        let rot = (a.rot + add) % N;
        let flip = a.flip ^ b.flip;
        Self { rot, flip }
    }
    #[inline]
    fn inverse(a: Self) -> Self {
        if !a.flip {
            Self {
                rot: (N - (a.rot % N)) % N,
                flip: false,
            }
        } else {
            Self {
                rot: a.rot % N,
                flip: true,
            }
        }
    }
}

/// Small, fixed-size permutation group S_K, represented as mapping [0..K) -> [0..K).
/// Useful for faces (triangles => S3) or element-local permutations in 3D.
/// Compose(p,q) = p ∘ q (apply q, then p).
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Perm<const K: usize>(pub [u8; K]);
impl<const K: usize> Default for Perm<K> {
    fn default() -> Self {
        let mut id = [0u8; K];
        let mut i = 0;
        while i < K {
            id[i] = i as u8;
            i += 1;
        }
        Perm(id)
    }
}
impl<const K: usize> Debug for Perm<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Perm").field(&self.0).finish()
    }
}
impl<const K: usize> Perm<K> {
    #[inline]
    pub fn new_unchecked(p: [u8; K]) -> Self {
        Perm(p)
    }
    #[inline]
    pub fn invert(&self) -> Self {
        let mut inv = [0u8; K];
        let mut i = 0;
        while i < K {
            inv[self.0[i] as usize] = i as u8;
            i += 1;
        }
        Perm(inv)
    }
}
impl<const K: usize> Orientation for Perm<K> {
    #[inline]
    fn compose(a: Self, b: Self) -> Self {
        let mut out = [0u8; K];
        let mut i = 0;
        while i < K {
            out[i] = a.0[b.0[i] as usize];
            i += 1;
        }
        Perm(out)
    }
    #[inline]
    fn inverse(a: Self) -> Self {
        a.invert()
    }
}

// Ergonomic aliases for meshes:
/// Cheap sign-flip alias.
pub use BitFlip as Sign; // 1D edge flip
/// Triangle face orientation.
pub type D3 = Dihedral<3>;
/// Quad face orientation.
pub type D4 = Dihedral<4>;
/// Triangle vertex permutations.
pub type S3 = Perm<3>;
/// Tetra permutations.
pub type S4 = Perm<4>;

/// Accumulate a sequence of orientation steps along a path, left-to-right.
/// Returns the total orientation from the seed to the end of the path.
/// Identity is `O::default()`.
#[inline]
pub fn accumulate_path<O, I>(path: I) -> O
where
    O: Orientation,
    I: IntoIterator<Item = O>,
{
    path.into_iter()
        .fold(O::default(), |acc, step| O::compose(acc, step))
}

/// Extension for `OrientedSieve` call-sites (pure sugar).
pub trait AccumulatePathExt: Sized {
    fn accumulate_path<O, I>(path: I) -> O
    where
        O: Orientation,
        I: IntoIterator<Item = O>,
    {
        accumulate_path(path)
    }
}
impl<T> AccumulatePathExt for T {}
