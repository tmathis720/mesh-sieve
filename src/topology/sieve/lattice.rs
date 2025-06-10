// src/topology/sieve/lattice.rs

use crate::topology::sieve::sieve_trait::Sieve;

/// Minimal separator (meet) & dual separator (join) for any `Sieve`.
pub trait LatticeOps: Sieve {
    fn meet<'s>(&'s self, a: Self::Point, b: Self::Point) -> Box<dyn Iterator<Item=Self::Point> + 's>;
    fn join<'s>(&'s self, a: Self::Point, b: Self::Point) -> Box<dyn Iterator<Item=Self::Point> + 's>;
}

impl<S> LatticeOps for S
where
    S: Sieve + Sized,
    S::Point: Ord,
{
    fn meet<'s>(&'s self, a: Self::Point, b: Self::Point) -> Box<dyn Iterator<Item=Self::Point> + 's> {
        let mut ca: Vec<_> = self.closure(std::iter::once(a)).collect();
        let mut cb: Vec<_> = self.closure(std::iter::once(b)).collect();
        ca.sort_unstable();
        cb.sort_unstable();
        let mut inter = Vec::with_capacity(ca.len().min(cb.len()));
        let (mut i, mut j) = (0, 0);
        while i < ca.len() && j < cb.len() {
            match ca[i].cmp(&cb[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    inter.push(ca[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        let mut to_rm: Vec<_> = self.closure([a, b]).collect();
        to_rm.sort_unstable();
        to_rm.dedup();
        let filtered = inter.into_iter().filter(move |x| to_rm.binary_search(x).is_err());
        Box::new(filtered)
    }
    fn join<'s>(&'s self, a: Self::Point, b: Self::Point) -> Box<dyn Iterator<Item=Self::Point> + 's> {
        let mut sa: Vec<_> = self.star(std::iter::once(a)).collect();
        let mut sb: Vec<_> = self.star(std::iter::once(b)).collect();
        sa.sort_unstable();
        sb.sort_unstable();
        let mut out = Vec::with_capacity(sa.len() + sb.len());
        let (mut i, mut j) = (0, 0);
        while i < sa.len() && j < sb.len() {
            match sa[i].cmp(&sb[j]) {
                std::cmp::Ordering::Less => {
                    out.push(sa[i]);
                    i += 1
                }
                std::cmp::Ordering::Greater => {
                    out.push(sb[j]);
                    j += 1
                }
                std::cmp::Ordering::Equal => {
                    out.push(sa[i]);
                    i += 1;
                    j += 1
                }
            }
        }
        out.extend_from_slice(&sa[i..]);
        out.extend_from_slice(&sb[j..]);
        Box::new(out.into_iter())
    }
}
