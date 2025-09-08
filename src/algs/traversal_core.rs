use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

pub enum Dir {
    Down,
    Up,
    Both,
}
pub enum Strategy {
    DFS,
    BFS,
}

/// Neighbors provider abstraction so both traversal paths can reuse it.
pub trait Neigh<'a, P: 'a> {
    type Iter: Iterator<Item = P> + 'a;
    fn down(&'a self, p: P) -> Self::Iter;
    fn up(&'a self, p: P) -> Self::Iter;
}

pub struct Opts<'a, P: Copy + Ord + Hash + Eq> {
    pub dir: Dir,
    pub strat: Strategy,
    pub max_depth: Option<u32>,
    pub deterministic: bool,
    pub early_stop: Option<&'a dyn Fn(P) -> bool>,
}

pub fn traverse<'a, P, N>(
    neigh: &'a N,
    seeds: impl IntoIterator<Item = P>,
    opts: Opts<'a, P>,
) -> Vec<P>
where
    P: Copy + Ord + Hash + Eq + 'a,
    N: Neigh<'a, P>,
{
    let Opts {
        dir,
        strat,
        max_depth,
        deterministic,
        early_stop,
    } = opts;
    let mut seen: HashSet<P> = HashSet::new();

    match strat {
        Strategy::DFS => {
            let mut stack: Vec<(P, u32)> = seeds.into_iter().map(|p| (p, 0)).collect();
            seen.extend(stack.iter().map(|&(p, _)| p));
            while let Some((p, d)) = stack.pop() {
                if let Some(f) = early_stop
                    && f(p) {
                        break;
                    }
                if max_depth.is_some_and(|md| d >= md) {
                    continue;
                }
                let push =
                    |it: N::Iter, dnext: u32, seen: &mut HashSet<P>, stack: &mut Vec<(P, u32)>| {
                        for q in it {
                            if seen.insert(q) {
                                stack.push((q, dnext));
                            }
                        }
                    };
                match dir {
                    Dir::Down => push(neigh.down(p), d + 1, &mut seen, &mut stack),
                    Dir::Up => push(neigh.up(p), d + 1, &mut seen, &mut stack),
                    Dir::Both => {
                        push(neigh.down(p), d + 1, &mut seen, &mut stack);
                        push(neigh.up(p), d + 1, &mut seen, &mut stack);
                    }
                }
            }
        }
        Strategy::BFS => {
            let mut q: VecDeque<(P, u32)> = seeds.into_iter().map(|p| (p, 0)).collect();
            seen.extend(q.iter().map(|&(p, _)| p));
            while let Some((p, d)) = q.pop_front() {
                if let Some(f) = early_stop
                    && f(p) {
                        break;
                    }
                if max_depth.is_some_and(|md| d >= md) {
                    continue;
                }
                let push =
                    |it: N::Iter, dnext: u32, seen: &mut HashSet<P>, q: &mut VecDeque<(P, u32)>| {
                        for qn in it {
                            if seen.insert(qn) {
                                q.push_back((qn, dnext));
                            }
                        }
                    };
                match dir {
                    Dir::Down => push(neigh.down(p), d + 1, &mut seen, &mut q),
                    Dir::Up => push(neigh.up(p), d + 1, &mut seen, &mut q),
                    Dir::Both => {
                        push(neigh.down(p), d + 1, &mut seen, &mut q);
                        push(neigh.up(p), d + 1, &mut seen, &mut q);
                    }
                }
            }
        }
    }

    let mut out: Vec<P> = seen.into_iter().collect();
    if deterministic {
        out.sort_unstable();
        out.dedup();
    }
    out
}
