use crate::topology::sieve::Sieve;
use std::collections::HashMap;

/// Precomputed stratum information.
#[derive(Clone, Debug)]
pub struct StrataCache<P> {
    pub height:   HashMap<P,u32>,
    pub depth:    HashMap<P,u32>,
    pub strata:   Vec<Vec<P>>, // strata[h] = points at height h
    pub diameter: u32,
}

impl<P: Copy + Eq + std::hash::Hash + Ord> StrataCache<P> {
    pub fn new() -> Self {
        Self { height: HashMap::new(), depth: HashMap::new(), strata: Vec::new(), diameter: 0 }
    }
}

/// Compute strata for *any* S: on-the-fly, no cache.
pub fn compute_strata<S>(s: &S) -> StrataCache<S::Point>
where
    S: Sieve + ?Sized,
    S::Point: Copy + Eq + std::hash::Hash + Ord,
{
    // 1) collect in-degrees over s.points()
    let mut in_deg = HashMap::new();
    for p in s.points() {
        in_deg.entry(p).or_insert(0);
        for (q,_) in s.cone(p) { *in_deg.entry(q).or_insert(0) += 1; }
    }
    // 2) Kahn’s topo sort
    let mut stack: Vec<_> = in_deg.iter().filter(|&(_,d)| *d==0).map(|(&p,_)| p).collect();
    let mut topo = Vec::new();
    while let Some(p) = stack.pop() {
        topo.push(p);
        for (q,_) in s.cone(p) {
            let d = in_deg.get_mut(&q).unwrap();
            *d -= 1;
            if *d==0 { stack.push(q); }
        }
    }
    // 3) heights
    let mut height = HashMap::new();
    for &p in &topo {
        let h = s.support(p)
                .map(|(pred,_)| height.get(&pred).copied().unwrap_or(0))
                .max().map_or(0, |m| m+1);
        height.insert(p,h);
    }
    let max_h = *height.values().max().unwrap_or(&0);
    let mut strata = vec![Vec::new(); (max_h+1) as usize];
    for (&p,&h) in &height { strata[h as usize].push(p) }
    // 4) depths
    let mut depth = HashMap::new();
    for &p in topo.iter().rev() {
        let d = s.cone(p)
                .map(|(succ,_)| depth.get(&succ).copied().unwrap_or(0))
                .max().map_or(0, |m| m+1);
        depth.insert(p,d);
    }
    StrataCache { height, depth, strata, diameter: max_h }
}
