use mesh_sieve::algs::traversal::{Dir, TraversalBuilder};
use mesh_sieve::algs::traversal_ref::{closure_ref, star_ref};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

fn p(x: u64) -> PointId {
    PointId::new(x + 1).unwrap()
}

fn build_line() -> InMemorySieve<PointId, ()> {
    let mut s = InMemorySieve::<PointId, ()>::new();
    s.add_arrow(p(0), p(1), ());
    s.add_arrow(p(1), p(2), ());
    s.add_arrow(p(2), p(3), ());
    s
}

#[test]
fn dfs_down_max_depth() {
    let s = build_line();
    let v = TraversalBuilder::new(&s)
        .dir(Dir::Down)
        .dfs()
        .max_depth(Some(2))
        .seeds([p(0)])
        .run();
    assert_eq!(v, vec![p(0), p(1), p(2)]);
}

#[test]
fn parity_closure_star() {
    let s = build_line();
    let seeds = [p(1)];
    let cl_ref = closure_ref(&s, seeds);
    let cl = TraversalBuilder::new(&s)
        .dir(Dir::Down)
        .dfs()
        .seeds(seeds)
        .run();
    assert_eq!(cl_ref, cl);
    let star_ref_v = star_ref(&s, [p(2)]);
    let star_v = TraversalBuilder::new(&s)
        .dir(Dir::Up)
        .dfs()
        .seeds([p(2)])
        .run();
    assert_eq!(star_ref_v, star_v);
}

#[test]
fn dir_both_union() {
    let s = build_line();
    let seeds = [p(1)];
    let down = TraversalBuilder::new(&s)
        .dir(Dir::Down)
        .dfs()
        .seeds(seeds)
        .run();
    let up = TraversalBuilder::new(&s)
        .dir(Dir::Up)
        .dfs()
        .seeds(seeds)
        .run();
    let mut union = down.clone();
    union.extend(up.iter().copied());
    union.sort_unstable();
    union.dedup();
    let both = TraversalBuilder::new(&s)
        .dir(Dir::Both)
        .dfs()
        .seeds(seeds)
        .run();
    assert_eq!(both, union);
}

#[test]
fn determinism_shuffle() {
    let s = build_line();
    let mut seeds = vec![p(0), p(1)];
    let a = TraversalBuilder::new(&s)
        .dir(Dir::Down)
        .dfs()
        .seeds(seeds.clone())
        .run();
    seeds.reverse();
    let b = TraversalBuilder::new(&s)
        .dir(Dir::Down)
        .dfs()
        .seeds(seeds.clone())
        .run();
    assert_eq!(a, b);
    let c = closure_ref(&s, seeds.clone());
    let d = closure_ref(&s, seeds);
    assert_eq!(c, d);
}
