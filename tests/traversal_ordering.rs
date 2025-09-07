use mesh_sieve::algs::traversal::{Dir, TraversalBuilder, closure, closure_ordered, link, star};
use mesh_sieve::algs::traversal_ref::closure_ordered_ref;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;
use std::collections::HashSet;

#[test]
fn ordered_vs_unordered_chart_order() {
    let mut s = InMemorySieve::<u32, ()>::new();
    s.add_arrow(0, 1, ());
    s.add_arrow(1, 2, ());
    s.add_arrow(0, 3, ());

    let unordered = TraversalBuilder::new(&s)
        .dir(Dir::Down)
        .dfs()
        .seeds([0])
        .run();
    let ordered = closure_ordered(&mut s, [0]).unwrap();

    let mut u_sorted = unordered.clone();
    u_sorted.sort_unstable();
    let mut ord_sorted = ordered.clone();
    ord_sorted.sort_unstable();
    assert_eq!(u_sorted, ord_sorted);
    let strata = mesh_sieve::topology::sieve::strata::compute_strata(&s).unwrap();
    let idx = |p: &u32| *strata.chart_index.get(p).unwrap();
    assert!(ordered.windows(2).all(|w| idx(&w[0]) <= idx(&w[1])));
}

#[test]
fn unordered_is_deterministic() {
    let mut s = InMemorySieve::<u32, ()>::new();
    s.add_arrow(0, 2, ());
    s.add_arrow(0, 1, ());
    s.add_arrow(1, 3, ());
    let run = || {
        TraversalBuilder::new(&s)
            .dir(Dir::Down)
            .dfs()
            .seeds([0])
            .run()
    };
    let v1 = run();
    let v2 = run();
    let v3 = run();
    assert_eq!(v1, v2);
    assert_eq!(v2, v3);
}

#[test]
fn dir_both_equals_union() {
    let mut s = InMemorySieve::<u32, ()>::new();
    s.add_arrow(0, 1, ());
    s.add_arrow(1, 2, ());
    s.add_arrow(3, 1, ());

    let down = TraversalBuilder::new(&s)
        .dir(Dir::Down)
        .dfs()
        .seeds([1])
        .run();
    let up = TraversalBuilder::new(&s)
        .dir(Dir::Up)
        .dfs()
        .seeds([1])
        .run();
    let both = TraversalBuilder::new(&s)
        .dir(Dir::Both)
        .dfs()
        .seeds([1])
        .run();

    let mut union: Vec<u32> = down.iter().chain(up.iter()).copied().collect();
    union.sort_unstable();
    union.dedup();
    assert_eq!(both, union);
}

fn link_manual(s: &InMemorySieve<PointId, ()>, p: PointId) -> Vec<PointId> {
    let mut cl = closure(s, [p]);
    let mut st = star(s, [p]);
    cl.sort_unstable();
    st.sort_unstable();
    let cone: HashSet<_> = s.cone_points(p).collect();
    let sup: HashSet<_> = s.support_points(p).collect();
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < cl.len() && j < st.len() {
        match cl[i].cmp(&st[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let x = cl[i];
                if x != p && !cone.contains(&x) && !sup.contains(&x) {
                    out.push(x);
                }
                i += 1;
                j += 1;
            }
        }
    }
    out
}

#[test]
fn link_matches_expression_triangle() {
    let mut s = InMemorySieve::<u32, ()>::new();
    s.add_arrow(0, 1, ());
    s.add_arrow(1, 2, ());
    s.add_arrow(2, 0, ());
    let mut s = InMemorySieve::<PointId, ()>::new();
    let v = |i: u32| PointId::new((i + 1) as u64).unwrap();
    s.add_arrow(v(0), v(1), ());
    s.add_arrow(v(1), v(2), ());
    s.add_arrow(v(2), v(0), ());
    let l = link(&s, v(0));
    let manual = link_manual(&s, v(0));
    assert_eq!(l, manual);
}

#[test]
fn link_matches_expression_square_diagonal() {
    let mut s = InMemorySieve::<PointId, ()>::new();
    let v = |i: u32| PointId::new((i + 1) as u64).unwrap();
    s.add_arrow(v(0), v(1), ());
    s.add_arrow(v(0), v(2), ());
    s.add_arrow(v(1), v(3), ());
    s.add_arrow(v(2), v(3), ());
    s.add_arrow(v(1), v(2), ());
    let l = link(&s, v(1));
    let manual = link_manual(&s, v(1));
    assert_eq!(l, manual);
}

#[test]
fn ordered_ref_matches_ordered() {
    let mut s = InMemorySieve::<u32, ()>::new();
    s.add_arrow(0, 1, ());
    s.add_arrow(1, 2, ());
    let v1 = closure_ordered(&mut s, [0]).unwrap();
    let v2 = closure_ordered_ref(&mut s, [0]).unwrap();
    assert_eq!(v1, v2);
}
