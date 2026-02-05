use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use mesh_sieve::algs::traversal::{TraversalCache, closure, closure_cached, star, star_cached};
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

fn pid(raw: u32) -> PointId {
    PointId::new(u64::from(raw)).expect("nonzero PointId")
}

fn build_binary_tree(levels: u32) -> InMemorySieve<PointId> {
    let mut sieve = InMemorySieve::new();
    let mut start = 1u32;
    let mut end = 1u32;
    for _ in 1..levels {
        for parent in start..=end {
            let left = parent * 2;
            let right = parent * 2 + 1;
            sieve.add_arrow(pid(parent), pid(left), ());
            sieve.add_arrow(pid(parent), pid(right), ());
        }
        start = end + 1;
        end = end * 2 + 1;
    }
    sieve
}

fn bench_traversal_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("traversal_cache");

    for &levels in &[10u32, 12u32] {
        let sieve = build_binary_tree(levels);
        let root = pid(1u32);
        let leaf = pid(1u32 << (levels - 1));

        group.bench_with_input(
            BenchmarkId::new("closure_no_cache", levels),
            &levels,
            |b, _| {
                b.iter(|| {
                    let out = closure(&sieve, [root]);
                    black_box(out);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("closure_cached", levels),
            &levels,
            |b, _| {
                let mut cache = TraversalCache::new();
                b.iter(|| {
                    let out = closure_cached(&sieve, [root], Some(&mut cache));
                    black_box(out);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("star_no_cache", levels),
            &levels,
            |b, _| {
                b.iter(|| {
                    let out = star(&sieve, [leaf]);
                    black_box(out);
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("star_cached", levels), &levels, |b, _| {
            let mut cache = TraversalCache::new();
            b.iter(|| {
                let out = star_cached(&sieve, [leaf], Some(&mut cache));
                black_box(out);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_traversal_cache);
criterion_main!(benches);
