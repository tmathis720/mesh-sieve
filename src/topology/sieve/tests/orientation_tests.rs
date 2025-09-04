use crate::topology::orientation::*;
use crate::topology::sieve::oriented::Orientation;

// Group laws for BitFlip
#[test]
fn bitflip_group_laws() {
    let id = BitFlip::default();
    let a = BitFlip(true);
    // identity
    assert_eq!(BitFlip::compose(id, a), a);
    assert_eq!(BitFlip::compose(a, id), a);
    // inverse
    assert_eq!(BitFlip::compose(a, Orientation::inverse(a)), id);
    // associativity (exhaustive: 2^3)
    for &x in &[BitFlip(false), BitFlip(true)] {
        for &y in &[BitFlip(false), BitFlip(true)] {
            for &z in &[BitFlip(false), BitFlip(true)] {
                assert_eq!(
                    BitFlip::compose(x, BitFlip::compose(y, z)),
                    BitFlip::compose(BitFlip::compose(x, y), z)
                );
            }
        }
    }
}

// D3 sanity
#[test]
fn d3_compose_inverse() {
    let id = D3::default();
    let r1 = D3 {
        rot: 1,
        flip: false,
    };
    let r2 = D3 {
        rot: 2,
        flip: false,
    };
    let s = D3 { rot: 0, flip: true };
    // r1 * r2 = r3 mod 3 (i.e., rot 0)
    assert_eq!(
        D3::compose(r1, r2),
        D3 {
            rot: 0,
            flip: false
        }
    );
    // s*s = id
    assert_eq!(D3::compose(s, s), id);
    // (r * s) * (r * s) = id
    let rs = D3::compose(r1, s);
    assert_eq!(D3::compose(rs, rs), id);
    // inverse law
    for rot in 0..3 {
        for &flip in &[false, true] {
            let a = D3 { rot, flip };
            assert_eq!(D3::compose(a, D3::inverse(a)), id);
        }
    }
}

// Perm<3> sanity
#[test]
fn s3_permutation_laws() {
    let id = S3::default();
    let p: S3 = Perm([1, 2, 0]); // 0->1,1->2,2->0
    let q: S3 = Perm([0, 2, 1]); // swap 1,2
    // associativity with id
    assert_eq!(
        Perm::<3>::compose(Perm::<3>::compose(p, q), id),
        Perm::<3>::compose(p, Perm::<3>::compose(q, id))
    );
    // inverse
    let pinv = p.invert();
    let qinv = q.invert();
    assert_eq!(Perm::<3>::compose(p, pinv), id);
    assert_eq!(Perm::<3>::compose(q, qinv), id);
}

// Path accumulation
#[test]
fn accumulate_path_works() {
    let steps = [BitFlip(true), BitFlip(true), BitFlip(false)];
    let tot: BitFlip = accumulate_path(steps);
    assert_eq!(tot, BitFlip(false)); // true ^ true ^ false = false
}
