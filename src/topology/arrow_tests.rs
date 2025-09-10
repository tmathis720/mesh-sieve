#[cfg(test)]
mod tests {
    use crate::topology::arrow::Polarity;
    use crate::topology::arrow::Polarity::{Forward as F, Reverse as R};
    use crate::topology::orientation::Sign;

    #[test]
    fn polarity_ops() {
        assert_eq!(F.invert(), R);
        assert_eq!(R.invert(), F);

        assert_eq!(F ^ F, F);
        assert_eq!(F ^ R, R);
        assert_eq!(R ^ F, R);
        assert_eq!(R ^ R, F);

        assert_eq!(F.sign(), 1);
        assert_eq!(R.sign(), -1);

        assert_eq!(bool::from(F), false);
        assert_eq!(bool::from(R), true);
        assert_eq!(Polarity::from(false), F);
        assert_eq!(Polarity::from(true), R);
    }

    #[test]
    fn polarity_sign_roundtrip() {
        let s_f: Sign = F.into();
        let s_r: Sign = R.into();
        assert_eq!(s_f.0, false);
        assert_eq!(s_r.0, true);

        let pf: Polarity = s_f.into();
        let pr: Polarity = s_r.into();
        assert_eq!(pf, F);
        assert_eq!(pr, R);
    }

    #[test]
    fn polarity_serde_roundtrip() {
        let j = serde_json::to_string(&F).unwrap();
        assert!(j.contains("Forward"));
        let k = serde_json::to_string(&R).unwrap();
        assert!(k.contains("Reverse"));
        assert_eq!(serde_json::from_str::<Polarity>(&j).unwrap(), F);
        assert_eq!(serde_json::from_str::<Polarity>(&k).unwrap(), R);
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_orientation_alias_compiles() {
        let _ = crate::topology::arrow::Orientation::Forward;
    }
}
