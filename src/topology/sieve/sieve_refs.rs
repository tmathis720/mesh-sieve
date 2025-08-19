use super::sieve_trait::Sieve;

pub trait SieveRefs: Sieve {
    type ConeRefIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)> where Self: 'a;
    type SupportRefIter<'a>: Iterator<Item = (Self::Point, &'a Self::Payload)> where Self: 'a;

    fn cone_ref<'a>(&'a self, p: Self::Point) -> Self::ConeRefIter<'a>;
    fn support_ref<'a>(&'a self, p: Self::Point) -> Self::SupportRefIter<'a>;
}
