//! Mixed-type section storage with tagged scalar types.

use crate::data::atlas::Atlas;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::mesh_error::MeshSieveError;
use std::collections::BTreeMap;

/// Scalar type tag for mixed sections.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarType {
    F64,
    F32,
    I32,
    I64,
    U32,
    U64,
}

impl ScalarType {
    /// Returns a stable string label for the scalar type.
    pub fn as_str(self) -> &'static str {
        match self {
            ScalarType::F64 => "f64",
            ScalarType::F32 => "f32",
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::U32 => "u32",
            ScalarType::U64 => "u64",
        }
    }

    /// Parse a scalar type from a string label.
    pub fn parse(tag: &str) -> Option<Self> {
        match tag {
            "f64" => Some(ScalarType::F64),
            "f32" => Some(ScalarType::F32),
            "i32" => Some(ScalarType::I32),
            "i64" => Some(ScalarType::I64),
            "u32" => Some(ScalarType::U32),
            "u64" => Some(ScalarType::U64),
            _ => None,
        }
    }
}

/// Tagged, type-erased section storage for mixed scalar types.
#[derive(Clone, Debug)]
pub enum TaggedSection {
    F64(Section<f64, VecStorage<f64>>),
    F32(Section<f32, VecStorage<f32>>),
    I32(Section<i32, VecStorage<i32>>),
    I64(Section<i64, VecStorage<i64>>),
    U32(Section<u32, VecStorage<u32>>),
    U64(Section<u64, VecStorage<u64>>),
}

impl TaggedSection {
    /// Return the scalar type tag for this section.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            TaggedSection::F64(_) => ScalarType::F64,
            TaggedSection::F32(_) => ScalarType::F32,
            TaggedSection::I32(_) => ScalarType::I32,
            TaggedSection::I64(_) => ScalarType::I64,
            TaggedSection::U32(_) => ScalarType::U32,
            TaggedSection::U64(_) => ScalarType::U64,
        }
    }

    /// Return the atlas backing this tagged section.
    pub fn atlas(&self) -> &Atlas {
        match self {
            TaggedSection::F64(section) => section.atlas(),
            TaggedSection::F32(section) => section.atlas(),
            TaggedSection::I32(section) => section.atlas(),
            TaggedSection::I64(section) => section.atlas(),
            TaggedSection::U32(section) => section.atlas(),
            TaggedSection::U64(section) => section.atlas(),
        }
    }

    /// Gather values from this tagged section in atlas insertion order.
    pub fn gather_in_order(&self) -> TaggedSectionBuffer {
        match self {
            TaggedSection::F64(section) => TaggedSectionBuffer::F64(section.gather_in_order()),
            TaggedSection::F32(section) => TaggedSectionBuffer::F32(section.gather_in_order()),
            TaggedSection::I32(section) => TaggedSectionBuffer::I32(section.gather_in_order()),
            TaggedSection::I64(section) => TaggedSectionBuffer::I64(section.gather_in_order()),
            TaggedSection::U32(section) => TaggedSectionBuffer::U32(section.gather_in_order()),
            TaggedSection::U64(section) => TaggedSectionBuffer::U64(section.gather_in_order()),
        }
    }

    /// Scatter values into this tagged section in atlas insertion order.
    pub fn try_scatter_in_order(
        &mut self,
        buf: &TaggedSectionBuffer,
    ) -> Result<(), MeshSieveError> {
        match (self, buf) {
            (TaggedSection::F64(section), TaggedSectionBuffer::F64(data)) => {
                section.try_scatter_in_order(data)
            }
            (TaggedSection::F32(section), TaggedSectionBuffer::F32(data)) => {
                section.try_scatter_in_order(data)
            }
            (TaggedSection::I32(section), TaggedSectionBuffer::I32(data)) => {
                section.try_scatter_in_order(data)
            }
            (TaggedSection::I64(section), TaggedSectionBuffer::I64(data)) => {
                section.try_scatter_in_order(data)
            }
            (TaggedSection::U32(section), TaggedSectionBuffer::U32(data)) => {
                section.try_scatter_in_order(data)
            }
            (TaggedSection::U64(section), TaggedSectionBuffer::U64(data)) => {
                section.try_scatter_in_order(data)
            }
            (section, buf) => Err(MeshSieveError::TaggedSectionTypeMismatch {
                expected: section.scalar_type(),
                found: buf.scalar_type(),
            }),
        }
    }
}

/// Typed buffer for tagged section scatter/gather operations.
#[derive(Clone, Debug)]
pub enum TaggedSectionBuffer {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl TaggedSectionBuffer {
    /// Scalar type tag for this buffer.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            TaggedSectionBuffer::F64(_) => ScalarType::F64,
            TaggedSectionBuffer::F32(_) => ScalarType::F32,
            TaggedSectionBuffer::I32(_) => ScalarType::I32,
            TaggedSectionBuffer::I64(_) => ScalarType::I64,
            TaggedSectionBuffer::U32(_) => ScalarType::U32,
            TaggedSectionBuffer::U64(_) => ScalarType::U64,
        }
    }

    /// Length of the underlying flat buffer.
    pub fn len(&self) -> usize {
        match self {
            TaggedSectionBuffer::F64(data) => data.len(),
            TaggedSectionBuffer::F32(data) => data.len(),
            TaggedSectionBuffer::I32(data) => data.len(),
            TaggedSectionBuffer::I64(data) => data.len(),
            TaggedSectionBuffer::U32(data) => data.len(),
            TaggedSectionBuffer::U64(data) => data.len(),
        }
    }

    /// Return true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait to map scalar types to tagged sections for typed accessors.
pub trait MixedScalar: Sized + 'static {
    /// Scalar type tag for this concrete type.
    const SCALAR_TYPE: ScalarType;

    /// Wrap a typed section into a tagged container.
    fn wrap(section: Section<Self, VecStorage<Self>>) -> TaggedSection;
    /// Borrow a typed section if the tag matches.
    fn unwrap(section: &TaggedSection) -> Option<&Section<Self, VecStorage<Self>>>;
    /// Mutably borrow a typed section if the tag matches.
    fn unwrap_mut(section: &mut TaggedSection) -> Option<&mut Section<Self, VecStorage<Self>>>;
}

impl MixedScalar for f64 {
    const SCALAR_TYPE: ScalarType = ScalarType::F64;

    fn wrap(section: Section<Self, VecStorage<Self>>) -> TaggedSection {
        TaggedSection::F64(section)
    }

    fn unwrap(section: &TaggedSection) -> Option<&Section<Self, VecStorage<Self>>> {
        if let TaggedSection::F64(section) = section {
            Some(section)
        } else {
            None
        }
    }

    fn unwrap_mut(section: &mut TaggedSection) -> Option<&mut Section<Self, VecStorage<Self>>> {
        if let TaggedSection::F64(section) = section {
            Some(section)
        } else {
            None
        }
    }
}

impl MixedScalar for f32 {
    const SCALAR_TYPE: ScalarType = ScalarType::F32;

    fn wrap(section: Section<Self, VecStorage<Self>>) -> TaggedSection {
        TaggedSection::F32(section)
    }

    fn unwrap(section: &TaggedSection) -> Option<&Section<Self, VecStorage<Self>>> {
        if let TaggedSection::F32(section) = section {
            Some(section)
        } else {
            None
        }
    }

    fn unwrap_mut(section: &mut TaggedSection) -> Option<&mut Section<Self, VecStorage<Self>>> {
        if let TaggedSection::F32(section) = section {
            Some(section)
        } else {
            None
        }
    }
}

impl MixedScalar for i32 {
    const SCALAR_TYPE: ScalarType = ScalarType::I32;

    fn wrap(section: Section<Self, VecStorage<Self>>) -> TaggedSection {
        TaggedSection::I32(section)
    }

    fn unwrap(section: &TaggedSection) -> Option<&Section<Self, VecStorage<Self>>> {
        if let TaggedSection::I32(section) = section {
            Some(section)
        } else {
            None
        }
    }

    fn unwrap_mut(section: &mut TaggedSection) -> Option<&mut Section<Self, VecStorage<Self>>> {
        if let TaggedSection::I32(section) = section {
            Some(section)
        } else {
            None
        }
    }
}

impl MixedScalar for i64 {
    const SCALAR_TYPE: ScalarType = ScalarType::I64;

    fn wrap(section: Section<Self, VecStorage<Self>>) -> TaggedSection {
        TaggedSection::I64(section)
    }

    fn unwrap(section: &TaggedSection) -> Option<&Section<Self, VecStorage<Self>>> {
        if let TaggedSection::I64(section) = section {
            Some(section)
        } else {
            None
        }
    }

    fn unwrap_mut(section: &mut TaggedSection) -> Option<&mut Section<Self, VecStorage<Self>>> {
        if let TaggedSection::I64(section) = section {
            Some(section)
        } else {
            None
        }
    }
}

impl MixedScalar for u32 {
    const SCALAR_TYPE: ScalarType = ScalarType::U32;

    fn wrap(section: Section<Self, VecStorage<Self>>) -> TaggedSection {
        TaggedSection::U32(section)
    }

    fn unwrap(section: &TaggedSection) -> Option<&Section<Self, VecStorage<Self>>> {
        if let TaggedSection::U32(section) = section {
            Some(section)
        } else {
            None
        }
    }

    fn unwrap_mut(section: &mut TaggedSection) -> Option<&mut Section<Self, VecStorage<Self>>> {
        if let TaggedSection::U32(section) = section {
            Some(section)
        } else {
            None
        }
    }
}

impl MixedScalar for u64 {
    const SCALAR_TYPE: ScalarType = ScalarType::U64;

    fn wrap(section: Section<Self, VecStorage<Self>>) -> TaggedSection {
        TaggedSection::U64(section)
    }

    fn unwrap(section: &TaggedSection) -> Option<&Section<Self, VecStorage<Self>>> {
        if let TaggedSection::U64(section) = section {
            Some(section)
        } else {
            None
        }
    }

    fn unwrap_mut(section: &mut TaggedSection) -> Option<&mut Section<Self, VecStorage<Self>>> {
        if let TaggedSection::U64(section) = section {
            Some(section)
        } else {
            None
        }
    }
}

/// Store named sections with mixed scalar types.
#[derive(Clone, Debug, Default)]
pub struct MixedSectionStore {
    sections: BTreeMap<String, TaggedSection>,
}

impl MixedSectionStore {
    /// Create an empty mixed section store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a typed section into the store.
    pub fn insert<T: MixedScalar>(
        &mut self,
        name: impl Into<String>,
        section: Section<T, VecStorage<T>>,
    ) -> Option<TaggedSection> {
        self.sections.insert(name.into(), T::wrap(section))
    }

    /// Insert a tagged section into the store.
    pub fn insert_tagged(
        &mut self,
        name: impl Into<String>,
        section: TaggedSection,
    ) -> Option<TaggedSection> {
        self.sections.insert(name.into(), section)
    }

    /// Retrieve a typed section by name.
    pub fn get<T: MixedScalar>(&self, name: &str) -> Option<&Section<T, VecStorage<T>>> {
        self.sections.get(name).and_then(T::unwrap)
    }

    /// Retrieve a mutable typed section by name.
    pub fn get_mut<T: MixedScalar>(
        &mut self,
        name: &str,
    ) -> Option<&mut Section<T, VecStorage<T>>> {
        self.sections.get_mut(name).and_then(T::unwrap_mut)
    }

    /// Retrieve a tagged section by name.
    pub fn get_tagged(&self, name: &str) -> Option<&TaggedSection> {
        self.sections.get(name)
    }

    /// Iterate over all named tagged sections.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TaggedSection)> {
        self.sections.iter()
    }

    /// Mutably iterate over all named tagged sections.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut TaggedSection)> {
        self.sections.iter_mut()
    }

    /// Return true if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// Gather all tagged sections into flat buffers in atlas insertion order.
    pub fn gather_in_order(&self) -> BTreeMap<String, TaggedSectionBuffer> {
        self.sections
            .iter()
            .map(|(name, section)| (name.clone(), section.gather_in_order()))
            .collect()
    }

    /// Scatter all tagged sections from flat buffers in atlas insertion order.
    pub fn try_scatter_in_order(
        &mut self,
        buffers: &BTreeMap<String, TaggedSectionBuffer>,
    ) -> Result<(), MeshSieveError> {
        for (name, section) in &mut self.sections {
            let buf = buffers
                .get(name)
                .ok_or_else(|| MeshSieveError::MissingSectionName { name: name.clone() })?;
            section.try_scatter_in_order(buf)?;
        }
        Ok(())
    }
}
