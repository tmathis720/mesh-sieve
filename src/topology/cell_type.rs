//! Cell type metadata for mesh points.

/// Common cell types for mesh elements.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CellType {
    /// 0D vertex.
    Vertex,
    /// 1D segment/edge.
    Segment,
    /// 2D simplex (triangle).
    Triangle,
    /// 2D tensor-product cell (quad).
    Quadrilateral,
    /// 3D simplex (tet).
    Tetrahedron,
    /// 3D tensor-product cell (hex).
    Hexahedron,
    /// 3D wedge/prism.
    Prism,
    /// 3D pyramid.
    Pyramid,
    /// 2D polygon with `n` vertices.
    Polygon(u8),
    /// Generic simplex with dimension `d`.
    Simplex(u8),
    /// Generic polyhedron.
    Polyhedron,
}

impl Default for CellType {
    fn default() -> Self {
        CellType::Vertex
    }
}

impl CellType {
    /// Returns the topological dimension of the cell, when well-defined.
    pub fn dimension(self) -> u8 {
        match self {
            CellType::Vertex => 0,
            CellType::Segment => 1,
            CellType::Triangle | CellType::Quadrilateral | CellType::Polygon(_) => 2,
            CellType::Tetrahedron
            | CellType::Hexahedron
            | CellType::Prism
            | CellType::Pyramid
            | CellType::Polyhedron => 3,
            CellType::Simplex(d) => d,
        }
    }
}
