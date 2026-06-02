//! Fluent ASCII mesh reader.
//!
//! This reader imports a practical ASCII subset of ANSYS Fluent `.msh` files:
//! vertex coordinate sections `(10 ...)` and cell connectivity sections
//! `(12 ...)` with explicit node lists. It also accepts a compact fixture form
//! with `vertices`/`cells` records for tests and examples.

use crate::data::storage::VecStorage;
use crate::io::{MeshData, SieveSectionReader};
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;
use crate::topology::sieve::MeshSieve;
use std::io::Read;

/// Reader for ASCII Fluent meshes.
#[derive(Debug, Default, Clone)]
pub struct FluentReader;

impl SieveSectionReader for FluentReader {
    type Sieve = MeshSieve;
    type Value = f64;
    type Storage = VecStorage<f64>;
    type CellStorage = VecStorage<CellType>;

    fn read<R: Read>(
        &self,
        mut reader: R,
    ) -> Result<MeshData<Self::Sieve, Self::Value, Self::Storage, Self::CellStorage>, MeshSieveError>
    {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        parse_fluent_ascii(&text)
    }
}

fn parse_fluent_ascii(
    text: &str,
) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError> {
    if text.lines().any(|l| l.trim_start().starts_with("vertices")) {
        return parse_compact(text);
    }
    parse_sexpr_subset(text)
}

fn parse_compact(
    text: &str,
) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError> {
    let mut vertices = Vec::new();
    let mut cells = Vec::new();
    let mut labels = LabelSet::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["v", x, y, z] => vertices.push([parse_f64(x)?, parse_f64(y)?, parse_f64(z)?]),
            ["label", name, value, ids @ ..] if !ids.is_empty() => {
                let value = value.parse::<i32>().map_err(|_| {
                    MeshSieveError::MeshIoParse(format!("invalid Fluent label value: {value}"))
                })?;
                for id in ids {
                    labels.set_label(PointId::new(parse_u64(id)?)?, name, value);
                }
            }
            ["cell", rest @ ..] if !rest.is_empty() => {
                let conn = rest
                    .iter()
                    .map(|v| PointId::new(parse_u64(v)?))
                    .collect::<Result<Vec<_>, _>>()?;
                cells.push(conn);
            }
            ["vertices", _] | ["cells", _] => {}
            _ => {
                return Err(MeshSieveError::MeshIoParse(format!(
                    "unsupported Fluent compact line: {line}"
                )));
            }
        }
    }
    let mut mesh = crate::io::ply::build_mesh(vertices, cells)?;
    mesh.labels = (!labels.is_empty()).then_some(labels);
    Ok(mesh)
}

fn parse_sexpr_subset(
    text: &str,
) -> Result<MeshData<MeshSieve, f64, VecStorage<f64>, VecStorage<CellType>>, MeshSieveError> {
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut cells: Vec<Vec<PointId>> = Vec::new();
    let mut labels = LabelSet::new();
    let lines: Vec<_> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with("(10 (") && !line.contains(" 0 ") {
            i += 1;
            while i < lines.len() {
                let l = lines[i].trim().trim_matches(|c| c == '(' || c == ')');
                if l.is_empty() {
                    i += 1;
                    break;
                }
                let vals: Vec<_> = l.split_whitespace().collect();
                if vals.len() < 2 {
                    break;
                }
                let x = parse_hex_or_float(vals[0])?;
                let y = parse_hex_or_float(vals[1])?;
                let z = vals.get(2).map_or(Ok(0.0), |v| parse_hex_or_float(v))?;
                vertices.push([x, y, z]);
                if lines[i].contains("))") {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        if line.starts_with("(12 (") && !line.contains(" 0 ") {
            i += 1;
            while i < lines.len() {
                let l = lines[i].trim().trim_matches(|c| c == '(' || c == ')');
                if l.is_empty() {
                    i += 1;
                    break;
                }
                let vals: Vec<_> = l.split_whitespace().collect();
                if vals.len() < 2 {
                    break;
                }
                let conn = vals
                    .iter()
                    .map(|v| PointId::new(parse_hex_u64(v)?))
                    .collect::<Result<Vec<_>, _>>()?;
                cells.push(conn);
                if lines[i].contains("))") {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        if line.starts_with("(13 (") && !line.contains(" 0 ") {
            let zone_id = line
                .split_whitespace()
                .nth(1)
                .and_then(|raw| raw.trim_matches('(').parse::<i32>().ok())
                .unwrap_or(1);
            i += 1;
            while i < lines.len() {
                let l = lines[i].trim().trim_matches(|c| c == '(' || c == ')');
                if l.is_empty() {
                    i += 1;
                    break;
                }
                let vals: Vec<_> = l.split_whitespace().collect();
                if vals.len() < 4 {
                    break;
                }
                let vertex_tokens = &vals[..vals.len().saturating_sub(2)];
                for token in vertex_tokens {
                    let point = PointId::new(parse_hex_u64(token)?)?;
                    labels.set_label(point, "fluent:bc", zone_id);
                    labels.set_label(point, &format!("fluent:zone:{zone_id}"), 1);
                }
                if lines[i].contains("))") {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        i += 1;
    }
    if vertices.is_empty() {
        return Err(MeshSieveError::MeshIoParse(
            "no Fluent vertex coordinates found".into(),
        ));
    }
    let mut mesh = crate::io::ply::build_mesh(vertices, cells)?;
    mesh.labels = (!labels.is_empty()).then_some(labels);
    Ok(mesh)
}

fn parse_f64(token: &str) -> Result<f64, MeshSieveError> {
    token
        .parse::<f64>()
        .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid float: {token}")))
}
fn parse_u64(token: &str) -> Result<u64, MeshSieveError> {
    token
        .parse::<u64>()
        .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid integer: {token}")))
}
fn parse_hex_u64(token: &str) -> Result<u64, MeshSieveError> {
    u64::from_str_radix(token, 16)
        .or_else(|_| token.parse::<u64>())
        .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid Fluent integer: {token}")))
}
fn parse_hex_or_float(token: &str) -> Result<f64, MeshSieveError> {
    token
        .parse::<f64>()
        .or_else(|_| u64::from_str_radix(token, 16).map(|v| v as f64))
        .map_err(|_| MeshSieveError::MeshIoParse(format!("invalid Fluent number: {token}")))
}
