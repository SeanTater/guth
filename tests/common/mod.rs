//! Shared test utilities for fixture loading and tensor operations.

use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use serde::Deserialize;

pub type TestBackend = NdArray<f32>;

pub const FIXTURE_DIR: &str = "tests/fixtures";

/// Load and deserialize a JSON fixture file.
pub fn read_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = format!("{FIXTURE_DIR}/{name}");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    serde_json::from_str(&data).unwrap_or_else(|e| panic!("failed to parse fixture {path}: {e}"))
}

/// Assert two tensor data slices are element-wise close within tolerance.
pub fn assert_close(a: &TensorData, b: &TensorData, tol: f32) {
    let a_slice = a.as_slice::<f32>().expect("a slice");
    let b_slice = b.as_slice::<f32>().expect("b slice");
    assert_eq!(
        a_slice.len(),
        b_slice.len(),
        "tensor length mismatch: {} vs {}",
        a_slice.len(),
        b_slice.len()
    );
    for (idx, (x, y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        if (x - y).abs() > tol {
            panic!(
                "mismatch at {idx}: {x} vs {y} (diff: {}, tol: {tol})",
                (x - y).abs()
            );
        }
    }
}

/// Create a 1D tensor from a Vec.
pub fn tensor1(data: Vec<f32>, device: &NdArrayDevice) -> Tensor<TestBackend, 1> {
    let len = data.len();
    Tensor::from_data(TensorData::new(data, [len]), device)
}

/// Create a 2D tensor from nested Vecs.
pub fn tensor2(data: Vec<Vec<f32>>, device: &NdArrayDevice) -> Tensor<TestBackend, 2> {
    let rows = data.len();
    let cols = data.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f32> = data.into_iter().flatten().collect();
    Tensor::from_data(TensorData::new(flat, [rows, cols]), device)
}

/// Create a 2D int tensor from nested Vecs.
pub fn tensor2_int(data: Vec<Vec<i64>>, device: &NdArrayDevice) -> Tensor<TestBackend, 2, Int> {
    let rows = data.len();
    let cols = data.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<i64> = data.into_iter().flatten().collect();
    Tensor::from_data(TensorData::new(flat, [rows, cols]), device)
}

/// Create a 3D tensor from nested Vecs.
pub fn tensor3(data: Vec<Vec<Vec<f32>>>, device: &NdArrayDevice) -> Tensor<TestBackend, 3> {
    let d0 = data.len();
    let d1 = data.first().map(|v| v.len()).unwrap_or(0);
    let d2 = data
        .first()
        .and_then(|v| v.first())
        .map(|v| v.len())
        .unwrap_or(0);
    let flat: Vec<f32> = data.into_iter().flatten().flatten().collect();
    Tensor::from_data(TensorData::new(flat, [d0, d1, d2]), device)
}

/// Create a 3D bool tensor from nested Vecs.
pub fn tensor3_bool(
    data: Vec<Vec<Vec<bool>>>,
    device: &NdArrayDevice,
) -> Tensor<TestBackend, 3, Bool> {
    let d0 = data.len();
    let d1 = data.first().map(|v| v.len()).unwrap_or(0);
    let d2 = data
        .first()
        .and_then(|v| v.first())
        .map(|v| v.len())
        .unwrap_or(0);
    let flat: Vec<bool> = data.into_iter().flatten().flatten().collect();
    Tensor::from_data(TensorData::new(flat, [d0, d1, d2]), device)
}

/// Create a 4D tensor from nested Vecs.
pub fn tensor4(data: Vec<Vec<Vec<Vec<f32>>>>, device: &NdArrayDevice) -> Tensor<TestBackend, 4> {
    let d0 = data.len();
    let d1 = data.first().map(|v| v.len()).unwrap_or(0);
    let d2 = data
        .first()
        .and_then(|v| v.first())
        .map(|v| v.len())
        .unwrap_or(0);
    let d3 = data
        .first()
        .and_then(|v| v.first())
        .and_then(|v| v.first())
        .map(|v| v.len())
        .unwrap_or(0);
    let flat: Vec<f32> = data.into_iter().flatten().flatten().flatten().collect();
    Tensor::from_data(TensorData::new(flat, [d0, d1, d2, d3]), device)
}
