use numpy::ndarray::{Array1, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

// import numpy as np; np.random.seed(42); arr = np.random.randint(100_000, size=(100, 1_000_000)); import rustpy; rustpy.find_max(arr)

#[pymodule]
#[pyo3(name = "rustpy")]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // example using complex numbers
    fn find_max(arr: ArrayView2<'_, i64>) -> Array1<i64> {
        let mut out = Array1::zeros(arr.ncols());
        for (i, col) in arr.axis_iter(Axis(1)).enumerate() {
            let mut val = i64::MIN;
            for x in col {
                if val <= *x {
                    val = *x;
                }
            }

            out[i] = val;
        }

        out
    }

    // wrapper of `find_max`
    #[pyfn(m)]
    #[pyo3(name = "find_max")]
    fn find_max_py<'py>(py: Python<'py>, x: PyReadonlyArray2<'_, i64>) -> &'py PyArray1<i64> {
        find_max(x.as_array()).into_pyarray(py)
    }

    Ok(())
}
