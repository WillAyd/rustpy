use ndarray::parallel::prelude::*;
use numpy::ndarray::{Array1, ArrayView2, Axis, Zip};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

// import numpy as np; np.random.seed(42); arr = np.random.randint(100_000, size=(100, 1_000_000)); import rustpy; rustpy.find_max(arr)

// maturin develop --release

#[pymodule]
#[pyo3(name = "rustpy")]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn find_max(arr: ArrayView2<'_, i64>) -> Array1<i64> {
        let mut out = Array1::default(arr.ncols());

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
        let start = SystemTime::now();
        let result = find_max(x.as_array()).into_pyarray(py);
        let end = SystemTime::now();
        let duration = end.duration_since(start).unwrap();
        println!("rustpy took {} milliseconds", duration.as_millis());
        result
    }

    // example using complex numbers
    fn find_max_parallel(arr: ArrayView2<'_, i64>) -> Array1<i64> {
        let mutex = Arc::new(Mutex::new(Array1::default(arr.ncols())));

        // parallel iterator is not implemented, so some hacks
        // https://github.com/rust-ndarray/ndarray/issues/1043
        // https://github.com/rust-ndarray/ndarray/issues/1093
        Zip::indexed(arr.axis_iter(Axis(1)))
            .into_par_iter()
            .for_each(|(i, col)| {
                let mut val = i64::MIN;
                for x in col {
                    if val <= *x {
                        val = *x;
                    }
                }

                let mut guard = mutex.lock().unwrap();
                guard[i] = val;
            });

        // https://stackoverflow.com/questions/29177449/how-to-take-ownership-of-t-from-arcmutext
        let lock = Arc::try_unwrap(mutex).expect("Lock still have multiple owners");
        lock.into_inner().expect("Mutex cannot be locked")
    }

    // wrapper of `find_max`
    #[pyfn(m)]
    #[pyo3(name = "find_max_parallel")]
    fn find_max_py_parallel<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'_, i64>,
    ) -> &'py PyArray1<i64> {
        let start = SystemTime::now();
        let result = find_max_parallel(x.as_array()).into_pyarray(py);
        let end = SystemTime::now();
        let duration = end.duration_since(start).unwrap();
        println!("rustpy took {} milliseconds", duration.as_millis());
        result
    }

    Ok(())
}
