//! Python bindings for `trxscan` — the module `trxscan._core`.
//!
//! A deliberately thin layer over the std-only simulation core (`kspace,par` features; no file
//! I/O, no trx-rs, no HDF5). Conventions, shared with `trxscan/_arrays.py`:
//!
//! * **Arrays cross the boundary flat.** 3-D volumes are 1-D `float32` buffers in TRXScan's
//!   `x + nx*(y + ny*z)` order, which is the memory order of an F-contiguous `(nx, ny, nz)` numpy
//!   array (what nibabel hands out). 4-D data are `(x + nx*(y + ny*z))*ngrad + g`, the memory
//!   order of a C-contiguous `(nz, ny, nx, ngrad)` array. K-space per slice is `kx + nx*ky`, a
//!   C-contiguous `(ny, nx)` array. Python reshapes with `order="F"`/`"C"` at zero cost.
//! * **Complex data are `float32` pairs**: a buffer of `2n` floats that Python views as
//!   `complex64`.
//! * **Large Rust-owned state is an opaque handle** (`Mixture`, `Compartments`, `GnlField`,
//!   `GradCoef`); parameter bundles are Python dicts with the Rust field names.
//! * **Every heavy call releases the GIL** (`py.detach`).
//! * Dimension mismatches are validated here and raised as `ValueError`; the core's own
//!   `assert!`s are never reached from Python.

use std::sync::{Arc, Mutex};

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use trxscan::compartments::{self as comp, CompartmentParams, TissueFractions};
use trxscan::gnl;
use trxscan::gre;
use trxscan::kspace::{self, Acquisition, AcquisitionOptions, KspaceWindow, PartialFourierMode, SliceCapture};
use trxscan::microstructure::{field_scalars, voxel_scalars, watson_odi, ScalarConfig, VoxelMixture, SCALAR_NAMES};
use trxscan::signal::{Ball, FiberSignalModel, IsotropicSignalModel, Stick, Tensor};
use trxscan::mixture::MixtureField;
use trxscan::motion::{self, MotionEvent, MotionMode, Pose};
use trxscan::orient::Reorient;
use trxscan::phase::PhaseModel;
use trxscan::raster::Grid;
use trxscan::scheme::GradientScheme;
use trxscan::sphere::HemiSphere;

// ─── helpers ────────────────────────────────────────────────────────────────

fn verr<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn f32s(a: &PyReadonlyArray1<'_, f32>) -> PyResult<Vec<f32>> {
    a.as_slice().map(|s| s.to_vec()).map_err(verr)
}

fn f64s(a: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    a.as_slice().map(|s| s.to_vec()).map_err(verr)
}

fn u32s(a: &PyReadonlyArray1<'_, u32>) -> PyResult<Vec<u32>> {
    a.as_slice().map(|s| s.to_vec()).map_err(verr)
}

fn rows3(a: &PyReadonlyArray2<'_, f64>, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let v = a.as_array();
    if v.ncols() != 3 {
        return Err(verr(format!("{what} must have shape (n, 3), got {:?}", v.shape())));
    }
    Ok(v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn affine4(a: &PyReadonlyArray2<'_, f64>) -> PyResult<[[f64; 4]; 4]> {
    let v = a.as_array();
    if v.shape() != [4, 4] {
        return Err(verr(format!("affine must be (4, 4), got {:?}", v.shape())));
    }
    let mut m = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = v[[r, c]];
        }
    }
    Ok(m)
}

fn dims3(d: [usize; 3]) -> (usize, usize, usize) {
    (d[0], d[1], d[2])
}

fn dims_of(t: (usize, usize, usize)) -> [usize; 3] {
    [t.0, t.1, t.2]
}

fn nvox(d: [usize; 3]) -> usize {
    d[0] * d[1] * d[2]
}

fn check_len(name: &str, got: usize, want: usize) -> PyResult<()> {
    if got != want {
        Err(verr(format!("{name}: expected {want} values, got {got}")))
    } else {
        Ok(())
    }
}

fn flat_pairs(v: Vec<[f32; 2]>) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for p in v {
        out.push(p[0]);
        out.push(p[1]);
    }
    out
}

fn affine_to_py<'py>(py: Python<'py>, m: &[[f64; 4]; 4]) -> Bound<'py, PyArray1<f64>> {
    let mut v = Vec::with_capacity(16);
    for r in m {
        v.extend_from_slice(r);
    }
    PyArray1::from_vec(py, v)
}

fn get_f64(d: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract::<f64>().map_err(|_| verr(format!("{key}: expected a number"))),
        _ => Ok(default),
    }
}

fn get_usize(d: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract::<usize>().map_err(|_| verr(format!("{key}: expected a non-negative integer"))),
        _ => Ok(default),
    }
}

fn get_bool(d: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract::<bool>().map_err(|_| verr(format!("{key}: expected a bool"))),
        _ => Ok(default),
    }
}

fn get_str(d: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract::<String>().map_err(|_| verr(format!("{key}: expected a string"))),
        _ => Ok(default.to_string()),
    }
}

/// `kspace::Acquisition` from a dict of its field names. Missing keys keep `kspace::default_acquisition()`.
/// `window`: `None` | `"none"` | `"hann"` | `("tukey", alpha)` | `("fermi", radius, width)`;
/// `pf_mode`: `"scanner"` (default) | `"contiguous"` | `"fiberfox"`.
fn acquisition_from_dict(d: &Bound<'_, PyDict>) -> PyResult<Acquisition> {
    let base = kspace::default_acquisition();
    let window = match d.get_item("window")? {
        None => KspaceWindow::None,
        Some(v) if v.is_none() => KspaceWindow::None,
        Some(v) => {
            if let Ok(s) = v.extract::<String>() {
                match s.as_str() {
                    "none" => KspaceWindow::None,
                    "hann" => KspaceWindow::Hann,
                    o => return Err(verr(format!("window: unknown name {o:?}; use 'none', 'hann', ('tukey', a) or ('fermi', r, w)"))),
                }
            } else {
                let t = v.cast::<PyTuple>().map_err(|_| verr("window: expected a string or tuple"))?;
                let kind: String = t.get_item(0)?.extract()?;
                match kind.as_str() {
                    "tukey" => KspaceWindow::Tukey { alpha: t.get_item(1)?.extract()? },
                    "fermi" => KspaceWindow::Fermi { radius: t.get_item(1)?.extract()?, width: t.get_item(2)?.extract()? },
                    o => return Err(verr(format!("window: unknown kind {o:?}"))),
                }
            }
        }
    };
    let pf_mode = match get_str(d, "pf_mode", "scanner")?.as_str() {
        "scanner" => PartialFourierMode::Scanner,
        "contiguous" => PartialFourierMode::Contiguous,
        "fiberfox" => PartialFourierMode::FiberfoxCompatible,
        o => return Err(verr(format!("pf_mode: expected 'scanner', 'contiguous' or 'fiberfox', got {o:?}"))),
    };
    for key in d.keys().iter() {
        let k: String = key.extract()?;
        const KNOWN: &[&str] = &[
            "t_line", "t_echo", "t_inhom", "signal_scale", "reverse_phase", "do_distortions", "do_relaxation",
            "noise_variance", "partial_fourier", "pf_mode", "ghost_offset", "eddy_strength", "eddy_quad",
            "eddy_phase", "eddy_tau", "n_spikes", "spike_amplitude", "window", "n_coils", "accel", "acs_lines", "seed",
        ];
        if !KNOWN.contains(&k.as_str()) {
            return Err(verr(format!("acquisition: unknown field {k:?}")));
        }
    }
    Ok(Acquisition {
        t_line: get_f64(d, "t_line", base.t_line)?,
        t_echo: get_f64(d, "t_echo", base.t_echo)?,
        t_inhom: get_f64(d, "t_inhom", base.t_inhom)?,
        signal_scale: get_f64(d, "signal_scale", base.signal_scale)?,
        reverse_phase: get_bool(d, "reverse_phase", base.reverse_phase)?,
        do_distortions: get_bool(d, "do_distortions", base.do_distortions)?,
        do_relaxation: get_bool(d, "do_relaxation", base.do_relaxation)?,
        noise_variance: get_f64(d, "noise_variance", base.noise_variance)?,
        partial_fourier: get_f64(d, "partial_fourier", base.partial_fourier)?,
        pf_mode,
        ghost_offset: get_f64(d, "ghost_offset", base.ghost_offset)?,
        eddy_strength: get_f64(d, "eddy_strength", base.eddy_strength)?,
        eddy_quad: get_f64(d, "eddy_quad", base.eddy_quad)?,
        eddy_phase: get_f64(d, "eddy_phase", base.eddy_phase)?,
        eddy_tau: get_f64(d, "eddy_tau", base.eddy_tau)?,
        n_spikes: get_usize(d, "n_spikes", base.n_spikes)?,
        spike_amplitude: get_f64(d, "spike_amplitude", base.spike_amplitude)?,
        window,
        n_coils: get_usize(d, "n_coils", base.n_coils)?,
        accel: get_usize(d, "accel", base.accel)?,
        acs_lines: get_usize(d, "acs_lines", base.acs_lines)?,
        seed: get_usize(d, "seed", base.seed as usize)? as u64,
        // mrsim-acq's fields TRXScan does not set (the echo formation: spin echo)
        ..base
    })
}

/// `CompartmentParams` from a dict: `preset` (`"neonatal"` | `"adult"` | `"infant"`, default
/// neonatal = `CompartmentParams::default()`) plus any field-name overrides (`d_extra` as a
/// 3-tuple).
fn params_from_dict(d: &Bound<'_, PyDict>) -> PyResult<CompartmentParams> {
    let mut p = match get_str(d, "preset", "neonatal")?.as_str() {
        "neonatal" => CompartmentParams::default(),
        "adult" => CompartmentParams::adult(),
        "infant" => CompartmentParams::infant(),
        o => return Err(verr(format!("preset: expected 'neonatal', 'adult' or 'infant', got {o:?}"))),
    };
    p.b_value = get_f64(d, "b_value", p.b_value)?;
    p.fiber_radius_mm = get_f64(d, "fiber_radius_mm", p.fiber_radius_mm)?;
    p.intra_frac = get_f64(d, "intra_frac", p.intra_frac)?;
    p.extra_frac = get_f64(d, "extra_frac", p.extra_frac)?;
    p.d_intra = get_f64(d, "d_intra", p.d_intra)?;
    if let Some(v) = d.get_item("d_extra")? {
        if !v.is_none() {
            let t: (f64, f64, f64) = v.extract().map_err(|_| verr("d_extra: expected a 3-tuple of eigenvalues"))?;
            p.d_extra = t;
        }
    }
    p.d_gm = get_f64(d, "d_gm", p.d_gm)?;
    p.d_csf = get_f64(d, "d_csf", p.d_csf)?;
    p.t2_fiber = get_f64(d, "t2_fiber", p.t2_fiber as f64)? as f32;
    p.t2_gm = get_f64(d, "t2_gm", p.t2_gm as f64)? as f32;
    p.t2_csf = get_f64(d, "t2_csf", p.t2_csf as f64)? as f32;
    p.gm_restricted_frac = get_f64(d, "gm_restricted_frac", p.gm_restricted_frac)?;
    p.d_soma = get_f64(d, "d_soma", p.d_soma)?;
    Ok(p)
}

fn params_to_dict<'py>(py: Python<'py>, p: &CompartmentParams) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("b_value", p.b_value)?;
    d.set_item("fiber_radius_mm", p.fiber_radius_mm)?;
    d.set_item("intra_frac", p.intra_frac)?;
    d.set_item("extra_frac", p.extra_frac)?;
    d.set_item("d_intra", p.d_intra)?;
    d.set_item("d_extra", (p.d_extra.0, p.d_extra.1, p.d_extra.2))?;
    d.set_item("d_gm", p.d_gm)?;
    d.set_item("d_csf", p.d_csf)?;
    d.set_item("t2_fiber", p.t2_fiber as f64)?;
    d.set_item("t2_gm", p.t2_gm as f64)?;
    d.set_item("t2_csf", p.t2_csf as f64)?;
    d.set_item("gm_restricted_frac", p.gm_restricted_frac)?;
    d.set_item("d_soma", p.d_soma)?;
    Ok(d)
}

fn tissue_from_flat(
    dims: [usize; 3],
    wm: &PyReadonlyArray1<'_, f32>,
    gm: &PyReadonlyArray1<'_, f32>,
    csf: &PyReadonlyArray1<'_, f32>,
    mask: &PyReadonlyArray1<'_, f32>,
) -> PyResult<TissueFractions> {
    let n = nvox(dims);
    let (wm, gm, csf, mask) = (f32s(wm)?, f32s(gm)?, f32s(csf)?, f32s(mask)?);
    check_len("wm", wm.len(), n)?;
    check_len("gm", gm.len(), n)?;
    check_len("csf", csf.len(), n)?;
    check_len("mask", mask.len(), n)?;
    Ok(TissueFractions { dims, wm, gm, csf, mask: mask.iter().map(|&m| (m > 0.5) as u8).collect() })
}

fn scheme_from(bvals: &PyReadonlyArray1<'_, f64>, bvecs: &PyReadonlyArray2<'_, f64>) -> PyResult<GradientScheme> {
    let bvals = f64s(bvals)?;
    let mut bvecs = rows3(bvecs, "bvecs")?;
    check_len("bvecs", bvecs.len(), bvals.len())?;
    // the same normalisation GradientScheme::from_str applies: unit directions, b0 -> zero
    for (b, v) in bvals.iter().zip(bvecs.iter_mut()) {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if *b <= 0.0 || n < 1e-6 {
            *v = [0.0; 3];
        } else {
            *v = [v[0] / n, v[1] / n, v[2] / n];
        }
    }
    // A b0-only scheme has no reference b-value; any positive one gives the same (unit) signal
    // because every gradient is scaled by sqrt(b_i / b_max) = 0. Guard the division.
    let b_max = bvals.iter().cloned().fold(0.0, f64::max);
    let b_max = if b_max > 0.0 { b_max } else { 1000.0 };
    Ok(GradientScheme { bvals, bvecs, b_max })
}

fn poses_from(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Pose>> {
    let v = a.as_array();
    if v.ncols() != 6 {
        return Err(verr(format!("poses must be (n, 6) [tx, ty, tz mm, rx, ry, rz deg], got {:?}", v.shape())));
    }
    Ok(v.rows()
        .into_iter()
        .map(|r| Pose { trans_mm: [r[0], r[1], r[2]], rot_deg: [r[3], r[4], r[5]] })
        .collect())
}

fn poses_to_py<'py>(py: Python<'py>, poses: &[Pose]) -> Bound<'py, PyArray1<f64>> {
    let mut v = Vec::with_capacity(poses.len() * 6);
    for p in poses {
        v.extend_from_slice(&p.trans_mm);
        v.extend_from_slice(&p.rot_deg);
    }
    PyArray1::from_vec(py, v)
}

fn events_from(list: &Bound<'_, PyAny>) -> PyResult<Vec<MotionEvent>> {
    let mut out = Vec::new();
    for item in list.try_iter()? {
        let d = item?;
        let d = d.cast::<PyDict>().map_err(|_| verr("events must be dicts"))?;
        let jm: (f64, f64, f64) = d.get_item("jump_mm")?.ok_or_else(|| verr("event: jump_mm missing"))?.extract()?;
        let jd: (f64, f64, f64) = d.get_item("jump_deg")?.ok_or_else(|| verr("event: jump_deg missing"))?.extract()?;
        out.push(MotionEvent {
            volume: get_usize(d, "volume", 0)?,
            shot: get_usize(d, "shot", 0)?,
            severity: get_f64(d, "severity", 1.0)? as f32,
            jump_mm: [jm.0, jm.1, jm.2],
            jump_deg: [jd.0, jd.1, jd.2],
        });
    }
    Ok(out)
}

fn events_to_py<'py>(py: Python<'py>, evs: &[MotionEvent]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for e in evs {
        let d = PyDict::new(py);
        d.set_item("volume", e.volume)?;
        d.set_item("shot", e.shot)?;
        d.set_item("severity", e.severity as f64)?;
        d.set_item("jump_mm", (e.jump_mm[0], e.jump_mm[1], e.jump_mm[2]))?;
        d.set_item("jump_deg", (e.jump_deg[0], e.jump_deg[1], e.jump_deg[2]))?;
        out.append(d)?;
    }
    Ok(out)
}

fn reorient_from(d: &Bound<'_, PyDict>) -> PyResult<Reorient> {
    let src: (usize, usize, usize) = d.get_item("src")?.ok_or_else(|| verr("reorient: src missing"))?.extract()?;
    let flip: (bool, bool, bool) = d.get_item("flip")?.ok_or_else(|| verr("reorient: flip missing"))?.extract()?;
    let in_dims: (usize, usize, usize) = d.get_item("in_dims")?.ok_or_else(|| verr("reorient: in_dims missing"))?.extract()?;
    let out_dims: (usize, usize, usize) = d.get_item("out_dims")?.ok_or_else(|| verr("reorient: out_dims missing"))?.extract()?;
    Ok(Reorient { src: dims_of(src), flip: [flip.0, flip.1, flip.2], in_dims: dims_of(in_dims), out_dims: dims_of(out_dims) })
}

fn reorient_to_py<'py>(py: Python<'py>, r: &Reorient) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("src", dims3(r.src))?;
    d.set_item("flip", (r.flip[0], r.flip[1], r.flip[2]))?;
    d.set_item("in_dims", dims3(r.in_dims))?;
    d.set_item("out_dims", dims3(r.out_dims))?;
    Ok(d)
}

// ─── handles ────────────────────────────────────────────────────────────────

/// The per-voxel orientation mixture (`MixtureField`): the single object both the clean signal
/// and the closed-form ground truth derive from. Immutable.
#[pyclass(frozen, module = "trxscan._core")]
struct Mixture {
    inner: Arc<MixtureField>,
}

#[pymethods]
impl Mixture {
    #[getter]
    fn dims(&self) -> (usize, usize, usize) {
        dims3(self.inner.dims)
    }
    #[getter]
    fn nvert(&self) -> usize {
        self.inner.nvert()
    }
    #[getter]
    fn has_myelin(&self) -> bool {
        self.inner.myelin.is_some()
    }
    /// Hemisphere vertices, `(nvert*3,)` row-major.
    fn sphere_vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let mut v = Vec::with_capacity(self.inner.nvert() * 3);
        for vert in &self.inner.sphere.verts {
            v.extend_from_slice(vert);
        }
        PyArray1::from_vec(py, v)
    }
    /// The orientation histogram, `(nvox*nvert,)`, voxel-major (`vox*nvert + v`), unnormalised.
    fn odf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.odf)
    }
    /// Normalised (wm, gm, csf) fractions, each `(nvox,)` flat F-order.
    fn fractions<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
        (
            PyArray1::from_slice(py, &self.inner.wm),
            PyArray1::from_slice(py, &self.inner.gm),
            PyArray1::from_slice(py, &self.inner.csf),
        )
    }
    /// 1 where a masked WM voxel has no fibre support (isotropic fallback), `(nvox,)`.
    fn fallback<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        PyArray1::from_slice(py, &self.inner.fallback)
    }
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        params_to_dict(py, &self.inner.params)
    }
    /// A copy of this mixture with a per-voxel myelination map (0..1, `(nvox,)`) attached.
    fn with_myelin(&self, myelin: PyReadonlyArray1<'_, f32>) -> PyResult<Mixture> {
        let m = f32s(&myelin)?;
        check_len("myelin", m.len(), self.inner.nvox())?;
        let mut field = MixtureField {
            dims: self.inner.dims,
            sphere: self.inner.sphere.clone(),
            odf: self.inner.odf.clone(),
            wm: self.inner.wm.clone(),
            gm: self.inner.gm.clone(),
            csf: self.inner.csf.clone(),
            fallback: self.inner.fallback.clone(),
            params: self.inner.params,
            myelin: None,
        };
        field.myelin = Some(m);
        Ok(Mixture { inner: Arc::new(field) })
    }
    /// Up to `npeaks` ground-truth fibre peaks per ACQUIRED voxel (the mixture grid is `o` times
    /// finer in-plane): `((nx/o)*(ny/o)*nz*3*npeaks,)`, `vox*3*npeaks + 3k + c`, unit direction
    /// scaled by mass fraction.
    fn truth_peaks<'py>(&self, py: Python<'py>, o: usize, npeaks: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if o == 0 || self.inner.dims[0] % o != 0 || self.inner.dims[1] % o != 0 {
            return Err(verr(format!("oversampling {o} does not divide the mixture grid {:?}", self.inner.dims)));
        }
        let inner = self.inner.clone();
        let v = py.detach(move || trxscan::truth::truth_peaks(&inner, o, npeaks));
        Ok(PyArray1::from_vec(py, v))
    }
    /// The 27 closed-form microstructure maps (see `scalar_names()`), each `(nvox,)`.
    #[pyo3(signature = (tau=None, d_perp_floor=None, ng_floor=None, ng_top_k=None))]
    fn scalars<'py>(
        &self, py: Python<'py>, tau: Option<f64>, d_perp_floor: Option<f64>, ng_floor: Option<f64>, ng_top_k: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let base = ScalarConfig::default();
        let cfg = ScalarConfig {
            tau: tau.unwrap_or(base.tau),
            d_perp_floor: d_perp_floor.unwrap_or(base.d_perp_floor),
            ng_floor: ng_floor.unwrap_or(base.ng_floor),
            ng_top_k: ng_top_k.unwrap_or(base.ng_top_k),
        };
        let inner = self.inner.clone();
        let maps = py.detach(move || field_scalars(&inner, &cfg));
        let out = PyList::empty(py);
        for m in maps {
            out.append(PyArray1::from_vec(py, m))?;
        }
        Ok(out)
    }
}

/// Per-compartment clean signal images (fiber, GM, CSF), 4-D interleaved, plus their T2s.
/// Mutated in place by the stages that follow the signal stage (they are large).
#[pyclass(module = "trxscan._core")]
struct Compartments {
    inner: Mutex<comp::Compartments>,
}

impl Compartments {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, comp::Compartments>> {
        self.inner.lock().map_err(|_| PyRuntimeError::new_err("Compartments mutex poisoned"))
    }
}

#[pymethods]
impl Compartments {
    /// Build from explicit images: `images` is a list of flat 4-D interleaved arrays (one per
    /// compartment, each `nvox*ngrad`), `t2` one T2 (ms) per compartment.
    #[staticmethod]
    fn from_images(dims: (usize, usize, usize), ngrad: usize, images: &Bound<'_, PyAny>, t2: Vec<f32>) -> PyResult<Compartments> {
        let dims = dims_of(dims);
        let n = nvox(dims) * ngrad;
        let mut imgs = Vec::new();
        for item in images.try_iter()? {
            let a: PyReadonlyArray1<f32> = item?.extract()?;
            let v = f32s(&a)?;
            check_len("compartment image", v.len(), n)?;
            imgs.push(v);
        }
        check_len("t2", t2.len(), imgs.len())?;
        Ok(Compartments { inner: Mutex::new(comp::Compartments { dims, ngrad, images: imgs, t2 }) })
    }
    #[getter]
    fn dims(&self) -> PyResult<(usize, usize, usize)> {
        Ok(dims3(self.lock()?.dims))
    }
    #[getter]
    fn ngrad(&self) -> PyResult<usize> {
        Ok(self.lock()?.ngrad)
    }
    #[getter]
    fn n_compartments(&self) -> PyResult<usize> {
        Ok(self.lock()?.images.len())
    }
    #[getter]
    fn t2(&self) -> PyResult<Vec<f32>> {
        Ok(self.lock()?.t2.clone())
    }
    /// Copies of the compartment images, each flat `nvox*ngrad` interleaved.
    fn images<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let g = self.lock()?;
        let out = PyList::empty(py);
        for img in &g.images {
            out.append(PyArray1::from_slice(py, img))?;
        }
        Ok(out)
    }
    /// The compartment sum (S/S0 without T2), flat `nvox*ngrad`.
    fn mixed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let g = self.lock()?;
        Ok(PyArray1::from_vec(py, g.mixed().data))
    }
    /// Scale each compartment by `[fiber, gm, csf]` (the CLI's `--tissue-s0`).
    fn apply_s0(&self, s0: (f32, f32, f32)) -> PyResult<()> {
        self.lock()?.apply_s0([s0.0, s0.1, s0.2]);
        Ok(())
    }
    /// Within-volume multiband motion + slice dropout (`motion::apply_multiband_motion`), in
    /// place; returns the dropped-shot ground truth as dicts.
    #[pyo3(signature = (affine, mb, interleaved, bvals, b_max, events, slice_z=None, nz_full=None))]
    #[allow(clippy::too_many_arguments)]
    fn apply_multiband_motion<'py>(
        &self, py: Python<'py>, affine: PyReadonlyArray2<'_, f64>, mb: usize, interleaved: bool,
        bvals: PyReadonlyArray1<'_, f64>, b_max: f64, events: &Bound<'_, PyAny>,
        slice_z: Option<Vec<usize>>, nz_full: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let v2w = affine4(&affine)?;
        let bvals = f64s(&bvals)?;
        let events = events_from(events)?;
        let mut g = self.lock()?;
        check_len("bvals", bvals.len(), g.ngrad)?;
        let (dims, ngrad) = (g.dims, g.ngrad);
        if let Some(sz) = &slice_z {
            check_len("slice_z", sz.len(), dims[2])?;
        }
        let c: &mut comp::Compartments = &mut g;
        let dropped = py.detach(|| motion::apply_multiband_motion_slab(
            &mut c.images, dims, ngrad, v2w, mb, interleaved, &bvals, b_max, &events, slice_z.as_deref(), nz_full,
        ));
        let out = PyList::empty(py);
        for d in dropped {
            let e = PyDict::new(py);
            e.set_item("volume", d.volume)?;
            e.set_item("shot", d.shot)?;
            e.set_item("slices", d.slices)?;
            e.set_item("attenuation", d.attenuation as f64)?;
            out.append(e)?;
        }
        Ok(out)
    }
    /// A new `Compartments` holding only the given local slices, in that order.
    fn select_slices(&self, slices: Vec<usize>) -> PyResult<Compartments> {
        let g = self.lock()?;
        if let Some(bad) = slices.iter().find(|&&z| z >= g.dims[2]) {
            return Err(verr(format!("slice {bad} out of range for {} slices", g.dims[2])));
        }
        Ok(Compartments { inner: Mutex::new(g.select_slices(&slices)) })
    }
    /// Apply the gradient-nonlinearity spatial warp to every compartment image, in place.
    fn warp_gnl(&self, py: Python<'_>, field: &GnlField, modulate: bool) -> PyResult<()> {
        let mut g = self.lock()?;
        if field.inner.dims != g.dims {
            return Err(verr(format!("GNL field grid {:?} != compartment grid {:?}", field.inner.dims, g.dims)));
        }
        let f = field.inner.clone();
        let c: &mut comp::Compartments = &mut g;
        py.detach(|| {
            let ngrad = c.ngrad;
            for img in c.images.iter_mut() {
                *img = f.warp_4d(img, ngrad, modulate);
            }
        });
        Ok(())
    }
}

/// A gradient-coil coefficient set (`gnl::GradCoef`). Immutable; `scaled` returns a new one.
#[pyclass(frozen, module = "trxscan._core")]
struct GradCoef {
    inner: Arc<gnl::GradCoef>,
}

#[pymethods]
impl GradCoef {
    /// From a CLI-style spec: `"whole-body-80"`, `"connectom-300"`, or the path of a Siemens `.grad` file.
    #[staticmethod]
    fn from_spec(spec: &str) -> PyResult<GradCoef> {
        Ok(GradCoef { inner: Arc::new(gnl::GradCoef::from_spec(spec).map_err(verr)?) })
    }
    /// From the text of a Siemens `.grad` coefficient file.
    #[staticmethod]
    fn parse_siemens(text: &str) -> PyResult<GradCoef> {
        Ok(GradCoef { inner: Arc::new(gnl::GradCoef::parse_siemens(text).map_err(verr)?) })
    }
    /// Every nonlinear term multiplied by `s` (the severity knob).
    fn scaled(&self, s: f64) -> GradCoef {
        let mut c = (*self.inner).clone();
        c.scale_nonlinear(s);
        GradCoef { inner: Arc::new(c) }
    }
    fn to_siemens(&self) -> String {
        self.inner.write_siemens()
    }
    #[getter]
    fn n_terms(&self) -> usize {
        self.inner.terms.len()
    }
    #[getter]
    fn r0_mm(&self) -> f64 {
        self.inner.r0_mm
    }
}

/// The gradient-nonlinearity field cached on a grid (`gnl::GnlField`). Immutable.
#[pyclass(frozen, module = "trxscan._core")]
struct GnlField {
    inner: Arc<gnl::GnlField>,
}

#[pymethods]
impl GnlField {
    /// Evaluate `coef` at every voxel centre of the grid (scanner isocentre at the world origin).
    #[staticmethod]
    fn on_grid(py: Python<'_>, coef: &GradCoef, dims: (usize, usize, usize), affine: PyReadonlyArray2<'_, f64>) -> PyResult<GnlField> {
        let grid = Grid { dims: dims_of(dims), voxel_to_world: affine4(&affine)? };
        let c = coef.inner.clone();
        let f = py.detach(move || gnl::GnlField::on_grid(&c, &grid));
        Ok(GnlField { inner: Arc::new(f) })
    }
    #[getter]
    fn dims(&self) -> (usize, usize, usize) {
        dims3(self.inner.dims)
    }
    /// Forward displacement `phi(r) - r` per voxel, RAS mm, `(nvox*3,)`.
    fn disp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.disp.iter().flat_map(|d| d.iter().copied()).collect())
    }
    /// Continuous voxel coordinate of `phi^-1(x)` per voxel, `(nvox*3,)`.
    fn src_vox<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.src_vox.iter().flat_map(|d| d.iter().copied()).collect())
    }
    /// The 9-volume gradient-deviation image in HCP/FSL layout, `(nvox*9,)`.
    fn graddev<'py>(&self, py: Python<'py>, affine: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let grid = Grid { dims: self.inner.dims, voxel_to_world: affine4(&affine)? };
        let f = self.inner.clone();
        let v = py.detach(move || f.graddev_volumes(&grid));
        Ok(PyArray1::from_vec(py, v))
    }
    fn envelope<'py>(&self, py: Python<'py>, affine: PyReadonlyArray2<'_, f64>, radius_mm: f64) -> PyResult<Bound<'py, PyDict>> {
        let grid = Grid { dims: self.inner.dims, voxel_to_world: affine4(&affine)? };
        let e = self.inner.envelope(&grid, radius_mm);
        let d = PyDict::new(py);
        d.set_item("n_voxels", e.n_voxels)?;
        d.set_item("max_disp_mm", e.max_disp_mm)?;
        d.set_item("max_gradient_dev", e.max_gradient_dev)?;
        d.set_item("max_angle_deg", e.max_angle_deg)?;
        Ok(d)
    }
    /// Warp one 3-D volume (flat `(nvox,)`) into the apparent frame.
    fn warp_volume<'py>(&self, py: Python<'py>, vol: PyReadonlyArray1<'_, f32>, modulate: bool) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let v = f32s(&vol)?;
        check_len("volume", v.len(), nvox(self.inner.dims))?;
        let f = self.inner.clone();
        Ok(PyArray1::from_vec(py, py.detach(move || f.warp_volume(&v, modulate))))
    }
    /// Warp a 4-D interleaved image (flat `(nvox*ngrad,)`).
    fn warp_4d<'py>(&self, py: Python<'py>, vol: PyReadonlyArray1<'_, f32>, ngrad: usize, modulate: bool) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let v = f32s(&vol)?;
        check_len("4-D image", v.len(), nvox(self.inner.dims) * ngrad)?;
        let f = self.inner.clone();
        Ok(PyArray1::from_vec(py, py.detach(move || f.warp_4d(&v, ngrad, modulate))))
    }
}

// ─── signal stage ───────────────────────────────────────────────────────────

/// Rasterize streamlines into the per-voxel orientation mixture (`compartments::generate_mixture`).
/// Tissue arrays are flat `(nvox,)` F-order float32; `mask` is thresholded at 0.5 like the CLI.
/// `positions` `(n, 3)` world RAS mm, `offsets` `(n_streamlines+1,)` uint32, `weights`
/// `(n_streamlines,)` float32 or None.
#[pyfunction]
#[pyo3(signature = (dims, wm, gm, csf, mask, affine, positions, offsets, weights, params, kappa=None, myelin=None, sphere_level=3))]
#[allow(clippy::too_many_arguments)]
fn build_mixture(
    py: Python<'_>,
    dims: (usize, usize, usize),
    wm: PyReadonlyArray1<'_, f32>, gm: PyReadonlyArray1<'_, f32>, csf: PyReadonlyArray1<'_, f32>, mask: PyReadonlyArray1<'_, f32>,
    affine: PyReadonlyArray2<'_, f64>,
    positions: PyReadonlyArray2<'_, f64>, offsets: PyReadonlyArray1<'_, u32>, weights: Option<PyReadonlyArray1<'_, f32>>,
    params: &Bound<'_, PyDict>, kappa: Option<f64>, myelin: Option<PyReadonlyArray1<'_, f32>>, sphere_level: usize,
) -> PyResult<Mixture> {
    let dims = dims_of(dims);
    let tissue = tissue_from_flat(dims, &wm, &gm, &csf, &mask)?;
    let grid = Grid { dims, voxel_to_world: affine4(&affine)? };
    let positions = rows3(&positions, "positions")?;
    let offsets = u32s(&offsets)?;
    if offsets.is_empty() || *offsets.last().unwrap() as usize != positions.len() {
        return Err(verr("offsets must be CSR: length n_streamlines+1, last entry == number of points"));
    }
    let weights = match weights {
        Some(w) => {
            let w = f32s(&w)?;
            check_len("weights", w.len(), offsets.len() - 1)?;
            Some(w)
        }
        None => None,
    };
    let params = params_from_dict(params)?;
    let kappa = kappa.filter(|k| *k > 0.0);
    let myelin = match myelin {
        Some(m) => {
            let m = f32s(&m)?;
            check_len("myelin", m.len(), nvox(dims))?;
            Some(m)
        }
        None => None,
    };
    let field = py.detach(move || {
        let mut mix = comp::generate_mixture(
            &grid, &positions, &offsets, weights.as_deref(), &tissue, &params, kappa, HemiSphere::icosphere(sphere_level),
        );
        mix.myelin = myelin;
        mix
    });
    Ok(Mixture { inner: Arc::new(field) })
}

/// A mixture from an explicit fibre list (`compartments::mixture_from_fibers`): entry `i`
/// deposits `weight[i]` of orientation mass along `dirs[i]` in voxel `voxel[i]` (flat index).
#[pyfunction]
#[pyo3(signature = (dims, voxel, dirs, weight, wm, gm, csf, mask, params, kappa=None, sphere_level=3))]
#[allow(clippy::too_many_arguments)]
fn mixture_from_fibers(
    py: Python<'_>,
    dims: (usize, usize, usize),
    voxel: PyReadonlyArray1<'_, u32>, dirs: PyReadonlyArray2<'_, f64>, weight: PyReadonlyArray1<'_, f64>,
    wm: PyReadonlyArray1<'_, f32>, gm: PyReadonlyArray1<'_, f32>, csf: PyReadonlyArray1<'_, f32>, mask: PyReadonlyArray1<'_, f32>,
    params: &Bound<'_, PyDict>, kappa: Option<f64>, sphere_level: usize,
) -> PyResult<Mixture> {
    let dims = dims_of(dims);
    let tissue = tissue_from_flat(dims, &wm, &gm, &csf, &mask)?;
    let voxel = u32s(&voxel)?;
    let dirs = rows3(&dirs, "dirs")?;
    let weight = f64s(&weight)?;
    check_len("dirs", dirs.len(), voxel.len())?;
    check_len("weight", weight.len(), voxel.len())?;
    let n = nvox(dims);
    if let Some(bad) = voxel.iter().find(|&&v| v as usize >= n) {
        return Err(verr(format!("fibre voxel index {bad} out of range for {dims:?}")));
    }
    let params = params_from_dict(params)?;
    let kappa = kappa.filter(|k| *k > 0.0);
    let field = py.detach(move || {
        comp::mixture_from_fibers(dims, &voxel, &dirs, &weight, &tissue, &params, kappa, HemiSphere::icosphere(sphere_level))
    });
    Ok(Mixture { inner: Arc::new(field) })
}

/// The per-compartment clean signal from a mixture (`signal_from_mixture_gnl`).
#[pyfunction]
#[pyo3(signature = (mix, bvals, bvecs, gnl=None))]
fn signal_from_mixture(
    py: Python<'_>, mix: &Mixture, bvals: PyReadonlyArray1<'_, f64>, bvecs: PyReadonlyArray2<'_, f64>, gnl: Option<&GnlField>,
) -> PyResult<Compartments> {
    let scheme = scheme_from(&bvals, &bvecs)?;
    if let Some(f) = gnl {
        if f.inner.dims != mix.inner.dims {
            return Err(verr(format!("GNL field grid {:?} != mixture grid {:?}", f.inner.dims, mix.inner.dims)));
        }
    }
    let m = mix.inner.clone();
    let f = gnl.map(|g| g.inner.clone());
    let c = py.detach(move || comp::signal_from_mixture_gnl(&m, &scheme, f.as_deref()));
    Ok(Compartments { inner: Mutex::new(c) })
}

/// The faithful motion path (`generate_compartments_moving`): per volume, streamlines and tissue
/// are rigidly moved by that volume's pose and re-rasterized. `poses` is `(ngrad, 6)`
/// `[tx, ty, tz mm, rx, ry, rz deg]`.
#[pyfunction]
#[pyo3(signature = (dims, wm, gm, csf, mask, affine, positions, offsets, bvals, bvecs, params, poses))]
#[allow(clippy::too_many_arguments)]
fn signal_moving(
    py: Python<'_>,
    dims: (usize, usize, usize),
    wm: PyReadonlyArray1<'_, f32>, gm: PyReadonlyArray1<'_, f32>, csf: PyReadonlyArray1<'_, f32>, mask: PyReadonlyArray1<'_, f32>,
    affine: PyReadonlyArray2<'_, f64>,
    positions: PyReadonlyArray2<'_, f64>, offsets: PyReadonlyArray1<'_, u32>,
    bvals: PyReadonlyArray1<'_, f64>, bvecs: PyReadonlyArray2<'_, f64>,
    params: &Bound<'_, PyDict>, poses: PyReadonlyArray2<'_, f64>,
) -> PyResult<Compartments> {
    let dims = dims_of(dims);
    let tissue = tissue_from_flat(dims, &wm, &gm, &csf, &mask)?;
    let grid = Grid { dims, voxel_to_world: affine4(&affine)? };
    let positions = rows3(&positions, "positions")?;
    let offsets = u32s(&offsets)?;
    if offsets.is_empty() || *offsets.last().unwrap() as usize != positions.len() {
        return Err(verr("offsets must be CSR: length n_streamlines+1, last entry == number of points"));
    }
    let scheme = scheme_from(&bvals, &bvecs)?;
    let params = params_from_dict(params)?;
    let poses = poses_from(&poses)?;
    check_len("poses", poses.len(), scheme.len())?;
    let c = py.detach(move || comp::generate_compartments_moving(&grid, &positions, &offsets, &tissue, &scheme, &params, &poses));
    Ok(Compartments { inner: Mutex::new(c) })
}

// ─── motion helpers ─────────────────────────────────────────────────────────

#[pyfunction]
fn dropout_seed(run_seed: u64) -> u64 {
    motion::dropout_seed(run_seed)
}

#[pyfunction]
fn dropout_events<'py>(py: Python<'py>, bvals: PyReadonlyArray1<'_, f64>, n_shots: usize, rate: f64, seed: u64) -> PyResult<Bound<'py, PyList>> {
    let bvals = f64s(&bvals)?;
    events_to_py(py, &motion::dropout_events(&bvals, n_shots, rate, seed))
}

/// Resolve a motion mode to `(n_units, 6)` absolute poses `[tx, ty, tz mm, rx, ry, rz deg]`.
/// `kind`: `"random"` / `"linear"` (with `trans_mm`, `rot_deg`, `volumes`) or `"trajectory"`
/// (with `poses`, padded with identity / truncated to `n_units`).
#[pyfunction]
#[pyo3(signature = (kind, n_units, seed=0, trans_mm=None, rot_deg=None, volumes=None, poses=None))]
fn resolve_poses<'py>(
    py: Python<'py>, kind: &str, n_units: usize, seed: u64,
    trans_mm: Option<(f64, f64, f64)>, rot_deg: Option<(f64, f64, f64)>, volumes: Option<Vec<usize>>,
    poses: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = trans_mm.map(|t| [t.0, t.1, t.2]).unwrap_or([0.0; 3]);
    let r = rot_deg.map(|r| [r.0, r.1, r.2]).unwrap_or([0.0; 3]);
    let vols = volumes.unwrap_or_else(|| (0..n_units).collect());
    let mode = match kind {
        "random" => MotionMode::Random { trans_mm: t, rot_deg: r, volumes: vols },
        "linear" => MotionMode::Linear { trans_mm: t, rot_deg: r, volumes: vols },
        "trajectory" => MotionMode::Trajectory { poses: poses_from(poses.as_ref().ok_or_else(|| verr("trajectory needs poses"))?)? },
        "off" => MotionMode::Off,
        o => return Err(verr(format!("kind: expected random/linear/trajectory/off, got {o:?}"))),
    };
    Ok(poses_to_py(py, &motion::resolve_poses(&mode, n_units, seed)))
}

/// `(n, 6)` poses -> flat `(n*16,)` row-major 4x4 world transforms about `center` (mm).
#[pyfunction]
fn poses_to_matrices<'py>(py: Python<'py>, poses: PyReadonlyArray2<'_, f64>, center: (f64, f64, f64)) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let poses = poses_from(&poses)?;
    let mut out = Vec::with_capacity(poses.len() * 16);
    for p in &poses {
        for row in p.to_matrix([center.0, center.1, center.2]) {
            out.extend_from_slice(&row);
        }
    }
    Ok(PyArray1::from_vec(py, out))
}

/// FOV centre (world mm) of a grid, the rotation centre motion uses.
#[pyfunction]
fn fov_center(dims: (usize, usize, usize), affine: PyReadonlyArray2<'_, f64>) -> PyResult<(f64, f64, f64)> {
    let c = motion::fov_center(dims_of(dims), affine4(&affine)?);
    Ok((c[0], c[1], c[2]))
}

// ─── acquisition stage ──────────────────────────────────────────────────────

/// Run the k-space stage (`kspace::simulate_acquisition_complex`) on a `Compartments` object.
///
/// Returns a dict: `re`, `im`, `mag`, `phase` (flat 4-D interleaved on the ACQUIRED grid) and
/// `kspace` (None, or a dict with `slices`, `nx`, `ny`, `n_coils`, `mask` (bool, ky*nx+kx),
/// `sensitivities` (coil,y,x), `combined` (vol,slice,y,x as float32 pairs) and, when captured,
/// `acquired` / `reconstructed` / `coil_images` (vol,slice,coil,ky,kx pairs)).
#[pyfunction]
#[pyo3(signature = (comp, fmap, acq_dims, bvals, bvecs, acquisition, phase_model="hbcd", seed=0, noise_sigma=None,
                    te_per_volume=None, slice_z=None, nz_full=None, kspace_slices=None, capture=(false, false, false), progress=None))]
#[allow(clippy::too_many_arguments)]
fn simulate_acquisition<'py>(
    py: Python<'py>,
    comp: &Compartments,
    fmap: PyReadonlyArray1<'_, f32>,
    acq_dims: (usize, usize, usize),
    bvals: PyReadonlyArray1<'_, f64>, bvecs: PyReadonlyArray2<'_, f64>,
    acquisition: &Bound<'_, PyDict>,
    phase_model: &str,
    seed: u64,
    noise_sigma: Option<PyReadonlyArray1<'_, f32>>,
    te_per_volume: Option<Vec<f64>>,
    slice_z: Option<Vec<usize>>,
    nz_full: Option<usize>,
    kspace_slices: Option<Vec<usize>>,
    capture: (bool, bool, bool),
    progress: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let acq = acquisition_from_dict(acquisition)?;
    let scheme = scheme_from(&bvals, &bvecs)?;
    let phase = match phase_model {
        "hbcd" => PhaseModel::hbcd_like(),
        "none" => PhaseModel::none(),
        o => return Err(verr(format!("phase_model: expected 'hbcd' or 'none', got {o:?}"))),
    };
    let acq_dims = dims_of(acq_dims);
    let g = comp.lock()?;
    let sim_dims = g.dims;
    let ngrad = g.ngrad;
    check_len("bvals", scheme.len(), ngrad)?;
    if sim_dims[2] != acq_dims[2] {
        return Err(verr(format!("slice count differs: signal grid {sim_dims:?} vs acquisition grid {acq_dims:?}")));
    }
    if acq_dims[0] == 0 || acq_dims[1] == 0 || sim_dims[0] % acq_dims[0] != 0 || sim_dims[1] % acq_dims[1] != 0
        || sim_dims[0] / acq_dims[0] != sim_dims[1] / acq_dims[1]
    {
        return Err(verr(format!("signal grid {sim_dims:?} must be an integer multiple of the acquisition grid {acq_dims:?} in-plane")));
    }
    let fmap = f32s(&fmap)?;
    check_len("fieldmap (signal grid)", fmap.len(), nvox(sim_dims))?;
    let noise_sigma = match noise_sigma {
        Some(n) => {
            let n = f32s(&n)?;
            check_len("noise_sigma (acquisition grid)", n.len(), nvox(acq_dims))?;
            Some(n)
        }
        None => None,
    };
    if let Some(t) = &te_per_volume {
        check_len("te_per_volume", t.len(), ngrad)?;
    }
    if let Some(sz) = &slice_z {
        check_len("slice_z", sz.len(), acq_dims[2])?;
        let nzf = nz_full.ok_or_else(|| verr("slice_z requires nz_full"))?;
        if let Some(bad) = sz.iter().find(|&&z| z >= nzf) {
            return Err(verr(format!("slice_z entry {bad} >= nz_full {nzf}")));
        }
    }
    if let Some(ks) = &kspace_slices {
        if let Some(bad) = ks.iter().find(|&&z| z >= acq_dims[2]) {
            return Err(verr(format!("kspace_slices entry {bad} out of range for {} slices", acq_dims[2])));
        }
    }
    let cap = SliceCapture { acquired: capture.0, reconstructed: capture.1, coil_images: capture.2 };
    let err_slot: Mutex<Option<PyErr>> = Mutex::new(None);
    let cb = progress;
    let call_progress = |done: usize, total: usize| {
        if let Some(cb) = &cb {
            Python::attach(|py| {
                if let Err(e) = cb.call1(py, (done, total)) {
                    if let Ok(mut s) = err_slot.lock() {
                        if s.is_none() {
                            *s = Some(e);
                        }
                    }
                }
            });
        }
    };
    let out = py.detach(|| {
        let inp = kspace::SimulationInput {
            sim_dims,
            acq_dims,
            ngrad,
            images: &g.images,
            t2: &g.t2,
            fmap: &fmap,
            bvals: &scheme.bvals,
            bvecs: &scheme.bvecs,
            phase: &phase,
            seed,
            noise_sigma: noise_sigma.as_deref(),
        };
        let opts = AcquisitionOptions {
            slice_z: slice_z.as_deref(),
            nz_full,
            kspace_slices: kspace_slices.as_deref(),
            capture: cap,
            t_echo_per_volume: te_per_volume.as_deref(),
            progress: if cb.is_some() { Some(&call_progress) } else { None },
        };
        kspace::simulate_acquisition_complex(&inp, &acq, &opts)
    });
    drop(g);
    if let Some(e) = err_slot.lock().ok().and_then(|mut s| s.take()) {
        return Err(e);
    }
    let n = out.re.len();
    let (mut mag, mut ph) = (vec![0.0f32; n], vec![0.0f32; n]);
    for i in 0..n {
        let (re, im) = (out.re[i], out.im[i]);
        mag[i] = (re * re + im * im).sqrt();
        ph[i] = im.atan2(re);
    }
    let d = PyDict::new(py);
    d.set_item("re", PyArray1::from_vec(py, out.re))?;
    d.set_item("im", PyArray1::from_vec(py, out.im))?;
    d.set_item("mag", PyArray1::from_vec(py, mag))?;
    d.set_item("phase", PyArray1::from_vec(py, ph))?;
    match out.kspace {
        None => d.set_item("kspace", py.None())?,
        Some(k) => {
            let kd = PyDict::new(py);
            kd.set_item("slices", k.slices)?;
            kd.set_item("nx", k.nx)?;
            kd.set_item("ny", k.ny)?;
            kd.set_item("n_coils", k.n_coils)?;
            kd.set_item("mask", PyArray1::from_vec(py, k.mask.iter().map(|&b| b as u8).collect()))?;
            kd.set_item("sensitivities", PyArray1::from_vec(py, k.sensitivities))?;
            kd.set_item("combined", PyArray1::from_vec(py, flat_pairs(k.combined)))?;
            match k.acquired {
                Some(v) => kd.set_item("acquired", PyArray1::from_vec(py, flat_pairs(v)))?,
                None => kd.set_item("acquired", py.None())?,
            }
            match k.reconstructed {
                Some(v) => kd.set_item("reconstructed", PyArray1::from_vec(py, flat_pairs(v)))?,
                None => kd.set_item("reconstructed", py.None())?,
            }
            match k.coil_images {
                Some(v) => kd.set_item("coil_images", PyArray1::from_vec(py, flat_pairs(v)))?,
                None => kd.set_item("coil_images", py.None())?,
            }
            d.set_item("kspace", kd)?;
        }
    }
    Ok(d)
}

/// Which k-space samples an acquisition acquires on an `nx` x `ny` matrix: `(ny*nx,)` uint8, `ky*nx + kx`.
#[pyfunction]
fn sampling_mask<'py>(py: Python<'py>, nx: usize, ny: usize, acquisition: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let acq = acquisition_from_dict(acquisition)?;
    Ok(PyArray1::from_vec(py, kspace::sampling_mask(nx, ny, &acq).iter().map(|&b| b as u8).collect()))
}

/// Per-ky-line readout timing (`kspace::epi_timing`) as a dict of arrays and scalars.
#[pyfunction]
fn epi_timing<'py>(py: Python<'py>, nx: usize, ny: usize, acquisition: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let acq = acquisition_from_dict(acquisition)?;
    let t = kspace::epi_timing(nx, ny, &acq);
    let d = PyDict::new(py);
    d.set_item("t_ms", PyArray1::from_vec(py, t.t_ms))?;
    d.set_item("t_rf_ms", PyArray1::from_vec(py, t.t_rf_ms))?;
    d.set_item("t_read_ms", PyArray1::from_vec(py, t.t_read_ms))?;
    d.set_item("order", PyArray1::from_vec(py, t.order.iter().map(|&o| o as u32).collect()))?;
    d.set_item("t_line", t.t_line)?;
    d.set_item("t_echo", t.t_echo)?;
    d.set_item("dt", t.dt)?;
    Ok(d)
}

/// The sample-by-sample trajectory: `(kx, ky, t_ms)` as three `(nx*ny,)` arrays in acquisition order.
#[pyfunction]
fn epi_trajectory<'py>(
    py: Python<'py>, nx: usize, ny: usize, acquisition: &Bound<'_, PyDict>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f64>>)> {
    let acq = acquisition_from_dict(acquisition)?;
    let tr = kspace::epi_trajectory(nx, ny, &acq);
    let kx = tr.iter().map(|t| t.0 as u32).collect();
    let ky = tr.iter().map(|t| t.1 as u32).collect();
    let t = tr.iter().map(|t| t.2).collect();
    Ok((PyArray1::from_vec(py, kx), PyArray1::from_vec(py, ky), PyArray1::from_vec(py, t)))
}

/// Echo the resolved `kspace::Acquisition` for a dict (defaults filled in), for tests and sidecars.
#[pyfunction]
fn acquisition_defaults<'py>(py: Python<'py>, acquisition: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let a = acquisition_from_dict(acquisition)?;
    let d = PyDict::new(py);
    d.set_item("t_line", a.t_line)?;
    d.set_item("t_echo", a.t_echo)?;
    d.set_item("t_inhom", a.t_inhom)?;
    d.set_item("signal_scale", a.signal_scale)?;
    d.set_item("reverse_phase", a.reverse_phase)?;
    d.set_item("do_distortions", a.do_distortions)?;
    d.set_item("do_relaxation", a.do_relaxation)?;
    d.set_item("noise_variance", a.noise_variance)?;
    d.set_item("partial_fourier", a.partial_fourier)?;
    d.set_item("pf_mode", match a.pf_mode { PartialFourierMode::Scanner => "scanner", PartialFourierMode::Contiguous => "contiguous", PartialFourierMode::FiberfoxCompatible => "fiberfox" })?;
    d.set_item("ghost_offset", a.ghost_offset)?;
    d.set_item("eddy_strength", a.eddy_strength)?;
    d.set_item("eddy_quad", a.eddy_quad)?;
    d.set_item("eddy_phase", a.eddy_phase)?;
    d.set_item("eddy_tau", a.eddy_tau)?;
    d.set_item("n_spikes", a.n_spikes)?;
    d.set_item("spike_amplitude", a.spike_amplitude)?;
    d.set_item("n_coils", a.n_coils)?;
    d.set_item("accel", a.accel)?;
    d.set_item("acs_lines", a.acs_lines)?;
    d.set_item("seed", a.seed)?;
    Ok(d)
}

/// `kspace::hbcd_acquisition(ny)` field by field (the CLI protocol), for the Python `Protocol.HBCD` test.
#[pyfunction]
fn acquisition_hbcd<'py>(py: Python<'py>, ny: usize) -> PyResult<Bound<'py, PyDict>> {
    let a = kspace::hbcd_acquisition(ny);
    let d = PyDict::new(py);
    d.set_item("t_line", a.t_line)?;
    d.set_item("t_echo", a.t_echo)?;
    d.set_item("t_inhom", a.t_inhom)?;
    d.set_item("partial_fourier", a.partial_fourier)?;
    d.set_item("pf_mode", match a.pf_mode { PartialFourierMode::Scanner => "scanner", PartialFourierMode::Contiguous => "contiguous", PartialFourierMode::FiberfoxCompatible => "fiberfox" })?;
    d.set_item("ghost_offset", a.ghost_offset)?;
    d.set_item("acs_lines", a.acs_lines)?;
    d.set_item("signal_scale", a.signal_scale)?;
    Ok(d)
}

// ─── GRE fieldmap ───────────────────────────────────────────────────────────

/// Synthesize the dual-echo GRE fieldmap (`gre::synthesize`). Fractions are on the signal grid.
#[pyfunction]
#[pyo3(signature = (sig_dims, sig_affine, acq_dims, acq_affine, fiber, gm, csf, s0, t2, fmap, signal_scale, seed,
                    te_s=(4.92e-3, 7.38e-3), snr=50.0, res_mm=None, snr_vol_exp=1.0, output="phasediff", rx_phase_rad=6.0,
                    b0_field="b0gre", tr_s=0.5, flip_deg=60.0, t1_ms=(830.0, 1330.0, 4000.0), pd=(0.7, 0.85, 1.0), bias=0.3,
                    ringing=true, t2_head_ms=70.0, head=None, z_offset=0, nz_full=None, warp=None, warp_modulate=true))]
#[allow(clippy::too_many_arguments)]
fn gre_synthesize<'py>(
    py: Python<'py>,
    sig_dims: (usize, usize, usize), sig_affine: PyReadonlyArray2<'_, f64>,
    acq_dims: (usize, usize, usize), acq_affine: PyReadonlyArray2<'_, f64>,
    fiber: PyReadonlyArray1<'_, f32>, gm: PyReadonlyArray1<'_, f32>, csf: PyReadonlyArray1<'_, f32>,
    s0: (f32, f32, f32), t2: Vec<f32>, fmap: PyReadonlyArray1<'_, f32>, signal_scale: f64, seed: u64,
    te_s: (f64, f64), snr: f64, res_mm: Option<f64>, snr_vol_exp: f64, output: &str, rx_phase_rad: f64, b0_field: &str,
    tr_s: f64, flip_deg: f64, t1_ms: (f32, f32, f32), pd: (f32, f32, f32), bias: f64, ringing: bool, t2_head_ms: f32,
    head: Option<PyReadonlyArray1<'_, f32>>, z_offset: usize, nz_full: Option<usize>, warp: Option<&GnlField>, warp_modulate: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let sig = Grid { dims: dims_of(sig_dims), voxel_to_world: affine4(&sig_affine)? };
    let acq = Grid { dims: dims_of(acq_dims), voxel_to_world: affine4(&acq_affine)? };
    let n = nvox(sig.dims);
    let (fiber, gm, csf, fmap) = (f32s(&fiber)?, f32s(&gm)?, f32s(&csf)?, f32s(&fmap)?);
    check_len("fiber fraction", fiber.len(), n)?;
    check_len("gm fraction", gm.len(), n)?;
    check_len("csf fraction", csf.len(), n)?;
    check_len("fieldmap", fmap.len(), n)?;
    check_len("t2", t2.len(), 3)?;
    let head: Option<Vec<f32>> = match head {
        Some(h) => {
            let h = f32s(&h)?;
            check_len("head", h.len(), n)?;
            Some(h)
        }
        None => None,
    };
    let output = match output {
        "phasediff" => gre::GreOutput::Phasediff,
        "phase" => gre::GreOutput::Phase,
        o => return Err(verr(format!("output: expected 'phasediff' or 'phase', got {o:?}"))),
    };
    let p = gre::GreParams {
        te_s: [te_s.0, te_s.1], snr, res_mm, snr_vol_exp, output, rx_phase_rad, b0_field: b0_field.to_string(),
        tr_s, flip_deg, t1_ms: [t1_ms.0, t1_ms.1, t1_ms.2], pd: [pd.0, pd.1, pd.2], bias, ringing, t2_head_ms,
    };
    let (tr_s_out, flip_deg_out) = (p.tr_s, p.flip_deg);
    let warp_field = warp.map(|w| w.inner.clone());
    if let Some(w) = &warp_field {
        if w.dims != sig.dims {
            return Err(verr(format!("GNL field grid {:?} != signal grid {:?}", w.dims, sig.dims)));
        }
    }
    let g = py.detach(move || {
        gre::synthesize(
            &gre::GreObject {
                sig_grid: &sig,
                acq_grid: &acq,
                fractions: [&fiber, &gm, &csf],
                s0: [s0.0, s0.1, s0.2],
                t2_ms: &t2,
                fmap_hz: &fmap,
                warp: warp_field.as_deref().map(|f| (f, warp_modulate)),
                signal_scale,
                seed,
                head: head.as_deref(),
                z_offset,
                nz_full: nz_full.unwrap_or(sig.dims[2]),
            },
            &p,
        )
    });
    let d = PyDict::new(py);
    d.set_item("dims", dims3(g.grid.dims))?;
    d.set_item("affine", affine_to_py(py, &g.grid.voxel_to_world))?;
    let [m1, m2] = g.magnitude;
    d.set_item("magnitude1", PyArray1::from_vec(py, m1))?;
    d.set_item("magnitude2", PyArray1::from_vec(py, m2))?;
    let phases = PyList::empty(py);
    for p in g.phase {
        phases.append(PyArray1::from_vec(py, p))?;
    }
    d.set_item("phase", phases)?;
    d.set_item("te_s", (g.te_s[0], g.te_s[1]))?;
    d.set_item("tr_s", tr_s_out)?;
    d.set_item("flip_deg", flip_deg_out)?;
    d.set_item("output", match g.output { gre::GreOutput::Phasediff => "phasediff", gre::GreOutput::Phase => "phase" })?;
    d.set_item("b0_field", g.b0_field)?;
    d.set_item("sigma", g.sigma)?;
    d.set_item("stamped", g.stamped)?;
    Ok(d)
}

// ─── geometry, streamlines, orientation ─────────────────────────────────────

/// Weighted streamline subsampling (`streamlines::subsample_streamlines`): returns
/// `(positions (n,3) flat, offsets, weights|None, stats dict)`.
#[pyfunction]
#[pyo3(signature = (positions, offsets, weights, n, seed))]
fn subsample_streamlines<'py>(
    py: Python<'py>, positions: PyReadonlyArray2<'_, f64>, offsets: PyReadonlyArray1<'_, u32>,
    weights: Option<PyReadonlyArray1<'_, f32>>, n: usize, seed: u64,
) -> PyResult<Bound<'py, PyTuple>> {
    let positions = rows3(&positions, "positions")?;
    let offsets = u32s(&offsets)?;
    if offsets.is_empty() || *offsets.last().unwrap() as usize != positions.len() {
        return Err(verr("offsets must be CSR: length n_streamlines+1, last entry == number of points"));
    }
    let weights = match weights {
        Some(w) => {
            let w = f32s(&w)?;
            check_len("weights", w.len(), offsets.len() - 1)?;
            Some(w)
        }
        None => None,
    };
    let (p, o, w, st) = py.detach(move || trxscan::streamlines::subsample_streamlines(positions, offsets, weights, n, seed));
    let mut flat = Vec::with_capacity(p.len() * 3);
    for q in &p {
        flat.extend_from_slice(q);
    }
    let stats = PyDict::new(py);
    stats.set_item("kept", st.kept)?;
    stats.set_item("total", st.total)?;
    stats.set_item("vertex_fraction", st.vertex_fraction)?;
    stats.set_item("weight_fraction", st.weight_fraction)?;
    stats.set_item("summary", st.summary(seed))?;
    let w_py: Py<PyAny> = match w {
        Some(w) => PyArray1::from_vec(py, w).into_any().unbind(),
        None => py.None(),
    };
    PyTuple::new(py, [
        PyArray1::from_vec(py, flat).into_any().unbind(),
        PyArray1::from_vec(py, o).into_any().unbind(),
        w_py,
        stats.into_any().unbind(),
    ])
}

/// The simulation grid for oversampling `o`: `(affine flat 16, dims)`.
#[pyfunction]
fn hires_grid<'py>(py: Python<'py>, affine: PyReadonlyArray2<'_, f64>, dims: (usize, usize, usize), o: usize) -> PyResult<(Bound<'py, PyArray1<f64>>, (usize, usize, usize))> {
    if o == 0 {
        return Err(verr("oversampling factor must be positive"));
    }
    let g = Grid { dims: dims_of(dims), voxel_to_world: affine4(&affine)? }.hires(o);
    Ok((affine_to_py(py, &g.voxel_to_world), dims3(g.dims)))
}

/// The reorientation that takes an axis-aligned grid to radiological LAS (dcm2niix/FSL).
#[pyfunction]
fn reorient_to_las<'py>(py: Python<'py>, affine: PyReadonlyArray2<'_, f64>, dims: (usize, usize, usize)) -> PyResult<Bound<'py, PyDict>> {
    let r = Reorient::to_las(&affine4(&affine)?, dims_of(dims));
    reorient_to_py(py, &r)
}

#[pyfunction]
fn reorient_identity<'py>(py: Python<'py>, dims: (usize, usize, usize)) -> PyResult<Bound<'py, PyDict>> {
    reorient_to_py(py, &Reorient::identity(dims_of(dims)))
}

/// Apply a reorientation to a flat 4-D interleaved image (`ngrad` volumes; use 1 for 3-D).
#[pyfunction]
fn apply_reorient<'py>(py: Python<'py>, data: PyReadonlyArray1<'_, f32>, ngrad: usize, reorient: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let r = reorient_from(reorient)?;
    let v = f32s(&data)?;
    check_len("image", v.len(), nvox(r.in_dims) * ngrad)?;
    Ok(PyArray1::from_vec(py, py.detach(move || r.apply_volume(&v, ngrad))))
}

#[pyfunction]
fn reorient_affine<'py>(py: Python<'py>, affine: PyReadonlyArray2<'_, f64>, reorient: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r = reorient_from(reorient)?;
    Ok(affine_to_py(py, &r.apply_affine(&affine4(&affine)?)))
}

/// BIDS PhaseEncodingDirection for the written frame given the input PE axis and sign.
#[pyfunction]
fn out_ped(reorient: &Bound<'_, PyDict>, in_pe_axis: usize, in_pe_sign: i32) -> PyResult<String> {
    Ok(reorient_from(reorient)?.out_ped(in_pe_axis, in_pe_sign))
}

/// World-RAS unit directions -> FSL-convention bvecs for the given voxel->world affine, `(n*3,)` flat.
#[pyfunction]
fn fsl_bvecs<'py>(py: Python<'py>, bvecs: PyReadonlyArray2<'_, f64>, affine: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = affine4(&affine)?;
    let mut out = Vec::new();
    for b in rows3(&bvecs, "bvecs")? {
        out.extend_from_slice(&Reorient::fsl_bvec(b, &a));
    }
    Ok(PyArray1::from_vec(py, out))
}

// ─── synthetic objects ──────────────────────────────────────────────────────

/// A rectangle with sub-voxel edges on a `snx` x `sny` simulation slice (`kspace::box_hires`), flat `x + snx*y`.
#[pyfunction]
fn box_hires<'py>(py: Python<'py>, snx: usize, sny: usize, x0: f64, x1: f64, y0: f64, y1: f64) -> Bound<'py, PyArray1<f32>> {
    PyArray1::from_vec(py, kspace::box_hires(snx, sny, x0, x1, y0, y1))
}

/// A step edge at continuous position `edge` (sim-voxel units), flat `x + snx*y`.
#[pyfunction]
fn step_hires<'py>(py: Python<'py>, snx: usize, sny: usize, edge: f64) -> Bound<'py, PyArray1<f32>> {
    PyArray1::from_vec(py, kspace::step_hires(snx, sny, edge))
}


// ─── single voxel, exact directions (dipy.sims style) ───────────────────────

/// The clean per-compartment signal of ONE voxel with exact fibre directions (no hemisphere
/// quantisation): `(fiber, gm, csf)` arrays of length `n_vol`, the same arithmetic as
/// `signal_from_mixture` (`wm * mean_k w_k (intra stick + extra zeppelin)`, GM two-ball, CSF
/// ball; an empty fibre list is the hindered isotropic fallback).
#[pyfunction]
#[pyo3(signature = (bvals, bvecs, dirs, weights, wm, gm, csf, params))]
#[allow(clippy::too_many_arguments)]
fn voxel_signal<'py>(
    py: Python<'py>, bvals: PyReadonlyArray1<'_, f64>, bvecs: PyReadonlyArray2<'_, f64>,
    dirs: PyReadonlyArray2<'_, f64>, weights: PyReadonlyArray1<'_, f64>, wm: f64, gm: f64, csf: f64, params: &Bound<'_, PyDict>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let scheme = scheme_from(&bvals, &bvecs)?;
    let dirs = rows3(&dirs, "dirs")?;
    let w = f64s(&weights)?;
    check_len("weights", w.len(), dirs.len())?;
    let p = params_from_dict(params)?;
    let grads = scheme.fiberfox_gradients();
    let stick = Stick { b_value: p.b_value, diffusivity: p.d_intra };
    let extra = Tensor { b_value: p.b_value, eigenvalues: p.d_extra };
    let gmb = Ball { b_value: p.b_value, diffusivity: p.d_gm };
    let csfb = Ball { b_value: p.b_value, diffusivity: p.d_csf };
    let soma = Ball { b_value: p.b_value, diffusivity: p.d_soma };
    let md = (p.d_extra.0 + p.d_extra.1 + p.d_extra.2) / 3.0;
    let tot: f64 = w.iter().sum();
    let n = grads.len();
    let (mut fo, mut go, mut co) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for g in 0..n {
        let fiber_resp = if tot > 1e-12 {
            dirs.iter().zip(&w).map(|(d, &wk)| {
                let dn = trxscan::mat::normalize(*d);
                wk * (p.intra_frac * stick.simulate(grads[g], dn) + p.extra_frac * extra.simulate(grads[g], dn))
            }).sum::<f64>() / tot
        } else {
            (-p.b_value * trxscan::mat::dot(grads[g], grads[g]) * md).exp()
        };
        fo[g] = wm * fiber_resp;
        go[g] = gm * ((1.0 - p.gm_restricted_frac) * gmb.simulate(grads[g]) + p.gm_restricted_frac * soma.simulate(grads[g]));
        co[g] = csf * csfb.simulate(grads[g]);
    }
    Ok((PyArray1::from_vec(py, fo), PyArray1::from_vec(py, go), PyArray1::from_vec(py, co)))
}

/// The 27 closed-form scalars of ONE voxel with exact fibre directions: the `field_scalars`
/// per-voxel computation on a `VoxelMixture` whose vertices are the given directions.
#[pyfunction]
#[pyo3(signature = (dirs, weights, wm, gm, csf, params, tau=None, d_perp_floor=None, ng_floor=None, ng_top_k=None))]
#[allow(clippy::too_many_arguments)]
fn voxel_truth<'py>(
    py: Python<'py>, dirs: PyReadonlyArray2<'_, f64>, weights: PyReadonlyArray1<'_, f64>, wm: f64, gm: f64, csf: f64,
    params: &Bound<'_, PyDict>, tau: Option<f64>, d_perp_floor: Option<f64>, ng_floor: Option<f64>, ng_top_k: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dirs: Vec<[f64; 3]> = rows3(&dirs, "dirs")?.iter().map(|d| trxscan::mat::normalize(*d)).collect();
    let w = f64s(&weights)?;
    check_len("weights", w.len(), dirs.len())?;
    let p = params_from_dict(params)?;
    let base = ScalarConfig::default();
    let cfg = ScalarConfig {
        tau: tau.unwrap_or(base.tau), d_perp_floor: d_perp_floor.unwrap_or(base.d_perp_floor),
        ng_floor: ng_floor.unwrap_or(base.ng_floor), ng_top_k: ng_top_k.unwrap_or(base.ng_top_k),
    };
    let tot_frac = wm + gm + csf;
    if tot_frac <= 0.0 {
        return Err(verr("fractions must sum to a positive value"));
    }
    let (wf, gf, cf) = (wm / tot_frac, gm / tot_frac, csf / tot_frac);
    let tot: f64 = w.iter().sum();
    let (fi, fe) = (p.intra_frac, p.extra_frac);
    let fr = p.gm_restricted_frac;
    let mut iso = vec![(gf * (1.0 - fr), p.d_gm), (gf * fr, p.d_soma), (cf, p.d_csf)];
    let (mut intra, mut extra) = (vec![0.0; dirs.len()], vec![0.0; dirs.len()]);
    if tot <= 1e-12 || dirs.is_empty() {
        iso.push((wf, (p.d_extra.0 + p.d_extra.1 + p.d_extra.2) / 3.0));
    } else {
        for (i, &wk) in w.iter().enumerate() {
            let frac = wk / tot;
            intra[i] = wf * fi * frac;
            extra[i] = wf * fe * frac;
        }
    }
    let mix = VoxelMixture { verts: &dirs, intra: &intra, extra: &extra, d_par: p.d_intra, d_perp: p.d_extra.1, iso: &iso };
    let force = voxel_scalars(&mix, &cfg).values();
    let has_fibre = intra.iter().sum::<f64>() > 0.0;
    let tissue = wf + gf;
    let icvf = if has_fibre && tissue > 0.0 { wf * fi / tissue } else { 0.0 };
    let odi = if has_fibre { watson_odi(&dirs, &intra) } else { 0.0 };
    let mut out = force.to_vec();
    out.push(icvf);
    out.push(odi);
    out.push(cf);
    Ok(PyArray1::from_vec(py, out))
}

// ─── misc ───────────────────────────────────────────────────────────────────

#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The compartment parameters of a preset (`"neonatal"` | `"adult"` | `"infant"`) as a dict.
#[pyfunction]
fn tissue_preset<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    let p = match name {
        "neonatal" => CompartmentParams::default(),
        "adult" => CompartmentParams::adult(),
        "infant" => CompartmentParams::infant(),
        o => return Err(verr(format!("unknown tissue preset {o:?}; expected 'neonatal', 'adult' or 'infant'"))),
    };
    params_to_dict(py, &p)
}

/// The names of the 27 closed-form microstructure maps, in `Mixture.scalars()` order.
#[pyfunction]
fn scalar_names() -> Vec<&'static str> {
    SCALAR_NAMES.to_vec()
}

/// Size the rayon thread pool (once, before the first parallel call). Returns False if the pool
/// was already built.
#[pyfunction]
fn set_threads(n: usize) -> bool {
    rayon::ThreadPoolBuilder::new().num_threads(n).build_global().is_ok()
}

#[pyfunction]
fn current_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mixture>()?;
    m.add_class::<Compartments>()?;
    m.add_class::<GradCoef>()?;
    m.add_class::<GnlField>()?;
    m.add_function(wrap_pyfunction!(build_mixture, m)?)?;
    m.add_function(wrap_pyfunction!(mixture_from_fibers, m)?)?;
    m.add_function(wrap_pyfunction!(signal_from_mixture, m)?)?;
    m.add_function(wrap_pyfunction!(signal_moving, m)?)?;
    m.add_function(wrap_pyfunction!(dropout_seed, m)?)?;
    m.add_function(wrap_pyfunction!(dropout_events, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_poses, m)?)?;
    m.add_function(wrap_pyfunction!(poses_to_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(fov_center, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_acquisition, m)?)?;
    m.add_function(wrap_pyfunction!(sampling_mask, m)?)?;
    m.add_function(wrap_pyfunction!(epi_timing, m)?)?;
    m.add_function(wrap_pyfunction!(epi_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(acquisition_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(acquisition_hbcd, m)?)?;
    m.add_function(wrap_pyfunction!(gre_synthesize, m)?)?;
    m.add_function(wrap_pyfunction!(subsample_streamlines, m)?)?;
    m.add_function(wrap_pyfunction!(hires_grid, m)?)?;
    m.add_function(wrap_pyfunction!(reorient_to_las, m)?)?;
    m.add_function(wrap_pyfunction!(reorient_identity, m)?)?;
    m.add_function(wrap_pyfunction!(apply_reorient, m)?)?;
    m.add_function(wrap_pyfunction!(reorient_affine, m)?)?;
    m.add_function(wrap_pyfunction!(out_ped, m)?)?;
    m.add_function(wrap_pyfunction!(fsl_bvecs, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_signal, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_truth, m)?)?;
    m.add_function(wrap_pyfunction!(box_hires, m)?)?;
    m.add_function(wrap_pyfunction!(step_hires, m)?)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_names, m)?)?;
    m.add_function(wrap_pyfunction!(tissue_preset, m)?)?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(current_threads, m)?)?;
    Ok(())
}
