//! FORCE-style closed-form ground-truth scalars from the per-voxel Gaussian mixture.
//!
//! Port of `dipy/sims/_force_moments.py` (crash_force fork) plus the specific dipy scalar
//! functions it calls — `dipy/reconst/dki.py` (Carlson `R_F`/`R_D` :77/:151, Tabesh
//! `F1/F2` :241/:341, `G1/G2` :844/:911, MK :692, RK :977, AK :1130, MKT :1434, KFA :1595),
//! `dipy/reconst/dti.py` (FA :62), `dipy/reconst/qti.py` (Westin invariants :1202–:1401), and
//! `dipy/reconst/odf.py` (GFA :31). dipy is BSD-3; ported directly, with dipy as the fixture
//! oracle (`tools/gen_force_fixtures.py` → `tests/fixtures/force_moments.txt`).
//!
//! Everything here is per-voxel arithmetic on a Gaussian compartment mixture: the fibre
//! orientation histogram (stick + zeppelin per vertex) plus isotropic compartments. The only
//! special functions in the whole set are Carlson's elliptic integrals (exact MK). Pure std.
//!
//! Conventions faithfully replicated from dipy (the fixture diff is the referee):
//! - kurtosis clips: MK ∈ [-3/7, 3]; AK, RK, MKT ∈ [-3/7, 10]
//! - branch thresholds: F1/F2 use 2.5e-2 *relative*; G1/G2 use `eps·1e5` *absolute*;
//!   positive-eigenvalue gate 2e-7
//! - `d_perp_floor = 0.12e-3` regularizes singular sticks for MAP-MRI and NG/PA
//! - τ defaults to `1/(4π²)` (dipy's no-timing default → normalized units, contrast only)

use crate::mixture::MixtureField;

type M3 = [[f64; 3]; 3];
/// Full rank-4 tensor, `t[i][j][k][l]`.
type T4 = [[[[f64; 3]; 3]; 3]; 3];

const ID3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// --------------------------------------------------------------------------------------------
// small linear algebra
// --------------------------------------------------------------------------------------------

#[inline]
fn det3(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[inline]
fn madd(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][j] + b[i][j];
        }
    }
    o
}

#[inline]
fn mscale(a: &M3, s: f64) -> M3 {
    let mut o = *a;
    for row in o.iter_mut() {
        for v in row.iter_mut() {
            *v *= s;
        }
    }
    o
}

#[inline]
fn quad_form(a: &M3, v: [f64; 3]) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += v[i] * a[i][j] * v[j];
        }
    }
    s
}

/// Eigen-decomposition of a symmetric 3×3 by cyclic Jacobi rotations. Returns eigenvalues in
/// **descending** order with matching column eigenvectors (`evecs[row][col]`, col = index).
fn eigh3(m: &M3) -> ([f64; 3], M3) {
    let mut a = *m;
    let mut v = ID3;
    for _sweep in 0..64 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if off < 1e-30 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].partial_cmp(&a[i][i]).unwrap());
    let evals = [a[order[0]][order[0]], a[order[1]][order[1]], a[order[2]][order[2]]];
    let mut evecs = [[0.0; 3]; 3];
    for (col, &oi) in order.iter().enumerate() {
        for row in 0..3 {
            evecs[row][col] = v[row][oi];
        }
    }
    (evals, evecs)
}

/// Column `col` of a 3×3.
#[inline]
fn column(m: &M3, col: usize) -> [f64; 3] {
    [m[0][col], m[1][col], m[2][col]]
}

/// An orthonormal pair spanning the plane perpendicular to unit `axis`. NG⊥ only uses 2×2
/// determinants of projections, which are invariant to the in-plane basis choice, so any
/// orthonormal pair matches dipy's `all_tensor_evecs` construction.
fn perp_basis(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let pick = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let b1 = crate::mat::normalize(crate::mat::cross(axis, pick));
    let b2 = crate::mat::cross(axis, b1);
    (b1, b2)
}

// --------------------------------------------------------------------------------------------
// Carlson elliptic integrals (dipy/reconst/dki.py:77,151 — same errtol, same loop, so the
// truncation behaviour is bit-comparable)
// --------------------------------------------------------------------------------------------

/// Carlson `R_F(x, y, z)`, errtol 3e-4.
fn carlson_rf(x: f64, y: f64, z: f64) -> f64 {
    let errtol = 3e-4_f64;
    let (mut xn, mut yn, mut zn) = (x, y, z);
    let mut an = (xn + yn + zn) / 3.0;
    let q = (3.0 * errtol).powf(-1.0 / 6.0)
        * (an - xn).abs().max((an - yn).abs()).max((an - zn).abs());
    let mut n = 0i32;
    while 4.0_f64.powi(-n) * q > an.abs() {
        let (xr, yr, zr) = (xn.sqrt(), yn.sqrt(), zn.sqrt());
        let lamda = xr * (yr + zr) + yr * zr;
        n += 1;
        xn = (xn + lamda) * 0.25;
        yn = (yn + lamda) * 0.25;
        zn = (zn + lamda) * 0.25;
        an = (an + lamda) * 0.25;
    }
    let big_x = 1.0 - xn / an;
    let big_y = 1.0 - yn / an;
    let big_z = -big_x - big_y;
    let e2 = big_x * big_y - big_z * big_z;
    let e3 = big_x * big_y * big_z;
    an.powf(-0.5) * (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 / 44.0 * e2 * e3)
}

/// Carlson `R_D(x, y, z)`, errtol 1e-4.
fn carlson_rd(x: f64, y: f64, z: f64) -> f64 {
    let errtol = 1e-4_f64;
    let (mut xn, mut yn, mut zn) = (x, y, z);
    let a0 = (xn + yn + 3.0 * zn) / 5.0;
    let mut an = a0;
    let q = (errtol / 4.0).powf(-1.0 / 6.0)
        * (an - xn).abs().max((an - yn).abs()).max((an - zn).abs());
    let mut sum_term = 0.0;
    let mut n = 0i32;
    while 4.0_f64.powi(-n) * q > an.abs() {
        let (xr, yr, zr) = (xn.sqrt(), yn.sqrt(), zn.sqrt());
        let lamda = xr * (yr + zr) + yr * zr;
        sum_term += 4.0_f64.powi(-n) / (zr * (zn + lamda));
        n += 1;
        xn = (xn + lamda) * 0.25;
        yn = (yn + lamda) * 0.25;
        zn = (zn + lamda) * 0.25;
        an = (an + lamda) * 0.25;
    }
    let pow4n = 4.0_f64.powi(n);
    let big_x = (a0 - x) / (pow4n * an);
    let big_y = (a0 - y) / (pow4n * an);
    let big_z = -(big_x + big_y) / 3.0;
    let e2 = big_x * big_y - 6.0 * big_z * big_z;
    let e3 = (3.0 * big_x * big_y - 8.0 * big_z * big_z) * big_z;
    let e4 = 3.0 * (big_x * big_y - big_z * big_z) * big_z * big_z;
    let e5 = big_x * big_y * big_z * big_z * big_z;
    pow4n.recip() * an.powf(-1.5)
        * (1.0 - 3.0 / 14.0 * e2 + e3 / 6.0 + 9.0 / 88.0 * e2 * e2 - 3.0 / 22.0 * e4
            - 9.0 / 52.0 * e2 * e3 + 3.0 / 26.0 * e5)
        + 3.0 * sum_term
}

// --------------------------------------------------------------------------------------------
// Tabesh helper functions (dipy/reconst/dki.py:241,341,844,911 — same branch thresholds)
// --------------------------------------------------------------------------------------------

const POSITIVE_EVAL: f64 = 2e-7; // dki.py:49 `_positive_evals`

#[inline]
fn positive_evals(a: f64, b: f64, c: f64) -> bool {
    a > POSITIVE_EVAL && b > POSITIVE_EVAL && c > POSITIVE_EVAL
}

/// Tabesh `F1(a, b, c)` (dki.py:241). Branch threshold 2.5e-2 relative to `a`.
fn f1m(a: f64, b: f64, c: f64) -> f64 {
    let er = 2.5e-2;
    if !positive_evals(a, b, c) {
        return 0.0;
    }
    if (a - b).abs() >= a * er && (a - c).abs() >= a * er {
        let rf = carlson_rf(a / b, a / c, 1.0);
        let rd = carlson_rd(a / b, a / c, 1.0);
        return (a + b + c).powi(2) / (18.0 * (a - b) * (a - c))
            * ((b * c).sqrt() / a * rf
                + (3.0 * a * a - a * b - a * c - b * c) / (3.0 * a * (b * c).sqrt()) * rd
                - 1.0);
    }
    if (a - b).abs() < a * er && (a - c).abs() > a * er {
        return f2m(c, (a + b) / 2.0, (a + b) / 2.0) / 2.0;
    }
    if (a - c).abs() < a * er && (a - b).abs() > a * er {
        return f2m(b, (a + c) / 2.0, (a + c) / 2.0) / 2.0;
    }
    if (a - c).abs() < a * er && (a - b).abs() < a * er {
        return 1.0 / 5.0;
    }
    0.0 // measure-zero boundary crack — dipy leaves these at 0 too
}

/// Tabesh `F2(a, b, c)` (dki.py:341).
fn f2m(a: f64, b: f64, c: f64) -> f64 {
    let er = 2.5e-2;
    if !positive_evals(a, b, c) {
        return 0.0;
    }
    if (b - c).abs() > b * er {
        let rf = carlson_rf(a / b, a / c, 1.0);
        let rd = carlson_rd(a / b, a / c, 1.0);
        return (a + b + c).powi(2) / (3.0 * (b - c).powi(2))
            * ((b + c) / (b * c).sqrt() * rf
                + (2.0 * a - b - c) / (3.0 * (b * c).sqrt()) * rd
                - 2.0);
    }
    if (b - c).abs() < b * er && (a - b).abs() > b * er {
        let l3 = (b + c) / 2.0;
        let x = 1.0 - a / l3;
        let alpha = if x > 0.0 {
            x.sqrt().recip() * x.sqrt().atanh()
        } else {
            (-x).sqrt().recip() * (-x).sqrt().atan()
        };
        return 6.0 * (a + 2.0 * l3).powi(2) / (144.0 * l3 * l3 * (a - l3).powi(2))
            * (l3 * (a + 2.0 * l3) + a * (a - 4.0 * l3) * alpha);
    }
    if (b - c).abs() < b * er && (a - b).abs() < b * er {
        return 6.0 / 15.0;
    }
    0.0
}

/// Tabesh `G1(a, b, c)` (dki.py:844). Branch threshold `eps·1e5` **absolute**.
fn g1m(a: f64, b: f64, c: f64) -> f64 {
    let er = f64::EPSILON * 1e5;
    if !positive_evals(a, b, c) {
        return 0.0;
    }
    if (b - c).abs() > er {
        (a + b + c).powi(2) / (18.0 * b * (b - c).powi(2))
            * (2.0 * b + (c * c - 3.0 * b * c) / (b * c).sqrt())
    } else {
        (a + 2.0 * b).powi(2) / (24.0 * b * b)
    }
}

/// Tabesh `G2(a, b, c)` (dki.py:911).
fn g2m(a: f64, b: f64, c: f64) -> f64 {
    let er = f64::EPSILON * 1e5;
    if !positive_evals(a, b, c) {
        return 0.0;
    }
    if (b - c).abs() > er {
        (a + b + c).powi(2) / (3.0 * (b - c).powi(2)) * ((b + c) / (b * c).sqrt() - 2.0)
    } else {
        (a + 2.0 * b).powi(2) / (12.0 * b * b)
    }
}

// --------------------------------------------------------------------------------------------
// the per-voxel mixture and its moments (_force_moments.py:184 `moments_from_odfs`)
// --------------------------------------------------------------------------------------------

/// One voxel's Gaussian mixture: per-vertex intra (stick, `d_perp = 0`) and extra (zeppelin)
/// orientation weights, plus isotropic `(fraction, diffusivity)` compartments (GM, CSF, the
/// hindered fallback, soma/dot, …). All weights together must sum to 1.
pub struct VoxelMixture<'a> {
    pub verts: &'a [[f64; 3]],
    pub intra: &'a [f64],
    pub extra: &'a [f64],
    /// stick / zeppelin axial diffusivity
    pub d_par: f64,
    /// zeppelin radial diffusivity
    pub d_perp: f64,
    pub iso: &'a [(f64, f64)],
}

/// Mean tensor `D_app` and covariance `C = <D⊗D> − D_app⊗D_app` of the mixture, with **true**
/// sticks (intra `d_perp = 0`) — `_force_moments.py:184` with `intra_dperp=0`.
pub fn moments(mix: &VoxelMixture) -> (M3, T4) {
    let (a, p) = (mix.d_par, mix.d_perp);
    let (mut w_in, mut w_ex) = (0.0, 0.0);
    let mut om_in = [[0.0; 3]; 3];
    let mut om_ex = [[0.0; 3]; 3];
    let mut o4_in: T4 = [[[[0.0; 3]; 3]; 3]; 3];
    let mut o4_ex: T4 = [[[[0.0; 3]; 3]; 3]; 3];
    for (v, (&wi, &we)) in mix.verts.iter().zip(mix.intra.iter().zip(mix.extra)) {
        if wi == 0.0 && we == 0.0 {
            continue;
        }
        w_in += wi;
        w_ex += we;
        for i in 0..3 {
            for j in 0..3 {
                let vv = v[i] * v[j];
                om_in[i][j] += wi * vv;
                om_ex[i][j] += we * vv;
                for k in 0..3 {
                    for l in 0..3 {
                        let v4 = vv * v[k] * v[l];
                        o4_in[i][j][k][l] += wi * v4;
                        o4_ex[i][j][k][l] += we * v4;
                    }
                }
            }
        }
    }
    let iso_frac: f64 = mix.iso.iter().map(|&(f, _)| f).sum();
    debug_assert!(
        (w_in + w_ex + iso_frac - 1.0).abs() < 1e-2,
        "mixture weights must sum to 1 (got {})",
        w_in + w_ex + iso_frac
    );

    // D_app: sticks a·Om_in; zeppelins p·w·I + (a−p)·Om_ex; isotropics Σ f·d·I
    let iso_d: f64 = mix.iso.iter().map(|&(f, d)| f * d).sum();
    let mut d_app = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            d_app[i][j] = a * om_in[i][j] + (a - p) * om_ex[i][j];
            if i == j {
                d_app[i][j] += p * w_ex + iso_d;
            }
        }
    }

    // <D⊗D>: per _second_moment (dperp²·w·I⊗I + dperp·q·(I⊗Om + Om⊗I) + q²·O4), q = dpar−dperp;
    // intra has dperp = 0 → a²·O4_in alone. Isotropics add Σ f·d²·I⊗I.
    let iso_d2: f64 = mix.iso.iter().map(|&(f, d)| f * d * d).sum();
    let q_ex = a - p;
    let mut c: T4 = [[[[0.0; 3]; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    // intra (dperp = 0): a²·O4 alone; extra: dp²wI⊗I + dp·q(I⊗Om+Om⊗I) + q²O4
                    let mut dd = a * a * o4_in[i][j][k][l] + q_ex * q_ex * o4_ex[i][j][k][l];
                    if i == j {
                        dd += p * q_ex * om_ex[k][l];
                    }
                    if k == l {
                        dd += p * q_ex * om_ex[i][j];
                    }
                    if i == j && k == l {
                        dd += p * p * w_ex + iso_d2;
                    }
                    c[i][j][k][l] = dd - d_app[i][j] * d_app[k][l];
                }
            }
        }
    }
    (d_app, c)
}

// --------------------------------------------------------------------------------------------
// DTI / DKI scalars (dki_params_from_moments + dipy analytic solutions)
// --------------------------------------------------------------------------------------------

/// kt element ordering used by dipy (`_force_moments._KT_INDS` / `dki.Wcons`).
const KT_INDS: [(usize, usize, usize, usize); 15] = [
    (0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2),
    (0, 0, 0, 1), (0, 0, 0, 2), (0, 1, 1, 1),
    (1, 1, 1, 2), (0, 2, 2, 2), (1, 2, 2, 2),
    (0, 0, 1, 1), (0, 0, 2, 2), (1, 1, 2, 2),
    (0, 0, 1, 2), (0, 1, 1, 2), (0, 1, 2, 2),
];

/// `W = 3·sym(C)/MD²` — the kurtosis tensor (full, standard frame).
fn kurtosis_tensor(d_app: &M3, c: &T4) -> T4 {
    let md = (d_app[0][0] + d_app[1][1] + d_app[2][2]) / 3.0;
    let md2 = md.max(1e-10).powi(2);
    let mut w: T4 = [[[[0.0; 3]; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    // full symmetrization (C_ijkl + C_ikjl + C_iljk)/3 (_force_moments.py:72)
                    let sym = (c[i][j][k][l] + c[i][k][j][l] + c[i][l][j][k]) / 3.0;
                    w[i][j][k][l] = 3.0 * sym / md2;
                }
            }
        }
    }
    w
}

/// `Ŵ_ijkl`: `W` contracted into the eigenframe (`dki.Wrotate_element`, columns = eigenvectors).
fn w_rotated(w: &T4, evecs: &M3, i: usize, j: usize, k: usize, l: usize) -> f64 {
    let (ei, ej, ek, el) = (column(evecs, i), column(evecs, j), column(evecs, k), column(evecs, l));
    let mut s = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            for c_ in 0..3 {
                for d in 0..3 {
                    s += w[a][b][c_][d] * ei[a] * ej[b] * ek[c_] * el[d];
                }
            }
        }
    }
    s
}

pub struct DkiScalars {
    pub fa: f64,
    pub md: f64,
    pub rd: f64,
    pub ad: f64,
    pub ak: f64,
    pub rk: f64,
    pub mk: f64,
    pub mkt: f64,
    pub kfa: f64,
    /// eigenvalues (desc) — exposed for the fixture diff
    pub evals: [f64; 3],
    /// kt in dipy's 15-element order — exposed for the fixture diff
    pub kt: [f64; 15],
}

pub fn dki_scalars(d_app: &M3, c: &T4) -> DkiScalars {
    let w = kurtosis_tensor(d_app, c);
    let (evals, evecs) = eigh3(d_app);
    let (l1, l2, l3) = (evals[0], evals[1], evals[2]);
    let mut kt = [0.0; 15];
    for (n, &(i, j, k, l)) in KT_INDS.iter().enumerate() {
        kt[n] = w[i][j][k][l];
    }

    // DTI scalars (dti.py:62,171,201,234)
    let den = l1 * l1 + l2 * l2 + l3 * l3;
    let fa = if den == 0.0 {
        0.0
    } else {
        (0.5 * ((l1 - l2).powi(2) + (l2 - l3).powi(2) + (l3 - l1).powi(2)) / den).sqrt()
    };
    let md = (l1 + l2 + l3) / 3.0;
    let (ad, rd) = (l1, (l2 + l3) / 2.0);

    let clip = |v: f64, lo: f64, hi: f64| v.max(lo).min(hi);
    let (mut mk, mut rk, mut ak) = (0.0, 0.0, 0.0);
    if positive_evals(l1, l2, l3) {
        let w1111 = w_rotated(&w, &evecs, 0, 0, 0, 0);
        let w2222 = w_rotated(&w, &evecs, 1, 1, 1, 1);
        let w3333 = w_rotated(&w, &evecs, 2, 2, 2, 2);
        let w1122 = w_rotated(&w, &evecs, 0, 0, 1, 1);
        let w1133 = w_rotated(&w, &evecs, 0, 0, 2, 2);
        let w2233 = w_rotated(&w, &evecs, 1, 1, 2, 2);
        // MK (dki.py:819): F1/F2 assembly
        mk = f1m(l1, l2, l3) * w1111 + f1m(l2, l1, l3) * w2222 + f1m(l3, l2, l1) * w3333
            + f2m(l1, l2, l3) * w2233 + f2m(l2, l1, l3) * w1133 + f2m(l3, l2, l1) * w1122;
        // RK (dki.py:1089): G1/G2 assembly
        rk = g1m(l1, l2, l3) * w2222 + g1m(l1, l3, l2) * w3333 + g2m(l1, l2, l3) * w2233;
        // AK (dki.py:1236): Ŵ1111·MD²/λ1²
        ak = w1111 * md * md / (l1 * l1);
    }
    let mk = clip(mk, -3.0 / 7.0, 3.0);
    let rk = clip(rk, -3.0 / 7.0, 10.0);
    let ak = clip(ak, -3.0 / 7.0, 10.0);

    // MKT (dki.py:1488) + KFA (dki.py:1645)
    let mkt_raw = (kt[0] + kt[1] + kt[2] + 2.0 * (kt[9] + kt[10] + kt[11])) / 5.0;
    let mkt = clip(mkt_raw, -3.0 / 7.0, 10.0);
    let wbar = mkt_raw;
    let a_num = (kt[0] - wbar).powi(2) + (kt[1] - wbar).powi(2) + (kt[2] - wbar).powi(2)
        + 4.0 * (kt[3].powi(2) + kt[4].powi(2) + kt[5].powi(2) + kt[6].powi(2) + kt[7].powi(2)
            + kt[8].powi(2))
        + 6.0 * ((kt[9] - wbar / 3.0).powi(2) + (kt[10] - wbar / 3.0).powi(2)
            + (kt[11] - wbar / 3.0).powi(2))
        + 12.0 * (kt[12].powi(2) + kt[13].powi(2) + kt[14].powi(2));
    let b_den = kt[0].powi(2) + kt[1].powi(2) + kt[2].powi(2)
        + 4.0 * (kt[3].powi(2) + kt[4].powi(2) + kt[5].powi(2) + kt[6].powi(2) + kt[7].powi(2)
            + kt[8].powi(2))
        + 6.0 * (kt[9].powi(2) + kt[10].powi(2) + kt[11].powi(2))
        + 12.0 * (kt[12].powi(2) + kt[13].powi(2) + kt[14].powi(2));
    let kfa = if b_den > 0.0 && wbar > 1e-8 { (a_num / b_den).sqrt() } else { 0.0 };

    DkiScalars { fa, md, rd, ad, ak, rk, mk, mkt, kfa, evals, kt }
}

// --------------------------------------------------------------------------------------------
// QTI / DIVIDE invariants (qti.py:1202–:1401 via Frobenius contractions — the √2 Voigt inner
// products reduce to full rank-4 contractions with E_bulk/E_iso/E_shear)
// --------------------------------------------------------------------------------------------

/// `⟨A, E_bulk⟩ = Σ_ik A_iikk / 9`
fn bulk_contract(a: &T4) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            s += a[i][i][k][k];
        }
    }
    s / 9.0
}

/// `⟨A, E_iso⟩ = Σ_ij A_ijij / 3` (minor-symmetric A)
fn iso_contract(a: &T4) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += a[i][j][i][j];
        }
    }
    s / 3.0
}

pub struct QtiScalars {
    pub micro_fa: f64,
    pub coherence: f64,
    pub k_bulk: f64,
    pub k_shear: f64,
}

pub fn qti_scalars(d_app: &M3, c: &T4) -> QtiScalars {
    let mut dxd: T4 = [[[[0.0; 3]; 3]; 3]; 3];
    let mut dd: T4 = [[[[0.0; 3]; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    dxd[i][j][k][l] = d_app[i][j] * d_app[k][l];
                    dd[i][j][k][l] = c[i][j][k][l] + dxd[i][j][k][l];
                }
            }
        }
    }
    let shear = |a: &T4| iso_contract(a) - bulk_contract(a);
    let c_mu = 1.5 * shear(&dd) / iso_contract(&dd);
    let c_m = 1.5 * shear(&dxd) / iso_contract(&dxd);
    let k_bulk = 3.0 * bulk_contract(c) / bulk_contract(&dxd);
    let k_shear = 6.0 / 5.0 * shear(c) / bulk_contract(&dxd);
    // guards from _force_moments.py:627–:634
    let micro_fa = c_mu.max(0.0).sqrt();
    let coherence = if c_mu > 1e-12 { (c_m / c_mu).clamp(0.0, 1.0) } else { 0.0 };
    QtiScalars { micro_fa, coherence, k_bulk, k_shear }
}

// --------------------------------------------------------------------------------------------
// MAP-MRI closed-form indices (_force_moments.py:311)
// --------------------------------------------------------------------------------------------

pub struct MapmriScalars {
    pub rtop: f64,
    pub rtap: f64,
    pub rtpp: f64,
    pub msd: f64,
    pub qiv: f64,
}

/// `axis` is the principal eigenvector of the true-stick `D_app` (computed by the caller —
/// shared with the DKI eigendecomposition).
pub fn mapmri_scalars(
    mix: &VoxelMixture,
    d_app: &M3,
    axis: [f64; 3],
    tau: f64,
    d_perp_floor: f64,
) -> MapmriScalars {
    let (a, p, fl) = (mix.d_par, mix.d_perp, d_perp_floor);
    let w_in: f64 = mix.intra.iter().sum();
    let w_ex: f64 = mix.extra.iter().sum();
    let four_pi_tau = 4.0 * std::f64::consts::PI * tau;

    let msd = 2.0 * tau * (d_app[0][0] + d_app[1][1] + d_app[2][2]);

    // rotation-invariant reductions (floored intra)
    let det_in = a * fl * fl;
    let det_ex = a * p * p;
    let iso_rtop: f64 = mix.iso.iter().map(|&(f, d)| f * d.powf(-1.5)).sum();
    let rtop = four_pi_tau.powf(-1.5)
        * (w_in * det_in.powf(-0.5) + w_ex * det_ex.powf(-0.5) + iso_rtop);

    // QIV via the propagator Laplacian at 0 (_force_moments.py:367)
    let lap_iso = |w: f64, d: f64| {
        let s = 2.0 * tau * d;
        let n0 = (2.0 * std::f64::consts::PI * s).powf(-1.5);
        w * n0 * (3.0 / s)
    };
    let lap_aniso = |w: f64, ax: f64, pr: f64| {
        let (sa, sp) = (2.0 * tau * ax, 2.0 * tau * pr);
        let n0 = (2.0 * std::f64::consts::PI).powf(-1.5) * (sa * sp * sp).powf(-0.5);
        w * n0 * (1.0 / sa + 2.0 / sp)
    };
    let mut lap = lap_aniso(w_in, a, fl) + lap_aniso(w_ex, a, p);
    for &(f, d) in mix.iso {
        lap += lap_iso(f, d);
    }
    let qiv = 4.0 * std::f64::consts::PI.powi(2) / lap; // = −4π²/(−lap)

    // axis-dependent reductions (floored intra)
    let (mut rtpp_fib, mut rtap_fib) = (0.0, 0.0);
    for (v, (&wi, &we)) in mix.verts.iter().zip(mix.intra.iter().zip(mix.extra)) {
        if wi == 0.0 && we == 0.0 {
            continue;
        }
        let cos2 = (v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2]).powi(2);
        let sin2 = 1.0 - cos2;
        let dpar_in = a * cos2 + fl * sin2;
        let dpar_ex = a * cos2 + p * sin2;
        rtpp_fib += wi / dpar_in.sqrt() + we / dpar_ex.sqrt();
        let detperp_in = fl * (a * sin2 + fl * cos2);
        let detperp_ex = p * (a * sin2 + p * cos2);
        rtap_fib += wi / detperp_in.sqrt() + we / detperp_ex.sqrt();
    }
    let iso_rtpp: f64 = mix.iso.iter().map(|&(f, d)| f / d.sqrt()).sum();
    let iso_rtap: f64 = mix.iso.iter().map(|&(f, d)| f / d).sum();
    let rtpp = four_pi_tau.powf(-0.5) * (rtpp_fib + iso_rtpp);
    let rtap = four_pi_tau.recip() * (rtap_fib + iso_rtap);
    MapmriScalars { rtop, rtap, rtpp, msd, qiv }
}

// --------------------------------------------------------------------------------------------
// closed-form NG / PA (_force_moments.py:454,465,529)
// --------------------------------------------------------------------------------------------

pub struct NgPaScalars {
    pub ng: f64,
    pub ngpar: f64,
    pub ngperp: f64,
    pub pa: f64,
}

/// Real roots of `s³ + a s² + b s + c = 0` (depressed-cubic, trigonometric for the 3-real case),
/// each polished with one Newton step.
fn cubic_real_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
    let disc = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    let mut roots = Vec::with_capacity(3);
    if disc > 1e-300 {
        let sq = disc.sqrt();
        let t = (-q / 2.0 + sq).cbrt() + (-q / 2.0 - sq).cbrt();
        roots.push(t - a / 3.0);
    } else {
        let m = (-p / 3.0).sqrt().max(1e-300);
        let cosphi = ((3.0 * q) / (2.0 * p * m)).clamp(-1.0, 1.0);
        let phi = cosphi.acos();
        for k in 0..3 {
            let t = 2.0 * m * ((phi + 2.0 * std::f64::consts::PI * k as f64) / 3.0).cos();
            roots.push(t - a / 3.0);
        }
    }
    for r in roots.iter_mut() {
        let f = ((*r + a) * *r + b) * *r + c;
        let df = (3.0 * *r + 2.0 * a) * *r + b;
        if df.abs() > 1e-300 {
            *r -= f / df;
        }
    }
    roots
}

/// Ozarslan isotropic scale (_force_moments.py:454): largest positive real root of
/// `−3s³ − (X+Y+Z)s² + (XY+XZ+YZ)s + 3XYZ = 0`, falling back to `mean(evals)`.
fn isotropic_scale(evals: [f64; 3]) -> f64 {
    let (x, y, z) = (evals[0], evals[1], evals[2]);
    // divide by −3: s³ + (Σ/3)s² − (Σpairs/3)s − XYZ = 0
    let roots = cubic_real_roots(
        (x + y + z) / 3.0,
        -(x * y + x * z + y * z) / 3.0,
        -(x * y * z),
    );
    match roots.into_iter().filter(|r| *r > 0.0).fold(f64::NAN, f64::max) {
        best if best.is_nan() => (x + y + z) / 3.0,
        best => best,
    }
}

/// Generic mixture-vs-reference Gaussian angle over 3×3 compartments
/// (`gaussian_mixture_ng_pa._ng`).
fn ng_angle3(w: &[f64], t: &[M3], reference: &M3) -> f64 {
    let n = w.len();
    let mut ovpp = 0.0;
    for i in 0..n {
        for j in 0..n {
            ovpp += w[i] * w[j] * det3(&madd(&t[i], &t[j])).powf(-0.5);
        }
    }
    if ovpp <= 0.0 {
        return 0.0;
    }
    let ovp0: f64 =
        w.iter().zip(t).map(|(&wi, ti)| wi * det3(&madd(ti, reference)).powf(-0.5)).sum();
    let ov00 = det3(&mscale(reference, 2.0)).powf(-0.5);
    (1.0 - ovp0 * ovp0 / (ovpp * ov00)).max(0.0).sqrt()
}

/// 2×2 version for the perpendicular marginal.
fn ng_angle2(w: &[f64], t: &[[f64; 3]], reference: [f64; 3]) -> f64 {
    // 2×2 symmetric matrices as [xx, yy, xy]
    let det2 = |m: [f64; 3]| m[0] * m[1] - m[2] * m[2];
    let n = w.len();
    let mut ovpp = 0.0;
    for i in 0..n {
        for j in 0..n {
            let s = [t[i][0] + t[j][0], t[i][1] + t[j][1], t[i][2] + t[j][2]];
            ovpp += w[i] * w[j] * det2(s).powf(-0.5);
        }
    }
    if ovpp <= 0.0 {
        return 0.0;
    }
    let ovp0: f64 = w
        .iter()
        .zip(t)
        .map(|(&wi, ti)| {
            wi * det2([ti[0] + reference[0], ti[1] + reference[1], ti[2] + reference[2]])
                .powf(-0.5)
        })
        .sum();
    let ov00 = det2([2.0 * reference[0], 2.0 * reference[1], 2.0 * reference[2]]).powf(-0.5);
    (1.0 - ovp0 * ovp0 / (ovpp * ov00)).max(0.0).sqrt()
}

pub fn ng_pa_scalars(mix: &VoxelMixture, floor: f64, top_k: usize) -> NgPaScalars {
    // top-k selection on intra+extra (ng_pa_from_odfs:552–:555)
    let nv = mix.verts.len();
    let mut order: Vec<usize> = (0..nv).collect();
    order.sort_by(|&i, &j| {
        let (a, b) = (mix.intra[i] + mix.extra[i], mix.intra[j] + mix.extra[j]);
        b.partial_cmp(&a).unwrap()
    });
    order.truncate(top_k.min(nv));
    order.retain(|&i| mix.intra[i] + mix.extra[i] > 0.0);

    // compartment tensors with eigenvalues clipped at `floor` (gaussian_mixture_ng_pa:489–:491;
    // ours are axisymmetric so the clip is analytic)
    let (a, p) = (mix.d_par.max(floor), mix.d_perp.max(floor));
    let mut w: Vec<f64> = Vec::with_capacity(2 * order.len() + mix.iso.len());
    let mut t: Vec<M3> = Vec::with_capacity(w.capacity());
    let vv_tensor = |v: &[f64; 3], ax: f64, rad: f64| -> M3 {
        let mut m = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = (ax - rad) * v[i] * v[j];
                if i == j {
                    m[i][j] += rad;
                }
            }
        }
        m
    };
    for &vi in &order {
        w.push(mix.intra[vi]);
        t.push(vv_tensor(&mix.verts[vi], a, floor)); // stick: evals (a, floor, floor)
    }
    for &vi in &order {
        w.push(mix.extra[vi]);
        t.push(vv_tensor(&mix.verts[vi], a, p)); // zeppelin: evals (a, p, p)
    }
    for &(f, d) in mix.iso {
        w.push(f);
        t.push(mscale(&ID3, d.max(floor)));
    }
    // mask w > 1e-6, then normalize (gaussian_mixture_ng_pa:487–:488, ng_pa_from_odfs:564)
    let keep: Vec<usize> = (0..w.len()).filter(|&i| w[i] > 1e-6).collect();
    let wsum: f64 = keep.iter().map(|&i| w[i]).sum();
    let w: Vec<f64> = keep.iter().map(|&i| w[i] / wsum).collect();
    let t: Vec<M3> = keep.iter().map(|&i| t[i]).collect();

    // mixture mean of the clipped tensors (NOT the moments D_app)
    let mut d = [[0.0; 3]; 3];
    for (wi, ti) in w.iter().zip(&t) {
        for i in 0..3 {
            for j in 0..3 {
                d[i][j] += wi * ti[i][j];
            }
        }
    }

    let ng = ng_angle3(&w, &t, &d);

    let (evals, evecs) = eigh3(&d);
    let axis = column(&evecs, 0);

    // 1D axial marginal (gaussian_mixture_ng_pa:504–:509)
    let dpar: Vec<f64> = t.iter().map(|ti| quad_form(ti, axis)).collect();
    let da = quad_form(&d, axis);
    let mut ovpp = 0.0;
    for i in 0..w.len() {
        for j in 0..w.len() {
            ovpp += w[i] * w[j] * (dpar[i] + dpar[j]).powf(-0.5);
        }
    }
    let ovp0: f64 = w.iter().zip(&dpar).map(|(&wi, &di)| wi * (di + da).powf(-0.5)).sum();
    let ngpar = if ovpp > 0.0 {
        (1.0 - ovp0 * ovp0 / (ovpp * (2.0 * da).powf(-0.5))).max(0.0).sqrt()
    } else {
        0.0
    };

    // 2D perpendicular marginal (basis-invariant)
    let (b1, b2) = perp_basis(axis);
    let project = |m: &M3| -> [f64; 3] {
        let mv = |v: [f64; 3]| crate::mat::matvec(m, v);
        [
            crate::mat::dot(b1, mv(b1)),
            crate::mat::dot(b2, mv(b2)),
            crate::mat::dot(b1, mv(b2)),
        ]
    };
    let tp: Vec<[f64; 3]> = t.iter().map(project).collect();
    let ngperp = ng_angle2(&w, &tp, project(&d));

    // PA (gaussian_mixture_ng_pa:517–:522)
    let s_iso = isotropic_scale(evals);
    let si = mscale(&ID3, s_iso);
    let cos2 = det3(&madd(&d, &si)).recip()
        / (det3(&mscale(&d, 2.0)).powf(-0.5) * det3(&mscale(&si, 2.0)).powf(-0.5));
    let pa = (1.0 - cos2).max(0.0).sqrt();

    NgPaScalars { ng, ngpar, ngperp, pa }
}

// --------------------------------------------------------------------------------------------
// GQI-style ODF statistics (_force_moments.py:901)
// --------------------------------------------------------------------------------------------

pub struct GqiScalars {
    pub gfa: f64,
    pub qa: f64,
}

pub fn gqi_scalars(mix: &VoxelMixture) -> GqiScalars {
    let n = mix.verts.len();
    let wm: f64 = mix.intra.iter().sum::<f64>() + mix.extra.iter().sum::<f64>();
    let tot: Vec<f64> = mix.intra.iter().zip(mix.extra).map(|(a, b)| a + b).collect();
    let ssum: f64 = tot.iter().sum();
    let baseline = (1.0 - wm) / n as f64;
    let odf: Vec<f64> = tot
        .iter()
        .map(|&t| if ssum > 0.0 { wm * t / ssum + baseline } else { baseline })
        .collect();
    // dipy gfa (odf.py:31): sqrt(n·Σ(x−x̄)² / ((n−1)·Σx²))
    let mean = odf.iter().sum::<f64>() / n as f64;
    let num: f64 = odf.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() * n as f64;
    let den: f64 = odf.iter().map(|&x| x * x).sum::<f64>() * (n as f64 - 1.0);
    let gfa = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in &odf {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    GqiScalars { gfa, qa: hi - lo }
}

// --------------------------------------------------------------------------------------------
// NODDI-style parameters — not closed forms but direct properties of the mixture
// --------------------------------------------------------------------------------------------

/// Watson second moment `t(κ) = E[(µ·v)²] = ∫₀¹ u² e^{κ(u²−1)} du / ∫₀¹ e^{κ(u²−1)} du`
/// by Simpson quadrature (the e^{−κ} normalization keeps large κ finite).
fn watson_second_moment(kappa: f64) -> f64 {
    let n = 512usize; // even
    let h = 1.0 / n as f64;
    let f = |u: f64| (kappa * (u * u - 1.0)).exp();
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..=n {
        let u = i as f64 * h;
        let w = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        let fu = f(u);
        num += w * u * u * fu;
        den += w * fu;
    }
    num / den
}

/// Monotone κ → t(κ) table on a geometric κ grid, built once.
fn watson_table() -> &'static Vec<(f64, f64)> {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let (lo, hi, n) = (1e-3_f64, 1e3_f64, 240usize);
        (0..=n)
            .map(|i| {
                let k = lo * (hi / lo).powf(i as f64 / n as f64);
                (k, watson_second_moment(k))
            })
            .collect()
    })
}

/// NODDI orientation dispersion index of an orientation-weight set: moment-match a Watson
/// concentration κ to the principal eigenvalue `τ₁` of the weight-normalized orientation tensor
/// `Σ p̂_v vv^T`, then `ODI = (2/π)·atan(1/κ)`. `τ₁ → 1/3` (isotropic) gives ODI → 1;
/// `τ₁ → 1` (a single direction) gives ODI → 0 (floored by the κ table cap at 1e3).
pub fn watson_odi(verts: &[[f64; 3]], weights: &[f64]) -> f64 {
    let tot: f64 = weights.iter().sum();
    if tot <= 0.0 {
        return 0.0; // no orientation content — "not applicable" convention, see field_scalars
    }
    let mut om = [[0.0; 3]; 3];
    for (v, &w) in verts.iter().zip(weights) {
        for i in 0..3 {
            for j in 0..3 {
                om[i][j] += w / tot * v[i] * v[j];
            }
        }
    }
    let (evals, _) = eigh3(&om);
    let tau1 = evals[0];
    let table = watson_table();
    if tau1 <= table[0].1 {
        return 1.0; // at/below the isotropic moment → maximal dispersion
    }
    if tau1 >= table[table.len() - 1].1 {
        let kappa = table[table.len() - 1].0;
        return 2.0 / std::f64::consts::PI * kappa.recip().atan();
    }
    // binary search the monotone (κ, t) table, interpolate κ in log-space
    let idx = table.partition_point(|&(_, t)| t < tau1);
    let ((k0, t0), (k1, t1)) = (table[idx - 1], table[idx]);
    let frac = (tau1 - t0) / (t1 - t0);
    let kappa = (k0.ln() + frac * (k1.ln() - k0.ln())).exp();
    2.0 / std::f64::consts::PI * kappa.recip().atan()
}

// --------------------------------------------------------------------------------------------
// assembly
// --------------------------------------------------------------------------------------------

/// Configuration for the closed forms. Defaults match dipy's.
#[derive(Debug, Clone, Copy)]
pub struct ScalarConfig {
    /// diffusion time `Δ − δ/3` (s); dipy's no-timing default `1/(4π²)` gives normalized units
    pub tau: f64,
    /// intra-axonal radial floor for MAP-MRI (singular sticks)
    pub d_perp_floor: f64,
    /// eigenvalue floor for NG/PA compartments
    pub ng_floor: f64,
    /// strongest-orientation pruning for NG/PA (FORCE default 28)
    pub ng_top_k: usize,
}

impl Default for ScalarConfig {
    fn default() -> Self {
        ScalarConfig {
            tau: 1.0 / (4.0 * std::f64::consts::PI * std::f64::consts::PI),
            d_perp_floor: 0.12e-3,
            ng_floor: 0.12e-3,
            ng_top_k: 28,
        }
    }
}

/// All 24 closed-form scalars of one voxel's mixture.
pub struct Scalars {
    pub dki: DkiScalars,
    pub qti: QtiScalars,
    pub mapmri: MapmriScalars,
    pub ngpa: NgPaScalars,
    pub gqi: GqiScalars,
}

pub fn voxel_scalars(mix: &VoxelMixture, cfg: &ScalarConfig) -> Scalars {
    let (d_app, c) = moments(mix);
    let dki = dki_scalars(&d_app, &c);
    let qti = qti_scalars(&d_app, &c);
    let (_, evecs) = eigh3(&d_app);
    let axis = column(&evecs, 0);
    let mapmri = mapmri_scalars(mix, &d_app, axis, cfg.tau, cfg.d_perp_floor);
    let ngpa = ng_pa_scalars(mix, cfg.ng_floor, cfg.ng_top_k);
    let gqi = gqi_scalars(mix);
    Scalars { dki, qti, mapmri, ngpa, gqi }
}

/// Map names, in the order [`field_scalars`] returns them. The first 24 are the FORCE closed
/// forms ([`voxel_scalars`]); the last three are NODDI-style parameters read directly off the
/// mixture: `icvf = wf·intra_frac / (wf + gf)` (intra fraction of the non-CSF tissue — the GM
/// ball counts as non-neurite tissue), `odi` via Watson moment-matching of the orientation
/// histogram ([`watson_odi`]), `isovf = csf fraction`. Voxels with no fibre content (pure
/// GM/CSF, and WM fallback voxels, which the model represents with **zero** neurites) get
/// `icvf = 0` and, by the usual undefined→0 map convention, `odi = 0`.
pub const SCALAR_NAMES: [&str; 27] = [
    "fa", "md", "rd", "ad", "ak", "rk", "mk", "mkt", "kfa",
    "micro_fa", "coherence", "k_bulk", "k_shear",
    "rtop", "rtap", "rtpp", "msd", "qiv",
    "ng", "ngpar", "ngperp", "pa",
    "gfa", "qa",
    "icvf", "odi", "isovf",
];

impl Scalars {
    /// Values in the order of the first 24 [`SCALAR_NAMES`] entries (the FORCE closed forms;
    /// the NODDI-style trailing three are field-level, see [`field_scalars`]).
    pub fn values(&self) -> [f64; 24] {
        [
            self.dki.fa, self.dki.md, self.dki.rd, self.dki.ad, self.dki.ak, self.dki.rk,
            self.dki.mk, self.dki.mkt, self.dki.kfa,
            self.qti.micro_fa, self.qti.coherence, self.qti.k_bulk, self.qti.k_shear,
            self.mapmri.rtop, self.mapmri.rtap, self.mapmri.rtpp, self.mapmri.msd,
            self.mapmri.qiv,
            self.ngpa.ng, self.ngpa.ngpar, self.ngpa.ngperp, self.ngpa.pa,
            self.gqi.gfa, self.gqi.qa,
        ]
    }
}

/// Ground-truth scalar maps for every masked voxel of a [`MixtureField`] — one `nvox` map per
/// [`SCALAR_NAMES`] entry, 0 outside the mask. WM voxels with no streamline support
/// ([`MixtureField::fallback`]) are evaluated as an isotropic Gaussian at
/// [`MixtureField::md_fallback`], mirroring the signal fallback exactly.
pub fn field_scalars(field: &MixtureField, cfg: &ScalarConfig) -> Vec<Vec<f32>> {
    let nvox = field.nvox();
    let nvert = field.nvert();
    let verts = &field.sphere.verts;
    let (intra_frac, extra_frac) = (field.params.intra_frac, field.params.extra_frac);
    // myelin map: per-voxel lerp between the field's params (m = 0) and the adult (myelinated)
    // endpoint — mirrors signal_from_mixture so truth and signal remain one object.
    let adult = crate::compartments::CompartmentParams::adult();

    // per-voxel closure so the loop can go parallel: fully independent work, tiny scratch
    let compute = |vox: usize| -> Option<[f64; 27]> {
        if !field.is_masked(vox) {
            return None;
        }
        let (wf, gf, cf) = (field.wm[vox] as f64, field.gm[vox] as f64, field.csf[vox] as f64);
        let m = field.myelin.as_ref().map(|mm| mm[vox] as f64).unwrap_or(0.0);
        let li = |a: f64, b: f64| a + (b - a) * m;
        let (fi, fe) = (li(intra_frac, adult.intra_frac), li(extra_frac, adult.extra_frac));
        let row = field.odf_row(vox);
        let tot: f64 = row.iter().map(|&v| v as f64).sum();
        let mut intra = vec![0.0f64; nvert];
        let mut extra = vec![0.0f64; nvert];
        // GM = free ball + restricted (soma) ball — mirrors signal_from_mixture's two-ball GM
        let fr = field.params.gm_restricted_frac;
        let mut iso = vec![
            (gf * (1.0 - fr), field.params.d_gm),
            (gf * fr, field.params.d_soma),
            (cf, field.params.d_csf),
        ];
        if field.fallback[vox] == 1 || tot <= 1e-12 {
            // no streamline support: the whole WM fraction is the hindered isotropic Gaussian
            let md_m = (li(field.params.d_extra.0, adult.d_extra.0)
                + li(field.params.d_extra.1, adult.d_extra.1)
                + li(field.params.d_extra.2, adult.d_extra.2))
                / 3.0;
            iso.push((wf, md_m));
        } else {
            for (i, &v) in row.iter().enumerate() {
                let frac = v as f64 / tot;
                intra[i] = wf * fi * frac;
                extra[i] = wf * fe * frac;
            }
        }
        let mix = VoxelMixture {
            verts,
            intra: &intra,
            extra: &extra,
            d_par: li(field.params.d_intra, adult.d_intra),
            d_perp: li(field.params.d_extra.1, adult.d_extra.1),
            iso: &iso,
        };
        let force = voxel_scalars(&mix, cfg).values();
        // NODDI-style parameters: direct mixture properties (SCALAR_NAMES doc has conventions)
        let has_fibre = intra.iter().sum::<f64>() > 0.0;
        let tissue = wf + gf;
        let icvf = if has_fibre && tissue > 0.0 { wf * fi / tissue } else { 0.0 };
        let odi = if has_fibre { watson_odi(verts, &intra) } else { 0.0 };
        let mut out = [0.0f64; 27];
        out[..24].copy_from_slice(&force);
        out[24] = icvf;
        out[25] = odi;
        out[26] = cf;
        Some(out)
    };

    #[cfg(feature = "par")]
    let rows: Vec<Option<[f64; 27]>> = {
        use rayon::prelude::*;
        (0..nvox).into_par_iter().map(compute).collect()
    };
    #[cfg(not(feature = "par"))]
    let rows: Vec<Option<[f64; 27]>> = (0..nvox).map(compute).collect();

    let mut out = vec![vec![0.0f32; nvox]; SCALAR_NAMES.len()];
    for (vox, row) in rows.iter().enumerate() {
        if let Some(vals) = row {
            for (m, &v) in out.iter_mut().zip(vals.iter()) {
                m[vox] = v as f32;
            }
        }
    }
    out
}

// --------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rel(a: f64, b: f64, rtol: f64, what: &str) {
        let scale = a.abs().max(b.abs()).max(1e-30);
        assert!(
            (a - b).abs() <= rtol * scale,
            "{what}: {a} vs {b} (rel {})",
            (a - b).abs() / scale
        );
    }

    #[test]
    fn eigh_recovers_known_spectrum() {
        // diag(3,1,2) rotated by a known rotation
        let r = crate::mat::rotation_zyx_deg(20.0, -35.0, 55.0);
        let d = [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let rd = crate::mat::matmul(&r, &crate::mat::matmul(&d, &crate::mat::transpose(&r)));
        let (evals, evecs) = eigh3(&rd);
        assert_rel(evals[0], 3.0, 1e-12, "l1");
        assert_rel(evals[1], 2.0, 1e-12, "l2");
        assert_rel(evals[2], 1.0, 1e-12, "l3");
        // eigenvector property: A e = λ e
        for k in 0..3 {
            let e = column(&evecs, k);
            let ae = crate::mat::matvec(&rd, e);
            for i in 0..3 {
                assert!((ae[i] - evals[k] * e[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn carlson_degenerate_values() {
        // R_F(x,x,x) = x^{-1/2}; R_D(x,x,x) = x^{-3/2}
        for x in [0.5, 1.0, 2.5] {
            assert_rel(carlson_rf(x, x, x), x.powf(-0.5), 1e-10, "rf");
            assert_rel(carlson_rd(x, x, x), x.powf(-1.5), 1e-10, "rd");
        }
    }

    #[test]
    fn single_gaussian_compartment_has_zero_kurtosis_and_ufa_equals_fa() {
        // one zeppelin along z, nothing else → C = 0
        let verts = [[0.0, 0.0, 1.0]];
        let (intra, extra) = ([0.0], [1.0]);
        let mix = VoxelMixture {
            verts: &verts, intra: &intra, extra: &extra,
            d_par: 1.5e-3, d_perp: 0.6e-3, iso: &[],
        };
        let s = voxel_scalars(&mix, &ScalarConfig::default());
        assert!(s.dki.mk.abs() < 1e-9, "single Gaussian MK {}", s.dki.mk);
        assert!(s.dki.ak.abs() < 1e-9 && s.dki.rk.abs() < 1e-9);
        assert!(s.dki.mkt.abs() < 1e-9 && s.dki.kfa.abs() < 1e-9);
        // Westin µFA equals conventional FA for a single compartment; coherence = 1
        assert_rel(s.qti.micro_fa, s.dki.fa, 1e-9, "ufa=fa");
        assert_rel(s.qti.coherence, 1.0, 1e-9, "coherence");
        // analytic FA of (a, p, p)
        let (a, p): (f64, f64) = (1.5e-3, 0.6e-3);
        let fa = (0.5 * (2.0 * (a - p).powi(2)) / (a * a + 2.0 * p * p)).sqrt();
        assert_rel(s.dki.fa, fa, 1e-12, "fa analytic");
        // zeppelin evals exceed the NG floor, so the mixture is a single Gaussian → NG = 0
        assert!(s.ngpa.ng.abs() < 1e-9 && s.ngpa.ngpar.abs() < 1e-9);
        assert!(s.ngpa.pa > 0.1, "anisotropic single tensor should have PA > 0");
    }

    #[test]
    fn isotropic_ball_matches_analytic_rtop_and_zero_anisotropy() {
        let verts = [[0.0, 0.0, 1.0]];
        let zero = [0.0];
        let d = 3.0e-3;
        let cfg = ScalarConfig::default();
        let mix = VoxelMixture {
            verts: &verts, intra: &zero, extra: &zero,
            d_par: 1.5e-3, d_perp: 0.6e-3, iso: &[(1.0, d)],
        };
        let s = voxel_scalars(&mix, &cfg);
        assert!(s.dki.fa.abs() < 1e-12 && s.qti.micro_fa.abs() < 1e-9);
        assert!(s.dki.mk.abs() < 1e-9 && s.ngpa.ng.abs() < 1e-9);
        assert!(s.ngpa.pa.abs() < 1e-9, "isotropic PA {}", s.ngpa.pa);
        let four_pi_tau = 4.0 * std::f64::consts::PI * cfg.tau;
        assert_rel(s.mapmri.rtop, four_pi_tau.powf(-1.5) * d.powf(-1.5), 1e-12, "rtop");
        assert_rel(s.mapmri.msd, 2.0 * cfg.tau * 3.0 * d, 1e-12, "msd");
        assert_rel(s.mapmri.rtpp, four_pi_tau.powf(-0.5) / d.sqrt(), 1e-12, "rtpp");
        assert_rel(s.mapmri.rtap, four_pi_tau.recip() / d, 1e-12, "rtap");
    }

    #[test]
    fn two_ball_mixture_kurtosis_is_bulk_only() {
        // 50/50 free water + parenchyma-like ball: isotropic mixture with size variance
        let verts = [[0.0, 0.0, 1.0]];
        let zero = [0.0];
        let (d1, d2) = (3.0e-3, 0.8e-3);
        let mix = VoxelMixture {
            verts: &verts, intra: &zero, extra: &zero,
            d_par: 1.5e-3, d_perp: 0.6e-3, iso: &[(0.5, d1), (0.5, d2)],
        };
        let s = voxel_scalars(&mix, &ScalarConfig::default());
        let dbar = 0.5 * (d1 + d2);
        let var = 0.5 * (d1 - dbar).powi(2) + 0.5 * (d2 - dbar).powi(2);
        let k_iso = 3.0 * var / (dbar * dbar);
        assert_rel(s.qti.k_bulk, k_iso, 1e-9, "k_bulk");
        assert!(s.qti.k_shear.abs() < 1e-9, "isotropic mixture has no shear kurtosis");
        assert_rel(s.dki.mk, k_iso, 1e-9, "Tabesh MK equals bulk kurtosis for balls");
        assert!(s.qti.micro_fa < 1e-6);
        assert!(s.ngpa.ng > 0.05, "size heterogeneity → NG > 0, got {}", s.ngpa.ng);
    }

    #[test]
    fn crossing_drops_fa_but_not_micro_fa() {
        // two orthogonal stick+zeppelin populations, equal weight
        let verts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let intra = [0.275, 0.275];
        let extra = [0.225, 0.225];
        let mix = VoxelMixture {
            verts: &verts, intra: &intra, extra: &extra,
            d_par: 1.5e-3, d_perp: 0.4e-3, iso: &[],
        };
        let s = voxel_scalars(&mix, &ScalarConfig::default());
        let single_verts = [[1.0, 0.0, 0.0]];
        let si = [0.55];
        let se = [0.45];
        let single = voxel_scalars(
            &VoxelMixture {
                verts: &single_verts, intra: &si, extra: &se,
                d_par: 1.5e-3, d_perp: 0.4e-3, iso: &[],
            },
            &ScalarConfig::default(),
        );
        // orthogonal crossing → oblate mean tensor: FA drops substantially (measured ~0.63×)
        // but not to zero — the perpendicular d_perp keeps the plane anisotropic.
        assert!(s.dki.fa < 0.7 * single.dki.fa, "crossing FA {} vs single {}", s.dki.fa,
            single.dki.fa);
        assert_rel(s.qti.micro_fa, single.qti.micro_fa, 1e-9,
            "µFA is dispersion-invariant across the crossing");
        assert!(s.qti.coherence < 0.6, "crossing coherence {}", s.qti.coherence);
        assert!(s.dki.mk > 0.1, "crossing of Gaussian compartments has MK > 0");
    }

    #[test]
    fn watson_odi_limits_and_kernel_roundtrip() {
        let s = crate::sphere::HemiSphere::icosphere(3);
        // single direction → minimal dispersion; uniform → maximal
        let one_hot: Vec<f64> =
            (0..s.len()).map(|i| if i == 7 { 1.0 } else { 0.0 }).collect();
        assert!(watson_odi(&s.verts, &one_hot) < 0.02);
        let uniform = vec![1.0; s.len()];
        assert!(watson_odi(&s.verts, &uniform) > 0.95);
        // weights drawn from a Watson kernel must round-trip its κ (up to sphere discretization)
        for kappa in [5.0_f64, 20.0, 80.0] {
            let mu = s.verts[42];
            let w: Vec<f64> = s
                .verts
                .iter()
                .map(|v| {
                    let c = v[0] * mu[0] + v[1] * mu[1] + v[2] * mu[2];
                    (kappa * (c * c - 1.0)).exp()
                })
                .collect();
            let odi = watson_odi(&s.verts, &w);
            let expect = 2.0 / std::f64::consts::PI * kappa.recip().atan();
            assert!(
                (odi - expect).abs() < 0.012 + 0.05 * expect,
                "κ={kappa}: odi {odi} vs analytic {expect}"
            );
        }
    }

    #[test]
    fn field_noddi_maps_read_off_the_mixture() {
        use crate::mixture::MixtureField;
        let sphere = crate::sphere::HemiSphere::icosphere(2);
        let nvert = sphere.len();
        let mut odf = vec![0.0f32; nvert];
        odf[sphere.nearest([1.0, 0.0, 0.0])] = 2.5; // single-orientation WM voxel
        let field = MixtureField {
            dims: [1, 1, 1],
            sphere,
            odf,
            wm: vec![0.8],
            gm: vec![0.15],
            csf: vec![0.05],
            fallback: vec![0],
            params: crate::compartments::CompartmentParams::default(),
            myelin: None,
        };
        let maps = field_scalars(&field, &ScalarConfig::default());
        let at = |name: &str| {
            maps[SCALAR_NAMES.iter().position(|n| *n == name).unwrap()][0] as f64
        };
        // icvf = wf·intra_frac/(wf+gf); isovf = cf; single orientation → odi ≈ 0
        assert!((at("icvf") - 0.8 * 0.55 / 0.95).abs() < 1e-6, "icvf {}", at("icvf"));
        assert!((at("isovf") - 0.05).abs() < 1e-6);
        assert!(at("odi") < 0.02, "odi {}", at("odi"));
    }

    // ---- the oracle diff -------------------------------------------------------------------

    struct FixtureCase {
        verts: Vec<[f64; 3]>,
        intra: Vec<f64>,
        extra: Vec<f64>,
        iso: [(f64, f64); 3],
        d_par: f64,
        d_perp: f64,
        tau: f64,
        d_app: [f64; 9],
        c: [f64; 81],
        evals: [f64; 3],
        kt: [f64; 15],
        scalars: [f64; 9],
        qti: [f64; 4],
        mapmri: [f64; 5],
        ngpa: [f64; 4],
        gqi: [f64; 2],
    }

    fn parse_fixtures() -> Vec<FixtureCase> {
        let text = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/force_moments.txt"),
        )
        .expect("run tools/gen_force_fixtures.py first");
        let mut cases = Vec::new();
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        while let Some(line) = lines.next() {
            if !line.starts_with("case") {
                continue;
            }
            let mut grab = |tag: &str| -> Vec<f64> {
                let l = lines.next().unwrap_or_else(|| panic!("missing {tag}"));
                let mut toks = l.split_whitespace();
                assert_eq!(toks.next(), Some(tag), "expected {tag} in {l}");
                toks.map(|t| t.parse().unwrap()).collect()
            };
            let nverts = grab("nverts")[0] as usize;
            let vflat = grab("verts");
            let verts: Vec<[f64; 3]> =
                (0..nverts).map(|i| [vflat[3 * i], vflat[3 * i + 1], vflat[3 * i + 2]]).collect();
            let intra = grab("intra");
            let extra = grab("extra");
            let isov = grab("iso");
            let iso = [(isov[0], isov[1]), (isov[2], isov[3]), (isov[4], isov[5])];
            let d_par = grab("dpar")[0];
            let d_perp = grab("dperp")[0];
            let tau = grab("tau")[0];
            let d_app: [f64; 9] = grab("D_app").try_into().unwrap();
            let c: [f64; 81] = grab("C").try_into().unwrap();
            let params = grab("dki_params");
            let evals: [f64; 3] = params[0..3].try_into().unwrap();
            let kt: [f64; 15] = params[12..27].try_into().unwrap();
            let scalars: [f64; 9] = grab("scalars").try_into().unwrap();
            let qti: [f64; 4] = grab("qti").try_into().unwrap();
            let mapmri: [f64; 5] = grab("mapmri").try_into().unwrap();
            let ngpa: [f64; 4] = grab("ngpa").try_into().unwrap();
            let gqi: [f64; 2] = grab("gqi").try_into().unwrap();
            assert_eq!(lines.next(), Some("end"));
            cases.push(FixtureCase {
                verts, intra, extra, iso, d_par, d_perp, tau,
                d_app, c, evals, kt, scalars, qti, mapmri, ngpa, gqi,
            });
        }
        assert!(!cases.is_empty(), "no fixture cases parsed");
        cases
    }

    #[test]
    fn matches_dipy_closed_forms_on_fixtures() {
        let rtol = 1e-8;
        for (ci, case) in parse_fixtures().iter().enumerate() {
            let mix = VoxelMixture {
                verts: &case.verts,
                intra: &case.intra,
                extra: &case.extra,
                d_par: case.d_par,
                d_perp: case.d_perp,
                iso: &case.iso,
            };
            let (d_app, c) = moments(&mix);
            for i in 0..3 {
                for j in 0..3 {
                    assert_rel(d_app[i][j], case.d_app[3 * i + j], rtol,
                        &format!("case {ci} D_app[{i}][{j}]"));
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        for l in 0..3 {
                            assert_rel(c[i][j][k][l], case.c[27 * i + 9 * j + 3 * k + l], 1e-6,
                                &format!("case {ci} C[{i}{j}{k}{l}]"));
                        }
                    }
                }
            }
            let dki = dki_scalars(&d_app, &c);
            for k in 0..3 {
                assert_rel(dki.evals[k], case.evals[k], rtol, &format!("case {ci} eval {k}"));
            }
            for k in 0..15 {
                assert_rel(dki.kt[k], case.kt[k], 1e-6, &format!("case {ci} kt[{k}]"));
            }
            let got = [dki.fa, dki.md, dki.rd, dki.ad, dki.ak, dki.rk, dki.mk, dki.mkt, dki.kfa];
            let names = ["fa", "md", "rd", "ad", "ak", "rk", "mk", "mkt", "kfa"];
            for (g, (want, name)) in got.iter().zip(case.scalars.iter().zip(names)) {
                assert_rel(*g, *want, rtol, &format!("case {ci} {name}"));
            }
            let q = qti_scalars(&d_app, &c);
            for (g, (want, name)) in [q.micro_fa, q.coherence, q.k_bulk, q.k_shear]
                .iter()
                .zip(case.qti.iter().zip(["micro_fa", "coherence", "k_bulk", "k_shear"]))
            {
                assert_rel(*g, *want, rtol, &format!("case {ci} {name}"));
            }
            let (_, evecs) = eigh3(&d_app);
            let mp = mapmri_scalars(&mix, &d_app, column(&evecs, 0), case.tau, 0.12e-3);
            for (g, (want, name)) in [mp.rtop, mp.rtap, mp.rtpp, mp.msd, mp.qiv]
                .iter()
                .zip(case.mapmri.iter().zip(["rtop", "rtap", "rtpp", "msd", "qiv"]))
            {
                assert_rel(*g, *want, rtol, &format!("case {ci} {name}"));
            }
            let np = ng_pa_scalars(&mix, 0.12e-3, usize::MAX);
            for (g, (want, name)) in [np.ng, np.ngpar, np.ngperp, np.pa]
                .iter()
                .zip(case.ngpa.iter().zip(["ng", "ngpar", "ngperp", "pa"]))
            {
                assert_rel(*g, *want, rtol, &format!("case {ci} {name}"));
            }
            let gq = gqi_scalars(&mix);
            assert_rel(gq.gfa, case.gqi[0], rtol, &format!("case {ci} gfa"));
            assert_rel(gq.qa, case.gqi[1], rtol, &format!("case {ci} qa"));
        }
    }
}
