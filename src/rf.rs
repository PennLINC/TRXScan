//! RF pulse fidelity of the spin-echo EPI: the excitation and refocusing flip angles actually
//! applied, and the per-compartment signal weight they leave on the clean signal.
//!
//! A diffusion EPI is a single-refocused spin echo. With an excitation of `α` and a refocusing
//! of `β` (nominally 90° and 180°) the spin-echo pathway carries `sin α · sin²(β/2)` of the
//! longitudinal magnetization; the rest is spoiled by the diffusion gradients (crushers) and does
//! not return. The longitudinal magnetization itself is the steady state of the TR loop
//! (excitation, TE/2 recovery, refocusing, TR − TE/2 recovery), which for `α < 90°` saturates
//! less than the textbook `1 − 2E(TR − TE/2) + E(TR)`.
//!
//! The reason this exists: on Siemens VE11 the transmitter can apply less voltage than the
//! pulse asks for (`sTXSPEC.aRFPULSE[n].flAmplitude < flAmplitudeNL` in the CSA series
//! header), and the flip angle scales linearly with the voltage ratio, pulse by pulse. The
//! multiband refocusing pulse needs `MB` times the single-band voltage, so it clips first; the
//! applied angles are therefore an (excitation, refocusing) pair, not one factor.
//! [`SpinEchoRf::from_voltage_ratio`] turns the two DICOM ratios into the pair.
//!
//! Noise is set on the receive side and does not move with the transmit voltage, so under this
//! model the SNR falls with the weight while `S(b)/S(0)` in the clean signal is untouched: the
//! harm arrives through the Rician floor at high b.

/// Adult 3 T T1 (ms) per compartment `[fiber, gm, csf]` (Wansapura 1999: WM 830, GM 1330;
/// CSF ~4000). Shared with the GRE synthesis.
pub const ADULT_T1_MS: [f32; 3] = [830.0, 1330.0, 4000.0];

/// The applied RF pulses of a spin-echo EPI and the relaxation context that turns them into a
/// per-compartment amplitude. `tr_s = None` is an infinitely long TR (no T1 saturation).
#[derive(Clone, Debug, PartialEq)]
pub struct SpinEchoRf {
    /// Applied excitation flip angle (degrees; nominal 90).
    pub excite_deg: f64,
    /// Applied refocusing flip angle (degrees; nominal 180).
    pub refocus_deg: f64,
    /// Repetition time (s); `None` = fully relaxed.
    pub tr_s: Option<f64>,
    /// Echo time (s): the refocusing pulse sits at TE/2.
    pub te_s: f64,
    /// Per-compartment T1 (ms) `[fiber, gm, csf]`.
    pub t1_ms: [f32; 3],
}

impl SpinEchoRf {
    /// Ideal 90°/180° pulses at `tr_s`/`te_s` with adult T1s.
    pub fn nominal(tr_s: Option<f64>, te_s: f64) -> Self {
        SpinEchoRf { excite_deg: 90.0, refocus_deg: 180.0, tr_s, te_s, t1_ms: ADULT_T1_MS }
    }

    /// From the per-pulse applied/requested voltage ratios (`flAmplitude / flAmplitudeNL` of the
    /// excitation and refocusing pulses): the flip angle is linear in the pulse voltage.
    pub fn from_voltage_ratio(excite_ratio: f64, refocus_ratio: f64, tr_s: Option<f64>, te_s: f64) -> Self {
        SpinEchoRf { excite_deg: 90.0 * excite_ratio, refocus_deg: 180.0 * refocus_ratio, ..Self::nominal(tr_s, te_s) }
    }

    /// Steady-state longitudinal magnetization before the excitation, as a fraction of M0, per
    /// compartment: `[(1 − Er) + Er cos β (1 − Eh)] / (1 − Er Eh cos α cos β)` with
    /// `Eh = exp(−TE/2 / T1)`, `Er = exp(−(TR − TE/2) / T1)`. Reduces to the textbook
    /// `1 − 2 E(TR − TE/2) + E(TR)` at 90°/180° and to 1 for an infinite TR. The transverse
    /// magnetization the refocusing pulse tips back along z is taken as spoiled.
    pub fn steady_state(&self) -> [f64; 3] {
        let (ca, cb) = (self.excite_deg.to_radians().cos(), self.refocus_deg.to_radians().cos());
        let mut out = [1.0f64; 3];
        let Some(tr) = self.tr_s else { return out };
        assert!(tr > 0.0 && tr >= self.te_s, "TR must be positive and at least TE");
        for (c, o) in out.iter_mut().enumerate() {
            let t1 = self.t1_ms[c] as f64 / 1000.0;
            let eh = (-(0.5 * self.te_s) / t1).exp();
            let er = (-(tr - 0.5 * self.te_s) / t1).exp();
            *o = ((1.0 - er) + er * cb * (1.0 - eh)) / (1.0 - er * eh * ca * cb);
        }
        out
    }

    /// The spin-echo pathway's share of the longitudinal magnetization, `sin α · sin²(β/2)`
    /// (1 at 90°/180°).
    pub fn pathway(&self) -> f64 {
        let a = self.excite_deg.to_radians();
        let hb = 0.5 * self.refocus_deg.to_radians();
        a.sin() * hb.sin() * hb.sin()
    }

    /// Per-compartment amplitude to multiply the clean signal by: [`Self::steady_state`] times
    /// [`Self::pathway`]. The ideal pulses at an infinite TR give `[1, 1, 1]`.
    pub fn weights(&self) -> [f64; 3] {
        let p = self.pathway();
        self.steady_state().map(|m| m * p)
    }

    /// `self.weights()` relative to the nominal pulses at the same TR/TE/T1: the fraction of the
    /// correctly-adjusted scan's signal each compartment keeps.
    pub fn relative_weights(&self) -> [f64; 3] {
        let nominal = SpinEchoRf { excite_deg: 90.0, refocus_deg: 180.0, ..self.clone() }.weights();
        let w = self.weights();
        [w[0] / nominal[0], w[1] / nominal[1], w[2] / nominal[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn ideal_pulses_at_infinite_tr_are_unit_weight() {
        let rf = SpinEchoRf::nominal(None, 0.089);
        assert_eq!(rf.weights(), [1.0, 1.0, 1.0]);
        assert_eq!(rf.relative_weights(), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn nominal_steady_state_is_the_textbook_spin_echo() {
        let rf = SpinEchoRf::nominal(Some(4.2), 0.089);
        let ss = rf.steady_state();
        for c in 0..3 {
            let t1 = ADULT_T1_MS[c] as f64 / 1000.0;
            let want = 1.0 - 2.0 * (-(4.2 - 0.0445) / t1).exp() + (-4.2 / t1).exp();
            assert!(close(ss[c], want, 1e-12), "compartment {c}: {} vs {want}", ss[c]);
        }
        // WM and GM are nearly relaxed at TR 4.2 s; CSF is not
        assert!(ss[0] > 0.98 && ss[1] > 0.9 && ss[2] < 0.7, "{ss:?}");
    }

    #[test]
    fn pathway_is_sin_alpha_sin_squared_half_beta() {
        let rf = SpinEchoRf { excite_deg: 63.0, refocus_deg: 126.0, ..SpinEchoRf::nominal(None, 0.089) };
        let want = 63f64.to_radians().sin() * 63f64.to_radians().sin().powi(2);
        assert!(close(rf.pathway(), want, 1e-12));
        assert!(close(rf.weights()[0], want, 1e-12));
        // a 30 % voltage shortfall on both pulses costs ~29 % of the signal
        assert!(close(want, 0.707, 2e-3), "{want}");
    }

    #[test]
    fn voltage_ratio_scales_each_pulse_linearly() {
        let rf = SpinEchoRf::from_voltage_ratio(1.0, 0.8, Some(4.2), 0.089);
        assert_eq!((rf.excite_deg, rf.refocus_deg), (90.0, 144.0));
        // an intact 90 with a clipped 180: only the refocusing term moves
        let p = rf.pathway();
        assert!(close(p, 72f64.to_radians().sin().powi(2), 1e-12));
        assert!(p < 1.0);
    }

    #[test]
    fn t1_term_follows_the_sign_of_cos_alpha_cos_beta() {
        // both pulses clipped: the sub-90° leaves +cos α of Mz, which the sub-180° (cos β < 0)
        // inverts, so the steady state drops below nominal and long-T1 CSF loses the most
        let both = SpinEchoRf::from_voltage_ratio(0.7, 0.7, Some(4.2), 0.089);
        let rel = both.relative_weights();
        assert!(rel[2] < rel[1] && rel[1] < rel[0] && rel[0] < both.pathway(), "{rel:?}");
        assert!(rel[0] > 0.7 && rel[2] > 0.6, "{rel:?}");
        // refocusing alone clipped: cos α = 0 kills that term, and the weaker inversion of the
        // TE/2 recovery leaves CSF marginally *above* the pathway share
        let refocus = SpinEchoRf::from_voltage_ratio(1.0, 0.7, Some(4.2), 0.089);
        let rel = refocus.relative_weights();
        assert!(close(rel[0], refocus.pathway(), 1e-3) && rel[2] > refocus.pathway(), "{rel:?}");
        // at an infinite TR the relative weight is the pathway share exactly
        let inf = SpinEchoRf::from_voltage_ratio(0.7, 0.7, None, 0.089).relative_weights();
        assert!(inf.iter().all(|&r| close(r, both.pathway(), 1e-12)));
    }

    #[test]
    fn steady_state_is_bounded_and_monotone_in_tr() {
        for &tr in &[0.5, 1.0, 2.0, 4.2, 10.0] {
            let ss = SpinEchoRf::nominal(Some(tr), 0.089).steady_state();
            assert!(ss.iter().all(|&m| m > 0.0 && m <= 1.0), "TR {tr}: {ss:?}");
        }
        let a = SpinEchoRf::nominal(Some(2.0), 0.089).steady_state();
        let b = SpinEchoRf::nominal(Some(6.0), 0.089).steady_state();
        assert!((0..3).all(|c| b[c] > a[c]));
    }
}
