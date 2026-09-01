//! The factor grid is the acceptance suite's input; a missing arm silently weakens every
//! conclusion drawn from it, so its shape is pinned here rather than assumed.

use std::collections::BTreeSet;
use trxscan::benchmark::{factor_grid, PhaseKind};
use trxscan::kspace::KspaceWindow;

#[test]
fn factor_grid_is_complete_and_labelled() {
    let g = factor_grid();
    assert_eq!(g.len(), 3 * 4 * 2 * 2, "expected 48 grid points, got {}", g.len());

    let pf: BTreeSet<_> = g.iter().map(|p| format!("{:.3}", p.partial_fourier)).collect();
    assert_eq!(pf.len(), 3, "expected full / 6-8 / 7-8, got {pf:?}");
    assert!(pf.contains("1.000") && pf.contains("0.750") && pf.contains("0.875"));

    let ph: BTreeSet<_> = g.iter().map(|p| format!("{:?}", p.phase)).collect();
    assert_eq!(ph.len(), 4, "expected 4 phase conditions, got {ph:?}");

    let labels: BTreeSet<_> = g.iter().map(|p| p.label.clone()).collect();
    assert_eq!(labels.len(), g.len(), "labels must be unique: they name output directories");
    for p in &g {
        assert!(!p.label.is_empty(), "every grid point must be labelled for the results table");
    }
}

#[test]
fn grid_covers_the_shipping_configuration_and_the_clean_baseline() {
    let g = factor_grid();
    // 6/8 PF with diffusion phase and noise is what the CLI actually ships.
    assert!(
        g.iter().any(|p| p.partial_fourier == 0.75
            && p.phase == PhaseKind::Diffusion
            && p.noisy
            && p.window == KspaceWindow::None),
        "grid must include the shipping configuration"
    );
    // Full Fourier, no phase, noiseless, unapodized is the analytic baseline Kellner assumes.
    assert!(
        g.iter().any(|p| p.partial_fourier == 1.0
            && p.phase == PhaseKind::None
            && !p.noisy
            && p.window == KspaceWindow::None),
        "grid must include the clean analytic baseline"
    );
}
