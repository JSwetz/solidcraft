//! Where a face's surface has a pole, as the kernel's classifier says, asked once per iso-line.
//!
//! Whether a surface is collapsed along `d` at `q` is decided by the kernel on the iso-line
//! through `q` along `d` alone ([`solidcore::eval::collapse`]): it depends on `q` only through
//! the other parameter (wrapped into the surface's period) and on the reach. One question builds
//! that iso-line's whole control row, a de Boor evaluation per column of the net, while the
//! mesher asks at every node, and the nodes of a face share iso-lines (an edge on a patch side,
//! a grid's rows and columns, a cut grid's lines). [`Face`] keeps each answer under its iso-line
//! and asks the kernel once. Where the kernel cannot tell (a weight not certified non-zero, a
//! point off the net), the speed of the surface there decides, as SolidCraft's mesher did
//! ([`uv::degenerate_within`]).

use std::collections::HashMap;
use std::sync::Mutex;

use super::uv;
use crate::eval::EvalSurface;

/// A face's surface for the mesher: its display evaluator's form, the kernel's form of the same
/// surface, and the poles found on it.
pub(crate) struct Face {
    eval: EvalSurface,
    kernel: solidcore::eval::EvalSurface,
    dom: uv::Domain,
    /// `(d, other parameter, reach)`, the two numbers by their bits, to the kernel's answer.
    collapsed: Mutex<HashMap<(usize, u64, u64), bool>>,
    /// Calls of the kernel's classifier made, for the tests.
    #[cfg(test)]
    asked: std::sync::atomic::AtomicUsize,
}

impl Face {
    pub(crate) fn new(eval: EvalSurface) -> Face {
        let kernel = kernel_form(&eval);
        let dom = uv::domain(&eval);
        Face { eval, kernel, dom, collapsed: Mutex::new(HashMap::new()), #[cfg(test)] asked: std::sync::atomic::AtomicUsize::new(0) }
    }

    /// The display evaluator's form, for the plan.
    pub(crate) fn into_eval(self) -> EvalSurface {
        self.eval
    }

    /// The kernel's form of the surface (the pole pass's).
    pub(crate) fn kernel(&self) -> &solidcore::eval::EvalSurface {
        &self.kernel
    }

    /// Whether the surface is collapsed along `d` at `q`, or within `reach` mm of it.
    pub(crate) fn pole(&self, q: [f64; 2], d: usize, reach: f64) -> bool {
        self.classified(q, d, reach).unwrap_or_else(|| uv::degenerate_within(&self.eval, &self.dom, q, d, reach))
    }

    /// The kernel's answer, asked once per iso-line; `None` where it cannot tell.
    fn classified(&self, q: [f64; 2], d: usize, reach: f64) -> Option<bool> {
        let kernel = &self.kernel;
        if matches!(self.eval, EvalSurface::Plane { .. }) || d > 1 || !(q[0].is_finite() && q[1].is_finite()) {
            return solidcore::eval::uv::degenerate_within(kernel, q, d, reach).ok();
        }
        let other = *kernel.wrap(q).get(1 - d)?;
        let key = (d, other.to_bits(), reach.to_bits());
        if let Some(&known) = self.lock().get(&key) {
            return Some(known);
        }
        #[cfg(test)]
        self.asked.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let answer = solidcore::eval::uv::degenerate_within(kernel, q, d, reach).ok()?;
        self.lock().insert(key, answer);
        Some(answer)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<(usize, u64, u64), bool>> {
        // The map holds only finished answers, so one left by a panicking thread is sound.
        self.collapsed.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Calls of the kernel's classifier made so far.
    #[cfg(test)]
    pub(crate) fn asked(&self) -> usize {
        self.asked.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl std::ops::Deref for Face {
    type Target = EvalSurface;

    fn deref(&self) -> &EvalSurface {
        &self.eval
    }
}

/// `s` in the kernel's form: the same plane, net or revolution, value for value.
pub(crate) fn kernel_form(s: &EvalSurface) -> solidcore::eval::EvalSurface {
    use solidcore::eval::{EvalCurve, EvalSurface as K};
    let v = |p: solidcore_record::Vec3| solidcore::Vec3::new(p.x, p.y, p.z);
    match s {
        EvalSurface::Plane { origin, du, dv } => K::Plane { origin: v(*origin), du: v(*du), dv: v(*dv) },
        EvalSurface::Nurbs { degree, knots, count, ctrl, .. } => K::nurbs(*degree, knots.clone(), *count, ctrl.clone()),
        EvalSurface::Revolved { profile, origin, axis, transform, swap } => K::Revolved {
            profile: EvalCurve::new(profile.degree, profile.knots.clone(), profile.ctrl.clone()),
            origin: v(*origin),
            axis: v(*axis),
            transform: *transform,
            swap: *swap,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The memo gives the kernel's own answer at every node, and asks the kernel once per
    /// iso-line: on a 9 x 9 grid of each curved face of the coil and of a sphere (whose poles
    /// collapse a whole row), 2 directions x 9 iso-lines x 2 reaches, not 324 questions.
    #[test]
    fn the_pole_classifier_is_asked_once_per_iso_line() {
        for name in ["smlib_coil_lower_half", "sphere_r4"] {
            for face in &crate::fixtures::fixture(name).faces {
                let Ok((s, _)) = super::super::surface::eval_surface(face.parameters(), face.domain) else { panic!("{name}: evaluates") };
                if matches!(s, EvalSurface::Plane { .. }) {
                    continue;
                }
                let f = Face::new(s);
                let dom = uv::domain(&f);
                let [(u0, u1), (v0, v1)] = dom.range.map(|r| r.unwrap_or((0.0, 1.0)));
                let at = |a: f64, b: f64, i: usize| a + (b - a) * i as f64 / 8.0;
                for reach in [0.0, 1e-3] {
                    for i in 0..9 {
                        for j in 0..9 {
                            let q = [at(u0, u1, i), at(v0, v1, j)];
                            for d in 0..2 {
                                let Ok(want) = solidcore::eval::uv::degenerate_within(f.kernel(), q, d, reach) else { panic!("{name}: classifies {q:?}") };
                                assert_eq!(f.pole(q, d, reach), want, "{name} {q:?}");
                            }
                        }
                    }
                }
                assert_eq!(f.asked(), 2 * 9 * 2, "{name}");
            }
        }
    }
}
