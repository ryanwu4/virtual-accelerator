from typing import Any

import beamphysics
from lume.model import LUMEModel
from lume.staged_model import FinalParticlesMixIn
from lume.variables import IntVariable, ParticleGroupVariable, ScalarVariable


class TwoBunchSurrogate(LUMEModel, FinalParticlesMixIn):
    """
    FACET-II drive and witness bunches at the entrance of XTCAVF, sampled from
    the facet2-twobunch-cnf normalizing flow.

    The inputs are the flow's 37 knobs, under their FACET2-S2E simulation names
    and units (not PVs), and the sampling ``seed``: the same inputs give the
    same beam. The output ``XTCAVF_beam`` is the same beam as
    ``final_particles``. It follows the FACET2-S2E conventions: lost particles
    are removed, drive particles carry 0.2 % more weight than witness
    particles, and every particle has z at the FACET2-S2E s of the XTCAVF
    entrance, with t on the FACET2-S2E clock.

    Parameters
    ----------
    cnf : twobunch_cnf.cnf.TwoBunchCNF
        The flow, from ``twobunch_cnf.load_model``.
    n_particles : int, optional
        Number of macroparticles injected per draw; losses leave fewer. Default
        is 10000.
    seed : int, optional
        Initial value of the ``seed`` input. Default is 0.
    latent : str, optional
        "sample" draws the beam latent (the injector variation the knobs do not
        explain) from its prior, "mean" uses the prior mean. Default is
        "sample".
    """

    def __init__(
        self, cnf, n_particles: int = 10000, seed: int = 0, latent: str = "sample"
    ) -> None:
        super().__init__()
        self.cnf = cnf
        self.n_particles = n_particles
        self.latent = latent

        self._variables = {
            knob["name"]: ScalarVariable(
                name=knob["name"],
                description=knob["description"],
                unit=knob["unit"],
                default_value=knob["baseline"],
                value_range=(knob["low"], knob["high"]),
            )
            for knob in cnf.meta["knobs"]
        }
        self._variables["seed"] = IntVariable(name="seed", default_value=seed)
        self._variables["XTCAVF_beam"] = ParticleGroupVariable(
            name="XTCAVF_beam", read_only=True
        )
        self.reset()

    @property
    def supported_variables(self) -> dict[str, Any]:
        return self._variables

    @property
    def final_particles(self) -> beamphysics.ParticleGroup:
        # A copy, since downstream stages may modify the beam they are given.
        return self._beam.copy()

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {
            name: self.final_particles if name == "XTCAVF_beam" else self._inputs[name]
            for name in names
        }

    def _set(self, values: dict[str, Any]) -> None:
        inputs = {**self._inputs, **values}
        self._beam = self._sample(inputs)
        self._inputs = inputs

    def reset(self) -> None:
        self._inputs = {
            name: variable.default_value
            for name, variable in self._variables.items()
            if not variable.read_only
        }
        self._beam = self._sample(self._inputs)

    def _sample(self, inputs: dict[str, Any]) -> beamphysics.ParticleGroup:
        knobs = {name: value for name, value in inputs.items() if name != "seed"}
        return self.cnf.particle_group(
            knobs,
            n_particles=self.n_particles,
            seed=inputs["seed"],
            latent=self.latent,
        )
