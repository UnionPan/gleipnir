"""Compile, before serving, everything that would otherwise compile inside a request."""

from backend.services import bayesian_fits, heston_surface, lab_service


def warm_up() -> None:
    lab_service.warm_up()
    heston_surface.warm_up()
    bayesian_fits.warm_up()
