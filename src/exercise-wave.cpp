#include "WaveEquation.hpp"

// Initial displacement: Gaussian pulse centered at (0.5, 0.5).
//   u_0(x,y) = exp(-100 * ((x-0.5)^2 + (y-0.5)^2))
double
WaveEquation::FunctionU0::value(const Point<dim> &p, const unsigned int) const
{
  const double d2 = (p[0]-0.5)*(p[0]-0.5) + (p[1]-0.5)*(p[1]-0.5);
  return std::exp(-100.0 * d2);
}

// Initial velocity: starts from rest.
double
WaveEquation::FunctionV0::value(const Point<dim> & /*p*/, const unsigned int) const
{
  return 0.0;
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  constexpr unsigned int dim = WaveEquation::dim;

  // Homogeneous medium: rho (density) and c (wave speed) are both constant.
  const auto rho = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto c   = [](const Point<dim> & /*p*/) { return 1.0; };

  // No external forcing -- the wave is driven entirely by the initial
  // Gaussian displacement defined in FunctionU0 below.
  const auto f = [](const Point<dim> & /*p*/, const double & /*t*/) {
    return 0.0;
  };

  // Theta-method parameter. theta=0.5 is Crank-Nicolson: unconditionally
  // stable and second-order accurate in time (theta=1.0 would be backward
  // Euler, first-order; theta=0.0 forward Euler, conditionally stable).
  const double theta = 0.5;

  WaveEquation problem(
      /* mesh_file_name = */ "wave_domain.msh", // used for output naming; the mesh itself is generated internally
      /* degree         = */ 1,                 // linear (P1) elements
      /* T              = */ 2.0,               // long enough for the wave to reach the boundary and reflect back
      /* delta_t        = */ 0.01,
      /* theta          = */ theta,
      rho,
      c,
      f,
      /* n_subdivisions = */ 50,
      /* output_dir     = */ "./results-exercise"
  );

  problem.run();

  return 0;
}

// Expected behavior: the energy starts out purely potential (from the
// initial Gaussian displacement), the pulse expands outward in a circle,
// and the total energy printed each step should stay roughly constant.
// With the default homogeneous Dirichlet BCs (u=0 on the boundary), the
// wave reflects off the domain edges and bounces back inward.