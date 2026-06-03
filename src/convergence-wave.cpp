#include "WaveEquation.hpp"

#include <deal.II/base/convergence_table.h>

#include <cmath>
#include <fstream>
#include <iostream>

using namespace dealii;

// Manufactured solution:
//   u(x,y,t) = sin(pi*x) * sin(pi*y) * cos(2*pi*t)
//
// With rho=1, c=1:  u_tt - Delta u = f
//   u_tt    = -4*pi^2 * sin(pi*x)*sin(pi*y)*cos(2*pi*t)
//   Delta u = -2*pi^2 * sin(pi*x)*sin(pi*y)*cos(2*pi*t)
//   => f    = -2*pi^2 * sin(pi*x)*sin(pi*y)*cos(2*pi*t)

static constexpr unsigned int dim = WaveEquation::dim;

// Initial displacement: u_0(x,y) = sin(pi*x)*sin(pi*y)
double
WaveEquation::FunctionU0::value(const Point<dim> &p, const unsigned int) const
{
  return std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
}

// Initial velocity: v_0(x,y) = du/dt|_{t=0}.
//   du/dt = -2*pi * sin(pi*x)*sin(pi*y)*sin(2*pi*t),  which is 0 at t=0.
double
WaveEquation::FunctionV0::value(const Point<dim> & /*p*/, const unsigned int) const
{
  return 0.0;
}

class ExactSolution : public Function<dim>
{
public:
  ExactSolution() : Function<dim>() {}

  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) *
           std::cos(2.0 * M_PI * this->get_time());
  }

  Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override
  {
    const double c2t = std::cos(2.0 * M_PI * this->get_time());
    Tensor<1, dim> g;
    g[0] = M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]) * c2t;
    g[1] = M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]) * c2t;
    return g;
  }
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const double       theta = 0.5; // Crank-Nicolson
  const double       T     = 0.5;
  const unsigned int r     = 1;

  const auto rho = [](const Point<dim> &) { return 1.0; };
  const auto c   = [](const Point<dim> &) { return 1.0; };
  const auto f   = [](const Point<dim> &p, const double &t) {
    return -2.0 * M_PI * M_PI *
           std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) *
           std::cos(2.0 * M_PI * t);
  };

  ConvergenceTable table;

  std::ofstream csv("convergence-wave.csv");
  csv << "h,dt,eL2,eH1\n";

  for (const unsigned int N : {4u, 8u, 16u, 32u})
    {
      const double h = 1.0 / N;
      // Use dt ~ h in the convergence study so temporal error stays comparable
      // to spatial error and does not dominate for large N.
      // For Crank–Nicolson, this keeps the time discretization error O(h^2).
      const double delta_t = 0.5 * h;

      WaveEquation problem("mms", r, T, delta_t, theta, rho, c, f, N, "./results-convergence");
      problem.run();

      // Compare solution at final time T with the initial condition at t=0.
      // The solution should evolve in time, so the difference should be non-zero.
      // If eL2_t0 ~ 0, the solution is likely stuck at the initial condition
      // (time stepping not working or solver frozen).
      ExactSolution exact_t0;
      exact_t0.set_time(0.0);
      const double eL2_t0 = problem.compute_error(VectorTools::L2_norm, exact_t0);
      std::cout << "[Sanity check] N=" << N
                << "; ||u_h - u(t=0)||=" << eL2_t0 << std::endl
                << "  (~0 -> solver stuck at t=0, O(1) -> time evolution OK)"
                << std::endl;

      ExactSolution exact;
      exact.set_time(T);

      const double eL2 = problem.compute_error(VectorTools::L2_norm, exact);
      const double eH1 = problem.compute_error(VectorTools::H1_norm, exact);

      table.add_value("h",  h);
      table.add_value("dt", delta_t);
      table.add_value("L2", eL2);
      table.add_value("H1", eH1);
      std::cout << "[Info] h-Convergence Study (L^2 and H^1 Error Norms)" << std::endl;
      csv << h << "," << delta_t << "," << eL2 << "," << eH1 << "\n";
    }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
  table.set_scientific("L2", true);
  table.set_scientific("H1", true);
  table.write_text(std::cout);

  return 0;
}
