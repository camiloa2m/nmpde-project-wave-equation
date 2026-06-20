#include "WaveEquation.hpp"

#include <deal.II/base/convergence_table.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using namespace dealii;

// Manufactured solution:
//   u(x,y,t) = sin(pi*x) * sin(pi*y) * cos(omega*t)
//
// With rho=1, c=1:  u_tt - Delta u = f
//   u_tt    = -omega^2 * sin(pi*x)*sin(pi*y)*cos(omega*t)
//   Delta u = -2*pi^2  * sin(pi*x)*sin(pi*y)*cos(omega*t)
//   f       = (2*pi^2 - omega^2) * sin(pi*x)*sin(pi*y)*cos(omega*t)
//
// omega = pi*sqrt(2) zeroes out f (homogeneous case), omega = 2*pi gives
// a forced case.

static constexpr unsigned int dim = WaveEquation::dim;

// Runtime parameter for the MMS driver, settable from argv. Default 2*pi.
static double omega = 2.0 * M_PI;

// Initial displacement: u_0(x,y) = sin(pi*x)*sin(pi*y)
double
WaveEquation::FunctionU0::value(const Point<dim> &p, const unsigned int) const
{
  return std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
}

// Initial velocity: v_0(x,y) = du/dt|_{t=0}.
//   du/dt = -omega * sin(pi*x)*sin(pi*y)*sin(omega*t),  which is 0 at t=0.
double
WaveEquation::FunctionV0::value(const Point<dim> & /*p*/, const unsigned int) const
{
  return 0.0;
}

class ExactSolution : public Function<dim>
{
public:
  explicit ExactSolution(const double omega_) : Function<dim>(), omega(omega_) {}

  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) *
           std::cos(omega * this->get_time());
  }

  Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override
  {
    const double cwt = std::cos(omega * this->get_time());
    Tensor<1, dim> g;
    g[0] = M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]) * cwt;
    g[1] = M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]) * cwt;
    return g;
  }

private:
  const double omega;
};

// Exact velocity: v = du/dt = -omega * sin(pi*x)*sin(pi*y)*sin(omega*t)
class ExactVelocity : public Function<dim>
{
public:
  explicit ExactVelocity(const double omega_) : Function<dim>(), omega(omega_) {}

  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return -omega * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) *
           std::sin(omega * this->get_time());
  }

private:
  const double omega;
};

// Forcing term: f(x,y,t) = (2*pi^2 - omega^2) * sin(pi*x)*sin(pi*y)*cos(omega*t)
double
forcing(const Point<dim> &p, const double &t, const double omega_)
{
  return (2.0 * M_PI * M_PI - omega_ * omega_) *
         std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::cos(omega_ * t);
}

// Checks that u_tt - Laplacian(u) - f is zero at a few random points,
// to catch typos in the forcing term before running the full study.
void
check_manufactured_solution(const double omega_)
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> time_dist(0.0, 1.0);

  double max_residual = 0.0;
  for (int i = 0; i < 20; ++i)
    {
      const Point<dim> p(unit(rng), unit(rng));
      const double t = time_dist(rng);

      const double s   = std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
      const double cwt = std::cos(omega_ * t);

      const double u_tt_ex     = -omega_ * omega_ * s * cwt;
      const double laplacian_u = -2.0 * M_PI * M_PI * s * cwt;
      const double f           = forcing(p, t, omega_);

      const double residual = u_tt_ex - laplacian_u - f;
      max_residual = std::max(max_residual, std::abs(residual));
    }

  std::cout << "[Sanity check] omega = " << omega_ << std::endl
            << "  max|u_tt_ex - Laplacian(u_ex) - f| over 20 random points = "
            << max_residual << std::endl;
  AssertThrow(max_residual < 1e-10,
              ExcMessage("Manufactured solution residual check failed: "
                          "forcing term does not match u_tt - Laplacian(u)."));
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const double       theta = 0.5; // Crank-Nicolson
  const double       T     = 0.5;
  const unsigned int r     = 1;

  // argv[1]: "spatial" or "temporal" sweep (default spatial).
  // argv[2]: omega (e.g. 6.283185307 for 2*pi, 4.442882938 for pi*sqrt(2)).
  std::string sweep_kind = "spatial";
  if (argc > 1)
    sweep_kind = argv[1];
  if (argc > 2)
    omega = std::stod(argv[2]);

  const auto rho = [](const Point<dim> &) { return 1.0; };
  const auto c   = [](const Point<dim> &) { return 1.0; };
  const auto f   = [](const Point<dim> &p, const double &t) {
    return forcing(p, t, omega);
  };

  check_manufactured_solution(omega);

  // Tag for CSV/output file names so the two omega cases don't collide.
  std::ostringstream tag_stream;
  tag_stream << "omega" << omega;
  const std::string case_tag = tag_stream.str();

  ConvergenceTable table;

  if (sweep_kind == "spatial")
    {
      // Spatial (h) convergence: dt is fixed and small (Crank-Nicolson is
      // O(dt^2)) so temporal error stays negligible up to N=128.
      const double delta_t = 1e-4;

      const std::string csv_name = "convergence_spatial_" + case_tag + ".csv";
      std::ofstream csv(csv_name);
      csv << "case,refinement,N,h,dt,L2_error,H1_error\n";

      unsigned int level = 0;
      for (const unsigned int N : {4u, 8u, 16u, 32u, 64u, 128u})
        {
          const double h = 1.0 / N;

          // No VTU output needed here, it'd be costly.
          WaveEquation problem("mms", r, T, delta_t, theta, rho, c, f, N,
                                "./results-convergence-spatial-" + case_tag,
                                /* enable_output = */ false);
          problem.run();

          // u_h should differ from u(t=0): if eL2_t0 ~ 0 the solver is
          // stuck at the initial condition instead of timestepping.
          ExactSolution exact_t0(omega);
          exact_t0.set_time(0.0);
          const double eL2_t0 = problem.compute_error(VectorTools::L2_norm, exact_t0);
          std::cout << "[Sanity check] N=" << N
                    << "; ||u_h - u(t=0)||=" << eL2_t0 << std::endl
                    << "  (~0 -> solver stuck at t=0, O(1) -> time evolution OK)"
                    << std::endl;

          ExactSolution exact(omega);
          exact.set_time(T);

          const double eL2 = problem.compute_error(VectorTools::L2_norm, exact);
          const double eH1 = problem.compute_error(VectorTools::H1_seminorm, exact);

          table.add_value("h", h);
          table.add_value("dt", delta_t);
          table.add_value("L2", eL2);
          table.add_value("H1", eH1);

          csv << case_tag << "," << level << "," << N << "," << h << ","
              << delta_t << "," << eL2 << "," << eH1 << "\n";
          ++level;
        }

      table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
      table.set_scientific("L2", true);
      table.set_scientific("H1", true);
      std::cout << "[Info] h-Convergence Study (L2 and H1-seminorm Errors), "
                << case_tag << std::endl;
      table.write_text(std::cout);
    }
  else if (sweep_kind == "temporal")
    {
      // Temporal (dt) convergence: mesh is fixed and fine so spatial
      // error stays negligible compared to temporal error.
      const unsigned int N = 128;
      const double       h = 1.0 / N;

      const std::string csv_name = "convergence_temporal_" + case_tag + ".csv";
      std::ofstream csv(csv_name);
      csv << "case,refinement,N,h,dt,L2_error,H1_error\n";

      const double base_delta_t = 0.02;
      unsigned int level = 0;
      for (const double delta_t : {base_delta_t, base_delta_t / 2.0,
                                    base_delta_t / 4.0, base_delta_t / 8.0})
        {
          // No VTU output needed here, it'd be costly.
          WaveEquation problem("mms", r, T, delta_t, theta, rho, c, f, N,
                                "./results-convergence-temporal-" + case_tag,
                                /* enable_output = */ false);
          problem.run();

          ExactSolution exact(omega);
          exact.set_time(T);

          const double eL2 = problem.compute_error(VectorTools::L2_norm, exact);
          const double eH1 = problem.compute_error(VectorTools::H1_seminorm, exact);

          table.add_value("h", h);
          table.add_value("dt", delta_t);
          table.add_value("L2", eL2);
          table.add_value("H1", eH1);

          csv << case_tag << "," << level << "," << N << "," << h << ","
              << delta_t << "," << eL2 << "," << eH1 << "\n";
          ++level;
        }

      table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
      table.set_scientific("L2", true);
      table.set_scientific("H1", true);
      std::cout << "[Info] dt-Convergence Study (L2 and H1-seminorm Errors), "
                << case_tag << std::endl;
      table.write_text(std::cout);
    }
  else
    {
      AssertThrow(false, ExcMessage("Unknown sweep kind '" + sweep_kind +
                                     "'. Use 'spatial' or 'temporal'."));
    }

  return 0;
}
