#include "WaveEquation.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

// Long-time discrete energy check for the homogeneous manufactured-solution
// case (f == 0): omega = pi*sqrt(2), u0(x,y) = sin(pi*x)*sin(pi*y), u1 = 0.
//
// Discrete energy E^n = 0.5*(V^n)^T*M*V^n + 0.5*(U^n)^T*K*U^n is logged at
// every timestep by WaveEquation::compute_energy() (see enable_energy_log()).
// Mesh is fixed at N=64 (h=1/64) since this targets the time-stepping
// scheme, not spatial accuracy.

static constexpr unsigned int dim = WaveEquation::dim;

static const double omega = M_PI * std::sqrt(2.0);

// Initial displacement: u_0(x,y) = sin(pi*x)*sin(pi*y)
double
WaveEquation::FunctionU0::value(const Point<dim> &p, const unsigned int) const
{
  return std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
}

// Initial velocity: v_0(x,y) = 0
double
WaveEquation::FunctionV0::value(const Point<dim> & /*p*/, const unsigned int) const
{
  return 0.0;
}

void
run_case(const double theta, const std::string &csv_name)
{
  const unsigned int N        = 64; // h = 1/64
  const double       delta_t  = 0.02;
  const double       T_final  = 35.0;

  const auto rho = [](const Point<dim> &) { return 1.0; };
  const auto c   = [](const Point<dim> &) { return 1.0; };
  const auto f   = [](const Point<dim> &, const double &) { return 0.0; };

  std::cout << "=== Energy run: theta = " << theta << " ===" << std::endl;

  WaveEquation problem("energy", 1, T_final, delta_t, theta, rho, c, f, N,
                        "./results/energy", /* enable_output = */ false);
  problem.enable_energy_log(csv_name);
  problem.run();
}

// Reads the (t,energy) CSV written by enable_energy_log() and reports
// E(0), the final logged value, and the first time E^n exceeds
// ten_x_threshold * E(0) (0 if it never does).
void
report_energy_csv(const std::string &csv_path)
{
  std::ifstream csv(csv_path);
  std::string line;
  std::getline(csv, line); // header

  double e0 = std::numeric_limits<double>::quiet_NaN();
  double t_last = 0.0, e_last = 0.0;
  double ten_x_time = -1.0;
  bool first = true;

  while (std::getline(csv, line))
    {
      std::istringstream ss(line);
      std::string t_str, e_str;
      std::getline(ss, t_str, ',');
      std::getline(ss, e_str, ',');
      const double t = std::stod(t_str);
      const double e = std::stod(e_str);

      if (first)
        {
          e0 = e;
          first = false;
        }
      if (ten_x_time < 0.0 && e > 10.0 * e0)
        ten_x_time = t;

      t_last = t;
      e_last = e;
    }

  std::cout << "  E(0)              = " << e0 << std::endl;
  std::cout << "  Final logged E    = " << e_last << " (at t = " << t_last << ")" << std::endl;
  if (ten_x_time >= 0.0)
    std::cout << "  E^n first exceeded 10*E(0) at t = " << ten_x_time << std::endl;
  else
    std::cout << "  E^n never exceeded 10*E(0) over the logged run" << std::endl;
}

// theta=0 (fully explicit) is only conditionally stable, so this run stops
// early via the blow-up guard once E^n exceeds 100x E^0, instead of running
// a diverging simulation to T_final. Also reports when E^n first crossed
// 10x E^0, as an instability-onset marker.
void
run_theta0_case(const double delta_t, const std::string &csv_name)
{
  const unsigned int N        = 64; // h = 1/64
  const double       T_final  = 35.0;

  const auto rho = [](const Point<dim> &) { return 1.0; };
  const auto c   = [](const Point<dim> &) { return 1.0; };
  const auto f   = [](const Point<dim> &, const double &) { return 0.0; };

  std::cout << "=== Energy run: theta = 0 (explicit), dt = " << delta_t
            << " ===" << std::endl;

  WaveEquation problem("energy", 1, T_final, delta_t, /* theta = */ 0.0,
                        rho, c, f, N, "./results/energy",
                        /* enable_output = */ false);
  problem.enable_energy_log(csv_name);
  problem.enable_blowup_guard(/* blowup_factor = */ 100.0);
  problem.run();

  std::cout << "--- theta=0, dt=" << delta_t << " summary ---" << std::endl;
  if (problem.blew_up())
    std::cout << "  Status: BLEW UP (E^n > 100*E^0) at t = "
              << problem.blowup_time() << std::endl;
  else
    std::cout << "  Status: completed to T_final = " << T_final
              << " without blow-up" << std::endl;

  report_energy_csv("./results/energy/" + csv_name);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  std::cout << "[Info] omega = pi*sqrt(2) = " << omega
            << " (homogeneous case, f == 0)" << std::endl;

  bool with_theta1 = false;
  bool with_theta0 = false;
  for (int i = 1; i < argc; ++i)
    {
      if (std::string(argv[i]) == "--with-theta1")
        with_theta1 = true;
      else if (std::string(argv[i]) == "--with-theta0")
        with_theta0 = true;
    }

  // Primary case: Crank-Nicolson (theta = 0.5).
  run_case(0.5, "energy_homogeneous.csv");

  // Optional comparison: implicit Euler (theta = 1.0).
  if (with_theta1)
    run_case(1.0, "energy_theta1.csv");

  // theta = 0 (fully explicit) is only conditionally stable: run at two
  // time steps to observe/quantify the expected instability.
  if (with_theta0)
    {
      run_theta0_case(0.005, "energy_theta0_dt005.csv");
      run_theta0_case(0.05, "energy_theta0_dt05.csv");
    }

  return 0;
}
