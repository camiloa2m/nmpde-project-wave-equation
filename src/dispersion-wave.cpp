#include "WaveEquation.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Dispersion study: drive the solver with the exact plane wave
// u_ex(x,y,t) = cos(kx*x + ky*y - omega*t) via Dirichlet BCs, then measure
// how far u_h drifts from it in phase. Since u_ex solves the homogeneous
// equation exactly, any mismatch is pure numerical dispersion.
//
// Two directions, same k = 8*pi: axis (kx=k,ky=0) and diagonal (kx=ky=k/sqrt(2)).
//
// Phase error: find the zero-crossing of u_h closest to the domain center
// along a line in the propagation direction, compare to the exact crossing,
// convert the offset to radians via 2*pi*offset/wavelength. T_final has to
// stay small enough that the accumulated error is under pi, or the
// crossing search locks onto the wrong zero and the result jumps around.

static constexpr unsigned int dim = WaveEquation::dim;

static const double k = 8.0 * M_PI;

static double kx = 0.0;
static double ky = 0.0;
static double omega = 0.0;

double
WaveEquation::FunctionU0::value(const Point<dim> &p, const unsigned int) const
{
  return std::cos(kx * p[0] + ky * p[1]);
}

double
WaveEquation::FunctionV0::value(const Point<dim> &p, const unsigned int) const
{
  return omega * std::sin(kx * p[0] + ky * p[1]);
}

double
plane_wave_u(const Point<dim> &p, const double t)
{
  return std::cos(kx * p[0] + ky * p[1] - omega * t);
}

double
plane_wave_dudt(const Point<dim> &p, const double t)
{
  return omega * std::sin(kx * p[0] + ky * p[1] - omega * t);
}

// Sanity check: u_tt_ex - Laplacian(u_ex) should be 0 (f=0) everywhere.
void
check_plane_wave_residual(const double kx_, const double ky_, const double omega_)
{
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> time_dist(0.0, 1.0);

  double max_residual = 0.0;
  for (int i = 0; i < 20; ++i)
    {
      const double x = unit(rng), y = unit(rng), t = time_dist(rng);
      const double phase = kx_ * x + ky_ * y - omega_ * t;

      const double u_tt_ex     = -omega_ * omega_ * std::cos(phase);
      const double laplacian_u = -(kx_ * kx_ + ky_ * ky_) * std::cos(phase);
      const double residual    = u_tt_ex - laplacian_u;

      max_residual = std::max(max_residual, std::abs(residual));
    }

  std::cout << "Residual check (kx=" << kx_ << ", ky=" << ky_ << ", omega=" << omega_
            << "): max|u_tt - Laplacian(u)| over 20 random points = "
            << max_residual << std::endl;
  AssertThrow(max_residual < 1e-8,
              ExcMessage("Plane wave does not satisfy the wave equation: "
                         "check that omega^2 == kx^2 + ky^2."));
}

class DispersionProblem : public WaveEquation
{
public:
  using WaveEquation::WaveEquation;

  void
  run_to(const double T_final)
  {
    setup();
    assemble_matrices();
    while (time < T_final - 0.5 * delta_t)
      {
        time += delta_t;
        ++timestep_number;
        solve_timestep();
      }
  }

  // Each rank contributes solution(p) if it owns p, 0 otherwise; MPI sum
  // gives the right value everywhere without throwing on ranks that don't.
  double
  sample(const Point<dim> &p) const
  {
    double local_value = 0.0;
    try
      {
        local_value = VectorTools::point_value(*mapping, dof_handler, solution, p);
      }
    catch (const VectorTools::ExcPointNotAvailableHere &)
      {
        local_value = 0.0;
      }
    return Utilities::MPI::sum(local_value, MPI_COMM_WORLD);
  }

  double
  current_time() const
  {
    return time;
  }
};

double
nearest_zero_crossing(const std::vector<double> &coords,
                       const std::vector<double> &values,
                       const double                coord_center)
{
  double best_coord = std::numeric_limits<double>::quiet_NaN();
  double best_dist   = std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i + 1 < values.size(); ++i)
    {
      if (values[i] == 0.0 || (values[i] > 0.0) != (values[i + 1] > 0.0))
        {
          const double t = values[i] / (values[i] - values[i + 1]);
          const double crossing = coords[i] + t * (coords[i + 1] - coords[i]);
          const double dist = std::abs(crossing - coord_center);
          if (dist < best_dist)
            {
              best_dist  = dist;
              best_coord = crossing;
            }
        }
    }

  AssertThrow(!std::isnan(best_coord),
              ExcMessage("No zero-crossing found along the sampling line."));
  return best_coord;
}

struct SamplingLine
{
  std::vector<Point<dim>> points;
  std::vector<double>     coords;
  double                  coord_center;
};

// Axis: y=0.5, parametrized by x. Diagonal: y=x, parametrized by arclength
// s = x*sqrt(2), so coords always means distance along the propagation
// direction.
SamplingLine
make_sampling_line(const bool diagonal, const unsigned int n_samples = 4001)
{
  SamplingLine line;
  line.points.reserve(n_samples);
  line.coords.reserve(n_samples);

  const double eps = 1e-9;
  for (unsigned int i = 0; i < n_samples; ++i)
    {
      const double s = eps + (1.0 - 2.0 * eps) * static_cast<double>(i) /
                              static_cast<double>(n_samples - 1);
      if (!diagonal)
        {
          line.points.emplace_back(s, 0.5);
          line.coords.push_back(s);
        }
      else
        {
          line.points.emplace_back(s, s);
          line.coords.push_back(s * std::sqrt(2.0));
        }
    }
  line.coord_center = diagonal ? 0.5 * std::sqrt(2.0) : 0.5;
  return line;
}

double
measure_phase_error(const bool         diagonal,
                     const unsigned int n_subdivisions,
                     const double       delta_t,
                     const double       T_final)
{
  const auto rho = [](const Point<dim> &) { return 1.0; };
  const auto c   = [](const Point<dim> &) { return 1.0; };
  const auto f   = [](const Point<dim> &, const double &) { return 0.0; };
  const auto g   = [](const Point<dim> &p, const double &t) { return plane_wave_u(p, t); };
  const auto dg  = [](const Point<dim> &p, const double &t) { return plane_wave_dudt(p, t); };

  const double theta = 0.5; // Crank-Nicolson

  DispersionProblem problem("dispersion", /*r=*/1, T_final, delta_t, theta,
                             rho, c, f, n_subdivisions,
                             "./results-dispersion", /*enable_output=*/false,
                             g, dg);
  problem.run_to(T_final);
  const double t_actual = problem.current_time();

  const SamplingLine line = make_sampling_line(diagonal);

  std::vector<double> uh_values(line.points.size());
  for (std::size_t i = 0; i < line.points.size(); ++i)
    uh_values[i] = problem.sample(line.points[i]);

  std::vector<double> uex_values(line.points.size());
  for (std::size_t i = 0; i < line.points.size(); ++i)
    uex_values[i] = plane_wave_u(line.points[i], t_actual);

  const double crossing_h   = nearest_zero_crossing(line.coords, uh_values, line.coord_center);
  const double crossing_ex  = nearest_zero_crossing(line.coords, uex_values, line.coord_center);

  const double wavelength = 2.0 * M_PI / k;
  const double position_offset = crossing_h - crossing_ex;
  return 2.0 * M_PI * (position_offset / wavelength);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // argv[1]: "spatial", "temporal", "cancellation", or "all" (default).
  std::string experiment = "all";
  if (argc > 1)
    experiment = argv[1];
  AssertThrow(experiment == "all" || experiment == "spatial" ||
              experiment == "temporal" || experiment == "cancellation",
              ExcMessage("Unknown experiment '" + experiment +
                         "'. Use 'spatial', 'temporal', 'cancellation', or 'all'."));
  const bool run_spatial      = (experiment == "all" || experiment == "spatial");
  const bool run_temporal     = (experiment == "all" || experiment == "temporal");
  const bool run_cancellation = (experiment == "all" || experiment == "cancellation");

  const double wavelength = 2.0 * M_PI / k;

  const double kx_axis = k, ky_axis = 0.0;
  const double kx_diag = k / std::sqrt(2.0), ky_diag = k / std::sqrt(2.0);
  const double omega_common = k;

  std::cout << "k = 8*pi = " << k << ", wavelength = " << wavelength
            << ", omega = " << omega_common << std::endl;

  check_plane_wave_residual(kx_axis, ky_axis, omega_common);
  check_plane_wave_residual(kx_diag, ky_diag, omega_common);

  std::cout << "Points per wavelength (2*pi/(k*h)) for each mesh used in the "
            << "spatial sweep:" << std::endl;
  for (const unsigned int n : {40u, 50u, 70u, 100u})
    {
      const double h = 1.0 / n;
      std::cout << "  n=" << n << "  h=" << h
                << "  ppw=" << (2.0 * M_PI / (k * h)) << std::endl;
    }
  std::cout << "  n=20 and n=30 are left out: ppw=5.0 and 7.5, below the "
            << "~8 ppw we want even at the coarsest level." << std::endl;

  const double period = 2.0 * M_PI / omega_common;

  // theta=0.5 + P1 dispersion: relative phase-speed error ~ C*(k*h)^2
  // (space) + C*(omega*dt)^2 (time), phase_error ~ omega*T*relative_error.
  // C_diag_spatial=1/24 instead of the 4-fold-symmetric value 5/24 because
  // subdivided_hyper_cube_with_simplices always cuts along the same
  // diagonal; this predicts C_diag/C_axis=0.5, matching the measured 0.503.
  // There's still an unexplained ~0.48x gap between theory and measurement
  // in both directions.
  const double C_axis_spatial  = 1.0 / 12.0;
  const double C_diag_spatial  = 1.0 / 24.0;
  const double C_temporal      = 1.0 / 24.0;

  // pi/8 instead of pi/4: at k*h ~ 0.6 (coarsest mesh) the leading-order
  // estimate undershoots the real error, so use extra margin.
  const double safe_phase_budget = M_PI / 8.0;

  // Experiment 1: spatial dispersion vs. mesh resolution, both directions.
  if (run_spatial)
  {
    const double delta_t = 0.001;
    const double T_final = 0.5 * period; // worst case (n=40, axis) ~0.10 rad, well under budget

    {
      const double h_finest = 1.0 / 100;
      const double temporal_phase_error =
        C_temporal * std::pow(omega_common * delta_t, 2) * omega_common * T_final;
      const double spatial_phase_error_axis =
        C_axis_spatial * std::pow(k * h_finest, 2) * omega_common * T_final;
      const double spatial_phase_error_diag =
        C_diag_spatial * std::pow(k * h_finest, 2) * omega_common * T_final;

      std::cout << "Experiment 1, dt=" << delta_t
                << ": checking margin at the finest mesh (n=100, h=" << h_finest
                << ")" << std::endl;
      std::cout << "  predicted temporal phase error           = "
                << temporal_phase_error << " rad" << std::endl;
      std::cout << "  predicted spatial phase error (axis)     = "
                << spatial_phase_error_axis << " rad  ("
                << spatial_phase_error_axis / temporal_phase_error << "x the temporal error)" << std::endl;
      std::cout << "  predicted spatial phase error (diagonal) = "
                << spatial_phase_error_diag << " rad  ("
                << spatial_phase_error_diag / temporal_phase_error << "x the temporal error)" << std::endl;

      AssertThrow(spatial_phase_error_axis > 20.0 * temporal_phase_error &&
                  spatial_phase_error_diag > 20.0 * temporal_phase_error,
                  ExcMessage("Temporal phase error is not negligible (<20x smaller) "
                             "compared to the spatial phase error at the finest mesh; "
                             "use a smaller dt for Experiment 1."));
    }

    {
      const double h_coarsest = 1.0 / 40;
      const double worst_phase_error_axis =
        C_axis_spatial * std::pow(k * h_coarsest, 2) * omega_common * T_final;
      const double worst_phase_error_diag =
        C_diag_spatial * std::pow(k * h_coarsest, 2) * omega_common * T_final;
      const double worst_phase_error = std::max(worst_phase_error_axis, worst_phase_error_diag);
      std::cout << "Experiment 1, T_final=" << T_final << " (" << T_final / period
                << " periods): wraparound check at n=40 ("
                << (worst_phase_error_axis >= worst_phase_error_diag ? "axis" : "diagonal")
                << " is worse) gives a predicted phase error of "
                << worst_phase_error << " rad, budget is pi/8=" << safe_phase_budget
                << std::endl;
      AssertThrow(worst_phase_error < safe_phase_budget,
                  ExcMessage("Experiment 1 T_final is too close to the phase-wraparound "
                             "threshold at the worst-case mesh; use a smaller T_final."));
    }

    for (const bool diagonal : {false, true})
      {
        kx    = diagonal ? kx_diag : kx_axis;
        ky    = diagonal ? ky_diag : ky_axis;
        omega = omega_common;

        const std::string csv_name = diagonal ? "dispersion_spatial_diagonal.csv"
                                                : "dispersion_spatial_axis.csv";
        std::ofstream csv(csv_name);
        csv << "n_subdivisions,h,phase_error_radians\n";

        std::cout << "Running the spatial sweep, "
                  << (diagonal ? "diagonal" : "axis") << " direction:" << std::endl;

        for (const unsigned int n : {40u, 50u, 70u, 100u})
          {
            const double h = 1.0 / n;
            const double phase_error = measure_phase_error(diagonal, n, delta_t, T_final);
            std::cout << "  n=" << n << " h=" << h
                      << " phase_error_radians=" << phase_error << std::endl;
            csv << n << "," << h << "," << phase_error << "\n";
          }
        std::cout << "Saved " << csv_name << std::endl;
      }
  }

  // Experiment 2: temporal dispersion vs. dt, axis direction only.
  if (run_temporal)
  {
    kx    = kx_axis;
    ky    = ky_axis;
    omega = omega_common;

    const unsigned int n_finest = 100;

    const double T_final = 1.25 * period;
    const double temporal_safe_phase_budget = M_PI / 4.0; // looser than pi/8: estimate tracks measurement well here
    {
      const double dt_worst = 0.05;
      const double worst_phase_error =
        C_temporal * std::pow(omega_common * dt_worst, 2) * omega_common * T_final;
      std::cout << "Experiment 2, T_final=" << T_final << " (" << T_final / period
                << " periods): wraparound check at dt=" << dt_worst
                << " gives a predicted phase error of " << worst_phase_error
                << " rad, budget is pi/4=" << temporal_safe_phase_budget << std::endl;
      AssertThrow(worst_phase_error < temporal_safe_phase_budget,
                  ExcMessage("Experiment 2 T_final is too close to the phase-wraparound "
                             "threshold at the worst-case dt; use a smaller T_final."));
    }

    const std::string csv_name = "dispersion_temporal.csv";
    std::ofstream csv(csv_name);
    csv << "dt,phase_error_radians\n";

    std::cout << "Running the temporal sweep, axis direction, n=" << n_finest
              << ":" << std::endl;

    for (const double dt : {0.05, 0.025, 0.0125, 0.00625})
      {
        const double phase_error = measure_phase_error(/*diagonal=*/false, n_finest, dt, T_final);
        std::cout << "  dt=" << dt << " phase_error_radians=" << phase_error << std::endl;
        csv << dt << "," << phase_error << "\n";
      }
    std::cout << "Saved " << csv_name << std::endl;
  }

  // Experiment 3: cancellation between spatial and temporal error, both directions.
  if (run_cancellation)
  {
    const unsigned int n = 50;
    const double       h = 1.0 / n;

    const double dt_cancel_axis = h; // h*sqrt(12*C_spatial): where spatial/temporal errors cancel
    const double dt_cancel_diag = h / std::sqrt(2.0);

    std::cout << "Experiment 3 cancellation points: dt_cancel_axis=" << dt_cancel_axis
              << ", dt_cancel_diag=" << dt_cancel_diag << " (h=" << h << ")" << std::endl;

    const std::vector<double> ratios_axis = {0.5, 0.75, 1.0, 1.25, 1.5};
    const std::vector<double> ratios_diag = {0.35, 0.4, 0.5, 0.6, 0.707, 0.9, 1.1};

    // Bound = worse of the cancelling combination |sp-tp| and the larger
    // individual term, in case they don't cancel cleanly off-center.
    auto worst_bound = [&](const std::vector<double> &ratios, const double C_spatial) {
      double worst = 0.0;
      for (const double ratio : ratios)
        {
          const double dt = ratio * h;
          const double sp = C_spatial * std::pow(k * h, 2);
          const double tp = C_temporal * std::pow(omega_common * dt, 2);
          const double bound = std::max(std::abs(sp - tp), std::max(sp, tp));
          worst = std::max(worst, bound);
        }
      return worst;
    };

    // Separate T_final per direction: a shared value sized for diagonal
    // still caused wraparound in axis, whose coefficient is larger.
    const double T_final_axis = 0.75 * period;
    const double T_final_diag = 2.0 * period;

    for (const bool diagonal : {false, true})
      {
        kx    = diagonal ? kx_diag : kx_axis;
        ky    = diagonal ? ky_diag : ky_axis;
        omega = omega_common;

        const std::vector<double> &ratios = diagonal ? ratios_diag : ratios_axis;
        const double C_spatial = diagonal ? C_diag_spatial : C_axis_spatial;
        const double T_final = diagonal ? T_final_diag : T_final_axis;

        {
          const double worst = worst_bound(ratios, C_spatial);
          const double worst_phase_error = worst * omega_common * T_final;
          std::cout << "Experiment 3 (" << (diagonal ? "diagonal" : "axis")
                    << "), T_final=" << T_final << " (" << T_final / period
                    << " periods): wraparound check gives a predicted phase error of "
                    << worst_phase_error << " rad, budget is pi/8=" << safe_phase_budget
                    << std::endl;
          AssertThrow(worst_phase_error < safe_phase_budget,
                      ExcMessage("Experiment 3 (" + std::string(diagonal ? "diagonal" : "axis") +
                                 ") T_final is too close to the phase-wraparound threshold "
                                 "at the worst-case (dt,h) combination; use a smaller T_final."));
        }

        const std::string csv_name = diagonal ? "dispersion_cancellation_diagonal.csv"
                                                : "dispersion_cancellation_axis.csv";
        std::ofstream csv(csv_name);
        csv << "dt,h,dt_over_h,phase_error_radians\n";

        std::cout << "Running the cancellation sweep, "
                  << (diagonal ? "diagonal" : "axis") << " direction, n=" << n
                  << ":" << std::endl;

        for (const double ratio : ratios)
          {
            const double dt = ratio * h;
            const double phase_error = measure_phase_error(diagonal, n, dt, T_final);
            std::cout << "  dt/h=" << ratio << " dt=" << dt
                      << " phase_error_radians=" << phase_error << std::endl;
            csv << dt << "," << h << "," << ratio << "," << phase_error << "\n";
          }
        std::cout << "Saved " << csv_name << std::endl;
      }
  }

  return 0;
}