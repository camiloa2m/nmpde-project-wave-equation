#ifndef WAVE_EQUATION_HPP
#define WAVE_EQUATION_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using namespace dealii;

/**
 * Class managing the second-order wave equation problem.
 */
class WaveEquation
{
public:
  // Physical dimension (1D, 2D, 3D). Set to 2 for a 2D wave problem.
  static constexpr unsigned int dim = 2;

  // Initial displacement: u(x,y,0) = u_0(x,y)
  class FunctionU0 : public Function<dim>
  {
  public:
    FunctionU0() = default;
    virtual double
    value(const Point<dim> &p, const unsigned int = 0) const override;
  };

  // Initial velocity: du/dt(x,y,0) = v_0(x,y)
  class FunctionV0 : public Function<dim>
  {
  public:
    FunctionV0() = default;
    virtual double
    value(const Point<dim> &p, const unsigned int = 0) const override;
  };

  // Wrap a space-time lambda into a dealii::Function (time read via get_time()).
  class LambdaFunction : public Function<dim>
  {
  public:
    std::function<double(const Point<dim>&, double)> fn;
    LambdaFunction(std::function<double(const Point<dim>&, double)> f)
      : Function<dim>(), fn(std::move(f)) {}
    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return fn(p, this->get_time());
    }
  };

  // boundary_g_/boundary_dgdt_ prescribe a (possibly time-dependent)
  // Dirichlet condition U|_boundary = g(x,t), V|_boundary = dg/dt(x,t).
  // Left null (default), this falls back to homogeneous Dirichlet (g = 0).
  WaveEquation(const std::string                                               &mesh_file_name_,
               const unsigned int                                              &r_,
               const double                                                    &T_,
               const double                                                    &delta_t_,
               const double                                                    &theta_,
               const std::function<double(const Point<dim> &)>                 &rho_,
               const std::function<double(const Point<dim> &)>                 &c_,
               const std::function<double(const Point<dim> &, const double &)> &f_,
               const unsigned int                                               n_subdivisions_ = 50,
               const std::string                                               &output_dir_     = "./results",
               const bool                                                       enable_output_  = true,
               const std::function<double(const Point<dim> &, const double &)>  boundary_g_     = nullptr,
               const std::function<double(const Point<dim> &, const double &)>  boundary_dgdt_  = nullptr)
    : mesh_file_name(mesh_file_name_)
    , r(r_)
    , T(T_)
    , delta_t(delta_t_)
    , theta(theta_)
    , rho(rho_)
    , c(c_)
    , f(f_)
    , n_subdivisions(n_subdivisions_)
    , output_dir(output_dir_)
    , enable_output(enable_output_)
    , boundary_g(boundary_g_)
    , boundary_dgdt(boundary_dgdt_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , computing_timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    , mesh(MPI_COMM_WORLD)
  {}

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                const Function<dim>         &exact_solution) const;

  // Energy-norm error against exact u and v = du/dt, same rho/c
  // weighting as compute_energy().
  double
  compute_energy_norm_error(const Function<dim> &exact_u,
                             const Function<dim> &exact_v) const;

  // Run the time-dependent simulation.
  void
  run();

  // True if the blow-up guard stopped run() early instead of reaching T.
  bool
  blew_up() const { return blowup_triggered; }

  // Time the blow-up guard triggered. Meaningless unless blew_up() is true.
  double
  blowup_time() const { return blowup_time_value; }

  // Log energy per timestep to output_dir/csv_name (columns: t,energy).
  // Call before run(); at the end run() also prints E^0, max/min energy
  // and the relative variation.
  void
  enable_energy_log(const std::string &csv_name);

  // Stop run() early if E^n > blowup_factor * E^0, for unstable schemes
  // (e.g. theta=0). Call before run().
  void
  enable_blowup_guard(const double blowup_factor);

protected:
  // Initialization of mesh, FE space, DoFs, and linear algebra objects.
  void
  setup();

  // Builds M and K once
  void
  assemble_matrices();

  // Solves for U_{n+1} and V_{n+1} at the next timestep.
  void
  solve_timestep();

  // Computes and prints the total mechanical energy E_n, logging it to
  // the CSV file if enabled.
  double
  compute_energy();

  // Output to VTU/PVTU.
  void
  output() const;

  // Parameters
  const std::string mesh_file_name;
  const unsigned int r;
  const double T;
  const double delta_t;
  const double theta;

  // Physical properties
  std::function<double(const Point<dim> &)> rho;
  std::function<double(const Point<dim> &)> c;
  std::function<double(const Point<dim> &, const double &)> f;

  const unsigned int n_subdivisions;
  const std::string output_dir;
  // If false, run() skips VTU/PVTU writes (used in convergence studies).
  const bool enable_output;

  // Time-dependent Dirichlet boundary data: U|_boundary = g(x,t),
  // V|_boundary = dg/dt(x,t). Null means homogeneous Dirichlet (g = 0).
  std::function<double(const Point<dim> &, const double &)> boundary_g;
  std::function<double(const Point<dim> &, const double &)> boundary_dgdt;

  double time = 0.0;
  unsigned int timestep_number = 0;

  // MPI and Output
  const unsigned int mpi_size;
  const unsigned int mpi_rank;
  ConditionalOStream pcout;
  // Per-rank timer (no MPI synchronization). Report min/avg/max 
  // with print_wall_time_statistics(MPI_COMM_WORLD) after run.
  TimerOutput computing_timer;

  // FE structure
  parallel::fullydistributed::Triangulation<dim> mesh;
  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>> quadrature;
  std::unique_ptr<Mapping<dim>> mapping;
  DoFHandler<dim> dof_handler;

  // Linear Algebra Objects
  AffineConstraints<double> constraints;
  TrilinosWrappers::SparseMatrix mass_matrix;      // M
  TrilinosWrappers::SparseMatrix stiffness_matrix; // K
  TrilinosWrappers::SparseMatrix matrix_u;         // M + theta^2*dt^2*K  (assembled once, with Dirichlet BCs)
  TrilinosWrappers::SparseMatrix matrix_v;         // M                   (assembled once, with Dirichlet BCs)

  // Kinematic state vectors (owned for solving, full for evaluation/output)
  TrilinosWrappers::MPI::Vector solution_owned, solution;           // U_{n+1}
  TrilinosWrappers::MPI::Vector old_solution_owned, old_solution;   // U_n
  TrilinosWrappers::MPI::Vector velocity_owned, velocity;           // V_{n+1}
  TrilinosWrappers::MPI::Vector old_velocity_owned, old_velocity;   // V_n

private:
  // Sets rhs(dof) = value_fn(point, t) for every (dof, point) pair. Used to
  // impose a Dirichlet value at DOFs that already have an identity row in
  // the system matrix, so the RHS value IS the solution value there.
  // Falls back to 0 (homogeneous Dirichlet) if value_fn is null.
  static void
  apply_dirichlet_value(const std::vector<std::pair<types::global_dof_index, Point<dim>>> &dofs,
                         const std::function<double(const Point<dim> &, const double &)>   &value_fn,
                         const double                                                        t,
                         TrilinosWrappers::MPI::Vector                                       &rhs);

  // Temporary vectors for assembly and solving
  TrilinosWrappers::MPI::Vector rhs_owned;
  TrilinosWrappers::MPI::Vector tmp_owned;
  TrilinosWrappers::MPI::Vector force_terms;

  // Locally-owned Dirichlet-boundary DOFs paired with their support point,
  // so solve_timestep() can re-evaluate boundary_g/boundary_dgdt at the
  // right location each step without searching the boundary again.
  std::vector<std::pair<types::global_dof_index, Point<dim>>> boundary_dofs;

  // Energy CSV logging state, see enable_energy_log().
  bool energy_log_enabled = false;
  std::string energy_log_path;
  std::ofstream energy_log_stream;
  double energy_initial = 0.0;
  double energy_max = -std::numeric_limits<double>::infinity();
  double energy_min = std::numeric_limits<double>::infinity();

  // Blow-up guard state, see enable_blowup_guard().
  bool blowup_guard_enabled = false;
  double blowup_factor = 0.0;
  bool blowup_triggered = false;
  double blowup_time_value = 0.0;
};

#endif